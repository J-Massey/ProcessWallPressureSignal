import h5py
import numpy as np
from scipy.signal import welch, csd, get_window
import scipy.io as sio

from icecream import ic
from pathlib import Path

from scipy.signal import butter, sosfiltfilt
import torch
from tqdm import tqdm

from wiener_filter_torch import wiener_cancel_background_torch

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import scienceplots
plt.style.use(["science", "grid"])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")

from models import channel_model, bl_model

############################
# Constants & defaults
############################
FS = 50_000.0
NPERSEG = 2**14
WINDOW = "hann"
TRIM_CAL_SECS = 5  # seconds trimmed from the start of calibration runs (0 to disable)

nc_colour = '#1f77b4'  # matplotlib default blue
ph1_colour = "#c76713"  # matplotlib default orange
ph2_colour = "#9fda16"  # matplotlib default red
nkd_colour = '#2ca02c' # matplotlib default green

# --- constants (keep once, top of file) ---
R = 287.05        # J/kg/K
PSI_TO_PA = 6_894.76
P_ATM = 101_325.0
DELTA = 0.035  # m, bl-height of 'channel'
TDEG = [18, 20, 22]

TPLUS_CUT = 10  # picked so that we cut at half the inner peak

CHAIN_SENS_V_PER_PA = {
    'PH1': 50.9e-3,  # V/Pa
    'PH2': 51.7e-3,  # V/Pa
    'NC':  52.4e-3,  # V/Pa
}


def bandpass_filter(data, fs, f_low, f_high, order=4):
    sos = butter(order, [f_low, f_high], btype='band', fs=fs, output='sos')
    filtered = sosfiltfilt(sos, data)
    # protect against NaNs and negatives
    filtered = np.nan_to_num(filtered, nan=0.0)
    # filtered[filtered < 0] = 0
    return filtered

def volts_to_pa(x_volts: np.ndarray, channel: str, f_cut: float) -> np.ndarray:
    sens = CHAIN_SENS_V_PER_PA[channel]  # V/Pa
    # Band pass filter 0.1-f_cut Hz
    x_volts = bandpass_filter(x_volts, FS, 0.1, f_cut)
    return x_volts / sens

def f_cut_from_Tplus(u_tau: float, nu: float, Tplus_cut: float = TPLUS_CUT):
    return (u_tau**2 / nu) / Tplus_cut


def air_props_from_gauge(psi_gauge: float, T_K: float):
    """
    Return rho [kg/m^3], mu [Pa·s], nu [m^2/s] from gauge pressure [psi] and temperature [K].
    Sutherland's law for mu; nu = mu/rho.
    """
    p_abs = P_ATM + psi_gauge * PSI_TO_PA
    # Sutherland's
    mu0, T0, S = 1.716e-5, 273.15, 110.4
    mu = mu0 * (T_K/T0)**1.5 * (T0 + S)/(T_K + S)
    rho = p_abs / (R * T_K)
    nu = mu / rho
    return rho, mu, nu


def compute_spec(fs: float, x: np.ndarray, npsg : int = NPERSEG):
    """Welch PSD with sane defaults and shape guarding. Returns (f [Hz], Pxx [Pa^2/Hz])."""
    x = np.asarray(x, float)
    nseg = int(min(npsg, x.size))
    nov = nseg // 2
    w = get_window(WINDOW, nseg, fftbins=True)
    f, Pxx = welch(
        x,
        fs=fs,
        window=w,
        nperseg=nseg,
        noverlap=nov,
        detrend="constant",
        scaling="density",
        return_onesided=True,
    )
    return f, Pxx


def clean_raw(
    mat_path: str,
    psi_gauge: float,   # psig for this run
    T_K: float,         # Kelvin for this run
    u_tau: float,       # m/s for this run (your measured/estimated)
    field="channelData",
    out_stub="run",
    plot_flag=True
):
    # --- gas props & scales ---
    rho, mu, nu = air_props_from_gauge(psi_gauge, T_K)
    Re_tau = u_tau * 0.035 / nu              # if 0.035 m is your half-height
    f_cut  = f_cut_from_Tplus(u_tau, nu)     # matches your scaling
    Tplus_cut = (u_tau**2 / nu) / f_cut      # should equal TPLUS_CUT

    # --- load ---
    dat = sio.loadmat(mat_path)
    X = dat[field]  # shape [N, >=3], columns: 0=PH1,1=PH2,2=NKD as per your code
    ph1_V, ph2_V, nkd_V = X[:,0], X[:,1], X[:,2]

    # --- Volts -> Pa using fixed chain sensitivities ---
    ph1 = volts_to_pa(ph1_V, 'PH1', f_cut)
    ph1 -= ph1.mean()
    ph2 = volts_to_pa(ph2_V, 'PH2', f_cut)
    ph2 -= ph2.mean()
    nkd = volts_to_pa(nkd_V, 'NC', f_cut)
    nkd -= nkd.mean()

    # --- take out the mean ---
    ph1 -= np.mean(ph1)
    ph2 -= np.mean(ph2)
    nkd -= np.mean(nkd)

    # --- apply a band pass filter between 0.1 Hz and f_cut ---
    from scipy.signal import butter, sosfiltfilt
    def bandpass_filter(data, fs, f_low, f_high, order=3):
        sos = butter(order, [f_low, f_high], btype='band', fs=fs, output='sos')
        filtered = sosfiltfilt(sos, data)
        # protect against NaNs and negatives
        filtered = np.nan_to_num(filtered, nan=0.0)
        # filtered[filtered < 0] = 0
        return filtered
    
    ph1 = bandpass_filter(ph1, FS, 1, f_cut)
    ph2 = bandpass_filter(ph2, FS, 1, f_cut)
    nkd = bandpass_filter(nkd, FS, 1, f_cut)


    # --- spectra ---
    f, P_ph1 = compute_spec(FS, ph1)
    _, P_ph2 = compute_spec(FS, ph2)
    _, P_nkd = compute_spec(FS, nkd)

    # premultiplied, dimensionless: f * Φ_pp / (ρ^2 u_τ^4)
    prem = lambda Pf: (f * Pf) / (rho**2 * u_tau**4)

    # --- optional plotting (mirrors your style) ---
    if plot_flag:
        T_plus = (u_tau**2 / nu) / f
        g1, g2, rv   = bl_model(T_plus, Re_tau, 2*(u_tau**2)/14.0**2)
        g1c,g2c,rv_c = channel_model(T_plus, Re_tau, u_tau, 14)

        fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, tight_layout=True)
        fig.suptitle(out_stub + fr" $\delta^+ \approx$  ({Re_tau:.0f})")

        ax[0].semilogx(f, prem(P_nkd), label='NC', color=nkd_colour)
        ax[0].semilogx(f, prem(P_ph1), label='PH1', color=ph1_colour)
        ax[0].semilogx(f, prem(P_ph2), label='PH2', color=ph2_colour)
        ax[0].semilogx(f, rv*(g1+g2),      label='BL Model', color='k', linestyle='--')
        ax[0].semilogx(f, rv_c*(g1c+g2c),  label='Channel Model', color='k', linestyle='-.')
        ax[0].axvline(f_cut, color='red', linestyle='--')
        ax[0].grid(True, which='both', linestyle=':', linewidth=0.3, alpha=0.7)
        ax[0].set_xlabel("$f$ [Hz]"); ax[0].set_ylabel(r"${f \phi_{pp}}^+$")
        ax[0].set_xlim(1, 1e4); ax[0].set_ylim(0, 9)

        ax[1].semilogx(T_plus, prem(P_nkd), label='NC', color=nkd_colour)
        ax[1].semilogx(T_plus, prem(P_ph1), label='PH1', color=ph1_colour)
        ax[1].semilogx(T_plus, prem(P_ph2), label='PH2', color=ph2_colour)
        ax[1].semilogx(T_plus, rv*(g1+g2),     label='BL Model', color='k', linestyle='--')
        ax[1].semilogx(T_plus, rv_c*(g1c+g2c), label='Channel Model', color='k', linestyle='-.')
        ax[1].axvline(Tplus_cut, color='red', linestyle='--')
        ax[1].grid(True, which='both', linestyle=':', linewidth=0.3, alpha=0.7)
        ax[1].set_xlabel("$T^+$")

        ax[0].legend(); ax[1].legend()
        out_png = f'figures/final/{out_stub}.png'
        fig.savefig(out_png, dpi=410)

    # --- Wiener clean, save HDF5 with consistent metadata ---
    ph1_clean = wiener_cancel_background_torch(ph1, nkd, FS).cpu().numpy()
    ph2_clean = wiener_cancel_background_torch(ph2, nkd, FS).cpu().numpy()
    torch.cuda.empty_cache()
    
    with h5py.File(f'data/final_cleaned/{out_stub}_cleaned.h5', 'w') as hf:
        hf.create_dataset('ph1_clean', data=ph1_clean)
        hf.create_dataset('ph2_clean', data=ph2_clean)
        hf.attrs['rho']   = rho
        hf.attrs['mu']    = mu
        hf.attrs['nu']    = nu
        hf.attrs['u_tau'] = u_tau
        hf.attrs['Re_tau']= Re_tau
        hf.attrs['FS']    = FS
        hf.attrs['f_cut'] = f_cut
        hf.attrs['Tplus_cut'] = Tplus_cut
        hf.attrs['delta'] = 0.035
        hf.attrs['psi_gauge'] = psi_gauge
        hf.attrs['T_K']       = T_K
        hf.attrs['cf_2']      = 2*(u_tau**2)/14.0**2

    # return useful bits if you want to overlay/compare
    return dict(f=f, P_ph1=P_ph1, P_ph2=P_ph2, P_nkd=P_nkd,
                rho=rho, nu=nu, Re_tau=Re_tau, f_cut=f_cut)


def run_all_final():
    # --- ATM ---
    clean_raw(
        mat_path='data/20251031/close/0psig.mat',
        psi_gauge=0.0,
        T_K=273.15 + TDEG[0],
        u_tau=0.537,
        out_stub='0psig_close',
        plot_flag=True
    )

    # --- 50 psig ---
    clean_raw(
        mat_path='data/20251031/close/50psig.mat',
        psi_gauge=50.0,
        T_K=273.15 + TDEG[1],
        u_tau=0.522,
        out_stub='50psig_close',
        plot_flag=True
    )

    # --- 100 psig ---
    clean_raw(
        mat_path='data/20251031/close/100psig.mat',
        psi_gauge=100.0,
        T_K=273.15 + TDEG[2],
        u_tau=0.506,
        out_stub='100psig_close',
        plot_flag=True
    )

    clean_raw(
        mat_path='data/20251031/far/0psig.mat',
        psi_gauge=0.0,
        T_K=273.15 + TDEG[0],
        u_tau=0.537,
        out_stub='0psig_far',
        plot_flag=True
    )

    # --- 50 psig ---
    clean_raw(
        mat_path='data/20251031/far/50psig.mat',
        psi_gauge=50.0,
        T_K=273.15 + TDEG[1],
        u_tau=0.522,
        out_stub='50psig_far',
        plot_flag=True
    )

    # --- 100 psig ---
    clean_raw(
        mat_path='data/20251031/far/100psig.mat',
        psi_gauge=100.0,
        T_K=273.15 + TDEG[2],
        u_tau=0.506,
        out_stub='100psig_far',
        plot_flag=True
    )

if __name__ == "__main__":
    run_all_final()