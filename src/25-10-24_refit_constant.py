import h5py
import numpy as np
from scipy.signal import welch, csd, get_window
import scipy.io as sio
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import scienceplots
from icecream import ic
from pathlib import Path


import torch
from tqdm import tqdm

from wiener_filter_torch import wiener_cancel_background_torch

plt.style.use(["science", "grid"])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")

############################
# Constants & defaults
############################
FS = 50_000.0
NPERSEG = 2**17
WINDOW = "hann"
TRIM_CAL_SECS = 5  # seconds trimmed from the start of calibration runs (0 to disable)

nc_colour = '#1f77b4'  # matplotlib default blue
ph1_colour = "#c76713"  # matplotlib default orange
ph2_colour = "#9fda16"  # matplotlib default red
nkd_colour = '#2ca02c' # matplotlib default green

DEFAULT_UNITS = {
    'channelData_300_plug': ('Pa', 'Pa'),  # PH, NKD
    'channelData_300_nose': ('Pa', 'Pa'),  # NKD, NC
    'channelData_300':      ('Pa', 'Pa'),  # NC,  PH
}

# --- constants (keep once, top of file) ---
R = 287.05        # J/kg/K
PSI_TO_PA = 6_894.76
P_ATM = 101_325.0
DELTA = 0.035  # m, bl-height of 'channel'
TDEG = [18, 20, 22]

TPLUS_CUT = 10  # picked so that we cut at half the inner peak

CHAIN_SENS_V_PER_PA = 51.7  # V/Pa  

def volts_to_pa(x_volts: np.ndarray) -> np.ndarray:
    sens = CHAIN_SENS_V_PER_PA  # V/Pa
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


def bl_model(Tplus, Re_tau: float, cf_2: float) -> np.ndarray:
    A1 = 2.2
    sig1 = 3.9
    mean_Tplus = 20
    A2 = 1.4 * (np.log10(Re_tau) - 2.2)
    sig2 = 1.2
    mean_To = 0.82
    r1 = 0.5
    r2 = 7
    rv = np.exp(r1 * Tplus)/(np.exp(r1*r2) + np.exp(r1 * Tplus)) # correct
    rv = np.nan_to_num(rv, nan=1)  # replace NaNs with 0
    mean_To_plus = mean_To * Re_tau * np.sqrt(cf_2)
    g1 = A1 * np.exp(-sig1 * (np.log10(Tplus) - np.log10(mean_Tplus))**2)
    g2 = A2 * np.exp(-sig2 * (np.log10(Tplus) - np.log10(mean_To_plus))**2)
    return g1, g2, rv

def channel_model(Tplus, Re_tau: float, u_tau: float, u_cl) -> np.ndarray:
    A1 = 2.1*(1 - 100/Re_tau)
    sig1 = 4.4
    mean_Tplus = 12
    A2 = 0.9 * (np.log10(Re_tau) - 2.2)
    sig2 = 1.0
    mean_To = 0.6
    r1 = 0.5
    r2 = 3
    rv = np.exp(r1 * Tplus)/(np.exp(r1*r2) + np.exp(r1 * Tplus)) # correct
    rv = np.nan_to_num(rv, nan=1)  # replace NaNs with 0
    mean_To_plus = mean_To * Re_tau * u_tau/u_cl
    g1 = A1 * np.exp(-sig1 * (np.log10(Tplus) - np.log10(mean_Tplus))**2)
    g2 = A2 * np.exp(-sig2 * (np.log10(Tplus) - np.log10(mean_To_plus))**2)
    return g1, g2, rv


def estimate_frf(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    window: str = WINDOW,
    detrend: str = "false",
    npsg: int = NPERSEG,
):
    """
    Estimate H1 FRF and magnitude-squared coherence using Welch/CSD.

    Returns
    -------
    f : array_like [Hz]
    H : array_like (complex) = S_yx / S_xx  (x → y)
    gamma2 : array_like in [0, 1]
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    nseg = int(min(npsg, x.size, y.size))
    if nseg < 8:
        raise ValueError(f"Signal too short for FRF: n={min(x.size, y.size)}")
    nov = int(min(npsg // 2, nseg // 2))
    w = get_window(window, nseg, fftbins=True)

    f, Sxx = welch(x, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)
    _, Syy = welch(y, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)
    # SciPy convention: csd(x, y) = E{ X * conj(Y) }
    _, Sxy = csd(x, y, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)  # x→y

    H = np.conj(Sxy) / Sxx               # H1 = Syx / Sxx = conj(Sxy)/Sxx
    gamma2 = (np.abs(Sxy) ** 2) / (Sxx * Syy)
    gamma2 = np.clip(gamma2.real, 0.0, 1.0)
    return f, H, gamma2


def analyse_run(
    mat_path: str,
    psi_gauge: float,   # psig for this run
    T_K: float,         # Kelvin for this run
    u_tau: float,       # m/s for this run (your measured/estimated)
    field="channelData_LP",
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
    ph1 = volts_to_pa(ph1_V)
    ph2 = volts_to_pa(ph2_V)
    nkd = volts_to_pa(nkd_V)

    # --- take out the mean ---
    ph1 -= np.mean(ph1)
    ph2 -= np.mean(ph2)
    nkd -= np.mean(nkd)

    # --- apply a band pass filter between 0.1 Hz and f_cut ---
    from scipy.signal import butter, sosfiltfilt
    def bandpass_filter(data, fs, f_low, f_high, order=4):
        sos = butter(order, [f_low, f_high], btype='band', fs=fs, output='sos')
        filtered = sosfiltfilt(sos, data)
        # protect against NaNs and negatives
        filtered = np.nan_to_num(filtered, nan=0.0)
        # filtered[filtered < 0] = 0
        return filtered
    
    ph1 = bandpass_filter(ph1, FS, 0.1, f_cut)
    ph2 = bandpass_filter(ph2, FS, 0.1, f_cut)
    nkd = bandpass_filter(nkd, FS, 0.1, f_cut)


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
        ax[0].set_xlim(1, 1e4); ax[0].set_ylim(0, 7)

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
    
    with h5py.File(f'data/{out_stub}_cleaned.h5', 'w') as hf:
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

    # return useful bits if you want to overlay/compare
    return dict(f=f, P_ph1=P_ph1, P_ph2=P_ph2, P_nkd=P_nkd,
                rho=rho, nu=nu, Re_tau=Re_tau, f_cut=f_cut)

def run_all_final():
    # --- ATM ---
    analyse_run(
        mat_path='data/20251024/final/atm.mat',
        psi_gauge=0.0,
        T_K=273.15 + TDEG[0],
        u_tau=0.58,
        out_stub='0psig',
        plot_flag=True
    )

    # --- 50 psig ---
    analyse_run(
        mat_path='data/20251024/final/50psig.mat',
        psi_gauge=50.0,
        T_K=273.15 + TDEG[1],
        u_tau=0.47,
        out_stub='50psig',
        plot_flag=True
    )

    # --- 100 psig ---
    analyse_run(
        mat_path='data/20251024/final/100psig.mat',
        psi_gauge=100.0,
        T_K=273.15 + TDEG[2],
        u_tau=0.52,
        out_stub='100psig',
        plot_flag=True
    )


def plot_model_comparison():
    fn_atm = '0psig_cleaned.h5'
    fn_50psig = '50psig_cleaned.h5'
    fn_100psig = '100psig_cleaned.h5'
    labels = ['0 psig', '50 psig', '100 psig']
    colours = ['C0', 'C1', 'C2']

    fig, ax = plt.subplots(1, 1, figsize=(7, 3), tight_layout=True)
    for idxfn, fn in enumerate([fn_atm, fn_50psig, fn_100psig]):
        with h5py.File(f'data/{fn}', 'r') as hf:
            ph1_clean = hf['ph1_clean'][:]
            ph2_clean = hf['ph2_clean'][:]
            u_tau = hf.attrs['u_tau']
            nu = hf.attrs['nu']
            rho = hf.attrs['rho']
            f_cut = hf.attrs['f_cut']
            Re_tau = hf.attrs['Re_tau']
        f_clean, Pyy_ph1_clean = compute_spec(FS, ph1_clean)
        f_clean, Pyy_ph2_clean = compute_spec(FS, ph2_clean)
        T_plus = 1/f_clean * (u_tau**2)/nu

        g1_c, g2_c, rv_c = channel_model(T_plus, Re_tau, u_tau, 14)
        channel_fphipp_plus = rv_c*(g1_c+g2_c)
        
        data_fphipp_plus1 = (f_clean * Pyy_ph1_clean)/(rho**2 * u_tau**4)
        data_fphipp_plus2 = (f_clean * Pyy_ph2_clean)/(rho**2 * u_tau**4)
        ax.semilogx(T_plus, data_fphipp_plus1, label=labels[idxfn], alpha=0.6, color=colours[idxfn])
        ax.semilogx(T_plus, data_fphipp_plus2, label=labels[idxfn], alpha=0.6, color=colours[idxfn])
        ax.semilogx(T_plus, channel_fphipp_plus, label=f'Model {labels[idxfn]}', linestyle='--', color=colours[idxfn])

    ax.set_xlabel(r"$T^+$")
    ax.set_ylabel(r"${f \phi_{pp}}^+$")
    ax.set_xlim(1, 1e4)
    ax.set_ylim(0, 4)
    ax.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

    labels_handles = ['0 psig Data', '0 psig Model',
                      '50 psig Data', '50 psig Model',
                      '100 psig Data', '100 psig Model']
    label_colours = ['C0', 'C0', 'C1', 'C1', 'C2', 'C2']
    label_styles = ['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed']
    custom_lines = [Line2D([0], [0], color=label_colours[i], linestyle=label_styles[i]) for i in range(len(labels_handles))]
    ax.legend(custom_lines, labels_handles, loc='upper right', fontsize=8)
    fig.savefig('figures/final/spectra_comparison.png', dpi=410)


def plot_required_tfs():
    fn_atm = '0psig_cleaned.h5'
    fn_50psig = '50psig_cleaned.h5'
    fn_100psig = '100psig_cleaned.h5'
    labels = ['0 psig', '50 psig', '100 psig']
    colours = ['C0', 'C1', 'C2']

    fig, ax = plt.subplots(2, 1, figsize=(7, 4), tight_layout=True)
    for idxfn, fn in enumerate([fn_atm, fn_50psig, fn_100psig]):
        with h5py.File(f'data/{fn}', 'r') as hf:
            ph1_clean = hf['ph1_clean'][:]
            ph2_clean = hf['ph2_clean'][:]
            u_tau = hf.attrs['u_tau']
            nu = hf.attrs['nu']
            rho = hf.attrs['rho']
            f_cut = hf.attrs['f_cut']
            Re_tau = hf.attrs['Re_tau']
        f_clean, Pyy_ph1_clean = compute_spec(FS, ph1_clean)
        f_clean, Pyy_ph2_clean = compute_spec(FS, ph2_clean)
        T_plus = 1/f_clean * (u_tau**2)/nu

        g1_c, g2_c, rv_c = channel_model(T_plus, Re_tau, u_tau, 14)
        channel_fphipp_plus = rv_c*(g1_c+g2_c)
        
        data_fphipp_plus1 = (f_clean * Pyy_ph1_clean)/(rho**2 * u_tau**4)
        data_fphipp_plus2 = (f_clean * Pyy_ph2_clean)/(rho**2 * u_tau**4)

        # Cut off where the filter does
        T_plus_mask = T_plus >= TPLUS_CUT
        f_clean = f_clean[T_plus_mask]
        T_plus = T_plus[T_plus_mask]
        channel_fphipp_plus = channel_fphipp_plus[T_plus_mask]
        data_fphipp_plus1 = data_fphipp_plus1[T_plus_mask]
        data_fphipp_plus2 = data_fphipp_plus2[T_plus_mask]

        model_data_ratio1 = channel_fphipp_plus / data_fphipp_plus1 / (CHAIN_SENS_V_PER_PA**2)
        model_data_ratio2 = channel_fphipp_plus / data_fphipp_plus2 / (CHAIN_SENS_V_PER_PA**2)
        ax[0].loglog(T_plus, model_data_ratio1, label=f'Ratio 1 {labels[idxfn]}', alpha=0.6, color=colours[idxfn])
        ax[0].loglog(T_plus, model_data_ratio2, label=f'Ratio 2 {labels[idxfn]}', alpha=0.6, color=colours[idxfn])

        ax[1].loglog(f_clean, model_data_ratio1, label=f'Ratio 1 {labels[idxfn]}', alpha=0.6, color=colours[idxfn])
        ax[1].loglog(f_clean, model_data_ratio2, label=f'Ratio 2 {labels[idxfn]}', alpha=0.6, color=colours[idxfn])
    ax[0].set_xlabel(r"$T^+$")
    ax[0].set_ylabel(r"Model/Data Ratio")
    ax[0].set_xlim(1, 1.5e4)
    ax[0].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
    ax[0].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)
    ax[0].axhline(1, linestyle='--', linewidth=0.4, alpha=0.7)

    ax[1].set_xlabel(r"$f$ [Hz]")
    ax[1].set_ylabel(r"Model/Data Ratio")
    ax[1].set_xlim(1, 1.5e4)
    ax[1].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
    ax[1].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)
    ax[1].axhline(1, linestyle='--', linewidth=0.4, alpha=0.7)

    ax[0].legend(loc='upper left', fontsize=8)
    fig.savefig('figures/final/required_transfer_functions.png', dpi=410)



if __name__ == "__main__":
    run_all_final()
    plot_model_comparison()
    plot_required_tfs()
    
