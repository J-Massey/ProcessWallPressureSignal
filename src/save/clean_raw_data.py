"""
File 1: 
"""


import h5py
import numpy as np
from scipy.signal import welch, csd, get_window
import scipy.io as sio

from icecream import ic
from pathlib import Path

from scipy.signal import butter, sosfiltfilt
import torch
from tqdm import tqdm

from src.core.wiener_filter_torch import (
    wiener_cancel_background_torch,
    cancel_background_freq,
    wiener_cancel_hybrid,
)

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import scienceplots
plt.style.use(["science", "grid"])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")

from src.plot.models import channel_model, bl_model
from src.config_params import Config

# Load the config parameters (file paths, constants, etc.) from a central location to ensure consistency
cfg = Config()

############################
# Constants & defaults
############################
FS = cfg.FS
NPERSEG = cfg.NPERSEG
WINDOW = cfg.WINDOW
TRIM_CAL_SECS = cfg.TRIM_CAL_SECS  # seconds trimmed from the start of calibration runs (0 to disable)

nc_colour = '#1f77b4'  # matplotlib default blue
ph1_colour = "#c76713"  # matplotlib default orange
ph2_colour = "#9fda16"  # matplotlib default red
nkd_colour = '#2ca02c' # matplotlib default green

# --- constants (keep once, top of file) ---
R = cfg.R        # J/kg/K
PSI_TO_PA = cfg.PSI_TO_PA
P_ATM = cfg.P_ATM
DELTA = cfg.DELTA  # m, bl-height of 'channel'
TDEG = cfg.TDEG

TPLUS_CUT = cfg.TPLUS_CUT  # picked so that we cut at half the inner peak

# Data
PH_RAW_FILE = cfg.PH_RAW_FILE
NKD_RAW_FILE = cfg.NKD_PROCESSED_FILE
FINAL_CLEANED_DIR = cfg.FINAL_CLEANED_DIR
FIG_DIR = Path("figures") / "final"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def bandpass_filter(data, fs, f_low, f_high, order=4):
    sos = butter(order, [f_low, f_high], btype='band', fs=fs, output='sos')
    filtered = sosfiltfilt(sos, data)
    # protect against NaNs and negatives
    filtered = np.nan_to_num(filtered, nan=0.0)
    # filtered[filtered < 0] = 0
    return filtered

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
    psi_gauge: int,   # psig for this run
    T_K: float,         # Kelvin for this run
    u_tau: float,       # m/s for this run (your measured/estimated)
    plot_flag=False
):
    # --- gas props & scales ---
    rho, mu, nu = air_props_from_gauge(psi_gauge, T_K)
    Re_tau = u_tau * DELTA / nu              # if DELTA is your half-height
    f_cut  = f_cut_from_Tplus(u_tau, nu)     # matches your scaling
    Tplus_cut = (u_tau**2 / nu) / f_cut      # should equal TPLUS_CUT

    with h5py.File(f"{FINAL_CLEANED_DIR}/{psi_gauge}psig_cleaned.h5", "w") as hf:
        hf.attrs['rho']   = rho
        hf.attrs['mu']    = mu
        hf.attrs['nu']    = nu
        hf.attrs['u_tau'] = u_tau
        hf.attrs['Re_tau']= Re_tau
        hf.attrs['FS']    = cfg.FS
        hf.attrs['f_cut'] = f_cut
        hf.attrs['Tplus_cut'] = Tplus_cut
        hf.attrs["delta"] = DELTA
        hf.attrs['psi_gauge'] = psi_gauge
        hf.attrs['T_K']       = T_K
        hf.attrs["cf_2"] = 2 * (u_tau**2) / cfg.U_E**2
        spacing = ['close', 'far']

        for sp in spacing:
            sgrp = hf.create_group(sp)
            # --- load ---
            ph_raw = PH_RAW_FILE
            with h5py.File(ph_raw, 'r') as f_raw:
                ph1 = f_raw[f'wallp_raw/{int(psi_gauge)}psig/{sp}/PH1_Pa'][:]
                ph2 = f_raw[f'wallp_raw/{int(psi_gauge)}psig/{sp}/PH2_Pa'][:]
            nkd_raw = NKD_RAW_FILE
            with h5py.File(nkd_raw, 'r') as f_nkd:
                nkd = f_nkd[f'freestream_production/{int(psi_gauge)}psig/{sp}/NC_Pa'][:]

            # --- take out the mean ---
            ph1 -= np.mean(ph1)
            ph2 -= np.mean(ph2)
            nkd -= np.mean(nkd)

            # --- apply a band pass filter between 0.1 Hz and f_cut ---
            ph1 = bandpass_filter(ph1, FS, 1, f_cut)
            ph2 = bandpass_filter(ph2, FS, 1, f_cut)
            nkd = bandpass_filter(nkd, FS, 1, f_cut)


            # --- spectra ---
            f, P_ph1 = compute_spec(FS, ph1)
            _, P_ph2 = compute_spec(FS, ph2)
            _, P_nkd = compute_spec(FS, nkd)

            # premultiplied, dimensionless: f * phi_pp / (rho^2 * u_tau^4)
            prem = lambda Pf: (f * Pf) / (rho**2 * u_tau**4)

            # --- optional plotting checks ---
            if plot_flag:
                T_plus = (u_tau**2 / nu) / f
                g1, g2, rv   = bl_model(T_plus, Re_tau, 2 * (u_tau**2) / cfg.U_E**2)
                g1c, g2c, rv_c = channel_model(T_plus, Re_tau, u_tau, cfg.U_E)

                fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, tight_layout=True)
                fig.suptitle(str(psi_gauge) + fr"psig $\delta^+ \approx$  ({Re_tau:.0f})")

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
                out_png = FIG_DIR / f"{psi_gauge}psig.png"
                fig.savefig(out_png, dpi=410)

            # --- Wiener clean, save HDF5 with consistent metadata ---
            ph1_clean = wiener_cancel_hybrid(ph1, nkd, FS)
            ph2_clean = wiener_cancel_hybrid(ph2, nkd, FS)
            torch.cuda.empty_cache()
            sgrp.create_dataset('ph1_clean', data=ph1_clean)
            sgrp.create_dataset('ph2_clean', data=ph2_clean)

    # return useful bits if you want to overlay/compare
    return dict(f=f, P_ph1=P_ph1, P_ph2=P_ph2, P_nkd=P_nkd,
                rho=rho, nu=nu, Re_tau=Re_tau, f_cut=f_cut)


def run_all_final():
    psigs = cfg.PSIGS
    u_tau = cfg.U_TAU
    Tdeg = cfg.TDEG

    # --- ATM ---
    clean_raw(
        psi_gauge=psigs[0],
        T_K=273.15 + Tdeg[0],
        u_tau=u_tau[0],
        plot_flag=False
    )

    # --- 50 psig ---
    clean_raw(
        psi_gauge=psigs[1],
        T_K=273.15 + Tdeg[1],
        u_tau=u_tau[1],
        plot_flag=False
    )

    # --- 100 psig ---
    clean_raw(
        psi_gauge=psigs[2],
        T_K=273.15 + Tdeg[2],
        u_tau=u_tau[2],
        plot_flag=False
    )

if __name__ == "__main__":
    run_all_final()
