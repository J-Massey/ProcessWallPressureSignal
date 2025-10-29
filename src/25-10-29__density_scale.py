# tf_compute.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple
import inspect

import numpy as np
import h5py
from scipy.io import loadmat
from scipy.signal import welch, csd, get_window, iirnotch, sosfiltfilt
from scipy.interpolate import UnivariateSpline

from icecream import ic
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import scienceplots
plt.style.use(["science", "grid"])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")

# =============================================================================
# Constants & styling (exported so tf_plot.py can import them)
# =============================================================================
FS: float = 50_000.0
NPERSEG: int = 2**12
WINDOW: str = "hann"

# Colors (exported for plotting)
PH1_COLOR = "#c76713"  # orange
PH2_COLOR = "#9fda16"  # green-ish
NC_COLOR = "#2ca02c"   # matplotlib default green (kept for reference)

# --- constants (keep once, top of file) ---
R = 287.05        # J/kg/K
PSI_TO_PA = 6_894.76
P_ATM = 101_325.0
DELTA = 0.035  # m, bl-height of 'channel'
TDEG = [18, 20, 22]

# =============================================================================
# Units & optional conversions (kept for compatibility with other workflows)
# =============================================================================


SENSITIVITIES_V_PER_PA: dict[str, float] = {
    'nc': 50e-3,
    'PH1': 50e-3,
    'PH2': 50e-3,
    'NC': 50e-3,
}
PREAMP_GAIN: dict[str, float] = {"nc": 1.0, "PH1": 1.0, "PH2": 1.0, "NC": 1.0}
TONAL_BASE = "data/2025-10-28/tonal/"

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


def convert_to_pa(x: np.ndarray, units: str, *, channel_name: str = "unknown") -> np.ndarray:
    """Convert a pressure time series to Pa."""
    u = units.lower()
    x = np.asarray(x, float)
    if u == "pa":
        return x
    if u == "kpa":
        return x * 1e3
    if u == "mbar":
        return x * 100.0
    if u in ("v", "volt", "volts"):
        if channel_name not in SENSITIVITIES_V_PER_PA or SENSITIVITIES_V_PER_PA[channel_name] is None:
            raise ValueError(
                f"Sensitivity (V/Pa) for channel '{channel_name}' not provided; cannot convert V→Pa."
            )
        sens = float(SENSITIVITIES_V_PER_PA[channel_name])  # V/Pa
        gain = float(PREAMP_GAIN.get(channel_name, 1.0))
        return x / (gain * sens)
    raise ValueError(f"Unsupported units '{units}' for channel '{channel_name}'")


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



def estimate_frf(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    window: str = WINDOW,
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

def scale_with_loading(H0, psig, *, mu0=None, psig_cal=None, S_meas=None,
                       T_K=None, T0_K=None, patm_psia=14.6959, Z=None, Z0=None,
                       return_meta=False):
    """
    Scale a 0-psig transfer function using the 'loaded' blend:
        S = |H_P|/|H_0| ≈ R * (1 + mu0) / (1 + mu0 * R),
    with R = rho/rho0 from the same density model as above.

    If mu0 is unknown, estimate it from one calibration measurement (`psig_cal`, `S_meas`):
        mu0 ≈ (R_cal - S_meas) / (R_cal * (S_meas - 1)),
    where S_meas is the measured linear amplitude ratio |H_P|/|H_0| at one frequency.

    Parameters
    ----------
    H0 : array_like
        Complex or real frequency response at 0 psig (shape preserved).
    psig : float
        Target gauge pressure (psig) to predict.
    mu0 : float, optional
        Loading ratio at 0 psig. If not provided, supply `psig_cal` and `S_meas` to estimate it.
    psig_cal : float, optional
        Gauge pressure (psig) used for the calibration measurement.
    S_meas : float, optional
        Measured linear amplitude ratio at the calibration pressure (not dB).
    T_K, T0_K, patm_psia, Z, Z0 : see `scale_by_density`.
    return_meta : bool, default False
        If True, return (H, meta) with {'R','S','mu0','P_abs_psia','P0_psia'}.

    Returns
    -------
    H : ndarray
        Scaled transfer function at target pressure.
    (H, meta) : tuple, if return_meta=True
    """
    H0 = np.asarray(H0)

    # Internal density-ratio helper (kept local so only two top-level functions are defined)
    def _R(psig_local, T_K_local, T0_K_local, patm, Z_local, Z0_local):
        P_abs_local = float(patm) + float(psig_local)
        P0_local = float(patm)

        if T_K_local is None and T0_K_local is None:
            TR_local = 1.0
        elif T_K_local is None or T0_K_local is None:
            TR_local = 1.0
        else:
            if T_K_local <= 0 or T0_K_local <= 0:
                raise ValueError("Temperatures must be > 0 K.")
            TR_local = T0_K_local / T_K_local

        if Z_local is None and Z0_local is None:
            ZR_local = 1.0
        else:
            if Z_local is None:
                Z_local = Z0_local
            if Z0_local is None:
                Z0_local = Z_local
            if Z_local == 0:
                raise ValueError("Z cannot be zero.")
            ZR_local = Z0_local / Z_local

        if P_abs_local <= 0 or P0_local <= 0:
            raise ValueError("Absolute pressures must be positive.")
        return (P_abs_local / P0_local) * TR_local * ZR_local, P_abs_local, P0_local

    R, P_abs, P0 = _R(psig, T_K, T0_K, patm_psia, Z, Z0)

    # Estimate mu0 if not given
    if mu0 is None:
        if psig_cal is None or S_meas is None:
            raise ValueError("Provide either mu0 OR (psig_cal and S_meas) to estimate mu0.")
        if S_meas <= 0:
            raise ValueError("S_meas must be a positive linear ratio (not dB).")
        R_cal, _, _ = _R(psig_cal, T_K, T0_K, patm_psia, Z, Z0)
        if abs(S_meas - 1.0) < 1e-12:
            raise ValueError("S_meas too close to 1.0; cannot estimate mu0 (division by zero).")
        mu0 = (R_cal - S_meas) / (R_cal * (S_meas - 1.0))

    denom = (1.0 + mu0 * R)
    if abs(denom) < 1e-12:
        raise ZeroDivisionError("Denominator (1 + mu0*R) ~ 0; check inputs.")
    S = R * (1.0 + mu0) / denom

    H = H0 * S
    if return_meta:
        return H, {"R": R, "S": S, "mu0": mu0, "P_abs_psia": P_abs, "P0_psia": P0}
    return H

def scale_0psig(pressures):
    pgs = [0, 50, 100]
    colours = ['C0', 'C1', 'C2']

    rho0, *_  = air_props_from_gauge(pgs[0], TDEG[0]+273)

    dat = loadmat(TONAL_BASE + f"calib_0psig.mat")
    ph1, ph2, nc, _ = dat["channelData_WN"].T
    nc_pa = convert_to_pa(nc, "V", channel_name="NC")
    ph1_pa = convert_to_pa(ph1, "V", channel_name="PH1")
    f1, H1, _ = estimate_frf(ph1_pa, nc_pa, fs=FS)
    mask = (f1 >= 100) & (f1 <= 3000)
    f1 = f1[mask]
    H1 = H1[mask]
    np.save(TONAL_BASE + f"wn_frequencies_0psig.npy", f1)
    np.save(TONAL_BASE + f"wn_H1_0psig.npy", H1)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, pressure in enumerate(pressures):
        # frequencies = np.arange(100, 3100, 100)
        rho, *_  = air_props_from_gauge(pgs[i], TDEG[i]+273)
        R = rho/rho0
        ax.loglog(f1, np.abs(H1)*R, label="FRF from WN", color=colours[i])
    ax.set_xlabel(r"$f$ [Hz]")
    ax.set_ylabel("Amplitude ratio")
    ax.set_ylim(0.1, 50)
    ax.legend()
    fig.savefig(f"figures/tonal_ratios/scaled_0psig_tf.png", dpi=300)

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

def apply_frf(
    x: np.ndarray,
    fs: float,
    f: np.ndarray,
    H: np.ndarray,
    demean: bool = True,
    zero_dc: bool = True,
):
    """
    Apply a measured FRF H (x→y) to a time series x to synthesise y.
    This is the forward operation: Y = H · X in the frequency domain.
    """
    x = np.asarray(x, float)
    if demean:
        x = x - x.mean()

    N = x.size
    Nfft = int(2 ** np.ceil(np.log2(N)))
    X = np.fft.rfft(x, n=Nfft)
    fr = np.fft.rfftfreq(Nfft, d=1.0 / fs)

    mag = np.abs(H)
    phi = np.unwrap(np.angle(H))
    # Safer OOB behaviour: taper magnitude to zero outside measured band
    mag_i = np.interp(fr, f, mag, left=1.0, right=1.0)
    phi_i = np.interp(fr, f, phi, left=phi[0], right=phi[-1])
    Hi = mag_i * np.exp(1j * phi_i)

    if zero_dc:
        Hi[0] = 0.0
        if Nfft % 2 == 0:
            Hi[-1] = 0.0

    y = np.fft.irfft(X * Hi, n=Nfft)[:N]
    return y

def plot_tf_model_comparison():
    fn_atm = '0psig_cleaned.h5'
    fn_50psig = '50psig_cleaned.h5'
    fn_100psig = '100psig_cleaned.h5'
    labels = ['0psig', '50psig', '100psig']
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
            cf_2 = hf.attrs['cf_2']  # default if missing
        f_clean, Pyy_ph1_clean = compute_spec(FS, ph1_clean)
        f_clean, Pyy_ph2_clean = compute_spec(FS, ph2_clean)
        T_plus = 1/f_clean * (u_tau**2)/nu

        g1_c, g2_c, rv_c = channel_model(T_plus, Re_tau, u_tau, 14)
        g1_b, g2_b, rv_b = bl_model(T_plus, Re_tau, cf_2)
        channel_fphipp_plus = rv_c*(g1_c+g2_c)
        bl_fphipp_plus = rv_b*(g1_b+g2_b)

        ax.semilogx(f_clean, channel_fphipp_plus, label=f'Model {labels[idxfn]}', linestyle='--', color=colours[idxfn], lw=0.7)
        ax.semilogx(f_clean, bl_fphipp_plus, label=f'BL Model {labels[idxfn]}', linestyle='-.', color=colours[idxfn], lw=0.7)


        # Get the ratios from above

        f_new = np.load(TONAL_BASE + f"tonal_frequencies_{labels[idxfn]}.npy")
        H_new = np.load(TONAL_BASE + f"tonal_H1_{labels[idxfn]}.npy")

        ph1_clean_tf = apply_frf(ph1_clean, FS, f_new, H_new)
        ph2_clean_tf = apply_frf(ph2_clean, FS, f_new, H_new)
        f_clean_tf, Pyy_ph1_clean_tf = compute_spec(FS, ph1_clean_tf)
        f_clean_tf, Pyy_ph2_clean_tf = compute_spec(FS, ph2_clean_tf)

        T_plus_tf = 1/f_clean_tf * (u_tau**2)/nu
        data_fphipp_plus1_tf = (f_clean_tf * Pyy_ph1_clean_tf)/(rho**2 * u_tau**4)
        data_fphipp_plus2_tf = (f_clean_tf * Pyy_ph2_clean_tf)/(rho**2 * u_tau**4)
        ax.semilogx(f_clean_tf, data_fphipp_plus1_tf, linestyle='-', color=colours[idxfn], alpha=0.7, lw=0.7)
        ax.semilogx(f_clean_tf, data_fphipp_plus2_tf, linestyle='-', color=colours[idxfn], alpha=0.7, lw=0.7)

    ax.set_xlabel(r"$T^+$")
    ax.set_xlabel(r"$f$ [Hz]")
    ax.set_ylabel(r"${f \phi_{pp}}^+$")
    ax.set_xlim(1, 1e4)
    ax.set_ylim(0, 7)
    ax.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

    labels_handles = ['0 psig Data', '0 psig Model',
                      '50 psig Data', '50 psig Model',
                      '100 psig Data', '100 psig Model']
    label_colours = ['C0', 'C0', 'C1', 'C1', 'C2', 'C2']
    label_styles = ['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed']
    custom_lines = [Line2D([0], [0], color=label_colours[i], linestyle=label_styles[i]) for i in range(len(labels_handles))]
    ax.legend(custom_lines, labels_handles, loc='upper right', fontsize=8)
    fig.savefig('figures/tonal_ratios/spectra_comparison_tf_freq.png', dpi=410)



if __name__ == "__main__":
    scale_0psig(['0psig', '50psig', '100psig'])
    # plot_tf_model_comparison()