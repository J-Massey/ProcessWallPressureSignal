# tf_compute.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple
import inspect
import os

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
BAND: Tuple[float, float] = (100.0, 1000.0)  # <-- scaling band

# Colors (exported for plotting)
PH1_COLOR = "#c76713"  # orange
PH2_COLOR = "#9fda16"  # green-ish
NC_COLOR = "#2ca02c"   # matplotlib default green (kept for reference)

# --- constants (keep once, top of file) ---
R = 287.05        # J/kg/K
PSI_TO_PA = 6_894.76
P_ATM = 101_325.0
DELTA = 0.035  # m, bl-height of 'channel'
TDEG = [18, 20, 22]  # temperatures [°C] for 0/50/100 psig

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

def _ensure_dir(path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# FIX: always enforce the common frequency band on load, even from cached .npy
# -----------------------------------------------------------------------------
def _load_or_make_frf(label: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load (f, H) from npy if available; otherwise compute from calib_<label>.mat and save.
    Always returns arrays band-limited to [100, 3000] Hz so shapes line up.
    """
    f_np = Path(TONAL_BASE) / f"wn_frequencies_{label}.npy"
    H_np = Path(TONAL_BASE) / f"wn_H1_{label}.npy"

    def _bandmask(f: np.ndarray, H: np.ndarray, lo=100.0, hi=3000.0):
        f = np.asarray(f).ravel()
        H = np.asarray(H).astype(complex).ravel()
        m = (f >= lo) & (f <= hi)
        if m.sum() == 0:
            raise ValueError("No frequency points in requested band.")
        return f[m], H[m]

    if f_np.exists() and H_np.exists():
        f = np.load(f_np)
        H = np.load(H_np)
        # Enforce band on old caches too (robust against mixed save formats)
        return _bandmask(f, H)

    dat = loadmat(Path(TONAL_BASE) / f"calib_{label}.mat")
    ph1, ph2, nc, _ = dat["channelData_WN"].T
    nc_pa = convert_to_pa(nc, "V", channel_name="NC")
    ph1_pa = convert_to_pa(ph1, "V", channel_name="PH1")
    f, H, _ = estimate_frf(ph1_pa, nc_pa, fs=FS)
    f, H = _bandmask(f, H)                # band-limit before saving
    _ensure_dir(f_np)
    np.save(f_np, f); np.save(H_np, H)
    return f, H

def save_calibs(pressures):
    """
    (Optional) regenerate and save band-limited FRFs for all pressures
    to guarantee identical shapes across cache files.
    """
    for i, pressure in enumerate(pressures):
        dat = loadmat(TONAL_BASE + f"calib_{pressure}.mat")
        ph1, ph2, nc, _ = dat["channelData_WN"].T
        nc_pa = convert_to_pa(nc, "V", channel_name="NC")
        ph1_pa = convert_to_pa(ph1, "V", channel_name="PH1")
        f1, H1, _ = estimate_frf(ph1_pa, nc_pa, fs=FS)
        # enforce the same band as the loader
        m = (f1 >= 100.0) & (f1 <= 3000.0)
        f1 = f1[m]; H1 = H1[m]
        np.save(TONAL_BASE + f"wn_frequencies_{pressure}.npy", f1)
        np.save(TONAL_BASE + f"wn_H1_{pressure}.npy", H1)

def scale_0psig(pressures):
    pgs = [0, 50, 100]
    colours = ['C0', 'C1', 'C2']

    rho0, *_  = air_props_from_gauge(pgs[0], TDEG[0]+273)

    f1, H1 = _load_or_make_frf("0psig")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, pressure in enumerate(pressures):
        rho, *_  = air_props_from_gauge(pgs[i], TDEG[i]+273)
        Rrho = rho/rho0
        ax.loglog(f1, np.abs(H1)*Rrho, label=f"{pressure}", color=colours[i])
    ax.set_xlabel(r"$f$ [Hz]")
    ax.set_ylabel("Amplitude ratio")
    ax.set_ylim(0.1, 50)
    ax.legend()
    _ensure_dir("figures/tonal_ratios/scaled_0psig_tf.png")
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
    R: float = 1.0
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

    mag = np.abs(H) * R
    phi = np.unwrap(np.angle(H))
    # Safer OOB behaviour: unity gain outside measured band
    mag_i = np.interp(fr, f, mag, left=1.0, right=1.0)
    phi_i = np.interp(fr, f, phi, left=phi[0], right=phi[-1])
    Hi = mag_i * np.exp(1j * phi_i)

    if zero_dc:
        Hi[0] = 0.0
        if Nfft % 2 == 0:
            Hi[-1] = 0.0

    y = np.fft.irfft(X * Hi, n=Nfft)[:N]
    return y

# -----------------------------------------------------------------------------
# Simple band-limited scaling using a single μ0 (fitted from 50 psig)
# -----------------------------------------------------------------------------
def estimate_mu0_from_band(f: np.ndarray, H0: np.ndarray, Hcal: np.ndarray,
                           R_cal: float, band: Tuple[float, float] = BAND):
    """
    Estimate a scalar μ0 using only the given band. Robust to notches via median.
    μ0 is solved from S = R*(1+μ0)/(1+μ0*R), where S is the median |Hcal|/|H0| in-band.
    """
    f = np.asarray(f).ravel()
    H0 = np.asarray(H0).astype(complex).ravel()
    Hc = np.asarray(Hcal).astype(complex).ravel()
    if not (f.shape == H0.shape == Hc.shape):
        raise ValueError("f, H0, Hcal must have identical 1D shapes.")

    mask = (f >= band[0]) & (f <= band[1])
    if mask.sum() < 8:
        raise ValueError("Too few frequency points in selected band.")

    ratio = np.abs(Hc[mask]) / (np.abs(H0[mask]) + 1e-30)
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        raise ValueError("No finite ratios in band.")
    S_band = np.median(ratio)
    if np.isclose(S_band, 1.0, atol=1e-6):
        S_band = 1.0 + 1e-3  # avoid division by zero

    mu0 = (R_cal - S_band) / (R_cal * (S_band - 1.0))
    return float(mu0), {"S_band": float(S_band), "R_cal": float(R_cal), "n_used": int(mask.sum())}

def predict_tf_in_band(f: np.ndarray, H0: np.ndarray, R_target: float, mu0: float,
                       band: Tuple[float, float] = BAND):
    """
    Return band-limited prediction of FRF at a target density ratio R_target.
    Phase from H0 is preserved; only amplitude within `band` is scaled.
    """
    f = np.asarray(f).ravel()
    H0 = np.asarray(H0).astype(complex).ravel()
    if f.shape != H0.shape:
        raise ValueError("f and H0 must have identical shapes.")

    S_scalar = R_target * (1.0 + mu0) / (1.0 + mu0 * R_target)
    mag0 = np.abs(H0)
    phi0 = np.angle(H0)

    scale = np.ones_like(mag0)
    mask = (f >= band[0]) & (f <= band[1])
    scale[mask] = S_scalar

    H_pred = (mag0 * scale) * np.exp(1j * phi0)
    return H_pred, scale

# -----------------------------------------------------------------------------

def plot_tf_model_comparison():
    fn_atm = '0psig_cleaned.h5'
    fn_50psig = '50psig_cleaned.h5'
    fn_100psig = '100psig_cleaned.h5'
    labels = ['0psig', '50psig', '100psig']
    colours = ['C0', 'C1', 'C2']

    # --- densities (for R = rho/rho0) ---
    pgs = [0, 50, 100]
    rho0, *_  = air_props_from_gauge(pgs[0], TDEG[0]+273)
    rho50, *_ = air_props_from_gauge(pgs[1], TDEG[1]+273)
    rho100, *_= air_props_from_gauge(pgs[2], TDEG[2]+273)
    R50 = rho50 / rho0
    R100 = rho100 / rho0

    # --- load or compute FRFs (band-limited and shape-consistent) ---
    f0, H0    = _load_or_make_frf("0psig")
    f50, H50  = _load_or_make_frf("50psig")  # used to fit μ0

    # FIX: guard for mismatched grids before comparing/resampling
    if (f0.shape != f50.shape) or (not np.allclose(f0, f50)):
        # resample complex H50 onto f0
        H50 = np.interp(f0, f50, H50.real) + 1j*np.interp(f0, f50, H50.imag)
        f50 = f0  # keep grids consistent downstream

    # --- fit μ0 using only 100–1000 Hz band ---
    mu0, meta = estimate_mu0_from_band(f0, H0, H50, R_cal=R50, band=BAND)
    ic(mu0, meta)

    # Pre-build predicted FRFs for the three pressures (band-limited scaling)
    H_pred_0, _   = predict_tf_in_band(f0, H0, R_target=1.0,  mu0=mu0, band=BAND)
    H_pred_50, _  = predict_tf_in_band(f0, H0, R_target=R50, mu0=mu0, band=BAND)
    H_pred_100, _ = predict_tf_in_band(f0, H0, R_target=R100, mu0=mu0, band=BAND)
    H_for_pressure = {"0psig": H_pred_0, "50psig": H_pred_50, "100psig": H_pred_100}

    fig, ax = plt.subplots(1, 1, figsize=(7, 3), tight_layout=True)

    for idxfn, (label, fn) in enumerate(zip(labels, [fn_atm, fn_50psig, fn_100psig])):
        # --- load cleaned time series + metadata ---
        with h5py.File(f'data/{fn}', 'r') as hf:
            ph1_clean = hf['ph1_clean'][:]
            ph2_clean = hf['ph2_clean'][:]
            u_tau = hf.attrs['u_tau']
            nu = hf.attrs['nu']
            rho = hf.attrs['rho']
            f_cut = hf.attrs['f_cut']
            Re_tau = hf.attrs['Re_tau']
            cf_2 = hf.attrs.get('cf_2', 0.0)  # default if missing

        f_clean, Pyy_ph1_clean = compute_spec(FS, ph1_clean)
        f_clean, Pyy_ph2_clean = compute_spec(FS, ph2_clean)

        # --- choose predicted FRF for this pressure (already includes scaling) ---
        H_use = H_for_pressure[label]

        # Apply FRF (R=1.0 because H_use already carries scaling)
        ph1_clean_tf = apply_frf(ph1_clean, FS, f0, H_use, R=1.0)
        ph2_clean_tf = apply_frf(ph2_clean, FS, f0, H_use, R=1.0)

        f_clean_tf, Pyy_ph1_clean_tf = compute_spec(FS, ph1_clean_tf)
        f_clean_tf, Pyy_ph2_clean_tf = compute_spec(FS, ph2_clean_tf)

        data_fphipp_plus1_tf = (f_clean_tf * Pyy_ph1_clean_tf)/(rho**2 * u_tau**4)
        data_fphipp_plus2_tf = (f_clean_tf * Pyy_ph2_clean_tf)/(rho**2 * u_tau**4)

        # clip at the helmholtz resonance upper edge
        mask_plot = f_clean_tf < 1_000
        f_plot = f_clean_tf[mask_plot]
        s1 = data_fphipp_plus1_tf[mask_plot]
        s2 = data_fphipp_plus2_tf[mask_plot]

        ax.semilogx(f_plot, s1, linestyle='-', color=colours[idxfn], alpha=0.7, lw=0.9)
        ax.semilogx(f_plot, s2, linestyle='-', color=colours[idxfn], alpha=0.7, lw=0.9)

        # Now plot upper and lower bounds from ±10% u_tau error
        u_tau_bounds = (u_tau * (1 - 0.1), u_tau * (1 + 0.1))
        s1_u = ((f_clean_tf * Pyy_ph1_clean_tf)/(rho**2 * u_tau_bounds[0]**4))[mask_plot]
        s2_u = ((f_clean_tf * Pyy_ph2_clean_tf)/(rho**2 * u_tau_bounds[0]**4))[mask_plot]
        s1_l = ((f_clean_tf * Pyy_ph1_clean_tf)/(rho**2 * u_tau_bounds[-1]**4))[mask_plot]
        s2_l = ((f_clean_tf * Pyy_ph2_clean_tf)/(rho**2 * u_tau_bounds[-1]**4))[mask_plot]

        ax.fill_between(f_plot, s1_u, s1_l, color=colours[idxfn], alpha=0.25, edgecolor='none')
        ax.fill_between(f_plot, s2_u, s2_l, color=colours[idxfn], alpha=0.25, edgecolor='none')

    ax.set_xlabel(r"$f$ [Hz]")
    ax.set_ylabel(r"${f \phi_{pp}}^+$")
    ax.set_xlim(50, 1e4)
    ax.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

    labels_handles = ['0 psig Data', '50 psig Data', '100 psig Data']
    custom_lines = [Line2D([0], [0], color=colours[i], linestyle='solid') for i in range(len(labels_handles))]
    ax.legend(custom_lines, labels_handles, loc='upper right', fontsize=8)

    _ensure_dir('figures/tonal_ratios/spectra_comparison_tf_freq.png')
    fig.savefig('figures/tonal_ratios/spectra_comparison_tf_freq.png', dpi=410)

if __name__ == "__main__":
    # If *.npy FRFs are missing, they will be generated from calib_*.mat automatically
    plot_tf_model_comparison()
