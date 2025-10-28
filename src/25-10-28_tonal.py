# tf_compute.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple
import inspect

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch, csd, get_window, iirnotch, sosfiltfilt
from scipy.interpolate import UnivariateSpline

from icecream import ic

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
NPERSEG: int = 2**14
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
DEFAULT_UNITS = {
    "channelData_300_plug": ("Pa", "Pa"),
    "channelData_300_nose": ("Pa", "Pa"),
    "channelData_300": ("Pa", "Pa"),
}

SENSITIVITIES_V_PER_PA: dict[str, float] = {
    # 'nc': 0.05,
    # 'PH': 0.05,
    # 'NC': 0.05,
}
PREAMP_GAIN: dict[str, float] = {"nc": 1.0, "PH": 1.0, "NC": 1.0}
TONAL_BASE = "data/25-10-28_tonal/"


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


def concatenate_signals(frequencies) -> np.ndarray:
    """Concatenate multiple 1D arrays into a single 1D array."""
    f_array = []
    ratios = [] 
    for freq in frequencies:
        fn = TONAL_BASE + f"ChannelData_{freq}.mat"
        dat = loadmat(fn, squeeze_me=True)
        ic(dat.keys())
        nc, ph1 = dat["ChannelData_LP"].T
        nc_pa = convert_to_pa(nc, "V", channel_name="nc")
        ph1_pa = convert_to_pa(ph1, "V", channel_name="PH")
        # Find the amplitude of the tone in nc_pa
        f, Pxx = welch(nc_pa, fs=FS, window=WINDOW, nperseg=NPERSEG)
        tone_idx = np.argmax(Pxx)
        tone_freq = f[tone_idx]
        # Find the amplitude of the tone in ph1_pa
        f, Pxx = welch(ph1_pa, fs=FS, window=WINDOW, nperseg=NPERSEG)
        tone_idx = np.argmax(Pxx)
        tone_freq = f[tone_idx]
        ratio = np.sqrt(np.max(Pxx)) / np.sqrt(np.max(Pxx))
        ratios.append(ratio)
        f_array.append(ph1_pa)
    return np.array(f_array), np.array(ratios)

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



def plot_ratios():
    frequencies = np.arange(100, 3100, 100)
    dat = loadmat(TONAL_BASE + "ChannelData_wn.mat")
    nc, ph1 = dat["ChannelData_LP"].T
    nc_pa = convert_to_pa(nc, "V", channel_name="nc")
    ph1_pa = convert_to_pa(ph1, "V", channel_name="PH")
    f1, H1, _ = estimate_frf(nc_pa, ph1_pa, fs=FS)
    f_array, ratios = concatenate_signals(frequencies)
    # Fit a spline through the tonal ratios
    spline = UnivariateSpline(f_array, ratios, s=0)
    f_smooth = np.logspace(np.log10(f_array[0]), np.log10(f_array[-1]), 100)
    ratios_smooth = spline(f_smooth)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(f1, np.abs(H1), label="FRF from WN", color="C0")
    ax.semilogx(f_array, ratios, "o", label="Tonal ratios", color="C1")
    ax.semilogx(f_smooth, ratios_smooth, "-", label="Spline fit", color="C1", alpha=0.5)
    ax.set_xlabel(r"$f$ [Hz]")
    ax.set_ylabel("Amplitude ratio")
    fig.savefig("figures/tonal_ratios/tonal.png", dpi=300)

plot_ratios()
