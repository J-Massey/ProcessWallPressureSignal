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

from apply_frf import apply_frf
# from fit_speaker_scales import fit_speaker_scaling_from_files
from models import bl_model
from clean_raw_data import volts_to_pa, air_props_from_gauge
from fuse_anechoic import combine_anechoic_calibrations

# =============================================================================
# Constants & styling (exported so tf_plot.py can import them)
# =============================================================================
FS: float = 50_000.0
NPERSEG: int = 2**10
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

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Assumes these already exist in your module:
#   - FS, CLEANED_BASE, TARGET_BASE (paths/constants)
#   - compute_spec(fs, x)  -> (f, Pxx)   # Welch PSD (Pa^2/Hz)
#   - apply_frf(...)        # the version that accepts lem_params & phase_strategy

def plot_raw_vs_lem_corrected(
    label: str,
    *,
    cleaned_base: str = "data/final_cleaned/",
    target_base:  str = "data/final_target/",
    lem_param_file: str = "lem_params.h5",
    fmin: float = 100.0,
    fmax: float = 1000.0,
    phase_strategy: str = "lem",   # 'measured'|'lem'|'minphase'|'zero'
    H_meas: tuple[np.ndarray, np.ndarray] | None = None,  # (f_cal, H_cal) if you want measured phase
    savepath: str | None = None,
):
    """
    Plot pre-multiplied, normalized spectra of the RAW cleaned signals and the same
    signals after LEM correction (100–1000 Hz).

    RAW:  (f * Pyy_raw) / (rho^2 * u_tau^4)
    LEM:  (f * Pyy_corr) / (rho^2 * u_tau^4),  where 'corr' is apply_frf(..., lem_params)

    Parameters
    ----------
    label : e.g. '0psig_close', '50psig_far', ...
    H_meas : optional measured FRF (f_cal, H_cal) to borrow *phase* (amplitude still from LEM)
             Use with phase_strategy='measured' for best time‑domain fidelity.
    """

    # -----------------------------
    # 1) Load cleaned time series & meta
    # -----------------------------
    cleaned_fn = os.path.join(cleaned_base, f"{label}_cleaned.h5")
    if not os.path.exists(cleaned_fn):
        raise FileNotFoundError(f"Cleaned file not found: {cleaned_fn}")

    with h5py.File(cleaned_fn, "r") as hf:
        ph1 = np.asarray(hf["ph1_clean"][:], float)
        ph2 = np.asarray(hf["ph2_clean"][:], float)
        rho = float(hf.attrs["rho"])
        u_tau = float(hf.attrs["u_tau"])
        # nu, Re_tau etc. are not needed for this plot

    # -----------------------------
    # 2) Load LEM params for this dataset
    # -----------------------------
    lem_path = os.path.join(target_base, lem_param_file)
    if not os.path.exists(lem_path):
        raise FileNotFoundError(f"LEM parameter file not found: {lem_path}")

    with h5py.File(lem_path, "r") as hf:
        if label not in hf:
            raise KeyError(f"Group '{label}' not found in {lem_path}")
        grp = hf[label]
        lem_params = {
            "g_db":  float(grp.attrs["g_db"]),
            "fD_Hz": float(grp.attrs["fD_Hz"]),
            "QD":    float(grp.attrs["QD"]),
        }

    # -----------------------------
    # 3) Build an LEM design grid for magnitude (interpolated inside apply_frf)
    # -----------------------------
    fH = np.geomspace(max(1.0, fmin), fmax, 512)

    # If we want measured phase, we pass H_meas=(f_cal,H_cal) and tell apply_frf to use 'measured'
    f_cal, H_cal = (H_meas if H_meas is not None else (None, None))
    use_phase = ("measured" if (phase_strategy == "measured" and H_meas is not None) else phase_strategy)

    # -----------------------------
    # 4) Apply LEM FRF to both channels (amplitude from LEM, phase per strategy)
    # -----------------------------
    y1 = apply_frf(
        ph1, fs=FS, f=fH, H=H_cal,
        lem_params=lem_params, invert_target=True,
        phase_strategy=use_phase
    )
    y2 = apply_frf(
        ph2, fs=FS, f=fH, H=H_cal,
        lem_params=lem_params, invert_target=True,
        phase_strategy=use_phase
    )

    # -----------------------------
    # 5) PSDs and normalization to (f*Pyy)/(rho^2 u_tau^4)
    #     (same normalization used in your pipeline)
    # -----------------------------
    f1_raw, P11_raw = compute_spec(FS, ph1)
    f2_raw, P22_raw = compute_spec(FS, ph2)
    f1_cor, P11_cor = compute_spec(FS, y1)
    f2_cor, P22_cor = compute_spec(FS, y2)

    # Use the same grid for both channels (Welch settings identical, so f1==f2)
    if not np.allclose(f1_raw, f2_raw):  # very rare, but keep it robust
        P22_raw = np.interp(f1_raw, f2_raw, P22_raw)
    if not np.allclose(f1_cor, f2_cor):
        P22_cor = np.interp(f1_cor, f2_cor, P22_cor)

    def norm_pm(f, P):
        return (f * P) / (rho**2 * u_tau**4)

    f_raw = f1_raw
    Y1_raw = norm_pm(f_raw, P11_raw)
    Y2_raw = norm_pm(f_raw, P22_raw)

    f_cor = f1_cor
    Y1_cor = norm_pm(f_cor, P11_cor)
    Y2_cor = norm_pm(f_cor, P22_cor)

    # Band-limit to 100–1000 Hz for display
    mraw = (f_raw >= fmin) & (f_raw <= fmax)
    mcor = (f_cor >= fmin) & (f_cor <= fmax)

    # -----------------------------
    # 6) Plot
    # -----------------------------
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.4), tight_layout=True)
    # RAW
    ax.semilogx(f_raw[mraw], Y1_raw[mraw], color="C0", ls="--", lw=0.9, alpha=0.9, label="PH1 raw")
    ax.semilogx(f_raw[mraw], Y2_raw[mraw], color="C1", ls="--", lw=0.9, alpha=0.9, label="PH2 raw")
    # LEM‑corrected
    ax.semilogx(f_cor[mcor], Y1_cor[mcor], color="C0", ls="-",  lw=1.2, alpha=0.95, label="PH1 LEM-corr")
    ax.semilogx(f_cor[mcor], Y2_cor[mcor], color="C1", ls="-",  lw=1.2, alpha=0.95, label="PH2 LEM-corr")

    ax.set_xlim(fmin, fmax)
    ax.set_xlabel(r"$f$ [Hz]")
    ax.set_ylabel(r"$f\,\phi_{pp}^+$")
    ax.grid(True, which="major", ls="--", lw=0.4, alpha=0.7)
    ax.grid(True, which="minor", ls=":",  lw=0.25, alpha=0.6)
    ax.legend(ncol=2, fontsize=8, loc="best")

    ttl = (f"{label} | LEM: g={lem_params['g_db']:+.2f} dB, "
           f"fD={lem_params['fD_Hz']:.0f} Hz, QD={lem_params['QD']:.1f} | "
           f"phase={use_phase}")
    ax.set_title(ttl, fontsize=9)

    if savepath:
        fig.savefig(savepath, dpi=350)
    return fig, ax



def plot_lem_tf_vs_target(
    label: str,
    *,
    target_base: str = "data/final_target/",
    lem_param_file: str = "lem_params.h5",
    fmin: float = 100.0,
    fmax: float = 1000.0,
    to_db: bool = True,
    invert: bool = True,       # if True, plot "required |H|" = 1/target and 1/LEM
    savepath: str | None = None
):
    """
    Overlay the LEM-corrected TF magnitude against the saved target ratio.

    Parameters
    ----------
    label : e.g. '0psig_close', '50psig_far', ...
    target_base : folder that contains 'target_{label}.h5'
    lem_param_file : HDF5 file (in target_base) containing fitted params per label
                     with attrs: g_db, fD_Hz, QD
    fmin, fmax : display band (and RMSE band)
    to_db : plot in dB if True
    invert : if True, compare "required |H|" orientation: 1/scaling_ratio vs 1/LEM
             (by default, compares as-saved amplitude scaling_ratio vs LEM amplitude)
    Returns
    -------
    fig, ax, rmse_db
    """

    # -----------------------------
    # 0) Helpers
    # -----------------------------
    def _lem_mag_diaph(f: np.ndarray, fD: float, QD: float) -> np.ndarray:
        """|H| ∝ |(jω)/(1 - (ω/ωD)^2 + j ω/(Q_D ωD))|"""
        f = np.asarray(f, float)
        w  = 2.0 * np.pi * f
        wD = 2.0 * np.pi * float(fD)
        den = (1.0 - (w / wD)**2) + 1j * (w / (QD * wD))
        H = (1j * w) / den
        return np.abs(H)

    def _as_db(x):
        return 20.0 * np.log10(np.maximum(np.asarray(x, float), 1e-16))

    # -----------------------------
    # 1) Load target ratio (amplitude)
    # -----------------------------
    tgt_path = os.path.join(target_base, f"target_{label}.h5")
    if not os.path.exists(tgt_path):
        raise FileNotFoundError(f"Target file not found: {tgt_path}")

    with h5py.File(tgt_path, "r") as hf:
        f_tgt = np.asarray(hf["frequencies"][:], float)
        S_tgt = np.asarray(hf["scaling_ratio"][:], float)  # amplitude = sqrt(data/model)
        rho   = float(hf.attrs.get("rho", np.nan))

    # Band limit for clean overlay + RMSE
    mband = (f_tgt >= fmin) & (f_tgt <= fmax)
    fB = f_tgt[mband]
    S_tgtB = S_tgt[mband]
    if fB.size < 8:
        raise ValueError(f"Not enough points in [{fmin},{fmax}] Hz to plot.")

    # -----------------------------
    # 2) Load LEM parameters for this label
    # -----------------------------
    lem_path = os.path.join(target_base, lem_param_file)
    if not os.path.exists(lem_path):
        raise FileNotFoundError(f"LEM parameter file not found: {lem_path}")

    with h5py.File(lem_path, "r") as hf:
        if label not in hf:
            raise KeyError(f"Group '{label}' not found in {lem_path}")
        g_db  = float(hf[label].attrs["g_db"])
        fD    = float(hf[label].attrs["fD_Hz"])
        QD    = float(hf[label].attrs["QD"])

    # LEM magnitude on the target grid (same grid makes the comparison direct)
    H_lem = _lem_mag_diaph(fB, fD=fD, QD=QD)
    S_lem = (10.0**(g_db/20.0)) * H_lem  # amplitude that the fit learned

    # Optional: compare in the "required |H|" orientation
    if invert:
        S_tgt_plot = 1.0 / np.maximum(S_tgtB, 1e-16)
        S_lem_plot = 1.0 / np.maximum(S_lem, 1e-16)
    else:
        S_tgt_plot = S_tgtB
        S_lem_plot = S_lem

    # -----------------------------
    # 3) RMSE (in dB) over display band
    # -----------------------------
    err_db = _as_db(S_tgt_plot) - _as_db(S_lem_plot)
    rmse_db = float(np.sqrt(np.mean(err_db**2)))

    # -----------------------------
    # 4) Plot
    # -----------------------------
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.6), tight_layout=True)
    if to_db:
        ax.semilogx(fB, _as_db(S_tgt_plot), lw=1.6, label="target_ratio (dB)")
        ax.semilogx(fB, _as_db(S_lem_plot), lw=1.2, ls="--", label="LEM (dB)")
        ax.set_ylabel("Magnitude [dB]")
    else:
        ax.semilogx(fB, S_tgt_plot, lw=1.6, label="target_ratio")
        ax.semilogx(fB, S_lem_plot, lw=1.2, ls="--", label="LEM")
        ax.set_ylabel("Magnitude [linear]")

    ax.set_xlim(fmin, fmax)
    ax.set_xlabel("f [Hz]")
    ax.grid(True, which="both", ls=":", alpha=0.7)
    mode = "required |H| (inverted)" if invert else "as saved (sqrt(data/model))"
    ax.set_title(f"{label} | LEM: g={g_db:+.2f} dB, fD={fD:.0f} Hz, QD={QD:.1f} | "
                 f"RMSE={rmse_db:.2f} dB | mode={mode}")
    ax.legend(loc="best", fontsize=8)

    if savepath:
        fig.savefig(savepath, dpi=350)
    return fig, ax, rmse_db


if __name__ == "__main__":
    # plot_raw_vs_lem_corrected("50psig_close",
    #                       savepath="figures/raw_vs_lem_50psig_close.png")
    plot_lem_tf_vs_target("100psig_close",
                      to_db=True, invert=True,
                      savepath="figures/lem_vs_target_0psig_close.png")