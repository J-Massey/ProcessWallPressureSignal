#!/usr/bin/env python3
"""
Vanilla FRF (H1) calculator & plotter.

- Minimal dependencies: numpy, scipy, matplotlib
- No de-biasing, no power-aware fusion, no in-situ blending
- Produces per-run FRF plots and a combined overlay
- Saves FRFs as: figures/tf_vanilla/700_{x}psig_vanilla_{spacing}_ph{1|2}_{f|H|gamma2}.npy

Usage:
  1) Edit the RUNS list in __main__ with your .mat paths / keys / columns.
  2) Run: python vanilla_frf.py
  3) Outputs go to: figures/tf_vanilla/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import welch, csd, get_window

# ------------------------------ Defaults -------------------------------------
FS: float = 50_000.0           # sampling rate [Hz]
NPERSEG_DEFAULT: int = 2**9    # increase for smoother curves (e.g., 2**12)
WINDOW: str = "hann"
OUTDIR = Path("figures/tf_vanilla")
SAVE_NUMPY = True              # save f, H, gamma2 as .npy next to the figures

# ------------------------------ Core math ------------------------------------
def estimate_frf(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    *,
    nperseg: int = NPERSEG_DEFAULT,
    window: str = WINDOW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Welch/CSD-based H1 FRF and gamma^2.
      H1 = S_yx / S_xx  (maps x -> y)
    Returns: f [Hz], H (complex), gamma2 [0..1]
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    nseg = int(min(nperseg, x.size, y.size))
    if nseg < 8:
        raise ValueError(f"Signal too short for FRF: n={min(x.size, y.size)} (need >= 8 samples)")

    nov = int(min(nseg // 2, nseg - 1))
    w = get_window(window, nseg, fftbins=True)

    f, Sxx = welch(x, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)
    _, Syy = welch(y, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)
    _, Sxy = csd(x, y, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)  # x->y

    H = np.conj(Sxy) / Sxx
    gamma2 = np.clip((np.abs(Sxy) ** 2) / (Sxx * Syy), 0.0, 1.0)
    return f, H, gamma2


# ------------------------------ I/O helpers ----------------------------------
def load_columns_from_mat(
    mat_path: str | Path,
    *,
    key: str,
    x_col: int,
    y_col: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load two 1-D arrays from a MATLAB .mat file under 'key', using column indices.
    Accepts shape (N, K) or (K, N). Returns (x, y) as float arrays.
    """
    mat = loadmat(Path(mat_path), squeeze_me=True)
    if key not in mat:
        raise KeyError(f"Key '{key}' not found in {mat_path}. Available: {list(mat.keys())}")
    arr = np.asarray(mat[key])
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array under '{key}', got shape {arr.shape}")

    if arr.shape[0] >= arr.shape[1]:  # (N, K)
        x = arr[:, x_col].astype(float)
        y = arr[:, y_col].astype(float)
    else:                              # (K, N)
        x = arr[x_col, :].astype(float)
        y = arr[y_col, :].astype(float)
    return x, y


# ------------------------------ Plotting -------------------------------------
def _nice_title(s: str) -> str:
    return s.replace("_", r"\_")

def plot_frf(
    f: np.ndarray,
    H: np.ndarray,
    *,
    title: str,
    outfile: Path,
    phase_units: str = "rad",
) -> None:
    """
    Save a two-panel figure: |H(f)| (log-log) and angle(H) (semilogx).
    """
    mag = np.abs(H)
    phase = np.unwrap(np.angle(H))
    if phase_units.lower().startswith("deg"):
        phase = np.degrees(phase)
        phase_ylabel = r"$\angle H(f)$ [deg]"
    else:
        phase_ylabel = r"$\angle H(f)$ [rad]"

    fig, (ax_mag, ax_ph) = plt.subplots(2, 1, sharex=True, figsize=(8, 4), dpi=160)

    ax_mag.loglog(f, mag, lw=1.3)
    ax_mag.set_ylabel(r"$|H(f)|$")
    ax_mag.grid(True, which="both", ls=":", alpha=0.4)
    ax_mag.set_title(_nice_title(title))

    ax_ph.semilogx(f, phase, lw=1.0)
    ax_ph.set_ylabel(phase_ylabel)
    ax_ph.set_xlabel(r"$f$ [Hz]")
    ax_ph.grid(True, which="both", ls=":", alpha=0.4)

    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def plot_overlay(
    curves: Sequence[tuple[str, np.ndarray, np.ndarray]] ,
    *,
    title: str,
    outfile: Path,
    phase_units: str = "rad",
) -> None:
    """
    Overlay multiple FRFs on one figure.
    curves: list of (label, f, H)
    """
    fig, (ax_mag, ax_ph) = plt.subplots(2, 1, sharex=True, figsize=(8, 4.5), dpi=160)

    for label, f, H in curves:
        mag = np.abs(H)
        ph = np.unwrap(np.angle(H))
        if phase_units.lower().startswith("deg"):
            ph = np.degrees(ph)
            phase_ylabel = r"$\angle H(f)$ [deg]"
        else:
            phase_ylabel = r"$\angle H(f)$ [rad]"
        ax_mag.loglog(f, mag, lw=1.2, label=label)
        ax_ph.semilogx(f, ph, lw=1.0, label=label)

    ax_mag.set_ylabel(r"$|H(f)|$")
    ax_mag.grid(True, which="both", ls=":", alpha=0.4)
    ax_mag.legend(fontsize=8, ncol=2)

    ax_ph.set_ylabel(phase_ylabel)
    ax_ph.set_xlabel(r"$f$ [Hz]")
    ax_ph.grid(True, which="both", ls=":", alpha=0.4)

    ax_mag.set_title(_nice_title(title))
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


# ------------------------------ Run descriptor --------------------------------
@dataclass
class FRFRun:
    """Description of one FRF computation."""
    label: str            # e.g., "0psi PH1->NC (far)"
    mat_path: Path        # path to .mat file
    key: str              # e.g., "channelData_LP"
    x_col: int            # input column (PH1=0, PH2=1, etc.)
    y_col: int            # output column (NC=2, etc.)
    nperseg: int = NPERSEG_DEFAULT
    ph: int | None = None # pinhole id (1 or 2). If None, will try to infer.


# ------------------------------ Main workflow --------------------------------
def _infer_tokens_for_saving(run: FRFRun) -> tuple[str, str, int]:
    """
    From run metadata/label, infer:
      - psi token like '0psig', '50psig'
      - spacing token like 'far', 'close', 'unknown'
      - ph id (1 or 2); prefer run.ph else from label, else from x_col
    """
    label_lc = run.label.lower()
    parts = run.label.split()
    psi_part_raw = next((p for p in parts if "psi" in p.lower()), "Xpsi").lower()
    psi_token = psi_part_raw.replace("psi", "psig")

    spacing = "unknown"
    for w in ("far", "close", "mid", "anechoic"):
        if w in label_lc:
            spacing = w
            break

    if run.ph in (1, 2):
        ph_id = run.ph
    elif "ph1" in label_lc:
        ph_id = 1
    elif "ph2" in label_lc:
        ph_id = 2
    else:
        ph_id = 1 if run.x_col == 0 else (2 if run.x_col == 1 else 0)

    return psi_token, spacing, ph_id


def compute_and_plot_runs(runs: Sequence[FRFRun], *, fs: float = FS) -> None:
    """Compute per-run FRFs and save per-run + overlay plots (and .npy as requested)."""
    overlay_curves: list[tuple[str, np.ndarray, np.ndarray]] = []

    for run in runs:
        x, y = load_columns_from_mat(run.mat_path, key=run.key, x_col=run.x_col, y_col=run.y_col)
        f, H, gamma2 = estimate_frf(x, y, fs, nperseg=run.nperseg)

        # Per-run figure (kept simple)
        out_png = OUTDIR / f"{run.label.replace(' ', '_')}.png"
        plot_frf(f, H, title=run.label, outfile=out_png)

        # Save NumPy arrays using your convention:
        #   700_{x}psig_vanilla_{spacing}_ph{1|2}_{f|H|gamma2}.npy
        if SAVE_NUMPY:
            psi_token, spacing, ph_id = _infer_tokens_for_saving(run)
            base = OUTDIR / f"700_{psi_token}_vanilla_{spacing}_ph{ph_id}"
            np.save(f"{base}_f.npy", f)
            np.save(f"{base}_H.npy", H)
            np.save(f"{base}_gamma2.npy", gamma2)

        overlay_curves.append((run.label, f, H))

    # Overlay (all runs)
    if overlay_curves:
        plot_overlay(
            overlay_curves,
            title="FRF overlay",
            outfile=OUTDIR / "overlay_all_runs.png",
        )


# --------------------------------- __main__ ----------------------------------
if __name__ == "__main__":
    # Example layout reminder for 'channelData_LP':
    #   column 0 -> PH1, column 1 -> PH2, column 2 -> NC
    #
    # Fill in your runs here. Example entries (uncomment & adjust paths):
    runs = [
        FRFRun("0psi PH1->NC (far)",
               Path("data/20251014/tf_calib/0psi_lp_16khz_ph1.mat"),
               key="channelData_LP", x_col=0, y_col=2, ph=1),
        FRFRun("0psi PH2->NC (far)",
               Path("data/20251014/tf_calib/0psi_lp_16khz_ph2.mat"),
               key="channelData_LP", x_col=1, y_col=2, ph=2),
        FRFRun("50psi PH1->NC (far)",
               Path("data/20251016/tf_calib/50psi_lp_16khz_ph1.mat"),
               key="channelData_LP", x_col=0, y_col=2, ph=1),
        FRFRun("50psi PH2->NC (far)",
               Path("data/20251016/tf_calib/50psi_lp_16khz_ph2.mat"),
               key="channelData_LP", x_col=1, y_col=2, ph=2),
        FRFRun("100psi PH1->NC (far)",
               Path("data/20251016/tf_calib/100psi_lp_16khz_ph1.mat"),
               key="channelData_LP", x_col=0, y_col=2, ph=1),
        FRFRun("100psi PH2->NC (far)",
               Path("data/20251016/tf_calib/100psi_lp_16khz_ph2.mat"),
               key="channelData_LP", x_col=1, y_col=2, ph=2),
    ]

    if not runs:
        print("Edit the 'runs' list in __main__ with your .mat paths/columns and re-run.")
    else:
        compute_and_plot_runs(runs, fs=FS)
