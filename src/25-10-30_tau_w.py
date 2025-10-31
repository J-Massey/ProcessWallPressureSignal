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

FS: float = 50_000.0
NPERSEG: int = 2**10
WINDOW: str = "hann"

# --- constants (keep once, top of file) ---
R = 287.05        # J/kg/K
PSI_TO_PA = 6_894.76
P_ATM = 101_325.0
DELTA = 0.035  # m, bl-height of 'channel'
TDEG = [18, 20, 20]

u_taus = [0.537, 0.522, 0.506]

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

def load_file(fn):
    dat = loadmat(fn)
    ic(dat.keys())
    s1 = dat['tau_w_1'].squeeze()
    s2 = dat['tau_w_2'].squeeze()
    return s1, s2

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

# def _plot_tauw():
#     fns = ['data/2025-10-30/atm_tauw.mat',
#            'data/2025-10-30/50psi_tauw.mat',
#            'data/2025-10-30/100psi_tauw.mat']
#     psigs = [0, 50, 100]
#     fig, ax = plt.subplots(1, 1, figsize=(7, 2.4), sharex=True)
#     for fn in fns:
#         s1, s2 = load_file(fn)
#         f, Ptt 1= compute_spec(FS, s1)
#         rho, mu, nu = air_props_from_gauge(psigs[fns.index(fn)], TDEG[fns.index(fn)]+273.15)
#         T_plus = u_taus[fns.index(fn)]**2/nu/f
#         ax.semilogx(T_plus, f*Ptt/(rho**2 * u_taus[fns.index(fn)]**4))
#     # ax.legend()

#     plt.savefig("figures/tau_w/spectra.png")

def _plot_tauw_speed_frequency():
    """
    Plot a contour map of inferred convective speed U [m/s] vs frequency f [Hz]
    from two time series measured at sensors separated by a fixed spacing.

    Method (phase-based, alias-aware):
      • Compute cross-spectrum G12(f) = P1(f) * conj(P2(f)) and its phase φ12(f).
      • For each candidate speed U on a grid, the predicted phase at spacing d is
            φ_pred(f; U) = (2π f d) / U  (wrapped to [-π, π]).
      • Build a likelihood L(f, U) by how well φ_pred matches φ12 (after wrapping),
        weighted by magnitude-squared coherence γ²(f).
      • Plot L as a contour map and overlay the ridge Û(f) = argmax_U L(f, U).

    Notes:
      • Requires globals: FS (sampling rate) and a `load_file(fn)` that returns (s1, s2).
      • The speed grid [umin, umax] and smoothing width `sigma` can be tuned per setup.
    """

    def _wrap_to_pi(x):
        # Wrap real-valued phase to [-π, π]
        return (x + np.pi) % (2 * np.pi) - np.pi

    # --- Config ---------------------------------------------------------------
    spacing = 0.1  # [m] sensor spacing
    fns   = ['data/2025-10-30/atm_tauw.mat',
             'data/2025-10-30/50psi_tauw.mat',
             'data/2025-10-30/100psi_tauw.mat']
    psigs = [0, 50, 100]  # labels for legend

    # Frequency / speed / weighting settings
    fmin = 5.0                      # ignore very low-frequency bins [Hz]
    fmax = 700.0                    # ignore very high-frequency bins [Hz]
    gamma_min = 0.05                # coherence threshold
    umin, umax = 5.0, 100.0         # speed bounds [m/s] (adjust as needed)
    n_u = 256                        # number of speed bins
    sigma = 0.35                    # [rad] phase error width for Gaussian likelihood
    u_grid = np.linspace(umin, umax, n_u)

    # Plot appearance
    cmaps  = ['Blues', 'Oranges', 'Greens']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    alpha  = 0.45                   # transparency for overlapping contours
    n_levels = 30


    ridge_lines = []

    for fn, psig, cmap, color in zip(fns, psigs, cmaps, colors):
        fig, ax = plt.subplots(1, 1, figsize=(7, 2.4))
        # --- Load signals -----------------------------------------------------
        s1, s2 = load_file(fn)
        s1 = np.asarray(s1, dtype=float)
        s2 = np.asarray(s2, dtype=float)

        # Basic conditioning (keep equal length, remove DC)
        n = min(len(s1), len(s2))
        if n % 2 == 1:
            n -= 1
        s1 = s1[:n] - np.mean(s1[:n])
        s2 = s2[:n] - np.mean(s2[:n])

        # --- Spectra ----------------------------------------------------------
        f = np.fft.rfftfreq(n, d=1.0 / FS)
        P1 = np.fft.rfft(s1)
        P2 = np.fft.rfft(s2)

        G12 = P1 * np.conj(P2)      # cross-spectrum
        S11 = (np.abs(P1) ** 2)
        S22 = (np.abs(P2) ** 2)

        # magnitude-squared coherence (unwindowed estimate; good enough for weighting)
        coh = (np.abs(G12) ** 2) / (S11 * S22 + 1e-30)
        coh = np.clip(coh, 0.0, 1.0)

        phi = np.angle(G12)

        # --- Select useful frequency bins ------------------------------------
        mask = (f > fmax) & (f >= fmin) & np.isfinite(phi) & (coh >= gamma_min)
        f_sel   = f[mask]
        phi_sel = phi[mask]
        coh_sel = coh[mask]

        if f_sel.size == 0:
            continue

        # --- Build likelihood L(f, U) ----------------------------------------
        # Predicted phase surface for all (f, U): φ_pred = (2π f d)/U
        omega = 2.0 * np.pi * f_sel           # [rad/s], shape (Nf,)
        phi_pred = (omega[:, None] * spacing) / u_grid[None, :]  # (Nf, Nu)

        # Phase misfit wrapped to [-π, π]
        dphi = _wrap_to_pi(phi_pred - phi_sel[:, None])

        # Gaussian phase likelihood, weighted by coherence
        # You can raise coherence to a power (e.g., 1.0) to sharpen/soften the weighting.
        L = np.exp(-0.5 * (dphi / sigma) ** 2) * (coh_sel[:, None])

        # Normalize per-dataset for plotting; log-like stretch improves contrast.
        Ln = L / (np.max(L) + 1e-12)
        # Plot as filled contours (speed on y, frequency on x)
        ax.contourf(f_sel, u_grid, Ln.T, levels=n_levels, cmap=cmap, alpha=alpha, antialiased=True)

        ax.set_xscale('log')
        ax.set_yscale('log')

        # Ridge (peak speed per frequency)
        u_hat = u_grid[np.argmax(L, axis=1)]
        ridge_lines.append((f_sel, u_hat, color, psig))

    # # --- Overlay ridges and finalize plot -------------------------------------
    # for fline, uline, color, psig in ridge_lines:
    #     ax.plot(fline, uline, color=color, lw=1.5, label=f'{psig} psig ridge')

        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Inferred speed [m/s]')
        ax.set_ylim(umin, umax)
        ax.set_xlim(fmin, fmax)
        ax.set_title(rf'Convective speed vs frequency from two sensors ($d={spacing:.3f}$ m)')
        ax.legend(loc='upper right', frameon=False)
        ax.grid(True, which='both', ls=':', lw=0.6)
        fig.tight_layout()

        plt.savefig(f"figures/tau_w/spectra_speed{psig}.png", dpi=600)



_plot_tauw_speed_frequency()
