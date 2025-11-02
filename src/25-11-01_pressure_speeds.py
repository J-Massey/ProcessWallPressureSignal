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
import os


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
NPERSEG: int = 2**15
WINDOW: str = "hann"

# --- constants (keep once, top of file) ---
R = 287.05        # J/kg/K
PSI_TO_PA = 6_894.76
P_ATM = 101_325.0
DELTA = 0.035  # m, bl-height of 'channel'
TDEG = [18, 20, 20]

u_taus = [0.537, 0.522, 0.506]

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

def _plot_tauw():
    fns = ['data/20251030/atm_tauw.mat',
           'data/20251030/50psi_tauw.mat',
           'data/20251030/100psi_tauw.mat']
    psigs = [0, 50, 100]
    fig, ax = plt.subplots(1, 1, figsize=(7, 2.4), sharex=True)
    for fn in fns:
        s1, s2 = load_file(fn)
        f, Ptt1= compute_spec(FS, s1)
        f, Ptt2= compute_spec(FS, s2)
        rho, mu, nu = air_props_from_gauge(psigs[fns.index(fn)], TDEG[fns.index(fn)]+273.15)
        T_plus = u_taus[fns.index(fn)]**2/nu/f
        ax.semilogx(T_plus, f*Ptt1/(rho**2 * u_taus[fns.index(fn)]**4))
        ax.semilogx(T_plus, f*Ptt2/(rho**2 * u_taus[fns.index(fn)]**4))
    # ax.legend()
    ax.set_xlabel('$T^+$')
    ax.set_ylabel(r'${f \Phi_{\tau_w\tau_w}}^+ = f \Phi_{\tau_w\tau_w} / \rho^2 u_\tau^4$')

    plt.savefig("figures/tau_w/spectra.png")

def _plot_tauw_speed_frequency():
    """
    Plot a contour of *energy carried by structures* convecting at speed U vs. T+.

    Uses two-sensor steered coherent power:
        P_coh(f,U) = 2 * Re{ G12(f) * exp(-j * phi_pred(f,U)) }
    and optionally normalizes by (S11+S22) to get a bounded fraction in [0,1].

    Requires:
      - FS (Hz)
      - load_file(fn) -> (s1, s2)
      - air_props_from_gauge(psig, T_kelvin) -> (rho, mu, nu)
      - arrays: TDEG (°C per condition), u_taus (m/s per condition)
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    def _wrap_to_pi(x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    # ---------------- Config ----------------
    spacing = 0.035 * 3  # [m] sensor spacing
    fns   = ['data/20251030/atm_tauw.mat',
             'data/20251030/50psi_tauw.mat',
             'data/20251030/100psi_tauw.mat']
    psigs = [0, 50, 100]

    # Frequency / speed grids
    fmin, fmax = 10.0, 700.0
    umin, umax = 2.0, 17.0
    n_u = 256
    u_grid = np.linspace(umin, umax, n_u)

    # Gates / display
    gamma_min = 0.10   # looser than 1.0 if you want to see energy where γ² isn't perfect
    # Choose how to display energy:
    energy_mode = 'fraction'  # 'fraction' (0..1) or 'absolute'
    coherence_power = 0.0     # optional extra weight: multiply by γ^(coherence_power)
                               # set 0.0 to avoid double-counting coherence

    cmaps  = ['Blues', 'Oranges', 'Greens']
    alpha  = 0.85            # a bit more opaque since we plot one condition per file
    n_levels = 40

    os.makedirs("figures/tau_w", exist_ok=True)

    for fn, psig, cmap in zip(fns, psigs, cmaps):
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 2.8))

        # ------------- Load & condition signals -------------
        s1, s2 = load_file(fn)
        s1 = np.asarray(s1, float)
        s2 = np.asarray(s2, float)
        n = min(len(s1), len(s2))
        if n % 2 == 1:
            n -= 1
        s1 = s1[:n] - np.mean(s1[:n])
        s2 = s2[:n] - np.mean(s2[:n])

        # ------------- Spectra -------------
        f  = np.fft.rfftfreq(n, d=1.0 / FS)
        X1 = np.fft.rfft(s1)
        X2 = np.fft.rfft(s2)

        S11 = (np.abs(X1) ** 2)
        S22 = (np.abs(X2) ** 2)
        G12 = X1 * np.conj(X2)

        # Magnitude-squared coherence for gating/optionally weighting
        coh = (np.abs(G12) ** 2) / (S11 * S22 + 1e-30)
        coh = np.clip(coh, 0.0, 1.0)

        # ------------- Select bins -------------
        mask = (f >= fmin) & (f <= fmax) & np.isfinite(coh) & (coh >= gamma_min)
        f_sel   = f[mask]
        S11_sel = S11[mask]
        S22_sel = S22[mask]
        G12_sel = G12[mask]
        coh_sel = coh[mask]

        if f_sel.size < 2:
            ax.text(0.5, 0.5,
                    f'No usable bins after mask\n{psig} psig\n'
                    f'f range: {f.min():.1f}–{f.max():.1f} Hz\n'
                    f'mask: {fmin:.1f}–{fmax:.1f} Hz, γ² ≥ {gamma_min:.2f}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            fig.tight_layout()
            plt.savefig(f"figures/tau_w/spectra_speed{psig}.png", dpi=600)
            plt.close(fig)
            continue

        # ------------- Energy vs speed (beamformer view) -------------
        # Predicted phase for every (f,U)
        omega = 2.0 * np.pi * f_sel     # [rad/s]
        phi_pred = (omega[:, None] * spacing) / u_grid[None, :]  # shape (Nf, Nu)

        # Cross term that carries the U-dependence:
        #   cross = 2 * Re{ G12 * exp(-j * phi_pred) }
        cross_term = 2.0 * np.real(G12_sel[:, None] * np.exp(-1j * phi_pred))

        # Keep only positive (matched) energy; optionally apply a small coherence weight
        cross_pos = np.maximum(0.0, cross_term)
        if coherence_power != 0.0:
            cross_pos = cross_pos * (coh_sel[:, None] ** coherence_power)

        if energy_mode == 'absolute':
            # Absolute coherent cross-power (arb. units of your periodogram)
            E_map = cross_pos
            cbar_label = 'Coherent cross-power (arb.)'
        else:
            # Fraction of total single-sensor power that is coherently convecting at U
            denom = (S11_sel[:, None] + S22_sel[:, None]) + 1e-30
            E_map = np.clip(cross_pos / denom, 0.0, 1.0)
            cbar_label = 'Coherent power fraction'

        # ------------- Convert axes to (T+, u+) -------------
        # T+ = u_tau^2 / (nu * f) ; u+ = U / u_tau
        idx = psigs.index(psig)
        _, _, nu = air_props_from_gauge(psig, TDEG[idx] + 273.15)
        u_tau = u_taus[idx]

        T_plus_sel = (u_tau**2 / (nu * f_sel))
        u_grid_plus = u_grid / u_tau

        # ------------- Plot -------------
        # Z needs shape (len(y), len(x)) so transpose
        # For 'fraction' the range is [0,1]; lock levels for comparability
        if energy_mode == 'fraction':
            levels = np.linspace(0.0, 1.0, n_levels)
            vmin, vmax = 0.0, 1.0
        else:
            # Absolute scale varies by condition; auto levels are OK
            levels, vmin, vmax = n_levels, None, None

        cf = ax.contourf(T_plus_sel, u_grid_plus, E_map.T,
                         levels=levels, cmap=cmap, alpha=alpha,
                         vmin=vmin, vmax=vmax, antialiased=True)
        cb = fig.colorbar(cf, ax=ax, pad=0.01)
        cb.set_label(cbar_label)

        # Ridge of max coherent energy (optional but useful)
        u_hat = u_grid[np.argmax(E_map, axis=1)]

        # Smooth ridge for visibility
        spline = UnivariateSpline(np.log(T_plus_sel), u_hat, s=5.0)
        u_hat_smooth = spline(np.log(T_plus_sel))
        ax.plot(T_plus_sel, (u_hat_smooth / u_tau), lw=1.0, color='k', alpha=0.7, label='max energy ridge')
        # ax.plot(T_plus_sel, (u_hat / u_tau), lw=0.5, color='k', alpha=0.3, label='max energy ridge')

        ax.set_xscale('log')
        ax.set_yscale('log')

        # T+ bounds: show a practical window
        ax.set_xlim(250, 1e4)
        ax.set_ylim(umin / u_tau, umax / u_tau)

        ax.set_xlabel(r'$T^+ = u_\tau^2/(\nu f)$')
        ax.set_ylabel(r'Convective speed $u^+=U/u_\tau$')
        ax.set_title(rf'Coherent energy vs $T^+$, $u^+$ ($d={spacing:.3f}$ m, {psig} psig)')
        ax.legend(loc='upper right', frameon=False)
        ax.grid(True, which='both', ls=':', lw=0.6)
        fig.tight_layout()

        plt.savefig(f"figures/tau_w/spectra_speed{psig}.png", dpi=600)
        plt.close(fig)


def _plot_tauw_speed_frequency_welch():
    """
    Welch-averaged map of *energy carried by structures* convecting at speed U,
    shown vs. T+ and u+ for each condition.

    Core quantity (beamformer view):
        P_coh(f,U) = 2 * Re{ G12(f) * exp(-j * phi_pred(f,U)) }
        where phi_pred(f,U) = 2π f d / U

    Display:
      - 'fraction' (default): P_coh / (S11+S22) clipped to [0,1]
      - 'absolute': P_coh (same units as the Welch PSDs)

    Requires globals:
      - FS (Hz)
      - load_file(fn) -> (s1, s2)
      - air_props_from_gauge(psig, T_kelvin) -> (rho, mu, nu)
      - arrays: TDEG (°C per condition), u_taus (m/s per condition)
    """
    EPS = 1e-30

    # ---------------- Config ----------------
    spacing = 0.035 * 3  # [m] sensor spacing
    fns   = ['data/20251030/atm_tauw.mat',
             'data/20251030/50psi_tauw.mat',
             'data/20251030/100psi_tauw.mat']
    psigs = [0, 50, 100]

    # Frequency / speed grids
    fmin, fmax = 10.0, 700.0
    umin, umax = 2.0, 17.0
    n_u = 512
    u_grid = np.linspace(umin, umax, n_u)

    # Coherence gate & plotting
    gamma_min = 0.07        # relax gate to avoid throwing away meaningful energy
    energy_mode = 'fraction' # 'fraction' (bounded) or 'absolute'
    cmaps  = ['Greens', 'Oranges', 'Purples']
    alpha  = 0.95
    n_levels = 40

    for fn, psig, cmap in zip(fns, psigs, cmaps):
        # -------- Load & condition ----------
        s1, s2 = load_file(fn)
        s1 = np.asarray(s1, float)
        s2 = np.asarray(s2, float)
        n = min(len(s1), len(s2))
        if n % 2 == 1:
            n -= 1
        s1 = s1[:n] - np.mean(s1[:n])
        s2 = s2[:n] - np.mean(s2[:n])


        # -------- Welch spectra ----------
        f, S11 = welch(s1, fs=FS, window=WINDOW, nperseg=NPERSEG,
                       detrend='constant', return_onesided=True, scaling='density')
        _, S22 = welch(s2, fs=FS, window=WINDOW, nperseg=NPERSEG,
                       detrend='constant', return_onesided=True, scaling='density')
        _, G12 = csd(s1, s2, fs=FS, window=WINDOW, nperseg=NPERSEG,
                     detrend='constant', return_onesided=True, scaling='density')
        # Keep only desired frequency band
        band = (f >= fmin) & (f <= fmax)
        f_sel   = f[band]
        S11_sel = S11[band]
        S22_sel = S22[band]
        G12_sel = G12[band]

        # Coherence (with Welch-averaged spectra)
        coh_sel = (np.abs(G12_sel) ** 2) / (S11_sel * S22_sel + EPS)
        coh_sel = np.clip(coh_sel, 0.0, 1.0)

        # Optionally gate by coherence
        mask = (coh_sel >= gamma_min)
        f_sel   = f_sel[mask]
        S11_sel = S11_sel[mask]
        S22_sel = S22_sel[mask]
        G12_sel = G12_sel[mask]
        coh_sel = coh_sel[mask]

        fig, ax = plt.subplots(1, 1, figsize=(7.2, 2.8))
        # -------- Energy vs speed (beamformer view) ----------
        # phi_pred for all (f, U)
        omega = 2.0 * np.pi * f_sel
        phi_pred = (omega[:, None] * spacing) / u_grid[None, :]  # (Nf, Nu)

        # Coherent cross-power term steered to speed U
        cross = 2.0 * np.real(G12_sel[:, None] * np.exp(-1j * phi_pred))
        cross_pos = np.maximum(0.0, cross)  # keep only matched (positive) energy

        if energy_mode == 'absolute':
            E_map = cross_pos
            cbar_label = 'Coherent cross-power (arb.)'
            levels, vmin, vmax = n_levels, None, None
        else:
            denom = (S11_sel[:, None] + S22_sel[:, None]) + EPS
            E_map = np.clip(cross_pos / denom, 0.0, 1.0)
            cbar_label = 'Coherent power fraction'
            levels = np.linspace(0.0, 1.0, n_levels)
            vmin, vmax = 0.0, 1.0

        # -------- Convert axes to (T+, u+) ----------
        idx = psigs.index(psig)
        _, _, nu = air_props_from_gauge(psig, TDEG[idx] + 273.15)
        u_tau = u_taus[idx]
        T_plus_sel = (u_tau**2) / (nu * f_sel)   # (Nf,)
        u_grid_plus = u_grid / u_tau             # (Nu,)

        # -------- Plot ----------
        cf = ax.contourf(T_plus_sel, u_grid_plus, E_map.T,
                         levels=levels, cmap=cmap, alpha=alpha,
                         vmin=vmin, vmax=vmax, antialiased=True)
        cb = fig.colorbar(cf, ax=ax, pad=0.01)
        cb.set_label(cbar_label)
        cb.ax.tick_params(labelsize=8)
        cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # Ridge of max coherent energy (useful guide)
        u_hat = u_grid[np.argmax(E_map, axis=1)] / u_tau
        # ax.plot(T_plus_sel, u_hat, lw=1.0, color='k', alpha=0.6, label='max energy ridge')

        ax.set_xscale('log')
        ax.set_yscale('log')

        # Practical T+ window; clamp to available data
        xmin, xmax = T_plus_sel.min(), T_plus_sel.max()
        lo, hi = 300.0, 7000
        ax.set_xlim(lo, hi)
        ax.set_ylim(umin / u_tau, umax / u_tau)

        ax.set_xlabel(r'$T^+ = u_\tau^2/(\nu f)$')
        ax.set_ylabel(r'$c^+=U/u_\tau$')
        ax.set_title(rf'Coherent energy vs $T^+$, $u^+$ ($d={spacing:.3f}$ m, {psig} psig)')
        ax.legend(loc='upper right', frameon=False)
        ax.grid(True, which='both', ls=':', lw=0.6)
        fig.tight_layout()

        plt.savefig(f"figures/tau_w/spectra_speed{psig}.png", dpi=600)
        plt.close(fig)

def _plot_speed_vs_Tplus_from_csd():
    """
    Raw two-sensor speed estimation from the CSD phase:
      For each Welch frequency f, compute alias speeds U_m(f) that satisfy the
      measured cross-phase, then plot (T+, c+=U/u_tau) colored by the CSD.

    color_mode:
      - 'absG12'   : |G12(f)| (log color scale)
      - 'fraction' : 2|G12|/(S11+S22) in [0,1]
      - 'coh'      : gamma^2 = |G12|^2/(S11*S22) in [0,1]
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from scipy.signal import welch, csd

    EPS = 1e-30
    TWOPI = 2.0 * np.pi

    # --- Config ---
    spacing = 0.035 * 2.8  # [m] sensor separation d
    fn = 'data/final_pressure/SU_2pt_pressure.h5'
    with h5py.File(fn, 'r') as f:
        grp = f['corrected_data']

        psigs = [0, 50, 100]

        fmin, fmax = 100.0, 1000.0
        Umin, Umax = 1.0, 500.0          # [m/s] display range for speeds
        gamma_min = 0.05                # coherence gate (to keep noisy bins out)
        color_mode = 'absG12'           # 'absG12', 'fraction', or 'coh'
        s_marker = 8                    # scatter marker size
        alpha = 0.8

        for psig in psigs:
            # --- Load & condition ---
            s1 = grp[f'{psig}psig_close/ph1'][:]
            s2 = grp[f'{psig}psig_close/ph2'][:]
            s1 = np.asarray(s1, float); s1 -= s1.mean()
            s2 = np.asarray(s2, float); s2 -= s2.mean()

            # --- Welch spectra ---
            f, S11 = welch(s1, fs=FS, window=WINDOW, nperseg=NPERSEG,
                        detrend='constant', return_onesided=True, scaling='density')
            _, S22 = welch(s2, fs=FS, window=WINDOW, nperseg=NPERSEG,
                        detrend='constant', return_onesided=True, scaling='density')
            _, G12 = csd(s1, s2, fs=FS, window=WINDOW, nperseg=NPERSEG,
                        detrend='constant', return_onesided=True, scaling='density')

            # Select band
            band = (f >= fmin) & (f <= fmax)
            f, S11, S22, G12 = f[band], S11[band], S22[band], G12[band]

            # Coherence & gate
            coh = (np.abs(G12)**2) / (S11 * S22 + EPS)
            good = (coh >= gamma_min)
            f, S11, S22, G12, coh = f[good], S11[good], S22[good], G12[good], coh[good]

            # Inner units for x/y
            idx = psigs.index(psig)
            _, _, nu = air_props_from_gauge(psig, TDEG[idx] + 273.15)
            u_tau = u_taus[idx]
            Tplus = (u_tau**2) / (nu * f)   # (Nf,)

            # Build point cloud (T+, c+) with color from CSD
            xs, ys, cs = [], [], []

            # Precompute color value per f (same for every alias of that f)
            if color_mode == 'absG12':
                cval = np.abs(G12)                      # positive, wide dynamic range
                cbar_label = r'$|G_{12}(f)|$ (arb.)'
                use_log = True
            elif color_mode == 'fraction':
                cval = np.clip(2.0*np.abs(G12)/(S11+S22+EPS), 0.0, 1.0)
                cbar_label = 'Coherent power fraction'
                use_log = False
            elif color_mode == 'coh':
                cval = np.clip(coh, 0.0, 1.0)
                cbar_label = r'$\gamma^2(f)$'
                use_log = False
            else:
                raise ValueError("color_mode must be 'absG12', 'fraction', or 'coh'")

            # For each f, compute alias speeds U_m in [Umin, Umax] and add scatter points
            for i, (fi, phi, cv) in enumerate(zip(f, np.angle(G12), cval)):
                # denominators that yield speeds within bounds:
                dmin = TWOPI * fi * spacing / Umax  # smallest denom
                dmax = TWOPI * fi * spacing / Umin  # largest denom
                m_min = int(np.ceil((dmin - phi) / TWOPI))
                m_max = int(np.floor((dmax - phi) / TWOPI))
                if m_max < m_min:
                    continue

                for m in range(m_min, m_max + 1):
                    denom = phi + TWOPI * m
                    if np.abs(denom) < 1e-12:
                        continue  # avoid infinite speed
                    U = (TWOPI * fi * spacing) / denom
                    if U <= 0 or U < Umin or U > Umax:
                        continue
                    xs.append(Tplus[i])
                    ys.append(U / u_tau)   # c+ = U / u_tau
                    cs.append(cv)

            # --- Plot ---
            fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.0))
            xs = np.asarray(xs); ys = np.asarray(ys); cs = np.asarray(cs)

            if color_mode == 'absG12':
                # log-normalize by robust percentiles to avoid a few outliers dominating
                lo = np.percentile(cs[cs > 0], 5)
                hi = np.percentile(cs, 95)
                lo = max(lo, 1e-16)
                norm = LogNorm(vmin=lo, vmax=hi)
                cmap = 'viridis'
            else:
                norm = Normalize(vmin=0.0, vmax=1.0)
                cmap = 'viridis'

            sc = ax.scatter(xs, ys, c=cs, s=s_marker, cmap=cmap, norm=norm, alpha=alpha, edgecolors='none')

            cb = fig.colorbar(sc, ax=ax, pad=0.01)
            cb.set_label(cbar_label)
            if color_mode != 'absG12':
                cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])

            # Optional alias-free boundary: c+ > 2 d+/T+
            d_plus = spacing * u_tau / nu
            T_line = np.logspace(np.log10(max(xs.min(), 200)), np.log10(min(xs.max(), 1e4)), 300)
            ax.plot(T_line, 2.0 * d_plus / T_line, 'k--', lw=0.9, label=r'alias-free $c^+>2d^+/T^+$')

            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlim(10, 7000)
            ax.set_ylim(Umin / u_tau, Umax / u_tau)

            ax.set_xlabel(r'$T^+ = u_\tau^2/(\nu f)$')
            ax.set_ylabel(r'$c^+=U/u_\tau$')
            ax.set_title(rf'Raw CSD-inferred speeds vs $T^+$ ($d={spacing:.3f}$ m, {psig} psig)')
            ax.legend(loc='upper right', frameon=False)
            ax.grid(True, which='both', ls=':', lw=0.6)
            fig.tight_layout()

            out = f"figures/pressure_speed/speed_vs_Tplus_raw_{color_mode}_{psig}.png"
            plt.savefig(out, dpi=600)
            plt.close(fig)
            print(f"[OK] saved {out}")



if __name__ == "__main__":
    # save_hdf5()
    # _plot_tauw()
    # _plot_tauw_speed_frequency()
    # _plot_tauw_speed_frequency_welch()
    _plot_speed_vs_Tplus_from_csd()