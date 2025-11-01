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
from fit_speaker_scales import fit_speaker_scaling_from_files
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


def save_calibs(pressures):
    f_cut = [2_100, 4_700, 14_100]
    for i, pressure in enumerate(pressures):
        frequencies = np.arange(100, 3100, 100)
        dat = loadmat(TONAL_BASE + f"calib_{pressure}.mat")
        ph1, ph2, nc, _ = dat["channelData_WN"].T
        nc_pa = volts_to_pa(nc, "NC", f_cut[i])
        ph1_pa = volts_to_pa(ph1, "PH1", f_cut[i])
        f1, H1, g2_1 = estimate_frf(ph1_pa, nc_pa, fs=FS)
        # add calibration for ph2
        dat = loadmat(TONAL_BASE + f"calib_{pressure}.mat")
        ph1, ph2, nc, _ = dat["channelData_WN"].T
        nc_pa = volts_to_pa(nc, "NC", f_cut[i])
        ph2_pa = volts_to_pa(ph2, "PH2", f_cut[i])
        f2, H2, g2_2 = estimate_frf(ph2_pa, nc_pa, fs=FS)
        # Fuse the two transfer functions
        f_fused, H_fused, g2_fused = combine_anechoic_calibrations(
            f1, H1, g2_1,
            f2, H2, g2_2,
            gmin=0.4,
            smooth_oct=1/6,
            points_per_oct=32,
            eps=1e-12
        )
        # save frf
        with h5py.File(TONAL_BASE + f"calibs_{pressure}.h5", 'w') as hf:
            hf.create_dataset('frequencies', data=f1)
            hf.create_dataset('H1', data=H1)
            hf.create_dataset('H2', data=H2)
            hf.create_dataset('H_fused', data=H_fused)
            hf.attrs['psig'] = pressure


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

def save_scaling_target():
    fn_atm = '0psig_cleaned.h5'
    fn_50psig = '50psig_cleaned.h5'
    fn_100psig = '100psig_cleaned.h5'
    labels = ['0psig', '50psig', '100psig']
    colours = ['C0', 'C1', 'C2']

    pgs = [0, 50, 100]
    rho0, *_  = air_props_from_gauge(pgs[0], TDEG[0]+273)
    R = 1, 2, 4

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

        # R = rho/rho0
        # ic(R)

        g1_b, g2_b, rv_b = bl_model(T_plus, Re_tau, cf_2)
        bl_fphipp_plus = rv_b*(g1_b+g2_b)

        f_clean_clipped = f_clean[f_clean < 1_000]
        bl_fphipp_plus = bl_fphipp_plus[f_clean < 1_000]

        f_clean_tf, Pyy_ph1_clean_tf = compute_spec(FS, ph1_clean)
        f_clean_tf, Pyy_ph2_clean_tf = compute_spec(FS, ph2_clean)

        T_plus_tf = 1/f_clean_tf * (u_tau**2)/nu

        data_fphipp_plus1_tf = (f_clean_tf * Pyy_ph1_clean_tf)/(rho**2 * u_tau**4)
        data_fphipp_plus2_tf = (f_clean_tf * Pyy_ph2_clean_tf)/(rho**2 * u_tau**4)

        # clip at the helmholtz resonance
        f_clean = f_clean_tf[f_clean_tf < 1_000]
        data_fphipp_plus1_tf_m = data_fphipp_plus1_tf[f_clean_tf < 1_000]
        data_fphipp_plus2_tf_m = data_fphipp_plus2_tf[f_clean_tf < 1_000]

        model_data_ratio1 = np.sqrt(data_fphipp_plus1_tf_m / bl_fphipp_plus)
        model_data_ratio2 = np.sqrt(data_fphipp_plus2_tf_m / bl_fphipp_plus)

        model_ratio_avg = (model_data_ratio1 + model_data_ratio2) / 2
        with h5py.File(TONAL_BASE + f"lumped_scaling_{labels[idxfn]}.h5", 'w') as hf:
            hf.create_dataset('frequencies', data=f_clean)
            hf.create_dataset('scaling_ratio', data=model_ratio_avg)
            hf.attrs['rho'] = rho
            hf.attrs['u_tau'] = u_tau
            hf.attrs['nu'] = nu
            hf.attrs['psig'] = pgs[idxfn]


def plot_target_calib_modeled(
    labels: tuple[str, ...] = ("0psig", "50psig", "100psig"),
    *,
    fmin: float = 100.0,
    fmax: float = 1000.0,
    f_ref: float = 500.0,
    rho_ref: float | None = None,
    invert_target: bool = True,   # saved "scaling_ratio" is model/data → required |H| = 1/scaling_ratio
    to_db: bool = False,          # set True to plot in dB
    colours: list[str] | None = None,
    savepath: str | None = None,
):
    # ----------------------------
    # 1) Fit both scaling models
    # ----------------------------
    # (a) your baseline power-law (ρ, f) model
    (c0_db, a_rho, b_f), scale_powerlaw, diag_pw = fit_speaker_scaling_from_files(
        labels=labels, fmin=fmin, fmax=fmax, f_ref=f_ref, rho_ref=rho_ref, invert_target=invert_target
    )
    ic(c0_db, a_rho, b_f)
    # ----------------------------
    # 2) helpers & styling
    # ----------------------------
    def as_db(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        return 20.0 * np.log10(np.maximum(x, 1e-16))

    if colours is None:
        colours = ['C0', 'C1', 'C2', 'C3', 'C4']

    fig, ax = plt.subplots(1, 1, figsize=(7, 3), tight_layout=True)

    # store RMSE (dB) per label & per model
    rmse_pw, rmse_stk = {}, {}

    # ----------------------------
    # 3) loop over labels
    # ----------------------------
    for i, L in enumerate(labels):
        color = colours[i % len(colours)]

        # --- load target (required |H|)
        with h5py.File(TONAL_BASE + f"lumped_scaling_{L}.h5", "r") as hf:
            f_tgt = np.asarray(hf["frequencies"][:], float)
            s_ratio = np.asarray(hf["scaling_ratio"][:], float)  # (model/data) in POWER
            rho = float(hf.attrs["rho"]) if "rho" in hf.attrs else np.nan
            nu = float(hf.attrs["nu"]) if "nu" in hf.attrs else np.nan  # preferred for Stokes
        tgt_mag = 1.0 / np.maximum(s_ratio, 1e-16) if invert_target else s_ratio  # required |H| (AMPLITUDE)
        mt = (f_tgt >= fmin) & (f_tgt <= fmax)
        f_tgt, tgt_mag = f_tgt[mt], tgt_mag[mt]

        # --- load measured calibration |H_cal|
        with h5py.File(TONAL_BASE + f"calibs_{L}.h5", "r") as hf:
            f_cal = np.asarray(hf["frequencies"][:], float)
            H_cal = np.asarray(hf["H_fused"][:], complex)
        cal_mag = np.abs(H_cal)
        mc = (f_cal >= fmin) & (f_cal <= fmax)
        f_cal, cal_mag = f_cal[mc], cal_mag[mc]

        # --- build modeled curves
        # power-law model uses (f, rho) directly
        S_pw = scale_powerlaw(f_cal, rho)

        modeled_pw = cal_mag * S_pw
        # --- plot (target, measured, both models)
        ax.semilogx(f_tgt, as_db(tgt_mag), color=color, lw=1.6)                     # target
        ax.semilogx(f_cal, as_db(cal_mag), color=color, lw=1.0, ls="--", alpha=0.9) # measured cal
        ax.semilogx(f_cal, as_db(modeled_pw), color=color, lw=1.0, ls=":", alpha=0.95)   # power-law modeled

        # --- RMSE on the target grid (in dB) for both models
        pw_on_tgt  = np.interp(f_tgt, f_cal, modeled_pw)
        e_pw  = as_db(tgt_mag) - as_db(pw_on_tgt)
        rmse_pw[L]  = float(np.sqrt(np.mean(e_pw**2)))

    # ----------------------------
    # 4) axes & legends
    # ----------------------------
    ax.set_xlabel(r"$f$ [Hz]")
    ax.set_ylabel("Magnitude (dB)" if to_db else "Magnitude")
    ax.set_xlim(fmin, fmax)
    ax.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

    # Legend 1: styles (one entry per curve type)
    style_handles = [
        Line2D([0], [0], color='k', lw=1.6, ls='-',  label="target |H| (required)"),
        Line2D([0], [0], color='k', lw=1.0, ls='--', label="|H_cal| (meas)"),
        Line2D([0], [0], color='k', lw=1.0, ls=':',  label="cal x S_powerlaw"),
    ]
    leg1 = ax.legend(handles=style_handles, loc="upper left", fontsize=8, framealpha=0.9)

    # Legend 2: colors (one entry per label)
    color_handles = [Line2D([0], [0], color=colours[i % len(colours)], lw=1.6, label=labels[i]) for i in range(len(labels))]
    leg2 = ax.legend(handles=color_handles, loc="lower right", fontsize=8, framealpha=0.9)
    ax.add_artist(leg1)

    # Optional: print compact RMSE summary to console
    print("RMSE (dB) power-law:", rmse_pw)

    if savepath:
        fig.savefig(savepath, dpi=350)
    return fig, ax



def plot_tf_model_comparison():
    """
    Apply the (rho, f)-scaled calibration FRF to measured time series and plot
    pre-multiplied, normalized spectra:  f * Pyy / (rho^2 * u_tau^4).
    """
    # --- fit rho–f scaling once from your saved target + calibration ---
    labels = ['0psig', '50psig', '100psig']
    (c0_db, a, b), scale, diag = fit_speaker_scaling_from_files(
        labels=tuple(labels),
        fmin=100.0, fmax=1000.0,   # fitting band
        f_ref=500.0,
        invert_target=True         # your "scaling_ratio" -> required |H| = 1/scaling_ratio
    )


    # files per label
    fn_map = {
        '0psig': '0psig_cleaned.h5',
        '50psig': '50psig_cleaned.h5',
        '100psig': '100psig_cleaned.h5',
    }
    colours = ['C0', 'C1', 'C2']

    # (rho0 only used if you want to compare densities directly elsewhere)
    pgs = [0, 50, 100]
    
    u_tau_uncertainty = [0.2, 0.1, 0.05]

    fig, ax = plt.subplots(3, 1, figsize=(7, 4), tight_layout=True, sharex=True, sharey=True)

    # --- main loop over datasets ---
    for i, L in enumerate(labels):
        fn = fn_map[L]
        color = colours[i]

        # Load cleaned signals and attributes
        with h5py.File(f'data/{fn}', 'r') as hf:
            ph1_clean = hf['ph1_clean'][:]
            ph2_clean = hf['ph2_clean'][:]
            u_tau = float(hf.attrs['u_tau'])
            nu = float(hf.attrs['nu'])
            rho = float(hf.attrs['rho'])
            # f_cut = hf.attrs.get('f_cut', np.nan)  # unused here
            Re_tau = hf.attrs.get('Re_tau', np.nan)
            # cf_2   = hf.attrs.get('cf_2',   np.nan)
        f_cal = np.load(TONAL_BASE + f"wn_frequencies_{L}.npy").astype(float)
        H_cal = np.load(TONAL_BASE + f"wn_H1_{L}.npy")

        # Plot bl model
        T_plus = (u_tau**2 / nu) / f_cal
        g1, g2, rv = bl_model(T_plus, Re_tau, 2*(u_tau**2)/14.0**2)
        ax[i].semilogx(f_cal, (g1+g2)*rv, color=colours[i], lw=0.6, ls='--')
        ic(Re_tau)

        # Load measured calibration FRF (frequency + complex H)

        # --- apply FRF with fitted rho–f magnitude scaling ---
        # (uses your updated apply_frf that accepts scale_fn and rho)
        ph1_filt = apply_frf(ph1_clean, FS, f_cal, H_cal, rho=rho, scale_fn=scale)
        ph2_filt = apply_frf(ph2_clean, FS, f_cal, H_cal, rho=rho, scale_fn=scale)

        # --- PSDs ---
        f1, Pyy1 = compute_spec(FS, ph1_filt)
        f2, Pyy2 = compute_spec(FS, ph2_filt)
        # unify grids (Welch outputs match with identical settings)
        if not np.allclose(f1, f2):
            # If extremely picky, interpolate one onto the other; here we pick f1
            Pyy2 = np.interp(f1, f2, Pyy2)
        f_sp = f1

        # --- pre-multiplied, normalized spectra ---
        Y1_pm = (f_sp * Pyy1) / (rho**2 * u_tau**4)
        Y2_pm = (f_sp * Pyy2) / (rho**2 * u_tau**4)

        # clip to display band (avoid Helmholtz etc.)
        band = (f_sp >= 50.0) & (f_sp < 1000.0)
        f_plot = f_sp[band]
        y1_plot = Y1_pm[band]
        y2_plot = Y2_pm[band]

        # plot PH1 & PH2 for this label
        ax[i].semilogx(f_plot, y1_plot, linestyle='-', color=colours[i], alpha=0.8, lw=0.8)
        ax[i].semilogx(f_plot, y2_plot, linestyle='-', color=colours[i], alpha=0.8, lw=0.8)



        # ±10% u_tau uncertainty bands (repeat for each channel)
        u_low, u_high = u_tau*(1 - u_tau_uncertainty[i]), u_tau*(1 + u_tau_uncertainty[i])
        y1_upper = ((f_sp * Pyy1) / (rho**2 * u_low**4))[band]
        y1_lower = ((f_sp * Pyy1) / (rho**2 * u_high**4))[band]
        y2_upper = ((f_sp * Pyy2) / (rho**2 * u_low**4))[band]
        y2_lower = ((f_sp * Pyy2) / (rho**2 * u_high**4))[band]
        ax[i].fill_between(f_plot, y1_upper, y1_lower, color=colours[i], alpha=0.25, edgecolor='none')
        ax[i].fill_between(f_plot, y2_upper, y2_lower, color=colours[i], alpha=0.25, edgecolor='none')
        ax[i].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
        ax[i].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

    # --- axes, legend, save ---
    ax[2].set_xlabel(r"$f$ [Hz]")
    ax[1].set_ylabel(r"${f \,\phi_{pp}}^+$")
    ax[0].set_xlim(50, 1_000)
    ax[0].set_ylim(0, 4)

    labels_handles = ['0 psig Data', '50 psig Data', '100 psig Data', 'Model']
    label_colours = ['C0', 'C1', 'C2', 'k']
    label_styles = ['solid',  'solid', 'solid', 'dashed']
    custom_lines = [Line2D([0], [0], color=label_colours[i], linestyle=label_styles[i]) for i in range(len(labels_handles))]
    ax[0].legend(custom_lines, labels_handles, loc='upper left', fontsize=8)

    # title_params = f"scaled FRF: c0={c0_db:.2f} dB, a={a:.3f}, b={b:.3f} | " \
    #                f"ρ_ref={diag['rho_ref']:.3g} kg/m³, f_ref={diag['f_ref']:.0f} Hz"
    # ax.set_title(title_params, fontsize=9)

    fig.savefig('figures/tonal_ratios/spectra_comparison_tf_freq_scaled.png', dpi=410)




if __name__ == "__main__":
    # scale_0psig(['0psig', '50psig', '100psig'])
    # plot_tf_model_comparison_stokes()
    plot_target_calib_modeled(
        labels=('0psig', '50psig', '100psig'),
        fmin=100.0,
        fmax=1000.0,
        to_db=True,
        savepath='figures/tonal_ratios/target_calib_modeled_comparison_db.png',
    )
    plot_tf_model_comparison()
    # psigs = [0, 50, 100]
    # labels = [f"{psig}psig" for psig in psigs]
    # save_calibs(labels)
    # save_scaling_target()
    # plot_TFs_and_ratios()