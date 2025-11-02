# tf_compute.py
from __future__ import annotations

import numpy as np
import h5py
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
from clean_raw_data import air_props_from_gauge
from save_calibs import save_calibs
from save_scaling_target import compute_spec, save_scaling_target

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
CALIB_BASE = "data/final_calibration/"
TARGET_BASE = "data/final_target/"
CLEANED_BASE = "data/final_cleaned/"


def plot_target_calib_modeled(
    labels: tuple[str, ...] = ("0psig", "50psig", "100psig"),
    *,
    fmin: float = 100.0,
    fmax: float = 1000.0,
    f_ref: float = 700.0,
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
        # labels=labels, fmin=fmin, fmax=fmax, f_ref=f_ref, rho_ref=rho_ref, invert_target=invert_target
    )
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
        with h5py.File(TARGET_BASE + f"target_{L}_close.h5", "r") as hf:
            f_tgt = np.asarray(hf["frequencies"][:], float)
            s_ratio = np.asarray(hf["scaling_ratio"][:], float)  # (model/data) in POWER
            rho = float(hf.attrs["rho"]) if "rho" in hf.attrs else np.nan
        tgt_mag = 1.0 / np.maximum(s_ratio, 1e-16) if invert_target else s_ratio  # required |H| (AMPLITUDE)
        mt = (f_tgt >= fmin) & (f_tgt <= fmax)
        f_tgt, tgt_mag = f_tgt[mt], tgt_mag[mt]

        # --- load measured calibration |H_cal|
        with h5py.File(CALIB_BASE + f"calibs_{L}.h5", "r") as hf:
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


def save_corrected_pressure():
    """
    Apply the (rho, f)-scaled calibration FRF to measured time series and plot
    pre-multiplied, normalized spectra:  f * Pyy / (rho^2 * u_tau^4).
    """
    # --- fit rho–f scaling once from your saved target + calibration ---
    labels = ['0psig', '50psig', '100psig']
    (c0_db, a, b), scale, diag = fit_speaker_scaling_from_files(
        labels=tuple(labels),
        fmin=100.0, fmax=1000.0,   # fitting band
        f_ref=700.0,
        invert_target=True         # your "scaling_ratio" -> required |H| = 1/scaling_ratio
    )

    with h5py.File("data/final_pressure/SU_2pt_pressure.h5", 'w') as hf_out:
        # Store some fit diagnostics
        hf_out.attrs['c0_db'] = c0_db
        hf_out.attrs['a_rho'] = a
        hf_out.attrs['b_f'] = b
        hf_out.attrs[f'close_spacing'] = 2.8*DELTA
        # hf_out.attrs = {
        #     'u_tau': u_tau,
        #     'nu': nu,
        #     'rho': rho,
        #     'Re_tau': Re_tau,
        # }

        #This is the raw data with the freestream noise removed, explain
        hf_out.create_group("raw_data")
        h_raw = hf_out["raw_data"]
        # This is where the data has been corrected using a LEM
        hf_out.create_group("corrected_data")
        h_corrected = hf_out["corrected_data"]

    
        u_tau_uncertainty = [0.2, 0.1, 0.05]
        # --- main loop over datasets ---
        for i, L in enumerate(labels):

            # Load cleaned signals and attributes
            with h5py.File(CLEANED_BASE +f'{L}_far_cleaned.h5', 'r') as hf:
                ph1_clean_far = hf['ph1_clean'][:]
                ph2_clean_far = hf['ph2_clean'][:]
                u_tau = float(hf.attrs['u_tau'])
                nu = float(hf.attrs['nu'])
                rho = float(hf.attrs['rho'])
                Re_tau = hf.attrs.get('Re_tau', np.nan)

            h_raw.create_dataset(f'{L}_far/ph1', data=ph1_clean_far)
            h_raw.create_dataset(f'{L}_far/ph2', data=ph2_clean_far)
            with h5py.File(CLEANED_BASE +f'{L}_close_cleaned.h5', 'r') as hf:
                ph1_clean_close = hf['ph1_clean'][:]
                ph2_clean_close = hf['ph2_clean'][:]
            
            h_raw.create_dataset(f'{L}_close/ph1', data=ph1_clean_close)
            h_raw.create_dataset(f'{L}_close/ph2', data=ph2_clean_close)
            

            with h5py.File(CALIB_BASE + f"calibs_{L}.h5", "r") as hf:
                f_cal = np.asarray(hf["frequencies"][:], float)
                H_cal = np.asarray(hf["H_fused"][:], complex)

            # --- apply FRF with fitted rho–f magnitude scaling ---
            # (uses your updated apply_frf that accepts scale_fn and rho)
            ph1_filt_far = apply_frf(ph1_clean_far, FS, f_cal, H_cal, rho=rho, scale_fn=scale)
            ph2_filt_far = apply_frf(ph2_clean_far, FS, f_cal, H_cal, rho=rho, scale_fn=scale)
            ph1_filt_close = apply_frf(ph1_clean_close, FS, f_cal, H_cal, rho=rho, scale_fn=scale)
            ph2_filt_close = apply_frf(ph2_clean_close, FS, f_cal, H_cal, rho=rho, scale_fn=scale)

            h_corrected.create_dataset(f'{L}_far/ph1', data=ph1_filt_far)
            h_corrected.create_dataset(f'{L}_far/ph2', data=ph2_filt_far)

            h_corrected.create_dataset(f'{L}_close/ph1', data=ph1_filt_close)
            h_corrected.create_dataset(f'{L}_close/ph2', data=ph2_filt_close)




if __name__ == "__main__":
    save_corrected_pressure()