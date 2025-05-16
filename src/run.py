"""
Wall- and Free-Stream Pressure Processing
Hard-coded inputs and parameters for transparency.
"""

import os
from icecream import ic

import numpy as np

from i_o import load_wallpressure
from plotting import *
from processing import *
from noise_rejection import *

from matplotlib import pyplot as plt
import seaborn as sns
import scienceplots

plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")

# === Hard-coded file paths ===
WALL_PRESSURE_MAT      = "data/wallpressure_booman_Pa.mat"
FREESTREAM_PRESSURE_MAT = "data/booman_wallpressure_fspressure_650sec_40khz.mat"
OUTPUT_DIR             = "figures"

# === Physical & processing parameters ===
SAMPLE_RATE = 40000        # Hz
NU0         = 1.52e-5      # m^2/s
RHO0        = 1.225        # kg/m^3
U_TAU0      = 0.58         # m/s
ERR_FRAC    = 0.03         # ±3% uncertainty
W, H        = 0.30, 0.152  # duct width & height (m)
L0          = 3.0          # duct length (m)
DELTA_L0    = 0.1 * L0     # low-frequency end correction
U           = 14.2         # flow speed (m/s)
C           = np.sqrt(1.4 * 101325 / RHO0)  # speed of sound (m/s)
MODE_M      = [0]
MODE_N      = [0]
MODE_L      = [0, 1, 4, 5, 8, 11, 15]

# ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # Find duct modes
    L = L0 + DELTA_L0
    duct_modes = compute_duct_modes(U, C, range(5), range(5), range(16), W, H, L, NU0, U_TAU0, ERR_FRAC)
    # Turn these into ranges by propogating error in normalisation quantities
    # load & PSD
    fs_w, p_w = load_wallpressure(WALL_PRESSURE_MAT)
    x_filt, f_nom_filt, phi_filt, f_nom, phi_nom, info =\
        notch_filter_timeseries(p_w, SAMPLE_RATE,
                                np.array(duct_modes['min'])*(U_TAU0**2/NU0),
                                np.array(duct_modes['max'])*(U_TAU0**2/NU0),
                                np.array(duct_modes["nom"])*(U_TAU0**2/NU0))
    # fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=600)
    # T_plus = 1 / (f_nom_filt * NU0 / U_TAU0**2)
    # ax.plot(T_plus, phi_filt*f_nom_filt, 'r-', lw=0.5, alpha=0.8, label="Filtered")
    # T_plus = 1 / (f_nom * NU0 / U_TAU0**2)
    # ax.plot(T_plus, phi_nom*f_nom, 'b-', lw=0.5, alpha=0.8, label="Wall Pressure")
    # ax.set_xscale("log")
    # # ax.set_xlim(1e2, 2e3)
    # plt.savefig(os.path.join(OUTPUT_DIR, "notched_wall_pressure_spectrum.png"))

    fs_fs, p_fs = load_wallpressure(FREESTREAM_PRESSURE_MAT)
    x_filt_fs, f_nom_filt_fs, phi_filt_fs, f_nom_fs, phi_nom_fs, info_fs =\
        notch_filter_timeseries(p_fs, SAMPLE_RATE,
                                np.array(duct_modes['min'])*(U_TAU0**2/NU0),
                                np.array(duct_modes['max'])*(U_TAU0**2/NU0),
                                np.array(duct_modes["nom"])*(U_TAU0**2/NU0))
    
    # fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=600)
    # T_plus = 1 / (f_nom_filt_fs * NU0 / U_TAU0**2)
    # ax.plot(T_plus, phi_filt_fs*f_nom_filt_fs, 'r-', lw=0.5, alpha=0.8, label="Filtered")
    # T_plus = 1 / (f_nom_fs * NU0 / U_TAU0**2)
    # ax.plot(T_plus, phi_nom_fs*f_nom_fs, 'b-', lw=0.5, alpha=0.8, label="Wall Pressure")
    # ax.set_xscale("log")
    # # ax.set_xlim(1e2, 2e3)
    # plt.savefig(os.path.join(OUTPUT_DIR, "notched_fs_pressure_spectrum.png"))

    # fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=600)
    # T_plus = 1 / (f_nom_filt * NU0 / U_TAU0**2)
    # ax.plot(T_plus, phi_nom*f_nom, 'b-', lw=0.5, alpha=0.8, label="Wall Pressure")
    # T_plus = 1 / (f_nom_filt_fs * NU0 / U_TAU0**2)
    # ax.plot(T_plus, phi_nom_fs*f_nom_fs, 'g-', lw=0.5, alpha=0.8, label="Free-Stream Pressure")
    # ax.set_xscale("log")
    # # ax.set_xlim(1e2, 2e3)
    # plt.savefig(os.path.join(OUTPUT_DIR, "notched_wall_vs_fs_pressure_spectrum.png"))

    # Wiener filter
    f, Phi_orig, Phi_clean = reject_noise_torch(x_filt_fs, x_filt, SAMPLE_RATE)
    fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=600)
    T_plus = 1 / (f * NU0 / U_TAU0**2)
    ax.semilogx(T_plus, Phi_orig*f, 'b-', lw=0.5, alpha=0.8, label="Original Wall PSD")
    ax.semilogx(T_plus, Phi_clean*f, 'r-', lw=0.5, alpha=0.8, label="Wiener Filtered")
    ax.set_xlabel("$T^+$")
    ax.set_ylabel("$f\\Phi_{pp}^+$")
    ax.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "wiener_filtered_wall_pressure_spectrum.png"))

    # compute modes
    # modes = compute_duct_modes(U, C, MODE_M, MODE_N, MODE_L, W, H, L, NU0, U_TAU0, ERR_FRAC)
    # plot_spectrum_and_modes(spec_w, modes, MODE_L,
    #                         os.path.join(OUTPUT_DIR, "wall_pressure_uncertainty.png"))

    # # free‐stream
    # fs_fs, p_fs = load_wallpressure(FREESTREAM_PRESSURE_MAT, var_name="wall_pressure_fluc_Pa")
    # f_raw_fs, psd_fs = compute_psd(p_fs, SAMPLE_RATE)
    # spec_fs = propagate_error(f_raw_fs, psd_fs, NU0, RHO0, U_TAU0, ERR_FRAC)
    # plot_spectrum_and_modes(spec_fs, modes, MODE_L,
    #                         os.path.join(OUTPUT_DIR, "fs_pressure_uncertainty.png"))
    
    # f_csd, Pyy_csd = compute_csd(p_fs, p_w, SAMPLE_RATE)
    # spec_csd = propagate_error(f_csd, Pyy_csd, NU0, RHO0, U_TAU0, ERR_FRAC)
    # plot_spectrum_and_modes(spec_csd, modes, MODE_L,
    #                         os.path.join(OUTPUT_DIR, "csd_pressure_uncertainty.png"))

    # raw signals overlay
    # plot_pw_p_fs(spec_w["f_nom"], spec_fs["f_nom"],
    #              spec_w["phi_nom"], spec_fs["phi_nom"],
    #              os.path.join(OUTPUT_DIR, "wall_vs_fs.png"))

    # notch & plot filtered
    # phi_filt_w, info_w   = notch_filter(spec_w["f_nom"], spec_w["phi_nom"],
    #                                     spec_w["f_min"], spec_w["f_max"], all_modes["nom"])
    # phi_filt_fs, info_fs = notch_filter(spec_fs["f_nom"], spec_fs["phi_nom"],
    #                                     spec_fs["f_min"], spec_fs["f_max"], all_modes["nom"])
    # phi_filt_csd, info_csd = notch_filter(spec_csd["f_nom"], spec_csd["phi_nom"],
    #                                       spec_csd["f_min"], spec_csd["f_max"], all_modes["nom"])
    
    # plot_filtered_spectrum(spec_csd, phi_filt_csd, info_csd,
    #                         os.path.join(OUTPUT_DIR, "csd_notched.png"),  'orange')
    # plot_filtered_spectrum(spec_w, phi_filt_w, info_w,
    #                         os.path.join(OUTPUT_DIR, "wall_notched.png"), 'b-')
    # plot_filtered_spectrum(spec_fs, phi_filt_fs, info_fs,
    #                         os.path.join(OUTPUT_DIR, "fs_notched.png"),  'g-')
    # plot_filtered_diff(spec_w, phi_filt_w, phi_filt_fs,
    #                    os.path.join(OUTPUT_DIR, "difference.png"))
    
def noise_rejection():
    fs_w, p_w = load_wallpressure(WALL_PRESSURE_MAT)
    fs_fs, p_fs = load_wallpressure(FREESTREAM_PRESSURE_MAT, var_name="wall_pressure_fluc_Pa")

    # Notch filter the wall-pressure time series
    all_modes = compute_duct_modes(U, C, range(5), range(5), range(16), W, H, L, NU0, U_TAU0, ERR_FRAC)

    # analyze coherence
    # f, coh, Pyy, Pyy_coh = compute_coherence_psd(p_fs, p_w, SAMPLE_RATE)
    # T_plus = 1 / (f * NU0 / U_TAU0**2)
    # plot_coherence(T_plus, coh, Pyy, Pyy_coh,
    #                os.path.join(OUTPUT_DIR, "coherence.png"))




if __name__ == "__main__":
    main()
    # noise_rejection()