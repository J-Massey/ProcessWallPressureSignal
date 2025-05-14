"""
Wall- and Free-Stream Pressure Processing
Hard-coded inputs and parameters for transparency.
"""

import os
import numpy as np

from io import load_wallpressure
from plotting import (plot_spectrum_and_modes, plot_pw_p_fs,
                       plot_filtered_spectrum, plot_filtered_diff)
from processing import (compute_psd, propagate_error,
                        compute_duct_modes, duct_mode_freq,
                        notch_filter)

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
    # load & PSD
    fs_w, p_w = load_wallpressure(WALL_PRESSURE_MAT)
    f_raw_w, psd_w = compute_psd(p_w, SAMPLE_RATE)
    spec_w = propagate_error(f_raw_w, psd_w, NU0, RHO0, U_TAU0, ERR_FRAC)

    # compute modes
    L = L0 + DELTA_L0
    modes = compute_duct_modes(U, C, MODE_M, MODE_N, MODE_L, W, H, L, NU0, U_TAU0, ERR_FRAC)
    plot_spectrum_and_modes(spec_w, modes, MODE_L,
                            os.path.join(OUTPUT_DIR, "wall_pressure_uncertainty.png"))

    # free‐stream
    fs_fs, p_fs = load_wallpressure(FREESTREAM_PRESSURE_MAT, var_name="wall_pressure_fluc_Pa")
    f_raw_fs, psd_fs = compute_psd(p_fs, SAMPLE_RATE)
    spec_fs = propagate_error(f_raw_fs, psd_fs, NU0, RHO0, U_TAU0, ERR_FRAC)
    modes_fs = compute_duct_modes(U, C, MODE_M, MODE_N, MODE_L, W, H, L, NU0, U_TAU0, ERR_FRAC)
    plot_spectrum_and_modes(spec_fs, modes_fs, MODE_L,
                            os.path.join(OUTPUT_DIR, "fs_pressure_uncertainty.png"))

    # raw signals overlay
    plot_pw_p_fs(spec_w["f_nom"], spec_fs["f_nom"],
                 spec_w["phi_nom"], spec_fs["phi_nom"],
                 os.path.join(OUTPUT_DIR, "wall_vs_fs.png"))

    # notch & plot filtered
    all_modes = compute_duct_modes(U, C, range(5), range(5), range(16), W, H, L, NU0, U_TAU0, ERR_FRAC)
    phi_filt_w, info_w   = notch_filter(spec_w["f_nom"], spec_w["phi_nom"],
                                        spec_w["f_min"], spec_w["f_max"], all_modes["nom"])
    phi_filt_fs, info_fs = notch_filter(spec_fs["f_nom"], spec_fs["phi_nom"],
                                        spec_fs["f_min"], spec_fs["f_max"], all_modes["nom"])
    plot_filtered_spectrum(spec_w, phi_filt_w, info_w,
                            os.path.join(OUTPUT_DIR, "wall_notched.png"), 'b-')
    plot_filtered_spectrum(spec_fs, phi_filt_fs, info_fs,
                            os.path.join(OUTPUT_DIR, "fs_notched.png"),  'g-')
    plot_filtered_diff(spec_w, phi_filt_w, phi_filt_fs,
                       os.path.join(OUTPUT_DIR, "difference.png"))

if __name__ == "__main__":
    main()