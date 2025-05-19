"""
Wall- and Free-Stream Pressure Processing
Hard-coded inputs and parameters for transparency.
"""

import os
from icecream import ic

import numpy as np
import scipy.io as sio

import numpy as np

from i_o import *
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
sns.set_palette("colorblind")

# === Hard-coded file paths ===
WALL_PRESSURE_MAT = "data/Pw_premul_spectra_07_Mar_2025.mat"
FREESTREAM_PRESSURE_MAT = "data/Pw_FS_premul_spectra_07_Mar_2025.mat"
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
    # 1) Identify duct modes
    L = L0 + DELTA_L0
    denom = (RHO0 * U_TAU0**2)**2
    duct_modes = compute_duct_modes(U, C, range(5), range(5), range(16), W, H, L, NU0, U_TAU0, ERR_FRAC)
    # Turn these into ranges by propogating error in normalisation quantities
    # 2) Load datasets
    data_W = sio.loadmat(WALL_PRESSURE_MAT)
    data_FS = sio.loadmat(FREESTREAM_PRESSURE_MAT)

    ic(data_W.keys(), data_W['prePspec_InnerNorm_all_ypos'].shape)

    fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=600)
    # ax.plot(data_FS['Pspec_FS_InnerNorm_avgd'][:, 0])
    ax.plot(data_W['prePspec_InnerNorm_all_ypos'][:, 0, 0], label="Wall Pressure")
    ax.set_xscale("log")
    fig.savefig(os.path.join(OUTPUT_DIR, "wall_pressure_spectrum_melb.png"))


    # 3) Compute wall pressure spectrum w. and w.out duct modes
    # x_wall_filt, f_wall_nom_filt, phi_wall_filt, f_wall_nom, phi_wall_nom, info_wall =\
    #     notch_filter_timeseries(p_w, SAMPLE_RATE,
    #                             np.array(duct_modes['min'])*(U_TAU0**2/NU0),
    #                             np.array(duct_modes['max'])*(U_TAU0**2/NU0),
    #                             np.array(duct_modes["nom"])*(U_TAU0**2/NU0))
    # x_filt_fs, f_nom_filt_fs, phi_filt_fs, f_nom_fs, phi_nom_fs, info_fs =\
    #     notch_filter_timeseries(p_fs, SAMPLE_RATE,
    #                             np.array(duct_modes['min'])*(U_TAU0**2/NU0),
    #                             np.array(duct_modes['max'])*(U_TAU0**2/NU0),
    #                             np.array(duct_modes["nom"])*(U_TAU0**2/NU0))
    # 4) Plot 
    # fig, axes = plt.subplots(1,2, figsize=(5.6, 2.5), dpi=600, sharex=True, sharey=True)
    # ax = axes[0]
    # T_plus = 1 / (f_wall_nom_filt * NU0 / U_TAU0**2)
    # ax.plot(T_plus, phi_wall_filt*f_wall_nom_filt, 'r-', lw=0.5, alpha=0.8, label="Filtered")
    # T_plus = 1 / (f_wall_nom * NU0 / U_TAU0**2)
    # ax.plot(T_plus, phi_wall_nom*f_wall_nom, 'g-', lw=0.5, alpha=0.8, label="Wall Pressure")
    # ax = axes[1]
    # T_plus = 1 / (f_nom_filt_fs * NU0 / U_TAU0**2)
    # ax.plot(T_plus, phi_filt_fs*f_nom_filt_fs, 'r-', lw=0.5, alpha=0.8, label="Filtered")
    # T_plus = 1 / (f_nom_fs * NU0 / U_TAU0**2)
    # ax.plot(T_plus, phi_nom_fs*f_nom_fs, 'b-', lw=0.5, alpha=0.8, label="F-S Pressure")
    # ax.set_xscale("log")
    # # ax.set_xlim(1e2, 2e3)
    # ax.legend()
    # plt.savefig(os.path.join(OUTPUT_DIR, "notched_pressure_spectrum.png"))

    # 5) Align signals to account for miss-alignment of wall and free-stream pressure mics
    # x_align_filt_fs, x_align_filt_wall, tau = align_signals(x_filt_fs, x_wall_filt, SAMPLE_RATE, max_lag_s=0.1)
    # x_align_fs, x_align_wall, tau = align_signals(p_fs, p_w, SAMPLE_RATE, max_lag_s=0.1)


    # 6) Compute coherence of all the processed signals
    # f_raw, coh_raw = compute_coherence(p_w, p_fs, SAMPLE_RATE)
    # f_filt, coh_filt = compute_coherence(x_wall_filt, x_filt_fs, SAMPLE_RATE)
    # f_align, coh_align = compute_coherence(x_align_wall, x_align_fs, SAMPLE_RATE)
    # f_align_notched, coh_align_notched = compute_coherence(x_align_filt_wall, x_align_filt_fs, SAMPLE_RATE)

    # 7) Plot coherence
    # fig, ax = plt.subplots(figsize=(3.5, 2.62), dpi=600)
    # T_plus = 1 / (f_raw * NU0 / U_TAU0**2)
    # ax.loglog(T_plus, coh_raw, 'g-', lw=0.5, alpha=0.8, label="Raw")
    # T_plus = 1 / (f_align * NU0 / U_TAU0**2)
    # ax.loglog(T_plus, coh_align, 'g--', lw=0.5, alpha=0.8, label="Aligned")
    # T_plus = 1 / (f_filt * NU0 / U_TAU0**2)
    # ax.loglog(T_plus, coh_filt, 'r-', lw=0.5, alpha=0.8, label="Notched")
    # T_plus = 1 / (f_align_notched * NU0 / U_TAU0**2)
    # ax.loglog(T_plus, coh_align_notched, 'r--', lw=0.5, alpha=0.8, label="Notched \& Aligned")
    # ax.set_xlabel("$T^+$")
    # ax.set_ylabel("$\\gamma^2$")
    # ax.legend()
    # ax.grid(True, which='both', ls='--', alpha=0.5)
    # plt.savefig(os.path.join(OUTPUT_DIR, "coherence_stan.png"))

    # 8) Coherence filtered wall-pressure PSD
    # f_raw, Phi_orig_raw, Phi_clean_raw = reject_noise_by_coherence(p_fs, p_w, SAMPLE_RATE, thresh=1e-3)
    # f_align, Phi_orig_align, Phi_clean_align = reject_noise_by_coherence(x_align_fs, x_align_wall, SAMPLE_RATE, thresh=1e-3)
    # f_filt, Phi_orig_filt, Phi_clean_filt = reject_noise_by_coherence(x_filt_fs, x_wall_filt, SAMPLE_RATE, thresh=1e-3)
    # f_align_notched, Phi_orig_align_notched, Phi_clean_align_notched = reject_noise_by_coherence(x_align_filt_fs, x_align_filt_wall, SAMPLE_RATE, thresh=1e-3)
    
    # 9) Plot original and coherence filtered wall-pressure PSDs
    # fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=600)
    # T_plus = 1 / (f_raw * NU0 / U_TAU0**2)
    # # ax.semilogx(T_plus, Phi_orig_raw*f_raw, 'b-', lw=0.5, alpha=0.8, label="Original Wall PSD")
    # # T_plus = 1 / (f_filt * NU0 / U_TAU0**2)
    # ax.semilogx(T_plus, Phi_clean_raw*f_raw/denom, 'g-', lw=0.5, alpha=0.8, label="Coherence Cleaned Wall PSD")
    # T_plus = 1 / (f_filt * NU0 / U_TAU0**2)
    # ax.semilogx(T_plus, Phi_clean_filt*f_filt/denom, 'r-', lw=0.5, alpha=0.8, label="Notched Wall PSD")
    # T_plus = 1 / (f_align_notched * NU0 / U_TAU0**2)
    # ax.semilogx(T_plus, Phi_clean_align_notched*f_align_notched/denom, 'b-', lw=0.5, alpha=0.8, label="Aligned \& Notched Wall PSD")
    # T_plus = 1 / (f_align * NU0 / U_TAU0**2)
    # ax.semilogx(T_plus, Phi_clean_align*f_align/denom, 'k-', lw=0.5, alpha=0.8, label="Aligned Wall PSD")
    # ax.set_xlabel("$T^+$")
    # ax.set_ylabel("$f\\Phi_{pp}^+$")
    # ax.legend()
    # ax.grid(True, which='both', ls='--', alpha=0.5)
    # ax.set_ylim(0, 10)
    # plt.savefig(os.path.join(OUTPUT_DIR, "wall_pressure_spectrum.png"))

    # 10) Fill and smooth the spectra
    # phi_nom_raw = fill_and_smooth_psd(f_raw, Phi_orig_raw)
    # phi_nom_filt = fill_and_smooth_psd(f_filt, Phi_clean_filt)
    # phi_nom_filt_fs = fill_and_smooth_psd(f_align_notched, Phi_clean_align_notched)
    # phi_nom = fill_and_smooth_psd(f_align, Phi_clean_align)

    # 11) Plot smoothed spectra
    # fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=600)

    # T_plus = 1 / (f_raw * NU0 / U_TAU0**2)
    # ax.semilogx(T_plus, phi_nom_raw*f_raw/denom, 'b-', lw=0.5, alpha=0.8, label="Original Wall PSD")

    # T_plus = 1 / (f_filt * NU0 / U_TAU0**2)
    # ax.semilogx(T_plus, phi_nom_filt*f_filt/denom, 'g-', lw=0.5, alpha=0.8, label="Notched Wall PSD")

    # T_plus = 1 / (f_align_notched * NU0 / U_TAU0**2)
    # ax.semilogx(T_plus, phi_nom_filt_fs*f_align_notched/denom, 'r-', lw=0.5, alpha=0.8, label="Aligned \& Notched Wall PSD")

    # T_plus = 1 / (f_align * NU0 / U_TAU0**2)
    # ax.semilogx(T_plus, phi_nom*f_align/denom, 'k-', lw=0.5, alpha=0.8, label="Aligned Wall PSD")

    # ax.set_xlabel("$T^+$")
    # ax.set_ylabel("$f\\Phi_{pp}^+$")
    # ax.legend()
    # ax.grid(True, which='both', ls='--', alpha=0.5)
    # ax.set_ylim(0, 5.5)
    # plt.savefig(os.path.join(OUTPUT_DIR, "smoothed_wall_pressure_spectrum.png"))


if __name__ == "__main__":
    main()
    # noise_rejection()