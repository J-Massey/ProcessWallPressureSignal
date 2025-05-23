"""
Wall- and Free-Stream Pressure Processing
This script demonstrates usage of :class:`WallPressureProcessor`.
"""

import os
import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import welch, coherence
import seaborn as sns
import scienceplots

from processor import WallPressureProcessor
from noise_rejection import *
from plotting import *

plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")
sns.set_palette("colorblind")

# === Hard-coded file paths ===
WALL_PRESSURE_MAT = "data/wallpressure_booman_Pa.mat"
FREESTREAM_PRESSURE_MAT = "data/booman_wallpressure_fspressure_650sec_40khz.mat"
OUTPUT_DIR = "figures"

# === Physical & processing parameters ===
SAMPLE_RATE = 40000        # Hz
NU0 = 1.52e-5              # m^2/s
RHO0 = 1.225               # kg/m^3
U_TAU0 = 0.58              # m/s
ERR_FRAC = 0.03            # ±3% uncertainty
W, H = 0.30, 0.152         # duct width & height (m)
L0 = 3.0                   # duct length (m)
DELTA_L0 = 0.1 * L0        # low-frequency end correction
U = 14.2                   # flow speed (m/s)
C = np.sqrt(1.4 * 101325 / RHO0)  # speed of sound (m/s)
MODE_M = [0]
MODE_N = [0]
MODE_L = [0, 1, 4, 5, 8, 11, 15]

# ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main(sanity=None):
    proc = WallPressureProcessor(
        sample_rate=SAMPLE_RATE,
        nu0=NU0,
        rho0=RHO0,
        u_tau0=U_TAU0,
        err_frac=ERR_FRAC,
        W=W,
        H=H,
        L0=L0,
        delta_L0=DELTA_L0,
        U=U,
        C=C,
        mode_m=MODE_M,
        mode_n=MODE_N,
        mode_l=MODE_L,
    )

    proc.load_data(WALL_PRESSURE_MAT, FREESTREAM_PRESSURE_MAT)
    proc.compute_duct_modes()
    proc.notch_filter()

    if sanity is not None:
        nperseg = len(proc.p_w) // 5000
        noverlap = nperseg // 2
        f, P_w = welch(proc.p_w, fs=SAMPLE_RATE, nperseg=nperseg,
                    noverlap=noverlap, window="hann")
        f, P_fs = welch(proc.p_fs, fs=SAMPLE_RATE, nperseg=nperseg,
                        noverlap=noverlap, window="hann")
        f, P_w_fs = csd(proc.p_w, proc.p_fs, fs=SAMPLE_RATE,
                        nperseg=nperseg, noverlap=noverlap, window="hann")
        f, P_w_fs_opt = csd(proc.p_w, proc.p_fs, fs=SAMPLE_RATE,
                            nperseg=nperseg, noverlap=noverlap, window="hann")

    # Find the transfer complex transfer function between the mics
    proc.phase_match(smoothing_len=1)

    if sanity is not None:
        f, P_w_fs_opt = csd(proc.p_w_matched, proc.p_fs, fs=SAMPLE_RATE,
                            nperseg=nperseg, noverlap=noverlap, window="hann")
        plot_phase_match_csd(f, P_w, P_fs, P_w_fs, P_w_fs_opt,
                            os.path.join(OUTPUT_DIR, "phase_match_csd.png"))

        f, coh = coherence(proc.p_w, proc.p_fs, fs=SAMPLE_RATE,
                        nperseg=nperseg, noverlap=noverlap)
        f_match, coh_match = coherence(proc.p_w_matched, proc.p_fs, fs=SAMPLE_RATE,
                                    nperseg=nperseg, noverlap=noverlap)
        # plot coherence
        plot_coherence(f, coh, f_match, coh_match,
                    os.path.join(OUTPUT_DIR, "coherence.png"))

        # Plot estimated transfer function
        f, H_match = phase_match_transfer(proc.p_w, proc.p_w_matched, SAMPLE_RATE, smoothing_len=7)
        plot_transfer(f, H_match, os.path.join(OUTPUT_DIR, "transfer_function.png"))

    # Wiener filter time series
    proc.reject_free_stream_noise()
    if sanity is not None:
        f, P_w_clean = welch(proc.p_w_clean, fs=SAMPLE_RATE, nperseg=nperseg,
                            noverlap=noverlap, window="hann")
        plot_wiener_filter(f, P_w, P_fs, P_w_clean,
                                    os.path.join(OUTPUT_DIR, "wiener_filtered_spectrum.png"))

    # Load reference spectrum for transfer function
    data = sio.loadmat("data/premultiplied_spectra_Pw_ReT2000_Deshpande_JFM_2025.mat")
    T_plus_ref = data["Tplus"][0]
    f_ref = 1 / T_plus_ref / (1.68e-5 / 0.358**2)
    denom_ref = (1.11 * 0.358**2) ** 2
    Pxx_ref = data["premul_Pw_plus"][0] * denom_ref / f_ref

    f_grid, H_mag = proc.compute_transfer_function(f_ref, Pxx_ref)
    if sanity is not None:
        fig, ax = plt.subplots(figsize=(5.6, 2.5), dpi=600)
        ax.plot(f_grid, H_mag, lw=0.5, alpha=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$f$ [Hz]")
        ax.set_ylabel("$|H(f)|$")
        ax.legend(["$|H(f)|$"])
        ax.grid(True, which="both", ls="--", alpha=0.5)
        plt.savefig(os.path.join(OUTPUT_DIR, "transfer_function.png"))
        plt.close()
    proc.apply_transfer_function()
    f, P_w_corrected = welch(proc.p_w_corrected, fs=SAMPLE_RATE,
                            nperseg=nperseg, noverlap=noverlap, window="hann")
    
    denom = (RHO0 * U_TAU0**2) ** 2
    
    fig, ax = plt.subplots(figsize=(5.6, 2.5), dpi=600)
    ax.plot(T_plus_ref, f_ref * Pxx_ref / denom_ref, lw=0.5, alpha=0.8)
    ax.plot(1/ (f * NU0 / U_TAU0**2), f * P_w_corrected / denom, lw=0.5, alpha=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$f^+$")
    ax.set_ylabel("$f\\Phi^+$")
    ax.legend(["$Deshpande et al. (2025)$", "$P_{ww}^{corrected}$"])
    ax.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(os.path.join(OUTPUT_DIR, "final_spectrum.png"))
    plt.close()

    # # Example: save corrected spectrum
    # f_corr, P_corr = welch(proc.p_w_corrected, fs=SAMPLE_RATE,
    #                        nperseg=len(proc.p_w_corrected)//2000,
    #                        noverlap=len(proc.p_w_corrected)//4000,
    #                        window="hann")
    # np.savez(os.path.join(OUTPUT_DIR, "wall_corrected_spectrum.npz"),
    #          freq=f_corr, pxx=P_corr)


if __name__ == "__main__":
    main(sanity=False)

