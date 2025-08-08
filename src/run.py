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
U_TAU0 = 0.358              # m/s
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


def main(sanity=False):
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
    ic(f"Loaded data: {proc.p_w.shape}, {proc.p_fs.shape}")
    # proc.compute_duct_modes()
    # proc.notch_filter()

    if sanity:
        p_w_org = proc.p_w.cpu()
        nperseg = len(proc.p_w.cpu()) // 5000
        noverlap = nperseg // 2
        f, P_w = welch(proc.p_w.cpu(), fs=SAMPLE_RATE, nperseg=nperseg,
                    noverlap=noverlap, window="hann")
        f, P_fs = welch(proc.p_fs.cpu(), fs=SAMPLE_RATE, nperseg=nperseg,
                        noverlap=noverlap, window="hann")
        f, P_w_fs = csd(proc.p_w.cpu(), proc.p_fs.cpu(), fs=SAMPLE_RATE,
                        nperseg=nperseg, noverlap=noverlap, window="hann")
        f, P_w_fs_opt = csd(proc.p_w.cpu(), proc.p_fs.cpu(), fs=SAMPLE_RATE,
                            nperseg=nperseg, noverlap=noverlap, window="hann")

    # Find the transfer complex transfer function between the mics
    proc.phase_match(smoothing_len=1)
    ic(f"Phase-matched: {proc.p_w.shape}")

    if sanity:
        f, H_match = phase_match_transfer(p_w_org, proc.p_w.cpu(), SAMPLE_RATE, smoothing_len=1)
        plot_transfer(f, H_match, os.path.join(OUTPUT_DIR, "complex_transfer_function.png"))

        f, P_w_fs_opt = csd(p_w_org, proc.p_fs.cpu(), fs=SAMPLE_RATE,
                            nperseg=nperseg, noverlap=noverlap, window="hann")
        plot_phase_match_csd(f, P_w, P_fs, P_w_fs, P_w_fs_opt,
                            os.path.join(OUTPUT_DIR, "phase_match_csd.png"))

        f, coh = coherence(p_w_org, proc.p_fs.cpu(), fs=SAMPLE_RATE,
                        nperseg=nperseg, noverlap=noverlap)
        f_match, coh_match = coherence(proc.p_w.cpu(), proc.p_fs.cpu(), fs=SAMPLE_RATE,
                                    nperseg=nperseg, noverlap=noverlap)
        # plot coherence
        plot_coherence(f, coh, f_match, coh_match,
                    os.path.join(OUTPUT_DIR, "coherence.png"))

        # Plot estimated transfer function

    # Wiener filter time series
    proc.reject_free_stream_noise()
    ic(f"Noise-rejected: {proc.p_w.shape}")
    if sanity:
        f, P_w_clean = welch(proc.p_w.cpu(), fs=SAMPLE_RATE, nperseg=nperseg,
                            noverlap=noverlap, window="hann")
        plot_wiener_filter(f, P_w, P_fs, P_w_clean,
                                    os.path.join(OUTPUT_DIR, "wiener_filtered_spectrum.png"))

    # Load reference spectrum for transfer function
    data  = sio.loadmat("data/premultiplied_spectra_Pw_ReT2000_Deshpande_JFM_2025.mat")
    T_plus_ref = data["Tplus"][0]
    f_ref_plus = 1/T_plus_ref  # convert T+ to f+
    f_Phi_ref_plus = data["premul_Pw_plus"][0]  # first column is the wall-pressure PSD
    u_tau_ref = 0.358
    rho_ref = 1.225
    nu_ref = 1.52e-5  # kinematic viscosity at reference conditions

    # Undo the normalisation to get the reference PSD in physical units
    denom_ref = (rho_ref * u_tau_ref**2)**2
    denom_ref = (RHO0 * U_TAU0**2) ** 2
    f_Pxx_ref = f_Phi_ref_plus * denom_ref  # premultiplied PSD in Pa^2/Hz^2
    f_ref = f_ref_plus / (nu_ref / u_tau_ref**2)  # dimensional frequency f [Hz]
    Pxx_ref = f_Pxx_ref / f_ref  # convert to physical units (Pa^2/Hz)

    f_grid, Phi_w_corrected, H_mag = proc.compute_transfer_function(f_ref, Pxx_ref)
    f_grid = f_grid.cpu()
    Phi_w_corrected = Phi_w_corrected.cpu()
    # Smooth
    Phi_w_corrected  = savgol_filter(Phi_w_corrected, 51, 1)  # window length 51, polynomial order 3
    H_mag = H_mag.cpu()


    plot_reference_transfer_function(
        f_grid, H_mag, os.path.join(OUTPUT_DIR, "reference_transfer_function.png")
    )
    
    denom = (RHO0 * U_TAU0**2) ** 2
    T_plus = 1/(f_grid * (nu_ref / U_TAU0**2))  # convert f to T+
    
    fig, ax = plt.subplots(figsize=(5.6, 2.5), dpi=600)
    ax.plot(T_plus_ref, f_Phi_ref_plus, lw=0.5, alpha=0.8)
    ax.plot(T_plus, f_grid * Phi_w_corrected/denom, lw=0.5, alpha=0.8)

    ax.set_xscale("log")
    ax.set_ylim(0, 4)
    # ax.set_yscale("log")
    ax.set_xlabel("$T^+$")
    ax.set_ylabel("$f\\Phi^+$")
    ax.legend(["Deshpande et al. (2025)", "$P_{ww}^{\\mathrm{corrected}}$"])
    ax.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(os.path.join(OUTPUT_DIR, "final_spectrum.png"))
    plt.close()


if __name__ == "__main__":
    main(sanity=1)
    # main(sanity=True)

