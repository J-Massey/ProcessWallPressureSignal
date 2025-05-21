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


def main():
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
    # proc.compute_duct_modes()
    # proc.notch_filter()

    # proc.p_fs = phase_match(proc.p_fs, proc.p_w)

    nperseg = len(proc.p_w) // 5000
    noverlap = nperseg // 2
    f, P_w = welch(proc.p_w, fs=SAMPLE_RATE, nperseg=nperseg,
                   noverlap=noverlap, window="hann")
    f, P_fs = welch(proc.p_fs, fs=SAMPLE_RATE, nperseg=nperseg,
                    noverlap=noverlap, window="hann")
    f, P_w_fs = csd(proc.p_w, proc.p_fs, fs=SAMPLE_RATE,
                    nperseg=nperseg, noverlap=noverlap, window="hann")
    # f, coh = coherence(proc.p_w, proc.p_fs, fs=SAMPLE_RATE,
    #                    nperseg=len(proc.p_w)//2000, noverlap=len(proc.p_w)//4000)
    fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=600)

    ax.plot(f, P_w_fs.real, lw=0.5, alpha=0.8)
    ax.plot(f, P_w.real, lw=0.5, alpha=0.8)
    ax.plot(f, P_fs.real, lw=0.5, alpha=0.8)
    ax.set_xscale("log")

    ax.set_xlabel("$f$ [Hz]")
    ax.set_ylabel("$\\Phi_{pp}$")
    ax.legend(["$P_{wf}$", "$P_{ww}$", "$P_{ff}$"])
    # ax.set_xlim(1e-1, 1e-4)
    # ax.set_ylim(0, 1)
    plt.savefig(os.path.join(OUTPUT_DIR, "coherence.png"))

    # # Load reference spectrum for transfer function
    # data = sio.loadmat("data/premultiplied_spectra_Pw_ReT2000_Deshpande_JFM_2025.mat")
    # T_plus_ref = data["Tplus"][0]
    # f_ref = 1 / T_plus_ref / (1.68e-5 / 0.358**2)
    # denom_ref = (1.11 * 0.358**2) ** 2
    # Pxx_ref = data["premul_Pw_plus"][0] * denom_ref / f_ref

    # proc.compute_transfer_function(f_ref, Pxx_ref)
    # proc.apply_transfer_function()

    # # Example: save corrected spectrum
    # f_corr, P_corr = welch(proc.p_w_corrected, fs=SAMPLE_RATE,
    #                        nperseg=len(proc.p_w_corrected)//2000,
    #                        noverlap=len(proc.p_w_corrected)//4000,
    #                        window="hann")
    # np.savez(os.path.join(OUTPUT_DIR, "wall_corrected_spectrum.npz"),
    #          freq=f_corr, pxx=P_corr)


if __name__ == "__main__":
    main()

