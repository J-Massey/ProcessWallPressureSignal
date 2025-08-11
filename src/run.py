"""
Wall- and Free-Stream Pressure Processing
This script demonstrates usage of :class:`WallPressureProcessor`.
"""

import os
import scipy.io as sio
import numpy as np
from scipy.signal import welch, coherence
from scipy.optimize import least_squares

from processor import WallPressureProcessor
from noise_rejection import *
from plotting import *

from matplotlib import pyplot as plt
import seaborn as sns
import scienceplots

plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")
sns.set_palette("colorblind")

# === Hard-coded file paths ===
WALL_PRESSURE_MAT = "data/wallpressure_booman_Pa.mat"
FREESTREAM_PRESSURE_MAT = "data/booman_wallpressure_fspressure_650sec_40khz.mat"

fn_train = [
    'Sensor1(naked)Sensor2(plug1).mat',
    'Sensor1(naked)Sensor3(plug2).mat',
    'Sensor1(naked)Sensor2(naked).mat',
    'Sensor1(naked)Sensor3(naked).mat',
    'Sensor1(naked)Sensor3(nosecone).mat',
]

idx = 0
TEST_MAT = f"data/calibration/{fn_train[1]}"
OUTPUT_DIR = "figures/calibration_08-11"

# === Physical & processing parameters ===
SAMPLE_RATE = 25000        # Hz
NU0 = 1.52e-5              # m^2/s
RHO0 = 1.225               # kg/m^3
U_TAU0 = 0.358              # m/s
ERR_FRAC = 0.03            # ±3% uncertainty
W, He = 0.30, 0.152         # duct width & height (m)
L0 = 3.0                   # duct length (m)
DELTA_L0 = 0.1 * L0        # low-frequency end correction
U = 14.2                   # flow speed (m/s)
C = np.sqrt(1.4 * 101325 / RHO0)  # speed of sound (m/s)
MODE_M = [0]
MODE_N = [0]
MODE_L = [0, 1, 4, 5, 8, 11, 15]

# ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 1) Transfer-function estimation (H1) ----------
def estimate_frf(x, y, fs, nperseg=4096, noverlap=2048, window='hann'):
    """
    x: reference mic (input); y: treated mic (output); fs: Hz.
    Returns: f [Hz], H1(f)=S_yx/S_xx (complex), coherence gamma^2(f).
    """
    f, Sxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)
    _, Syy = welch(y, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)
    _, Sxy = csd(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)  # x->y
    H = Sxy / Sxx
    coh2 = (np.abs(Sxy)**2) / (Sxx * Syy)
    return f, H, coh2

# ---------- 3) Inverse filtering (for Fig. 20-style time traces) ----------
def inverse_filter(y, fs, f, H, coh2=None, f_band=[0.1, 5000], reg=1e-4, pad=0):
    """
    Recover x from y when y ≈ H * x (+ noise).
    Assumes H was estimated for x→y (H = S_xy/S_xx).
    Uses Wiener inverse: W ≈ γ^2/H when coh2 is provided,
    else Tikhonov-like H* / (|H|^2 + λ).
    """
    y = np.asarray(y, float) - np.mean(y)
    N = len(y); Nfft = N + int(pad)
    Yr = np.fft.rfft(y, n=Nfft)
    fr = np.fft.rfftfreq(Nfft, d=1/fs)

    # interpolate H onto FFT grid (mag/phase to avoid wrap issues)
    mag = np.abs(H); phi = np.unwrap(np.angle(H))
    mag_i = np.interp(fr, f, mag, left=mag[0], right=mag[-1])
    phi_i = np.interp(fr, f, phi, left=phi[0], right=phi[-1])
    H_i = mag_i * np.exp(1j*phi_i)

    eps = np.finfo(float).eps
    if coh2 is not None:
        coh_i = np.clip(np.interp(fr, f, coh2, left=0.0, right=0.0), 0.0, 1.0)
        Hinv = coh_i * np.conj(H_i) / np.maximum(mag_i**2, eps)  # = γ^2 / H
    else:
        lam = reg * np.nanmax(mag_i**2)
        Hinv = np.conj(H_i) / (mag_i**2 + lam)

    # band-limit the inverse (e.g. (0, 3000] Hz as in the paper)
    if f_band is not None:
        Hinv *= (fr >= f_band[0]) & (fr <= f_band[1])

    # kill DC (almost always ill-conditioned after detrending)
    Hinv[0] = 0.0

    ycorr = np.fft.irfft(Yr * Hinv, n=Nfft)
    return ycorr[:N]

def main(idx):
    proc = WallPressureProcessor(
        sample_rate=SAMPLE_RATE,
        nu0=NU0,
        rho0=RHO0,
        u_tau0=U_TAU0,
        err_frac=ERR_FRAC,
        W=W,
        He=He,
        L0=L0,
        delta_L0=DELTA_L0,
        U=U,
        C=C,
        mode_m=MODE_M,
        mode_n=MODE_N,
        mode_l=MODE_L,
    )

    # proc.load_data(WALL_PRESSURE_MAT, FREESTREAM_PRESSURE_MAT)
    TEST_MAT = f"data/calibration/{fn_train[idx]}"
    proc.load_test(TEST_MAT)
    ic(f"Loaded data: {proc.p_w.shape}, {proc.p_fs.shape}")
    ref, trt, fs = proc.p_fs.cpu().numpy(), proc.p_w.cpu().numpy(), SAMPLE_RATE

    f, H, coh2 = estimate_frf(ref, trt, fs)
    plot_transfer(f, H, os.path.join(OUTPUT_DIR, f"transfer_function_{idx}.png"))
    trt_corr = inverse_filter(trt, fs, f, H)
    t = np.arange(len(ref))/fs
    plot_corrected_trace(t, ref, trt, trt_corr, os.path.join(OUTPUT_DIR, f"corrected_trace_{idx}.png"))



def main2(sanity=False):
    proc = WallPressureProcessor(
        sample_rate=SAMPLE_RATE,
        nu0=NU0,
        rho0=RHO0,
        u_tau0=U_TAU0,
        err_frac=ERR_FRAC,
        W=W,
        He=H,
        L0=L0,
        delta_L0=DELTA_L0,
        U=U,
        C=C,
        mode_m=MODE_M,
        mode_n=MODE_N,
        mode_l=MODE_L,
    )

    # proc.load_data(WALL_PRESSURE_MAT, FREESTREAM_PRESSURE_MAT)
    proc.load_test(TEST_MAT)
    ic(f"Loaded data: {proc.p_w.shape}, {proc.p_fs.shape}")
    # proc.compute_duct_modes()
    # proc.notch_filter()
    plot_raw_signals(proc.p_w[100:800], proc.p_fs[100:800],
                     os.path.join(OUTPUT_DIR, "raw_signals.png"))

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
        f, H_match, mag_diff = phase_match_transfer(p_w_org, proc.p_w.cpu(), SAMPLE_RATE, smoothing_len=1)
        plot_transfer(f, H_match, mag_diff, os.path.join(OUTPUT_DIR, "complex_transfer_function.png"))

        f, P_w_fs_opt = csd(p_w_org, proc.p_fs.cpu(), fs=SAMPLE_RATE,
                            nperseg=nperseg, noverlap=noverlap, window="hann")
        plot_phase_match_csd(f, P_w, P_fs, P_w_fs, P_w_fs_opt,
                            os.path.join(OUTPUT_DIR, "phase_match_csd.png"))
        
        std_corr = np.std(P_w_fs_opt)

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
    # main(sanity=1)
    for idx in range(len(fn_train)):
        main(idx)

