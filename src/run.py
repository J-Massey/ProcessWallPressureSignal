"""
Wall- and Free-Stream Pressure Processing
Hard-coded inputs and parameters for transparency.
"""

import os
from icecream import ic

import numpy as np
from scipy.signal import butter, filtfilt, welch, csd, coherence

from i_o import load_stan_wallpressure
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
WALL_PRESSURE_MAT      = "data/wallpressure_booman_Pa.mat"
# WALL_PRESSURE_MAT = "data/Pw_premul_spectra_07_Mar_2025.mat"
FREESTREAM_PRESSURE_MAT = "data/booman_wallpressure_fspressure_650sec_40khz.mat"
# FREESTREAM_PRESSURE_MAT = "data/Pw_FS_premul_spectra_07_Mar_2025.mat"
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
    fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=600)
    # 1) Identify duct modes
    L = L0 + DELTA_L0
    denom = (RHO0 * U_TAU0**2)**2
    duct_modes = compute_duct_modes(U, C, range(5), range(5), range(16), W, H, L, NU0, U_TAU0, ERR_FRAC)
    # Turn these into ranges by propogating error in normalisation quantities
    # 2) Load datasets
    fs_w, p_w = load_stan_wallpressure(WALL_PRESSURE_MAT)
    fs_fs, p_fs = load_stan_wallpressure(FREESTREAM_PRESSURE_MAT)


    # 3) Compute wall pressure spectrum w. and w.out duct modes
    x_wall_filt, f_wall_nom_nom_filt, phi_wall_filt, f_wall_nom, phi_wall_nom, info_wall =\
        notch_filter_timeseries(p_w, SAMPLE_RATE,
                                np.array(duct_modes['min'])*(U_TAU0**2/NU0),
                                np.array(duct_modes['max'])*(U_TAU0**2/NU0),
                                np.array(duct_modes["nom"])*(U_TAU0**2/NU0))
    # f_fs_nom, phi_fs_nom = compute_psd(p_fs, fs=SAMPLE_RATE)

    x_filt_fs, f_nom_filt_fs, phi_filt_fs, f_nom_fs, phi_nom_fs, info_fs =\
        notch_filter_timeseries(p_fs, SAMPLE_RATE,
                                np.array(duct_modes['min'])*(U_TAU0**2/NU0),
                                np.array(duct_modes['max'])*(U_TAU0**2/NU0),
                                np.array(duct_modes["nom"])*(U_TAU0**2/NU0))
    
    fig, axes = plt.subplots(1, 2, figsize=(5.6,3.2), dpi=600)
    title = "Wall Pressure Spectrum"
    color = "blue"
    ax = axes[0]
    ax.plot(1/f_wall_nom, phi_wall_nom, color, lw=0.5, alpha=0.8)
    for idx, f0 in enumerate(duct_modes["nom"]):
        label = "Duct Mode" if idx==0 else None
        ax.axvline(1/f0, color="red", linestyle="--", lw=0.8, alpha=0.5, label=label)
        ax.text(1/f0, ax.get_ylim()[1]*0.9, f"l={duct_modes['l'][idx]}",
                rotation=90, ha="right", va="center", fontsize=8, color="red")
    ax.set_xscale("log")
    ax.set_xlim(1/1e-1, 1/5e-4)
    ax.set_ylim(0, 25)
    ax.set_title(title)
    ax.set_xlabel("$T^+$")
    ax.set_ylabel("$f\\Phi_{pp}^+$")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "duct_wall_pressure_spectrum.png"))
    plt.close()

    p_w = x_wall_filt
    # 3) Compute wall pressure spectrum and free-stream pressure spectrum
    f_wall_nom, phi_wall_nom = compute_psd(p_w, fs=SAMPLE_RATE)
    
    T_plus = 1 / (f_wall_nom * NU0 / U_TAU0**2)
    ax.plot(f_wall_nom,  phi_wall_nom, 'g-', lw=0.5, alpha=0.8, label="Wall Pressure Spectrum")

    # 8) Load the reference wall-pressure PSD
    data  = sio.loadmat("data/premultiplied_spectra_Pw_ReT2000_Deshpande_JFM_2025.mat")
    T_plus_ref = data["Tplus"][0]
    f_ref_plus = 1/T_plus_ref  # convert T+ to f+ (Hz)
    f_Phi_ref_plus = data["premul_Pw_plus"][0]  # first column is the wall-pressure PSD
    u_tau_ref = 0.358
    rho_ref = 1.11
    nu_ref = 1.68e-5  # kinematic viscosity at reference conditions

    # Undo the normalisation to get the reference PSD in physical units
    denom_ref = (rho_ref * u_tau_ref**2)**2
    f_Pxx_ref = f_Phi_ref_plus * denom_ref  # premultiplied PSD in Pa^2/Hz
    f_ref = f_ref_plus / (nu_ref / u_tau_ref**2)  # non-dimensional frequency f+ (Hz)
    Pxx_ref = f_Pxx_ref / f_ref  # convert to physical units (Pa^2/Hz)

    ax.plot(f_ref, Pxx_ref, 'b-', lw=0.5, alpha=0.8, label="Reference Wall PSD")

    ax.set_xscale("log")

    plt.savefig(os.path.join(OUTPUT_DIR, "wall_pressure_spec.png"))

    # Interpolate spectra to a common logspaced frequency grid
    ic(np.log10(f_wall_nom.min()+1e-12), np.log10(f_wall_nom.max()))
    f_grid = np.logspace(
        np.log10(f_wall_nom.min()+1e-1),
        np.log10(f_wall_nom.max()),
        1024
    )

    phi_wall_grid = np.interp(f_grid, f_wall_nom, phi_wall_nom/denom, left=0, right=0)
    phi_wall_grid_smoothed = savgol_filter(phi_wall_grid, window_length=64, polyorder=1)
    Pxx_ref_grid = np.interp(f_grid, f_ref, Pxx_ref/denom_ref, left=0, right=0)

    # === 4. Compute the transfer function (magnitude) over the selected frequency range ===
    # Select the calibration frequency band (e.g., 0.1 Hz to 20000 Hz as per reliable range)
    f_min = 0.1   # Hz
    f_max = 20000.0  # Hz
    # Interpolate
    # Identify indices in f_wall_nom within this band
    band_idx = np.logical_and(f_grid >= f_min, f_grid <= f_max)

    # Calculate ratio of reference to measured PSD in this band
    H_power_ratio = np.ones_like(f_grid)
    H_power_ratio[band_idx] = Pxx_ref_grid[band_idx] / phi_wall_grid_smoothed[band_idx]

    # Take square root to get amplitude transfer function (since PSD ratio = (Amplitude ratio)^2)
    H_mag = np.sqrt(H_power_ratio)  # magnitude of transfer function to apply

    ax.cla()
    ax.semilogx(f_grid, H_mag, 'r-', lw=0.5, alpha=0.8, label="Transfer Function Magnitude")

    # Outside the band, H_mag remains 1 (no correction). This avoids extrapolating beyond known range.
    # (We will later apply a low-pass filter at f_max to handle high-frequency noise.)

    # Optionally, smooth H_mag within the band to avoid sharp fluctuations (e.g., using a moving average or Savitzky-Golay filter).
    H_mag = savgol_filter(H_mag, window_length=16, polyorder=1)  # smooth the transfer function
    ax.semilogx(f_grid, H_mag, 'g-', lw=0.5, alpha=0.8, label="Smoothed Transfer Function Magnitude")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("$|H|$")
    ax.set_title("Transfer Function Magnitude")

    plt.savefig(os.path.join(OUTPUT_DIR, "transfer_function_magnitude.png"))

    # === 5. Apply the inverse transfer function to the wall-pressure signal ===
    # Perform FFT on the full wall-pressure signal
    N = len(p_w)
    nperseg = N // 2000  # segment length for Welch's method
    # Use rfft for one-sided FFT since input is real
    freqs = np.fft.rfftfreq(N, 1/SAMPLE_RATE)        # frequency bins for the FFT result
    ic(freqs.max(), freqs.min(), f_grid.max())
    Wall_fft = np.fft.rfft(p_w)            # complex spectrum of measured wall pressure

    # Interpolate Wall_fft (from Welch frequencies f_wall_nom) onto the full FFT frequency grid if needed
    H_mag_fullres = np.interp(freqs, f_grid, H_mag, left=1.0, right=1.0)
    # replace NaNs or infinities with 1.0 (no correction)
    H_mag_fullres = np.nan_to_num(H_mag_fullres, nan=1.0, posinf=1.0, neginf=1.0)
    

    # Create the correction filter in frequency domain.
    # Since H_mag is the ratio (reference/measured) for amplitude, we multiply measured FFT by H_mag to get corrected spectrum.
    Corrected_fft = Wall_fft * H_mag_fullres

    # Convert back to time domain
    p_w_corrected = np.fft.irfft(Corrected_fft, n=N)
    # Plot the original and corrected wall pressure signals
    fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=600)
    ax.semilogx(freqs, Wall_fft, 'k-', lw=0.5, alpha=0.8, label="Wall Pressure Signal")
    ax.semilogx(freqs, Corrected_fft, 'm-', lw=0.5, alpha=0.8, label="Corrected Wall Pressure Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Corrected Wall Pressure Signal Spectrum")
    ax.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "corrected_wall_pressure_signal_spectrum.png"))

    # === 6. Post-filter the corrected signal (high-frequency noise attenuation) ===
    # Apply a zero-phase low-pass filter at f_max to remove any amplified noise above the reliable range
    # cutoff_norm = f_max / (0.5 * SAMPLE_RATE)  # normalize cutoff freq by Nyquist (fs/2)
    # b, a = butter(N=4, Wn=cutoff_norm, btype='low')  # 4th-order Butterworth low-pass
    # p_w_corrected = filtfilt(b, a, p_w_corrected)  # zero-phase filtering (no phase distortion)

    # The corrected time series is now in p_w_corrected. We can output or save this as needed:
    # e.g., np.save('corrected_wall_pressure.npy', p_w_corrected)

    # === 7. Compute corrected spectrum for verification ===
    f_corr, Pxx_corr = welch(p_w_corrected, fs=SAMPLE_RATE, nperseg=nperseg, noverlap=nperseg//2, window='hann')
    # Compute premultiplied spectra for plotting
    premult_wall = f_grid * phi_wall_grid
    premult_ref  = f_grid * Pxx_ref_grid
    premult_corr = f_corr * Pxx_corr

    # Smooth these spectra
    premult_wall = savgol_filter(premult_wall, window_length=64, polyorder=1)
    # premult_ref = savgol_filter(premult_ref, window_length=101, polyorder=1)
    # premult_corr = savgol_filter(premult_corr, window_length=101, polyorder=1)

    # Plot the original, reference, and corrected spectra
    fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=600)
    T_plus = 1 / (f_grid * NU0 / U_TAU0**2)
    ax.semilogx(T_plus, premult_wall, 'g-', lw=0.5, alpha=0.8, label="Original Wall PSD")
    T_plus_ref = 1 / (f_grid * NU0 / U_TAU0**2)
    ax.semilogx(T_plus_ref, premult_ref, 'b-', lw=0.5, alpha=0.8, label="Reference Wall PSD")
    T_plus_corr = 1 / (f_corr * NU0 / U_TAU0**2)
    # ax.semilogx(T_plus_corr, premult_corr/denom, 'm-', lw=0.5, alpha=0.8, label="Corrected Wall PSD")

    # Interpolate and smooth corrected spectrum for better visualization
    premult_corr_smooth = savgol_filter(np.interp(f_grid, f_corr, premult_corr/denom, left=0, right=0), window_length=64, polyorder=1)
    ax.semilogx(T_plus, premult_corr_smooth, 'm--', lw=0.5, alpha=0.8, label="Corrected Wall PSD")

    ax.set_title("Smoothed Wall Pressure Spectrum Comparison")
    ax.set_xlabel("$T^+$")
    ax.set_ylabel("$f\\Phi_{pp}^+$")
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.5)
    # ax.set_ylim(0, 5)
    # ax.set_xlim(1e1, 5e3)
    plt.savefig(os.path.join(OUTPUT_DIR, "corrected_wall_pressure_spectrum.png"))
    plt.close(fig)

    # 4) Plot 
    # fig, axes = plt.subplots(1,2, figsize=(5.6, 2.5), dpi=600, sharex=True, sharey=True)
    # ax = axes[0]
    # ax.set_title("Wall Pressure Spectrum")
    # T_plus = 1 / (f_wall_nom_nom_filt * NU0 / U_TAU0**2)
    # ax.plot(T_plus, phi_wall_filt*f_wall_nom_nom_filt, 'g-', lw=0.5, alpha=0.8, label="Filtered")
    # T_plus = 1 / (f_wall_nom_nom * NU0 / U_TAU0**2)
    # ax.plot(T_plus, phi_wall_nom*f_wall_nom_nom, 'r-', lw=0.5, alpha=0.8, label="Wall Pressure")
    # ax.legend()
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