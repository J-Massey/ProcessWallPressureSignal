import numpy as np
from scipy.signal import welch, csd, get_window
import scipy.io as sio

from icecream import ic
import os

from plotting import plot_spectrum, plot_transfer_NKD, plot_transfer_PH, plot_transfer_NC, plot_corrected_trace_NKD, plot_corrected_trace_NC, plot_corrected_trace_PH


def estimate_frf(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    nperseg: int = 4096,
    noverlap: int = 2048,
    window: str = "hann",
    detrend: str = "constant",
):
    """
    H1 FRF and coherence via Welch/CSD.
    Returns: f [Hz], H(f)=S_yx/S_xx (complex), gamma2(f).
    """
    w = get_window(window, nperseg, fftbins=True)
    f, Sxx = welch(x, fs=fs, window=w, nperseg=nperseg, noverlap=noverlap, detrend=detrend)
    _, Syy = welch(y, fs=fs, window=w, nperseg=nperseg, noverlap=noverlap, detrend=detrend)
    _, Sxy = csd(x, y, fs=fs, window=w, nperseg=nperseg, noverlap=noverlap, detrend=detrend)  # x→y
    H = Sxy / Sxx
    gamma2 = (np.abs(Sxy) ** 2) / (Sxx * Syy)
    gamma2 = np.clip(gamma2.real, 0.0, 1.0)
    return f, H, gamma2


def wiener_inverse(
    y_r: np.ndarray,
    fs: float,
    f: np.ndarray,
    H: np.ndarray,
    gamma2: np.ndarray,
    pad: int = 0,
    demean: bool = True,
    zero_dc: bool = True,
):
    """
    Reconstruct y from y_r using coherence-weighted inverse: H_inv = gamma^2 * H* / |H|^2.
    Returns: y (time series).
    """
    y = np.asarray(y_r, float)
    if demean:
        y = y - y.mean()
    N = y.size
    Nfft = N + int(pad)

    # FFT of measurement
    Yr = np.fft.rfft(y, n=Nfft)
    fr = np.fft.rfftfreq(Nfft, d=1.0 / fs)

    # Interpolate |H| and unwrapped phase to FFT grid
    mag = np.abs(H)
    phi = np.unwrap(np.angle(H))
    mag_i = np.interp(fr, f, mag, left=mag[0], right=mag[-1])
    phi_i = np.interp(fr, f, phi, left=phi[0], right=phi[-1])
    Hi = mag_i * np.exp(1j * phi_i)

    # Interpolate and clip coherence
    g2_i = np.clip(np.interp(fr, f, gamma2, left=gamma2[0], right=gamma2[-1]), 0.0, 1.0)

    # Inverse filter
    eps = np.finfo(float).eps
    Hinv = g2_i * np.conj(Hi) / np.maximum(mag_i**2, eps)
    if zero_dc:
        Hinv[0] = 0.0
        if Nfft % 2 == 0:  # real Nyquist bin exists
            Hinv[-1] = 0.0

    y = np.fft.irfft(Yr * Hinv, n=Nfft)[:N]
    return y


def load_mat(path):
    ic(sio.loadmat(path).keys())
    data = sio.loadmat(path)['channelData']
    x_r = np.array(data[:, 0])
    y_r = np.array(data[:, 1])
    return x_r, y_r


def compute_spec(fs, x: np.ndarray, nperseg: int = 4096):
    f, Pxx = welch(x, fs=fs, nperseg=nperseg)
    return f, Pxx


def main_PH():
    psi = ['atm', '10psi', '30psi', '50psi', '70psi', '100psi']
    root = 'data/15082025/P2S1_S2naked'
    fn_sweep = [f'{root}/data_{p}.mat' for p in psi]
    OUTPUT_DIR = "figures/PH-NKD"
    os.system(f"mkdir -p {OUTPUT_DIR}")  # ensure output directory exists
    NKD_path = "figures/S1-S2"
    
    for idx in range(len(psi)):
        ic(f"Processing {psi[idx]}...")

        # Load data
        x_r, y_r = load_mat(fn_sweep[idx])
        H_NKD = np.load(f"{NKD_path}/H_{psi[idx]}.npy")
        gamma2_NKD = np.load(f"{NKD_path}/gamma2_{psi[idx]}.npy")
        f_NKD = np.load(f"{NKD_path}/f_{psi[idx]}.npy")
        fs = 25000.0
        x = wiener_inverse(x_r, fs, f_NKD, H_NKD, gamma2_NKD)

        f, H, gamma2 = estimate_frf(x, y_r, fs)
        np.save(f"{OUTPUT_DIR}/H_{psi[idx]}.npy", H)
        np.save(f"{OUTPUT_DIR}/gamma2_{psi[idx]}.npy", gamma2)
        np.save(f"{OUTPUT_DIR}/f_{psi[idx]}.npy", f)
        ic(f.shape, H.shape, gamma2.shape)
        plot_transfer_PH(f, H, f"{OUTPUT_DIR}/H_{psi[idx]}", psi[idx])

        y = wiener_inverse(y_r, fs, f, H, gamma2)
        t = np.arange(len(y)) / fs
        plot_corrected_trace_PH(t, x_r, y_r, y, f"{OUTPUT_DIR}/y_{psi[idx]}", psi[idx])


def main_NC():
    psi = ['atm', '10psi', '30psi', '50psi', '70psi', '100psi']
    root = 'data/15082025/NCS2_S1naked'
    fn_sweep = [f'{root}/data_{p}.mat' for p in psi]
    OUTPUT_DIR = "figures/NC-NKD"
    os.system(f"mkdir -p {OUTPUT_DIR}")  # ensure output directory exists
    
    for idx in range(len(psi)):
        ic(f"Processing {psi[idx]}...")

        # Load data
        x_r, y_r = load_mat(fn_sweep[idx])
        fs = 25000.0 
        
        f, H, gamma2 = estimate_frf(x_r, y_r, fs)
        np.save(f"{OUTPUT_DIR}/H_{psi[idx]}.npy", H)
        np.save(f"{OUTPUT_DIR}/gamma2_{psi[idx]}.npy", gamma2)
        np.save(f"{OUTPUT_DIR}/f_{psi[idx]}.npy", f)
        ic(f.shape, H.shape, gamma2.shape)
        plot_transfer_NC(f, H, f"{OUTPUT_DIR}/H_{psi[idx]}", psi[idx])

        y = wiener_inverse(y_r, fs, f, H, gamma2)
        t = np.arange(len(y)) / fs
        plot_corrected_trace_NC(t, x_r, y_r, y, f"{OUTPUT_DIR}/y_{psi[idx]}", psi[idx])

def main_NKD():
    psi = ['atm', '10psi', '30psi', '50psi', '70psi', '100psi']
    root = 'data/15082025/S1naked_S2naked'
    fn_sweep = [f'{root}/data_{p}.mat' for p in psi]
    OUTPUT_DIR = "figures/S1-S2"
    os.system(f"mkdir -p {OUTPUT_DIR}")  # ensure output directory exists
    
    for idx in range(len(psi)):
        ic(f"Processing {psi[idx]}...")

        # Load data
        x_r, y_r = load_mat(fn_sweep[idx])
        fs = 25000.0 
        f, H, gamma2 = estimate_frf(x_r, y_r, fs)
        np.save(f"{OUTPUT_DIR}/H_{psi[idx]}.npy", H)
        np.save(f"{OUTPUT_DIR}/gamma2_{psi[idx]}.npy", gamma2)
        np.save(f"{OUTPUT_DIR}/f_{psi[idx]}.npy", f)
        ic(f.shape, H.shape, gamma2.shape)
        plot_transfer_NKD(f, H, f"{OUTPUT_DIR}/H_{psi[idx]}", psi[idx])

        y = wiener_inverse(y_r, fs, f, H, gamma2)
        t = np.arange(len(y)) / fs
        plot_corrected_trace_NKD(t, x_r, y_r, y, f"{OUTPUT_DIR}/y_{psi[idx]}", psi[idx])


def real_data():
    psi = ['atm', '10psi', '30psi', '50psi', '70psi', '100psi']
    root = 'data/14082025/flow/maxspeed'
    fn_sweep = [f'{root}/data_{p}.mat' for p in psi]
    OUTPUT_DIR = "figures/real"
    os.system(f"mkdir -p {OUTPUT_DIR}")  # ensure output directory exists
    PH_path = "figures/PH-NKD"
    NC_path = "figures/NC-NKD"
    
    for idx in range(len(psi)):
        ic(f"Processing {psi[idx]}...")

        # Load data
        x_r, y_r = load_mat(fn_sweep[idx])
        # plot raw data
        f, Pxx = compute_spec(25000.0, x_r)
        # plot_spectrum(f, f*Pxx, f"{OUTPUT_DIR}/Pxrxr_{psi[idx]}_raw")
        f, Pyy = compute_spec(25000.0, y_r)
        plot_spectrum(f, f*Pxx, f*Pyy, f"{OUTPUT_DIR}/spectra/P_{psi[idx]}_raw")
        ###
        # Correct NC
        H_NC = np.load(f"{NC_path}/H_{psi[idx]}.npy")
        gamma2_NC = np.load(f"{NC_path}/gamma2_{psi[idx]}.npy")
        f_NC = np.load(f"{NC_path}/f_{psi[idx]}.npy")
        fs = 25000.0
        x = wiener_inverse(x_r, fs, f_NC, H_NC, gamma2_NC)
        # Plot corrected NC
        f, Pxx = compute_spec(fs, x)
        # plot_spectrum(f, f*Pxx, f"{OUTPUT_DIR}/Pxx_{psi[idx]}_corr")
        ###
        # Correct PH
        H_PH = np.load(f"{PH_path}/H_{psi[idx]}.npy")
        gamma2_PH = np.load(f"{PH_path}/gamma2_{psi[idx]}.npy")
        f_PH = np.load(f"{PH_path}/f_{psi[idx]}.npy")
        fs = 25000.0
        y = wiener_inverse(y_r, fs, f_PH, H_PH, gamma2_PH)
        # Plot corrected PH
        f, Pyy = compute_spec(fs, y)
        # plot_spectrum(f, f*Pyy, f"{OUTPUT_DIR}/Pyryr_{psi[idx]}_corr")
        ###
        # TF between NC and PH
        f, H, gamma2 = estimate_frf(x, y, fs)
        plot_transfer_PH(f, H, f"{OUTPUT_DIR}/H_{psi[idx]}", psi[idx])

        y_rej = wiener_inverse(y, fs, f, H, gamma2)
        t = np.arange(len(y)) / fs
        plot_corrected_trace_PH(t, x, y, y_rej, f"{OUTPUT_DIR}/y_{psi[idx]}", psi[idx])
        f, Pyy_rej = compute_spec(fs, y_rej)
        plot_spectrum(f, f*Pxx, f*Pyy_rej, f"{OUTPUT_DIR}/P_{psi[idx]}_rej")


if __name__ == "__main__":
    # main_NC()
    # main_NKD()
    # main_PH()
    real_data()
