import numpy as np
from scipy.signal import welch, csd, get_window
import scipy.io as sio

from icecream import ic
import os

from plotting import (
    plot_spectrum,
    plot_transfer_NKD,
    plot_transfer_PH,
    plot_transfer_NC,
    plot_corrected_trace_NKD,
    plot_corrected_trace_NC,
    plot_corrected_trace_PH,
)

############################
# Constants & defaults
############################
FS = 25_000.0
NPERSEG = 2**11
WINDOW = "hann"


R = 287.0         # J/kg/K
T = 293.0         # K (adjust if you have per-case temps)
P_ATM = 101_325.0 # Pa
PSI_TO_PA = 6_894.76
psi_labels = ['atm', '10psi', '30psi', '50psi', '70psi', '100psi']
Re_taus = np.array([1500, 2500, 3500, 4500, 6000, 8000], dtype=float)
u_taus  = np.array([0.571, 0.532, 0.492, 0.515, 0.433, 0.481], dtype=float)
nu_atm  = 1.5e-5  # m^2/s


def inner_scales(Re_taus, u_taus, nu_atm):
    """Return delta (from the atm case) and nu for each case via Re_tau relation."""
    Re_taus = np.asarray(Re_taus, dtype=float)
    u_taus = np.asarray(u_taus, dtype=float)
    delta = Re_taus[0] * nu_atm / u_taus[0]
    nus = delta * u_taus / Re_taus
    return float(delta), nus


def estimate_frf(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    window: str = "hann",
    detrend: str = "constant",
):
    """
    Estimate H1 FRF and magnitude-squared coherence using Welch/CSD.

    Returns
    - f [Hz]
    - H(f) = S_yx / S_xx (complex, x→y)
    - gamma2(f) in [0, 1]
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    nseg = int(min(NPERSEG, x.size, y.size))
    if nseg < 8:
        raise ValueError(f"Signal too short for FRF: n={min(x.size, y.size)}")
    nov = int(min(NPERSEG/2, nseg // 2))
    w = get_window(window, nseg, fftbins=True)

    f, Sxx = welch(x, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=detrend)
    _, Syy = welch(y, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=detrend)
    _, Sxy = csd(x, y, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=detrend)  # x→y

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
    Reconstruct source-domain signal from a measured signal using a
    coherence-weighted inverse filter: H_inv = gamma^2 * H* / |H|^2.

    Parameters
    - y_r: measured time series (maps from source via H)
    - fs:  sample rate [Hz]
    - f,H,gamma2: FRF and coherence defined on frequency vector f
    - pad: zero-padding in samples for FFT length
    - demean: remove mean before FFT
    - zero_dc: zero DC (and Nyquist if present) in inverse filter
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


def load_mat(path: str, key: str = "channelData"):
    """Load an Nx2 array from a MATLAB .mat file under `key` robustly."""
    mat = sio.loadmat(path, squeeze_me=True)
    if key not in mat:
        raise KeyError(f"Key '{key}' not found in {path}. Available: {list(mat.keys())}")
    data = np.asarray(mat[key])
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array under '{key}', got shape {data.shape} in {path}")
    # Handle either (N,2) or (2,N)
    if data.shape[1] == 2:
        x_r = data[:, 0].astype(float)
        y_r = data[:, 1].astype(float)
    elif data.shape[0] == 2:
        x_r = data[0, :].astype(float)
        y_r = data[1, :].astype(float)
    else:
        raise ValueError(f"Unsupported shape for '{key}': {data.shape} in {path}")
    return x_r, y_r


def compute_spec(fs: float, x: np.ndarray, nperseg: int = 2**16):
    """Welch PSD with sane defaults and shape guarding."""
    x = np.asarray(x, float)
    nseg = int(min(nperseg, x.size))
    if nseg < 8:
        raise ValueError(f"Signal too short for PSD: n={x.size}")
    nov = nseg // 2
    w = get_window("hann", nseg, fftbins=True)
    f, Pxx = welch(
        x,
        fs=fs,
        window=w,
        nperseg=nseg,
        noverlap=nov,
        detrend="constant",
        scaling="density",
        return_onesided=True,
    )
    return f, Pxx


def main_PH():
    psi = ['atm', '10psi', '30psi', '50psi', '70psi', '100psi']
    root = 'data/15082025/P2S1_S2naked'
    fn_sweep = [f'{root}/data_{p}.mat' for p in psi]
    OUTPUT_DIR = "figures/PH-NKD"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    PH_path = "figures/PH-NKD"
    NC_path = "figures/NC-NKD"
    NKD_path = "figures/S1-S2"

    delta, nu_s = inner_scales(Re_taus, u_taus, nu_atm)
    pressures = np.array([P_ATM] + [P_ATM + float(p[:-3]) * PSI_TO_PA for p in psi_labels[1:]], dtype=float)
    rhos = pressures / (R * T)

    Times = []
    Pyys = []
    Pyryrs = []
    for idx in range(len(psi)):
        ic(f"Processing {psi[idx]}...")

        # Load data
        x_r, y_r = load_mat(fn_sweep[idx]) # s1 mic is naked, so no need to correct NKD1 to NKD2
        ###
        # Correct NC (invert NC→NKD mapping to bring NC into NKD domain)
        H_NC = np.load(f"{NC_path}/H_{psi[idx]}.npy")
        gamma2_NC = np.load(f"{NC_path}/gamma2_{psi[idx]}.npy")
        f_NC = np.load(f"{NC_path}/f_{psi[idx]}.npy")
        fs = 25000.0
        y = wiener_inverse(y_r, fs, f_NC, H_NC, gamma2_NC)
        # Plot corrected NC
        # f, Pxx = compute_spec(fs, y)
        # plot_spectrum(f, f*Pxx, f"{OUTPUT_DIR}/Pxx_{psi[idx]}_corr")
        ###
        # Correct PH (invert PH→NKD mapping to bring PH into NKD domain) # NKD mic is S2 mic, so we need to correct NKD2 to NKD1 first
        H_PH = np.load(f"{PH_path}/H_{psi[idx]}.npy")
        gamma2_PH = np.load(f"{PH_path}/gamma2_{psi[idx]}.npy")
        f_PH = np.load(f"{PH_path}/f_{psi[idx]}.npy")
        x_s2 = wiener_inverse(x_r, fs, f_PH, H_PH, gamma2_PH)
        # This has now been inverted to the equivalend of the NKD-S2 mic, we need to map it to the response of the pristine S1 mic
        H_NKD = np.load(f"{NKD_path}/H_{psi[idx]}.npy")
        gamma2_NKD = np.load(f"{NKD_path}/gamma2_{psi[idx]}.npy")
        f_NKD = np.load(f"{NKD_path}/f_{psi[idx]}.npy")
        x = wiener_inverse(x_s2, fs, f_NKD, H_NKD, gamma2_NKD)
        
        # Plot corrected PH
        f, Pxx = compute_spec(fs, x_r)
        f_plus = f * nu_s[idx] / (u_taus[idx] ** 2)
        Pxx_plus = Pxx / ((u_taus[idx] ** 2 * rhos[idx]) ** 2) 
        Times.append(1/f_plus)
        Pyys.append(f*Pxx_plus)
        # plot_spectrum(f, f*Pyy, f"{OUTPUT_DIR}/Pyryr_{psi[idx]}_corr")
        ###
        # TF between NC and PH
        f, H, gamma2 = estimate_frf(x, y, fs)
        # plot_transfer_PH(f, H, f"{OUTPUT_DIR}/H_{psi[idx]}", psi[idx])

        y_rej = wiener_inverse(y, fs, f, H, gamma2)
        t = np.arange(len(y)) / fs
        # plot_corrected_trace_PH(t, x, y, y_rej, f"{OUTPUT_DIR}/y_{psi[idx]}", psi[idx])
        f, Pyy_rej = compute_spec(fs, y_rej)
        f_plus = f * nu_s[idx] / (u_taus[idx] ** 2)
        Pyy_rej_plus = Pyy_rej / ((u_taus[idx] ** 2 * rhos[idx]) ** 2)
        Pyryrs.append(f*Pyy_rej_plus)

    plot_spectrum(Times, Pyys, Pyryrs, f"{OUTPUT_DIR}/spectra/Pyy_log.png")


if __name__ == "__main__":
    # main_NC()
    # main_NKD()
    # main_PH()
    real_data()
