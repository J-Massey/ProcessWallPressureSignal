import numpy as np
from scipy.signal import welch, csd, get_window
import scipy.io as sio
from matplotlib import pyplot as plt

from icecream import ic
import os

from plotting import (
    plot_spectrum,
    plot_raw_spectrum,
    plot_transfer_NKD,
    plot_transfer_PH,
    plot_transfer_NC,
    plot_corrected_trace_NKD,
    plot_corrected_trace_NC,
    plot_corrected_trace_PH,
    plot_time_series,
)

############################
# Constants & defaults
############################
FS = 50_000.0
NPERSEG = 2**11
WINDOW = "hann"
CALIB_BASE_DIR = "data/calibration"  # base directory to save calibration .npy files
TRIM_CAL_SECS = 30  # seconds to trim from the start of calibration runs (set to 0 to disable)

R = 287.0         # J/kg/K
T = 293.0         # K (adjust if you have per-case temps)
P_ATM = 101_325.0 # Pa
PSI_TO_PA = 6_894.76

# Labels for operating conditions; first entry is assumed 'atm'
psi_labels = ['atm']
Re_taus = np.array([1500], dtype=float)
u_taus  = np.array([0.571], dtype=float)
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
    window: str = WINDOW,
    detrend: str = "constant",
):
    """
    Estimate H1 FRF and magnitude-squared coherence using Welch/CSD.

    Returns
    -------
    f : array_like [Hz]
    H : array_like (complex) = S_yx / S_xx  (x → y)
    gamma2 : array_like in [0, 1]
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    nseg = int(min(NPERSEG, x.size, y.size))
    if nseg < 8:
        raise ValueError(f"Signal too short for FRF: n={min(x.size, y.size)}")
    nov = int(min(NPERSEG // 2, nseg // 2))
    w = get_window(window, nseg, fftbins=True)

    f, Sxx = welch(x, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=detrend)
    _, Syy = welch(y, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=detrend)
    _, Sxy = csd(x, y, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=detrend)  # x→y

    H = Sxy / Sxx
    gamma2 = (np.abs(Sxy) ** 2) / (Sxx * Syy)
    gamma2 = np.clip(gamma2.real, 0.0, 1.0)
    return f, H, gamma2


def _resolve_cal_dir(name: str) -> str:
    """
    Prefer new `data/calibration/<name>` if present; otherwise fall back to `figures/<name>`.
    This keeps backward compatibility with existing saved calibrations.
    """
    new_dir = os.path.join(CALIB_BASE_DIR, name)
    if os.path.isdir(new_dir):
        return new_dir
    old_dir = os.path.join("figures", name)
    return old_dir


def wiener_inverse(
    y_r: np.ndarray,
    fs: float,
    f: np.ndarray,
    H: np.ndarray,
    gamma2: np.ndarray,
    demean: bool = True,
    zero_dc: bool = True,
):
    """
    Reconstruct source-domain signal from a measured signal using a
    coherence-weighted inverse filter: H_inv = gamma^2 * H* / |H|^2.

    Parameters
    ----------
    y_r : array_like
        Measured time series (maps from source via H).
    fs : float
        Sample rate [Hz].
    f, H, gamma2 : arrays
        FRF and coherence tabulated on frequency vector f (as from estimate_frf).
        H must correspond to x→y (so this operation aims to recover x from y).
    demean : bool
        Remove mean before FFT.
    zero_dc : bool
        Zero DC (and Nyquist if present) in the inverse filter.

    Returns
    -------
    x_hat : array_like
        Reconstructed source-domain time series.
    """
    y = np.asarray(y_r, float)
    if demean:
        y = y - y.mean()
    N = y.size
    Nfft = int(2 ** np.ceil(np.log2(N)))  # next power of 2

    # FFT of measurement
    Yr = np.fft.rfft(y, n=Nfft)
    fr = np.fft.rfftfreq(Nfft, d=1.0 / fs)

    # Interpolate |H| and unwrapped phase to FFT grid
    mag = np.abs(H)
    phi = np.unwrap(np.angle(H))
    # Safer OOB behaviour: taper magnitude to zero outside measured band
    mag_i = np.interp(fr, f, mag, left=0.0, right=0.0)
    phi_i = np.interp(fr, f, phi, left=phi[0], right=phi[-1])
    Hi = mag_i * np.exp(1j * phi_i)

    # Interpolate and clip coherence; set OOB to 0 as well
    g2_i = np.clip(np.interp(fr, f, gamma2, left=0.0, right=0.0), 0.0, 1.0)

    # Inverse filter
    eps = np.finfo(float).eps
    Hinv = g2_i * np.conj(Hi) / np.maximum(mag_i**2, eps)
    if zero_dc:
        Hinv[0] = 0.0
        if Nfft % 2 == 0:  # real Nyquist bin exists
            Hinv[-1] = 0.0

    x_hat = np.fft.irfft(Yr * Hinv, n=Nfft)[:N]
    return x_hat


def apply_frf(
    x: np.ndarray,
    fs: float,
    f: np.ndarray,
    H: np.ndarray,
    demean: bool = True,
    zero_dc: bool = True,
):
    """
    Apply a measured FRF H (x→y) to a time series x to synthesise y.
    This is the forward operation: Y = H · X in the frequency domain.
    """
    x = np.asarray(x, float)
    if demean:
        x = x - x.mean()

    N = x.size
    Nfft = int(2 ** np.ceil(np.log2(N)))
    X = np.fft.rfft(x, n=Nfft)
    fr = np.fft.rfftfreq(Nfft, d=1.0 / fs)

    mag = np.abs(H)
    phi = np.unwrap(np.angle(H))
    # Safer OOB behaviour: taper magnitude to zero outside measured band
    mag_i = np.interp(fr, f, mag, left=0.0, right=0.0)
    phi_i = np.interp(fr, f, phi, left=phi[0], right=phi[-1])
    Hi = mag_i * np.exp(1j * phi_i)

    if zero_dc:
        Hi[0] = 0.0
        if Nfft % 2 == 0:
            Hi[-1] = 0.0

    y = np.fft.irfft(X * Hi, n=Nfft)[:N]
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


def compute_spec(fs: float, x: np.ndarray):
    """Welch PSD with sane defaults and shape guarding."""
    x = np.asarray(x, float)
    nseg = int(min(NPERSEG, x.size))
    if nseg < 8:
        raise ValueError(f"Signal too short for PSD: n={x.size}")
    nov = nseg // 2
    w = get_window(WINDOW, nseg, fftbins=True)
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


############################
# Calibration: PH→NKD
############################
def main_PH():
    """
    Calibrate, save and check the PH→NKD transfer function.
    x_ref: PH (input), y_meas: NKD (output)
    """
    root = 'data/11092025'
    fn_sweep = [f'{root}/cali.mat' for _ in psi_labels]
    FIG_DIR = "figures/cali_09/PH-NKD"
    CAL_DIR = os.path.join(CALIB_BASE_DIR, "PH-NKD")
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(CAL_DIR, exist_ok=True)

    for idx in range(len(psi_labels)):
        ic(f"Processing {psi_labels[idx]}...")

        # Load data: expect first column = PH, second = NKD
        ph, nkd = load_mat(fn_sweep[idx], key='channelData_300_plug')

        # Trim initial transient (consistent with NKD→NC calibration)
        if TRIM_CAL_SECS > 0:
            start = int(TRIM_CAL_SECS * FS)
            ph, nkd = ph[start:], nkd[start:]

        plot_time_series(np.arange(len(ph)) / FS, ph, f"{FIG_DIR}/ph_{psi_labels[idx]}")

        # FRF PH→NKD
        f, H, gamma2 = estimate_frf(ph, nkd, FS)
        np.save(f"{CAL_DIR}/H_{psi_labels[idx]}.npy", H)
        np.save(f"{CAL_DIR}/gamma2_{psi_labels[idx]}.npy", gamma2)
        np.save(f"{CAL_DIR}/f_{psi_labels[idx]}.npy", f)
        ic(f.shape, H.shape, gamma2.shape)
        plot_transfer_PH(f, H, f"{FIG_DIR}/H_{psi_labels[idx]}", psi_labels[idx])

        # Sanity: reconstruct PH from NKD using inverse (should resemble PH)
        ph_hat = wiener_inverse(nkd, FS, f, H, gamma2)
        t = np.arange(len(ph_hat)) / FS
        plot_corrected_trace_PH(t, ph, nkd, ph_hat, f"{FIG_DIR}/ph_recon_{psi_labels[idx]}", psi_labels[idx])


############################
# Calibration: NKD→NC
############################
def main_NC():
    """
    Calibrate NKD→NC transfer function.
    x_ref: NKD (input), y_meas: NC (output)
    """
    root = 'data/11092025'
    fn_sweep = [f'{root}/cali.mat' for _ in psi_labels]
    FIG_DIR = "figures/cali_09/NKD-NC"
    CAL_DIR = os.path.join(CALIB_BASE_DIR, "NKD-NC")
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(CAL_DIR, exist_ok=True)

    for idx in range(len(psi_labels)):
        ic(f"Processing {psi_labels[idx]}...")

        # Load data: expect first column = NKD, second = NC
        nkd, nc = load_mat(fn_sweep[idx], key='channelData_300_nose')
        plot_time_series(np.arange(len(nkd)) / FS, nkd, f"{FIG_DIR}/nkd_{psi_labels[idx]}")

        # Trim first TRIM_CAL_SECS to avoid initial transients
        if TRIM_CAL_SECS > 0:
            start = int(TRIM_CAL_SECS * FS)
            nkd, nc = nkd[start:], nc[start:]

        # FRF NKD→NC
        f, H, gamma2 = estimate_frf(nkd, nc, FS)
        np.save(f"{CAL_DIR}/H_{psi_labels[idx]}.npy", H)
        np.save(f"{CAL_DIR}/gamma2_{psi_labels[idx]}.npy", gamma2)
        np.save(f"{CAL_DIR}/f_{psi_labels[idx]}.npy", f)
        ic(f.shape, H.shape, gamma2.shape)
        plot_transfer_NC(f, H, f"{FIG_DIR}/H_{psi_labels[idx]}", psi_labels[idx])

        # Sanity: reconstruct NKD from NC via inverse
        nkd_hat = wiener_inverse(nc, FS, f, H, gamma2)
        t = np.arange(len(nkd_hat)) / FS
        plot_corrected_trace_NC(t, nkd, nc, nkd_hat, f"{FIG_DIR}/nkd_recon_{psi_labels[idx]}", psi_labels[idx])


############################
# Apply to flow data
############################
def real_data():
    root = 'data/11092025'
    fn_sweep = [f'{root}/data.mat' for _ in psi_labels]
    OUTPUT_DIR = "figures/cali_09/flow"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    PH_path = _resolve_cal_dir("PH-NKD")   # PH→NKD
    NC_path = _resolve_cal_dir("NKD-NC")   # NKD→NC   (correct orientation)

    RAW_DIR = f"{OUTPUT_DIR}/raw"
    os.makedirs(RAW_DIR, exist_ok=True)

    delta, nu_s = inner_scales(Re_taus, u_taus, nu_atm)
    pressures = np.array([P_ATM] + [P_ATM + float(p[:-3]) * PSI_TO_PA for p in psi_labels[1:]], dtype=float)
    rhos = pressures / (R * T)

    Times = []
    Pyys = []

    for idx in range(len(psi_labels)):
        ic(f"Processing {psi_labels[idx]}...")

        # Load data: NC (first col), PH (second col)
        nc, ph = load_mat(fn_sweep[idx], key='channelData_300')  # NC, PH

        # --- Raw PH spectrum (for comparison)
        f, Pphph = compute_spec(FS, ph)
        f_plus = f * nu_s[idx] / (u_taus[idx] ** 2)
        Pphph_plus = Pphph / ((u_taus[idx] ** 2 * rhos[idx]) ** 2)
        mask = f_plus > 0
        plot_raw_spectrum(1.0 / f_plus[mask], (f_plus[mask] * Pphph_plus[mask]),
                          f"{OUTPUT_DIR}/Pphph_{psi_labels[idx]}_raw")

        t = np.arange(len(ph)) / FS

        # --- Correct NC: invert NKD→NC to get NKD from NC
        H_NC = np.load(f"{NC_path}/H_{psi_labels[idx]}.npy")
        gamma2_NC = np.load(f"{NC_path}/gamma2_{psi_labels[idx]}.npy")
        f_NC = np.load(f"{NC_path}/f_{psi_labels[idx]}.npy")
        nkd_from_nc = wiener_inverse(nc, FS, f_NC, H_NC, gamma2_NC)
        # plot_time_series(t, nkd_from_nc, f"{OUTPUT_DIR}/nkd_from_nc_{psi_labels[idx]}")

        # --- Correct PH: apply PH→NKD forward to get NKD from PH
        H_PH = np.load(f"{PH_path}/H_{psi_labels[idx]}.npy")
        f_PH = np.load(f"{PH_path}/f_{psi_labels[idx]}.npy")
        nkd_from_ph = apply_frf(ph, FS, f_PH, H_PH)
        # plot_time_series(t, nkd_from_ph, f"{OUTPUT_DIR}/nkd_from_ph_{psi_labels[idx]}")

        # --- Spectrum of corrected PH (NKD domain)
        f, Pyy = compute_spec(FS, nkd_from_ph)
        f_plus = f * nu_s[idx] / (u_taus[idx] ** 2)
        Pyy_plus = Pyy / ((u_taus[idx] ** 2 * rhos[idx]) ** 2)
        mask = f_plus > 0
        Times.append(1.0 / f_plus[mask])
        Pyys.append(f_plus[mask] * Pyy_plus[mask])
        plot_raw_spectrum(1.0 / f_plus[mask], (f_plus[mask] * Pyy_plus[mask]),
                          f"{OUTPUT_DIR}/Pyy_{psi_labels[idx]}_corr")

        # --- Coherent rejection: remove component of nkd_from_ph coherent with nkd_from_nc
        # FRF x→y with x=nkd_from_nc, y=nkd_from_ph (both NKD domain)
        f_xy, H_xy, gamma2_xy = estimate_frf(nkd_from_nc, nkd_from_ph, FS)
        y_hat = apply_frf(nkd_from_nc, FS, f_xy, H_xy)   # predict coherent part of PH stream
        y_resid = nkd_from_ph - y_hat                    # coherent rejection residual

        f, Pyy_rej = compute_spec(FS, y_resid)
        f_plus = f * nu_s[idx] / (u_taus[idx] ** 2)
        Pyy_rej_plus = Pyy_rej / ((u_taus[idx] ** 2 * rhos[idx]) ** 2)
        mask = f_plus > 0
        plot_raw_spectrum(1.0 / f_plus[mask], (f_plus[mask] * Pyy_rej_plus[mask]),
                          f"{OUTPUT_DIR}/Pyy_{psi_labels[idx]}_rej")


if __name__ == "__main__":
    # Run calibrations if needed to (re)generate FRFs:
    main_PH()
    main_NC()

    # Apply to the real flow data:
    real_data()
