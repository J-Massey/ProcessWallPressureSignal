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
    plot_spectrum_pipeline,
)

############################
# Constants & defaults
############################
FS = 50_000.0
NPERSEG = 2**10
WINDOW = "hann"
CALIB_BASE_DIR = "data/calibration"  # base directory to save calibration .npy files
TRIM_CAL_SECS = 30  # seconds trimmed from the start of calibration runs (0 to disable)

R = 287.0         # J/kg/K
T = 298.0         # K (adjust if you have per-case temps)
P_ATM = 101_325.0 # Pa
PSI_TO_PA = 6_894.76

# Labels for operating conditions; first entry is assumed 'atm'
psi_labels = ['atm']
Re_taus = np.array([1500], dtype=float)
u_taus  = np.array([0.481/2], dtype=float)
nu_atm  = 1.52e-5  # m^2/s

# -------------------------------------------------------------------------
# >>> Units for loaded time-series (per MATLAB key, per column) <<<
# Many labs log pressures in kPa. If yours are already in Pa, change 'kPa'→'Pa'.
# If you have raw volts, set to 'V' and provide sensitivities below.
# Keys:
#   - channelData_300_plug : col1=NKD, col2=PH  (calibration sweep)
#   - channelData_300_nose : col1=NKD, col2=NC  (calibration sweep)
#   - channelData_300      : col1=NC,  col2=PH  (real flow data)
DEFAULT_UNITS = {
    'channelData_300_plug': ('Pa', 'Pa'),  # NKD, PH
    'channelData_300_nose': ('Pa', 'Pa'),  # NKD, NC
    'channelData_300':      ('Pa', 'Pa'),  # NC,  PH
}

# If using volts, specify sensitivities (V/Pa) and preamp gains here, e.g.:
SENSITIVITIES_V_PER_PA = {  # leave empty if not using 'V'
    # 'NKD': 0.05,  # 50 mV/Pa example
    # 'PH':  0.05,
    # 'NC':  0.05,
}
PREAMP_GAIN = {  # linear gain; leave 1.0 if unknown
    'NKD': 1.0,
    'PH':  1.0,
    'NC':  1.0,
}
# -------------------------------------------------------------------------


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

    H = np.conj(Sxy) / Sxx
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


def convert_to_pa(x: np.ndarray, units: str, *, channel_name: str = "unknown") -> np.ndarray:
    """
    Convert a pressure time series to Pa.
    Supported units: 'Pa', 'kPa', 'mbar', 'V'.
    For 'V', you must provide a sensitivity (V/Pa) and optional preamp gain via dicts above.
    """
    u = units.lower()
    if u == 'pa':
        return x.astype(float)
    elif u == 'kpa':
        return (x.astype(float) * 1e3)
    elif u == 'mbar':
        return (x.astype(float) * 100.0)  # 1 mbar = 100 Pa
    elif u in ('v', 'volt', 'volts'):
        if channel_name not in SENSITIVITIES_V_PER_PA or SENSITIVITIES_V_PER_PA[channel_name] is None:
            raise ValueError(
                f"Sensitivity (V/Pa) for channel '{channel_name}' not provided; cannot convert V→Pa."
            )
        sens = float(SENSITIVITIES_V_PER_PA[channel_name])  # V/Pa
        gain = float(PREAMP_GAIN.get(channel_name, 1.0))
        # Pa = V / (gain * (V/Pa))
        return x.astype(float) / (gain * sens)
    else:
        raise ValueError(f"Unsupported units '{units}' for channel '{channel_name}'")


def load_mat(path: str, key: str = "channelData"):
    """Load an Nx2 array from a MATLAB .mat file under `key` robustly (no unit conversion)."""
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


def load_mat_to_pa(path: str, key: str, ch1_name: str, ch2_name: str):
    """
    Load two channels and convert each to Pa using DEFAULT_UNITS[key].
    ch1_name/ch2_name are used only if units are 'V' (to look up sensitivity/gain).
    """
    x_r, y_r = load_mat(path, key=key)
    units_pair = DEFAULT_UNITS.get(key, ('Pa', 'Pa'))
    x_pa = convert_to_pa(x_r, units_pair[0], channel_name=ch1_name)
    y_pa = convert_to_pa(y_r, units_pair[1], channel_name=ch2_name)
    return x_pa, y_pa


def compute_spec(fs: float, x: np.ndarray):
    """Welch PSD with sane defaults and shape guarding. Returns (f [Hz], Pxx [Pa^2/Hz])."""
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


# --------- Inner scaling helpers (units & Jacobian are correct) ---------
def f_plus_from_f(f: np.ndarray, u_tau: float, nu: float) -> np.ndarray:
    """f⁺ = f * nu / u_tau²."""
    return f * (nu / (u_tau**2))


def phi_pp_plus_per_fplus(Pyy: np.ndarray, rho: float, u_tau: float, nu: float) -> np.ndarray:
    """
    Dimensionless PSD per unit f⁺:
    Φ_pp⁺(f⁺) = Φ_pp(f) / (ρ² u_τ² ν)
    (Jacobain df/df⁺ = u_τ²/ν has been applied.)
    """
    return Pyy / ((rho**2) * (u_tau**2) * nu)


def premultiplied_phi_pp_plus(f: np.ndarray, Pyy: np.ndarray, rho: float, u_tau: float, nu: float):
    """
    Return (1/f⁺, f⁺ Φ_pp⁺(f⁺)) with the zero bin safely excluded.
    Using the identity: f⁺ Φ_pp⁺ = f * Φ_pp / (ρ² u_τ⁴), but computed via f⁺·Φ_pp⁺ for clarity.
    """
    f = np.asarray(f, float)
    Pyy = np.asarray(Pyy, float)
    f_plus = f_plus_from_f(f, u_tau, nu)
    phi_plus = phi_pp_plus_per_fplus(Pyy, rho, u_tau, nu)   # per unit f⁺
    y = f_plus * phi_plus                                   # premultiplied
    mask = f_plus > 0
    return 1.0 / f_plus[mask], y[mask]


############################
# Calibration: PH→NKD
############################
def main_PH():
    """
    Calibrate, save and check the PH→NKD transfer function.
    NOTE: In channelData_300_plug, col1 = NKD (output), col2 = PH (input).
    We compute H as PH→NKD using (x=PH, y=NKD). All signals are converted to Pa.
    """
    root = 'data/11092025'
    fn_sweep = [f'{root}/cali.mat' for _ in psi_labels]
    FIG_DIR = "figures/cali_09/PH-NKD"
    CAL_DIR = os.path.join(CALIB_BASE_DIR, "PH-NKD")
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(CAL_DIR, exist_ok=True)

    for idx in range(len(psi_labels)):
        ic(f"Processing {psi_labels[idx]}...")

        # Load data (to Pa): col1 = NKD, col2 = PH
        nkd, ph = load_mat_to_pa(fn_sweep[idx], key='channelData_300_plug', ch1_name='NKD', ch2_name='PH')

        # Trim initial transient (consistent with NKD→NC calibration)
        if TRIM_CAL_SECS > 0:
            start = int(TRIM_CAL_SECS * FS)
            nkd, ph = nkd[start:], ph[start:]

        # Optional quick-look at PH input
        plot_time_series(np.arange(len(ph)) / FS, ph, f"{FIG_DIR}/ph_{psi_labels[idx]}")

        # FRF PH→NKD (x=PH, y=NKD)
        f, H, gamma2 = estimate_frf(nkd, ph, FS)
        np.save(f"{CAL_DIR}/H_{psi_labels[idx]}.npy", H)
        np.save(f"{CAL_DIR}/gamma2_{psi_labels[idx]}.npy", gamma2)
        np.save(f"{CAL_DIR}/f_{psi_labels[idx]}.npy", f)
        ic(f.shape, H.shape, gamma2.shape)
        plot_transfer_PH(f, H, f"{FIG_DIR}/H_{psi_labels[idx]}", psi_labels[idx])

        # Sanity: reconstruct PH from NKD using the inverse (should resemble PH)
        ph_hat = wiener_inverse(nkd, FS, f, H, gamma2)
        t = np.arange(len(ph_hat)) / FS
        plot_corrected_trace_PH(t, ph, nkd, ph_hat, f"{FIG_DIR}/ph_recon_{psi_labels[idx]}", psi_labels[idx])


############################
# Calibration: NKD→NC
############################
def main_NC():
    """
    Calibrate NKD→NC transfer function.
    In channelData_300_nose, col1 = NKD (input), col2 = NC (output).
    """
    root = 'data/11092025'
    fn_sweep = [f'{root}/cali.mat' for _ in psi_labels]
    FIG_DIR = "figures/cali_09/NKD-NC"
    CAL_DIR = os.path.join(CALIB_BASE_DIR, "NKD-NC")
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(CAL_DIR, exist_ok=True)

    for idx in range(len(psi_labels)):
        ic(f"Processing {psi_labels[idx]}...")

        # Load data (to Pa): col1 = NKD, col2 = NC
        nkd, nc = load_mat_to_pa(fn_sweep[idx], key='channelData_300_nose', ch1_name='NKD', ch2_name='NC')
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
    NC_path = _resolve_cal_dir("NKD-NC")   # NKD→NC

    RAW_DIR = f"{OUTPUT_DIR}/raw"
    os.makedirs(RAW_DIR, exist_ok=True)

    delta, nu_s = inner_scales(Re_taus, u_taus, nu_atm)
    pressures = np.array([P_ATM] + [P_ATM + float(p[:-3]) * PSI_TO_PA for p in psi_labels[1:]], dtype=float)
    rhos = pressures / (R * T)

    Times = []
    Pyys = []

    for idx in range(len(psi_labels)):
        ic(f"Processing {psi_labels[idx]}...")

        # Load real data (to Pa): channelData_300 -> col1=NC, col2=PH
        nc, ph = load_mat_to_pa(fn_sweep[idx], key='channelData_300', ch1_name='NC', ch2_name='PH')
        t = np.arange(len(ph)) / FS

        # --- Raw PH spectrum (for comparison) in inner-scaled premultiplied form
        f_raw, Pyy_raw = compute_spec(FS, ph)  # Pa^2/Hz
        # x_inv_fplus, y_pm = premultiplied_phi_pp_plus(f_raw, Pyy_raw, rhos[idx], u_taus[idx], nu_s[idx])
        # plot_raw_spectrum(f_raw, Pyy_raw, f"{OUTPUT_DIR}/Pphph_{psi_labels[idx]}_raw")


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

        # --- Spectrum of corrected PH (NKD domain), inner-scaled premultiplied
        f_corr, Pyy_corr = compute_spec(FS, nkd_from_ph)
        # x_inv_fplus, y_pm = premultiplied_phi_pp_plus(f_corr, Pyy_corr, rhos[idx], u_taus[idx], nu_s[idx])
        # Times.append(x_inv_fplus)
        # Pyys.append(y_pm)
        # plot_raw_spectrum(f_corr, f_corr*Pyy_corr, f"{OUTPUT_DIR}/Pyy_{psi_labels[idx]}_corr")

        # --- Coherent rejection: remove component of nkd_from_ph coherent with nkd_from_nc
        # FRF x→y with x=nkd_from_nc, y=nkd_from_ph (both NKD domain)
        f_xy, H_xy, gamma2_xy = estimate_frf(nkd_from_nc, nkd_from_ph, FS)
        y_hat = apply_frf(nkd_from_nc, FS, f_xy, H_xy)   # predict coherent part of PH stream
        y_resid = nkd_from_ph - y_hat                    # coherent rejection residual

        f_rej, Pyy_rej = compute_spec(FS, y_resid)
        # x_inv_fplus, y_pm_rej = premultiplied_phi_pp_plus(f_rej, Pyy_rej, rhos[idx], u_taus[idx], nu_s[idx])
        plot_raw_spectrum(f_rej, f_rej*Pyy_rej, f"{OUTPUT_DIR}/Pyy_{psi_labels[idx]}_rej")
        plot_spectrum_pipeline([f_raw, f_corr, f_rej], [f_raw*Pyy_raw, f_corr*Pyy_corr, 
                                                        f_rej*Pyy_rej], f"{OUTPUT_DIR}/Pyy_comparison_{psi_labels[idx]}")


if __name__ == "__main__":
    # Run calibrations if needed to (re)generate FRFs:
    main_PH()
    main_NC()

    # Apply to the real flow data:
    real_data()
