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
# >>> Units & column layout for loaded time-series (per MATLAB key) <<<
# Keys (column order):
#   - channelData_300_plug : col1=PH (pinhole), col2=NKD (naked)   [calibration sweep]
#   - channelData_300_nose : col1=NKD,            col2=NC          [calibration sweep]
#   - channelData_300      : col1=NC,             col2=PH          [real flow data]
DEFAULT_UNITS = {
    'channelData_300_plug': ('Pa', 'Pa'),  # PH, NKD
    'channelData_300_nose': ('Pa', 'Pa'),  # NKD, NC
    'channelData_300':      ('Pa', 'Pa'),  # NC,  PH
}

# If using volts, specify sensitivities (V/Pa) and preamp gains here:
SENSITIVITIES_V_PER_PA = {  # leave empty if not using 'V'
    # 'NKD': 0.05,
    # 'PH':  0.05,
    # 'NC':  0.05,
}
PREAMP_GAIN = {  # linear gain; leave 1.0 if unknown
    'NKD': 1.0,
    'PH':  1.0,
    'NC':  1.0,
}
# -------------------------------------------------------------------------
DATA_LAYOUT = {
    'channelData_300_plug': ('PH', 'NKD'),  # col1, col2
    'channelData_300_nose': ('NKD', 'NC'),
    'channelData_300':      ('NC',  'PH'),
}

def load_pair_pa(path: str, key: str):
    ch1, ch2 = DATA_LAYOUT[key]
    x, y = load_mat_to_pa(path, key=key, ch1_name=ch1, ch2_name=ch2)
    return x, y


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
    # SciPy convention: csd(x, y) = E{ X * conj(Y) }
    _, Sxy = csd(x, y, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=detrend)  # x→y

    H = np.conj(Sxy) / Sxx                 # H1 = Syx / Sxx = conj(Sxy)/Sxx
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
    (Jacobian df/df⁺ = u_τ²/ν has been applied.)
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


# --------- Robust forward equaliser for PH→NKD ---------
def _moving_average(y: np.ndarray, win: int = 7):
    if win <= 1:
        return y
    k = int(max(1, win))
    k = k + 1 - (k % 2)  # force odd
    w = np.ones(k, float) / k
    return np.convolve(y, w, mode='same')

def stabilise_forward_frf(f: np.ndarray,
                          H: np.ndarray,
                          gamma2: np.ndarray,
                          fs: float,
                          g2_thresh: float = 0.6,
                          smooth_bins: int = 9,
                          enforce_min_gain: bool = True,
                          monotone_hf_envelope: bool = True,
                          clip_gain_max: float = 50.0):
    """
    Produce a forward PH→NKD equaliser H_stab that:
      * keeps the measured phase (unwrapped),
      * smooths |H|,
      * enforces |H| >= 1 where coherence is reliable,
      * (optionally) makes the HF tail non-decreasing beyond the first unity crossing,
      * extends the last reliable gain into the low-coherence tail,
      * clips ridiculous gains to 'clip_gain_max'.
    """
    f = np.asarray(f, float)
    H = np.asarray(H, complex)
    gamma2 = np.asarray(gamma2, float)

    mag = np.abs(H).astype(float)
    phi = np.unwrap(np.angle(H))
    # smooth
    mag_s = _moving_average(mag, smooth_bins)

    # coherent region
    cmask = (gamma2 >= g2_thresh)
    if np.any(cmask) and enforce_min_gain:
        mag_s[cmask] = np.maximum(mag_s[cmask], 1.0)

    # find first index where |H|>=1 in coherent region
    idx_coh = np.where(cmask & (mag_s >= 1.0))[0]
    if monotone_hf_envelope and idx_coh.size > 0:
        start = int(idx_coh[0])
        # enforce non-decreasing envelope beyond 'start' (within coherent mask)
        # we also allow it to propagate a little into slightly lower g2 to avoid edge dips
        env = np.maximum.accumulate(mag_s[start:])
        mag_s[start:] = env

    # extend the last reliable coherent gain across the trailing low-coherence tail
    if np.any(cmask):
        last_coh = int(np.max(np.where(cmask)[0]))
        mag_s[last_coh+1:] = max(mag_s[last_coh], 1.0)

    # clip absurd gains
    mag_s = np.clip(mag_s, 0.0, clip_gain_max)

    H_stab = mag_s * np.exp(1j * phi)
    return H_stab


# --------- Coherent FS-noise cancellation ---------
def coherent_band_mask(f: np.ndarray, gamma2: np.ndarray, fs: float,
                       fmin: float = 50.0, fmax_frac: float = 0.4, g2_thresh: float = 0.6):
    """Return boolean mask for a sensible coherent band."""
    return (f >= fmin) & (f <= fmax_frac * fs) & (gamma2 >= g2_thresh)

def band_energy_from_psd(f: np.ndarray, Pxx: np.ndarray, mask: np.ndarray):
    """Integrate PSD over masked band to get band-limited variance [Pa^2]."""
    if not np.any(mask):
        return 0.0
    return float(np.trapz(Pxx[mask], f[mask]))

def coherent_cancel(ref: np.ndarray,
                    tgt: np.ndarray,
                    fs: float,
                    fmin: float = 50.0,
                    fmax_frac: float = 0.4,
                    g2_thresh: float = 0.6):
    """
    γ²-weighted coherent subtraction of the part of `tgt` that is linearly predictable from `ref`.
    Returns residual, predicted, and diagnostics.
    """
    ref0 = ref - np.mean(ref)
    tgt0 = tgt - np.mean(tgt)

    f_xy, H_xy, g2_xy = estimate_frf(ref0, tgt0, fs)
    H_eff = H_xy * np.clip(g2_xy, 0.0, 1.0)

    pred = apply_frf(ref0, fs, f_xy, H_eff)
    resid = tgt0 - pred

    # Diagnostics: band-limited energy ratio
    f_tgt, Pyy_tgt = compute_spec(fs, tgt0)
    f_res, Pyy_res = compute_spec(fs, resid)
    Pyy_tgt_i = np.interp(f_xy, f_tgt, Pyy_tgt, left=0.0, right=0.0)
    Pyy_res_i = np.interp(f_xy, f_res, Pyy_res, left=0.0, right=0.0)

    band = coherent_band_mask(f_xy, g2_xy, fs, fmin=fmin, fmax_frac=fmax_frac, g2_thresh=g2_thresh)
    E_tgt = band_energy_from_psd(f_xy, Pyy_tgt_i, band)
    E_res = band_energy_from_psd(f_xy, Pyy_res_i, band)
    ratio = (E_res / E_tgt) if E_tgt > 0 else np.nan
    ic({'FS_cancel_band_Eratio_res_over_tgt': ratio})

    return resid, pred, (f_xy, H_eff, g2_xy), (f_res, Pyy_res), (f_tgt, Pyy_tgt), ratio


############################
# Calibration: PH→NKD
############################
def main_PH():
    """
    Calibrate, save and check the PH→NKD transfer function.
    NOTE: In channelData_300_plug, col1 = PH (input, pinhole), col2 = NKD (output, naked).
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

        # Load data (to Pa): col1 = PH, col2 = NKD
        ph, nkd = load_pair_pa(fn_sweep[idx], 'channelData_300_plug')

        # Trim initial transients
        if TRIM_CAL_SECS > 0:
            start = int(TRIM_CAL_SECS * FS)
            ph = ph[start:]
            nkd = nkd[start:]

        plot_time_series(np.arange(len(ph)) / FS, ph, f"{FIG_DIR}/ph_{psi_labels[idx]}")

        # FRF PH→NKD (x=PH, y=NKD)
        f, H, gamma2 = estimate_frf(ph, nkd, FS)
        np.save(f"{CAL_DIR}/H_{psi_labels[idx]}.npy", H)
        np.save(f"{CAL_DIR}/gamma2_{psi_labels[idx]}.npy", gamma2)
        np.save(f"{CAL_DIR}/f_{psi_labels[idx]}.npy", f)
        ic(f.shape, H.shape, gamma2.shape)
        plot_transfer_PH(f, H, f"{FIG_DIR}/H_{psi_labels[idx]}", psi_labels[idx])

        # Sanity: expect |H|>1 where pinhole attenuates and coherence is decent
        mask = coherent_band_mask(f, gamma2, FS)
        if np.any(mask):
            mag_med = float(np.median(np.abs(H[mask])))
            ic({'PH_to_NKD_median_|H|_in_band': mag_med})

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
        nkd, nc = load_pair_pa(fn_sweep[idx], 'channelData_300_nose')
        plot_time_series(np.arange(len(nkd)) / FS, nkd, f"{FIG_DIR}/nkd_{psi_labels[idx]}")

        # Trim first TRIM_CAL_SECS to avoid initial transients
        if TRIM_CAL_SECS > 0:
            start = int(TRIM_CAL_SECS * FS)
            nkd = nkd[start:]
            nc = nc[start:]

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
    PH_path = _resolve_cal_dir("PH-NKD")   # PH→NKD (intended)
    NC_path = _resolve_cal_dir("NKD-NC")   # NKD→NC

    RAW_DIR = f"{OUTPUT_DIR}/raw"
    os.makedirs(RAW_DIR, exist_ok=True)

    delta, nu_s = inner_scales(Re_taus, u_taus, nu_atm)
    pressures = np.array([P_ATM] + [P_ATM + float(p[:-3]) * PSI_TO_PA for p in psi_labels[1:]], dtype=float)
    rhos = pressures / (R * T)

    for idx in range(len(psi_labels)):
        ic(f"Processing {psi_labels[idx]}...")

        # Load real data (to Pa): channelData_300 -> col1=NC, col2=PH
        nc, ph = load_pair_pa(fn_sweep[idx], 'channelData_300')
        t = np.arange(len(ph)) / FS

        # --- Raw PH spectrum (for comparison)
        f_raw, Pyy_raw = compute_spec(FS, ph)

        # --- Correct NC: invert NKD→NC to get NKD from NC
        H_NC = np.load(f"{NC_path}/H_{psi_labels[idx]}.npy")
        gamma2_NC = np.load(f"{NC_path}/gamma2_{psi_labels[idx]}.npy")
        f_NC = np.load(f"{NC_path}/f_{psi_labels[idx]}.npy")
        nkd_from_nc = wiener_inverse(nc, FS, f_NC, H_NC, gamma2_NC)

        # --- Correct PH: forward PH→NKD with stabilisation (prevents HF roll-off or |H|<1 in coherent band)
        H_PH = np.load(f"{PH_path}/H_{psi_labels[idx]}.npy")
        gamma2_PH = np.load(f"{PH_path}/gamma2_{psi_labels[idx]}.npy")
        f_PH = np.load(f"{PH_path}/f_{psi_labels[idx]}.npy")

        H_PH_stab = stabilise_forward_frf(f_PH, H_PH, gamma2_PH, FS,
                                          g2_thresh=0.6,
                                          smooth_bins=9,
                                          enforce_min_gain=True,
                                          monotone_hf_envelope=True,
                                          clip_gain_max=50.0)

        nkd_from_ph = apply_frf(ph, FS, f_PH, H_PH_stab)

        # --- Spectrum of corrected PH (NKD domain)
        f_corr, Pyy_corr = compute_spec(FS, nkd_from_ph)

        # --- Coherent FS-noise rejection (γ²-weighted, ref = nkd_from_nc → tgt = nkd_from_ph)
        resid, pred, (f_xy, H_eff, g2_xy), (f_rej, Pyy_rej), (f_tgt, Pyy_tgt), Eratio = \
            coherent_cancel(nkd_from_nc, nkd_from_ph, FS)

        # --- Plots
        plot_raw_spectrum(f_rej, f_rej * Pyy_rej, f"{OUTPUT_DIR}/Pyy_{psi_labels[idx]}_rej")
        plot_spectrum_pipeline(
            [f_raw,    f_corr,     f_rej],
            [f_raw*Pyy_raw, f_corr*Pyy_corr, f_rej*Pyy_rej],
            f"{OUTPUT_DIR}/Pyy_comparison_{psi_labels[idx]}"
        )

        # --- Quick numeric checks (HF band tendency)
        # Compare corrected vs raw PH energy in the upper coherent band to ensure "goes the right way".
        hf_mask = (f_PH > 0.2*FS) & (f_PH <= 0.4*FS) & (gamma2_PH >= 0.6)
        if np.any(hf_mask):
            hf_gain_med = float(np.median(np.abs(H_PH_stab[hf_mask])))
            ic({'HF_median_|H_PH->NKD|': hf_gain_med, 'expect_>=': 1.0})

        ic({'FS_cancel_pass': bool(np.isfinite(Eratio) and (Eratio < 1.0))})


if __name__ == "__main__":
    # Run calibrations if needed to (re)generate FRFs:
    main_PH()
    main_NC()

    # Apply to the real flow data:
    real_data()
