import numpy as np
from scipy.signal import welch, csd, get_window, iirnotch, sosfiltfilt, find_peaks
import scipy.signal as signal
import scipy.io as sio
from matplotlib import pyplot as plt
from icecream import ic
import os

from tqdm import tqdm

from wiener_filter_gib import wiener_cancel_background
from wiener_filter_torch import wiener_cancel_background_torch
from stft_wiener import wiener_cancel_background_stft_torch

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
#Plan
#    \item Raw TS
#    \item Raw spectra
#    \item PH calibration
#    \item TF corrected TS
#    \item TF corrected spectra
#    \item Notched electrical facility noise
#    \item Notched FS and PH spectra
#    \item Coherence correction
#    \item Cleaned spectra
############################

############################
# Constants & defaults
############################
FS = 50_000.0
NPERSEG = 2**14
WINDOW = "hann"
TRIM_CAL_SECS = 5  # seconds trimmed from the start of calibration runs (0 to disable)

R = 287.0         # J/kg/K
T = 298.0         # K (adjust if you have per-case temps)
P_ATM = 101_325.0 # Pa
PSI_TO_PA = 6_894.76

# Labels for operating conditions; first entry is assumed 'atm'
psi_labels = ['atm']
Re_tau = 5000
Re_tau = 1500
delta = 0.035
nu_utau = 1/(Re_tau / delta)
u_tau = 0.51 
# u_tau = 
nu = 1/(u_tau/nu_utau)
# at T+=20, what is f [Hz]? T^+\equiv T u_\tau^2/\nu

nc_colour = '#1f77b4'  # matplotlib default blue
ph1_colour = "#c76713"  # matplotlib default orange
ph2_colour = "#9fda16"  # matplotlib default red
nkd_colour = '#2ca02c' # matplotlib default green

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
    'channelData_300_nose': ('NC', 'NKD'),
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
    detrend: str = "false",
    npsg: int = NPERSEG,
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
    nseg = int(min(npsg, x.size, y.size))
    if nseg < 8:
        raise ValueError(f"Signal too short for FRF: n={min(x.size, y.size)}")
    nov = int(min(npsg // 2, nseg // 2))
    w = get_window(window, nseg, fftbins=True)

    f, Sxx = welch(x, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)
    _, Syy = welch(y, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)
    # SciPy convention: csd(x, y) = E{ X * conj(Y) }
    _, Sxy = csd(x, y, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)  # x→y

    H = np.conj(Sxy) / Sxx               # H1 = Syx / Sxx = conj(Sxy)/Sxx
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


def compute_spec(fs: float, x: np.ndarray, npsg : int = NPERSEG):
    """Welch PSD with sane defaults and shape guarding. Returns (f [Hz], Pxx [Pa^2/Hz])."""
    x = np.asarray(x, float)
    nseg = int(min(npsg, x.size))
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

def wiener_forward(x, fs, f, H, gamma2, nfft_pow=0, demean=True, zero_dc=True, taper_hz=0.0):
    """
    Forward FRF application: given x (PH) and H_{PH->NKD}, synthesize ŷ ≈ NKD.
    Uses coherence-weighted magnitude (sqrt(gamma2)) and (optionally) a gentle
    in-band edge taper over `taper_hz` near the measured band edges.

    Parameters
    ----------
    x : array_like
        Input time series (real).
    fs : float
        Sampling rate [Hz].
    f : 1D array
        Frequencies of H and gamma2 [Hz], strictly increasing, typically within [0, fs/2].
    H : 1D complex array
        FRF samples at frequencies `f`.
    gamma2 : 1D float array
        Magnitude-squared coherence at `f` in [0, 1].
    nfft_pow : int, default 0
        If 0: use next power-of-two >= len(x).
        If >0: use 2**nfft_pow, but never smaller than len(x).
    demean : bool, default True
        Subtract mean from x before processing.
    zero_dc : bool, default True
        Force the DC bin to zero after weighting/multiplication.
    taper_hz : float, default 0.0
        Width (in Hz) of a half-cosine taper *inside* the measured band edges.
        Set 0 to disable.
    """
    import numpy as np

    x = np.asarray(x, float)
    f = np.asarray(f, float)
    H = np.asarray(H)
    gamma2 = np.asarray(gamma2, float)

    # Basic checks (lightweight; raise early on obvious issues)
    if x.ndim != 1:
        raise ValueError("x must be 1-D")
    if f.ndim != 1 or H.ndim != 1 or gamma2.ndim != 1:
        raise ValueError("f, H, gamma2 must be 1-D")
    if not (len(f) == len(H) == len(gamma2)):
        raise ValueError("f, H, gamma2 must have the same length")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(H)) or np.any(~np.isfinite(gamma2)):
        raise ValueError("Inputs contain NaN/Inf")
    if np.any(np.diff(f) <= 0):
        raise ValueError("f must be strictly increasing")
    if f[0] < 0 or f[-1] > fs/2 + 1e-9:
        raise ValueError("f must lie within [0, fs/2]")

    if demean:
        x = x - x.mean()

    N = x.size
    # --- FIX 1: ensure Nfft >= N even if nfft_pow is given ---
    min_pow = int(np.ceil(np.log2(max(1, N))))
    if nfft_pow and nfft_pow > 0:
        Nfft = 2**max(nfft_pow, min_pow)
    else:
        Nfft = 2**min_pow

    X = np.fft.rfft(x, n=Nfft)
    fr = np.fft.rfftfreq(Nfft, d=1.0/fs)

    # Prefer complex-part interpolation to avoid manual unwrap
    Hr = np.interp(fr, f, np.real(H), left=0.0, right=0.0)
    Hi = np.interp(fr, f, np.imag(H), left=0.0, right=0.0)
    H_i = Hr + 1j*Hi

    # Coherence weighting (shrink toward zero where unreliable)
    g2_i = np.clip(np.interp(fr, f, gamma2, left=0.0, right=0.0), 0.0, 1.0)
    W = np.sqrt(g2_i)

    # Optional gentle in-band taper near measured band edges
    if taper_hz and taper_hz > 0.0:
        band_lo = f[0]
        band_hi = f[-1]
        # Apply half-cosine ramps INSIDE the measured band
        lo_edge = np.where((fr >= band_lo) & (fr < band_lo + taper_hz))[0]
        hi_edge = np.where((fr <= band_hi) & (fr > band_hi - taper_hz))[0]
        if lo_edge.size > 0:
            t = (fr[lo_edge] - band_lo) / taper_hz   # 0..1
            W[lo_edge] *= 0.5 * (1 - np.cos(np.pi * t))
        if hi_edge.size > 0:
            t = (band_hi - fr[hi_edge]) / taper_hz   # 0..1
            W[hi_edge] *= 0.5 * (1 - np.cos(np.pi * t))
        # Outside measured band W is already 0 via interpolation left/right=0.

    Y = W * H_i * X

    if zero_dc and Y.size > 0:
        Y[0] = 0.0

    y_hat = np.fft.irfft(Y, n=Nfft)[:N]  # guaranteed Nfft >= N
    return y_hat




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
    mag_i = np.interp(fr, f, mag, left=1.0, right=1.0)
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
    mag_i = np.interp(fr, f, mag, left=1.0, right=1.0)
    phi_i = np.interp(fr, f, phi, left=phi[0], right=phi[-1])
    Hi = mag_i * np.exp(1j * phi_i)

    if zero_dc:
        Hi[0] = 0.0
        if Nfft % 2 == 0:
            Hi[-1] = 0.0

    y = np.fft.irfft(X * Hi, n=Nfft)[:N]
    return y

def apply_frf_ph_corr(
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


def design_notches(fs, freqs, Q=30.0):
    """
    Make a cascade of IIR notch filters (as SOS).
    
    Parameters
    ----------
    fs : float
        Sampling rate [Hz].
    freqs : list of float
        Centre frequencies to notch [Hz].
    Q : float
        Quality factor (higher = narrower notch).
    """
    sos_list = []
    for f0 in freqs:
        w0 = f0 / (fs/2.0)   # normalised frequency (Nyquist=1)
        b, a = iirnotch(w0, Q)
        sos_list.append(np.hstack([b, a]))
    if not sos_list:
        return None
    return np.vstack(sos_list)

def apply_notches(x, sos):
    """Apply zero-phase notch filtering to signal x."""
    if sos is None:
        return x
    return sosfiltfilt(sos, x)



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
    return float(np.trapezoid(Pxx[mask], f[mask]))

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

def plot_white(ax):
    x = np.logspace(1, 4, 200)   # 10^1 to 10^4
    y = 1e-4 * (x / 1e1)         # slope +1 line, scaled to pass through (1e1, 1e-8)
    ax.loglog(x, y, '--', color='gray', label='White noise (slope +1)')


def calibration_700_atm():
    u_tau = 0.58
    nu_utau = 27e-6
    nu = nu_utau * u_tau

    f_cut = 2_100
    T_plus_fcut = 1/f_cut * (u_tau**2)/nu
    ic(T_plus_fcut)

    root = 'data/20251014/flow_data/far'
    fn_far = f'{root}/atm.mat'
    root = 'data/20251016/flow_data/close'
    fn_close = f'{root}/atm.mat'
    OUTPUT_DIR = "figures/raw_spectra"
    CAL_DIR = os.path.join('data/20251014/tf_calib', "tf_data")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CAL_DIR, exist_ok=True)


    dat = sio.loadmat(fn_far) # options are channelData_LP, channelData_NF
    ic(dat.keys())
    nkd_far = dat['channelData_flow'][:,2]
    ph1_far = dat['channelData_flow'][:,0]
    ph2_far = dat['channelData_flow'][:,1]
    f_far, Pyy_nkd_far = compute_spec(FS, nkd_far)
    f_far, Pyy_ph1_far = compute_spec(FS, ph1_far)
    f_far, Pyy_ph2_far = compute_spec(FS, ph2_far)

    fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, tight_layout=True)
    T_plus = 1/f_far * (u_tau**2)/nu

    fig.suptitle(r"$Re_\tau\approx$ 1,300 (700$\mu$m) - Far-spaced")

    ax[0].semilogx(f_far, f_far * Pyy_nkd_far, label='NC', color=nkd_colour)
    ax[0].semilogx(f_far, f_far * Pyy_ph1_far, label='PH1', color=ph1_colour)
    ax[0].semilogx(f_far, f_far * Pyy_ph2_far, label='PH2', color=ph2_colour)
    # ax.axvline(T_plus_fcut, color='gray', linestyle='--', label=r'$f_{\mathrm{LP}}'+r'={:.0f}$ [Hz]'.format(f_cut))
    #
    ax[0].set_xlabel("$f$ [Hz]")
    ax[1].semilogx(T_plus, f_far * Pyy_nkd_far, label='NC', color=nkd_colour)
    ax[1].semilogx(T_plus, f_far * Pyy_ph1_far, label='PH1', color=ph1_colour)
    ax[1].semilogx(T_plus, f_far * Pyy_ph2_far, label='PH2', color=ph2_colour)
    ax[1].set_xlabel("$T^+$")
    ax[0].set_ylabel(r"$f \phi_{pp}$")
    # ax[1].set_ylabel(r"$f \phi_{pp}$")

    ax[0].set_ylim(0, 2e-3)
    ax[1].set_ylim(0, 2e-3)
    # ax.set_xlim(1e0, 1e4)

    ax[0].legend()
    ax[1].legend()
    fig.savefig(f"{OUTPUT_DIR}/700_atm_raw_spec_far.png", dpi=410)

    dat = sio.loadmat(fn_close) # options are channelData_LP, channelData_NF
    ic(dat.keys())
    nkd_close = dat['channelData_flow'][:,2]
    ph1_close = dat['channelData_flow'][:,0]
    ph2_close = dat['channelData_flow'][:,1]
    f_close, Pyy_nkd_close = compute_spec(FS, nkd_close)
    f_close, Pyy_ph1_close = compute_spec(FS, ph1_close)
    f_close, Pyy_ph2_close = compute_spec(FS, ph2_close)

    # plot the raw spectra as f and T^+
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, tight_layout=True)
    T_plus = 1/f_close * (u_tau**2)/nu

    fig.suptitle(r"$Re_\tau\approx$ 1,300 (700$\mu$m) - Close-spaced")

    ax[0].semilogx(f_close, f_close * Pyy_nkd_close, label='NC', color=nkd_colour)
    ax[0].semilogx(f_close, f_close * Pyy_ph1_close, label='PH1', color=ph1_colour)
    ax[0].semilogx(f_close, f_close * Pyy_ph2_close, label='PH2', color=ph2_colour)
    # ax.axvline(T_plus_fcut, color='gray', linestyle='--', label=r'$f_{\mathrm{LP}}'+r'={:.0f}$ [Hz]'.format(f_cut))
    #
    ax[0].set_xlabel("$f$ [Hz]")
    ax[1].semilogx(T_plus, f_close * Pyy_nkd_close, label='NC', color=nkd_colour)
    ax[1].semilogx(T_plus, f_close * Pyy_ph1_close, label='PH1', color=ph1_colour)
    ax[1].semilogx(T_plus, f_close * Pyy_ph2_close, label='PH2', color=ph2_colour)
    ax[1].set_xlabel("$T^+$")
    ax[0].set_ylabel(r"$f \phi_{pp}$")
    # ax[1].set_ylabel(r"$f \phi_{pp}$")

    ax[0].set_ylim(0, 2e-3)
    ax[1].set_ylim(0, 2e-3)
    # ax.set_xlim(1e0, 1e4)

    ax[0].legend()
    ax[1].legend()
    fig.savefig(f"{OUTPUT_DIR}/700_atm_raw_spec_close.png", dpi=410)

    # # apply tf
    # TF_CORRECTED_OUT = 'figures/tf_corrected_spectra' 
    # f1 = np.load(f"{CAL_DIR}/f1_700_atm.npy")
    # H1 = np.load(f"{CAL_DIR}/H1_700_atm.npy")
    # gamma1 = np.load(f"{CAL_DIR}/gamma1_700_atm.npy")
    # ph1_nkd = apply_frf(ph1, FS, f1, H1)

    # f2 = np.load(f"{CAL_DIR}/f2_700_atm.npy")
    # H2 = np.load(f"{CAL_DIR}/H2_700_atm.npy")
    # gamma2 = np.load(f"{CAL_DIR}/gamma2_700_atm.npy")
    # ph2_nkd = apply_frf(ph2, FS, f2, H2)

    # # Compute the spectra of the TF-corrected signals
    # f, Pyy_ph1_nkd = compute_spec(FS, ph1_nkd)
    # f, Pyy_ph2_nkd = compute_spec(FS, ph2_nkd)

    # # plot the TF-corrected spectra as T^+
    # fig, ax = plt.subplots(1, 1, figsize=(5, 2.), sharex=True)
    # T_plus = 1/f * (u_tau**2)/nu
    # ax.semilogx(T_plus, f * Pyy_nkd, label='NC', color=nkd_colour)
    # ax.semilogx(T_plus, f * Pyy_ph1_nkd, label='PH1 (TF-corrected)', color=ph1_colour)
    # ax.semilogx(T_plus, f * Pyy_ph2_nkd, label='PH2 (TF-corrected)', color=ph2_colour)

    # ax.set_xlabel("$T^+$")
    # ax.set_ylabel(r"$f \phi_{pp}$")

    # ax.set_ylim(0, 1e-3)
    # ax.legend()
    # fig.savefig(f"{TF_CORRECTED_OUT}/700_atm_tf_spec.png", dpi=410)

def calibration_700_50psi():
    u_tau = 0.47
    nu_utau = 7.5e-6
    nu = nu_utau * u_tau
    f_cut = 4_700
    T_plus_fcut = 1/f_cut * (u_tau**2)/nu

    root = 'data/20251014/flow_data/far'
    fn_far = f'{root}/50psi.mat'
    root = 'data/20251016/flow_data/close'
    fn_close = f'{root}/50psi.mat'
    OUTPUT_DIR = "figures/raw_spectra"
    CAL_DIR = os.path.join('data/20251014/tf_calib', "tf_data")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CAL_DIR, exist_ok=True)


    dat = sio.loadmat(fn_far) # options are channelData_LP, channelData_NF
    ic(dat.keys())
    nkd_far = dat['channelData_flow'][:,2]
    ph1_far = dat['channelData_flow'][:,0]
    ph2_far = dat['channelData_flow'][:,1]
    f_far, Pyy_nkd_far = compute_spec(FS, nkd_far)
    f_far, Pyy_ph1_far = compute_spec(FS, ph1_far)
    f_far, Pyy_ph2_far = compute_spec(FS, ph2_far)

    fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, tight_layout=True)
    T_plus = 1/f_far * (u_tau**2)/nu

    fig.suptitle(r"$Re_\tau\approx$ 4,700 (700$\mu$m) - Far-spaced")

    ax[0].semilogx(f_far, f_far * Pyy_nkd_far, label='NC', color=nkd_colour)
    ax[0].semilogx(f_far, f_far * Pyy_ph1_far, label='PH1', color=ph1_colour)
    ax[0].semilogx(f_far, f_far * Pyy_ph2_far, label='PH2', color=ph2_colour)
    # ax[0].axvline(f_cut, color='red', linestyle='--', label=r'$f_{\mathrm{LP}}'+r'={:.0f}$ [Hz]'.format(f_cut))
    # ax.axvline(T_plus_fcut, color='gray', linestyle='--', label=r'$f_{\mathrm{LP}}'+r'={:.0f}$ [Hz]'.format(f_cut))
    #
    ax[0].set_xlabel("$f$ [Hz]")
    ax[1].semilogx(T_plus, f_far * Pyy_nkd_far, label='NC', color=nkd_colour)
    ax[1].semilogx(T_plus, f_far * Pyy_ph1_far, label='PH1', color=ph1_colour)
    ax[1].semilogx(T_plus, f_far * Pyy_ph2_far, label='PH2', color=ph2_colour)

    ax[1].set_xlabel("$T^+$")
    ax[0].set_ylabel(r"$f \phi_{pp}$")
    # ax[1].set_ylabel(r"$f \phi_{pp}$")

    ax[0].set_ylim(0, 1e-2)
    ax[1].set_ylim(0, 1e-2)
    ax[0].set_xlim(1e0, 1e4)
    ax[1].set_xlim(1e0, 1e4)

    ax[0].legend()
    ax[1].legend()
    fig.savefig(f"{OUTPUT_DIR}/700_50psi_raw_spec_far.png", dpi=410)

    dat = sio.loadmat(fn_close) # options are channelData_LP, channelData_NF
    ic(dat.keys())
    nkd_close = dat['channelData_flow'][:,2]
    ph1_close = dat['channelData_flow'][:,0]
    ph2_close = dat['channelData_flow'][:,1]
    f_close, Pyy_nkd_close = compute_spec(FS, nkd_close)
    f_close, Pyy_ph1_close = compute_spec(FS, ph1_close)
    f_close, Pyy_ph2_close = compute_spec(FS, ph2_close)

    # plot the raw spectra as f and T^+
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, tight_layout=True)
    T_plus = 1/f_close * (u_tau**2)/nu

    fig.suptitle(r"$Re_\tau\approx$ 4,700 (700$\mu$m) - Close-spaced")

    ax[0].semilogx(f_close, f_close * Pyy_nkd_close, label='NC', color=nkd_colour)
    ax[0].semilogx(f_close, f_close * Pyy_ph1_close, label='PH1', color=ph1_colour)
    ax[0].semilogx(f_close, f_close * Pyy_ph2_close, label='PH2', color=ph2_colour)
    # ax.axvline(T_plus_fcut, color='gray', linestyle='--', label=r'$f_{\mathrm{LP}}'+r'={:.0f}$ [Hz]'.format(f_cut))
    #
    ax[0].set_xlabel("$f$ [Hz]")
    ax[1].semilogx(T_plus, f_close * Pyy_nkd_close, label='NC', color=nkd_colour)
    ax[1].semilogx(T_plus, f_close * Pyy_ph1_close, label='PH1', color=ph1_colour)
    ax[1].semilogx(T_plus, f_close * Pyy_ph2_close, label='PH2', color=ph2_colour)
    ax[1].set_xlabel("$T^+$")
    ax[0].set_ylabel(r"$f \phi_{pp}$")
    # ax[1].set_ylabel(r"$f \phi_{pp}$")

    ax[0].set_ylim(0, 1e-2)
    ax[1].set_ylim(0, 1e-2)
    ax[0].set_xlim(1e0, 1e4)
    ax[1].set_xlim(1e0, 1e4)

    ax[0].legend()
    ax[1].legend()
    fig.savefig(f"{OUTPUT_DIR}/700_50psi_raw_spec_close.png", dpi=410)

def calibration_700_100psi(plot=[0,1]):
    u_tau = 0.52
    nu_utau = 3.7e-6
    nu = nu_utau * u_tau
    f_cut = 14_100
    T_plus_fcut = 1/f_cut * (u_tau**2)/nu
    
    OUTPUT_DIR = "figures/raw_spectra"
    TF_CORRECTED_OUT = 'figures/tf_corrected_spectra'
    os.makedirs(TF_CORRECTED_OUT, exist_ok=True)
    root = 'data/20251014/tf_calib'
    CAL_DIR_FAR = os.path.join(root, "tf_data")
    os.makedirs(CAL_DIR_FAR, exist_ok=True)
    root = 'data/20251016/tf_calib'
    CAL_DIR_CLOSE = os.path.join(root, "tf_data")
    os.makedirs(CAL_DIR_CLOSE, exist_ok=True)
    CAL_DIR_COMB = 'data/20251016/flow_data/tf_combined'

    root = 'data/20251014/flow_data/far'
    fn_far = f'{root}/100psi.mat'
    root = 'data/20251016/flow_data/close'
    fn_close = f'{root}/100psi.mat'

    dat = sio.loadmat(fn_far) # options are channelData_LP, channelData_NF
    ic(dat.keys())
    nkd_far = dat['channelData_flow'][:,2]
    ph1_far = dat['channelData_flow'][:,0]
    ph2_far = dat['channelData_flow'][:,1]
    f_far, Pyy_nkd_far = compute_spec(FS, nkd_far)
    f_far, Pyy_ph1_far = compute_spec(FS, ph1_far)
    f_far, Pyy_ph2_far = compute_spec(FS, ph2_far)

    if 0 in plot:
        fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, tight_layout=True)
        T_plus = 1/f_far * (u_tau**2)/nu

        fig.suptitle(r"$Re_\tau\approx$ 9,500 (700$\mu$m) - Far-spaced")

        ax[0].semilogx(f_far, f_far * Pyy_nkd_far, label='NC', color=nkd_colour)
        ax[0].semilogx(f_far, f_far * Pyy_ph1_far, label='PH1', color=ph1_colour)
        ax[0].semilogx(f_far, f_far * Pyy_ph2_far, label='PH2', color=ph2_colour)
        ax[0].axvline(f_cut, color='red', linestyle='--')
        #
        ax[0].set_xlabel("$f$ [Hz]")

        ax[1].semilogx(T_plus, f_far * Pyy_nkd_far, label='NC', color=nkd_colour)
        ax[1].semilogx(T_plus, f_far * Pyy_ph1_far, label='PH1', color=ph1_colour)
        ax[1].semilogx(T_plus, f_far * Pyy_ph2_far, label='PH2', color=ph2_colour)
        ax[1].axvline(T_plus_fcut, color='red', linestyle='--')

        ax[1].set_xlabel("$T^+$")
        ax[0].set_ylabel(r"$f \phi_{pp}$")
        # ax[1].set_ylabel(r"$f \phi_{pp}$")

        ax[0].set_ylim(0, 1e-2)
        ax[1].set_ylim(0, 1e-2)
        ax[0].set_xlim(1e0, 1e4)
        ax[1].set_xlim(1e0, 1e4)

        ax[0].legend()
        ax[1].legend()
        fig.savefig(f"{TF_CORRECTED_OUT}/700_100psi_raw_spec_far.png", dpi=410)

    dat = sio.loadmat(fn_close) # options are channelData_LP, channelData_NF
    ic(dat.keys())
    nkd_close = dat['channelData_flow'][:,2]
    ph1_close = dat['channelData_flow'][:,0]
    ph2_close = dat['channelData_flow'][:,1]
    f_close, Pyy_nkd_close = compute_spec(FS, nkd_close)
    f_close, Pyy_ph1_close = compute_spec(FS, ph1_close)
    f_close, Pyy_ph2_close = compute_spec(FS, ph2_close)

    if 1 in plot:

        # plot the raw spectra as f and T^+
        fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, tight_layout=True)
        T_plus = 1/f_close * (u_tau**2)/nu

        fig.suptitle(r"$Re_\tau\approx$ 9,500 (700$\mu$m) - Close-spaced")

        ax[0].semilogx(f_close, f_close * Pyy_nkd_close, label='NC', color=nkd_colour)
        ax[0].semilogx(f_close, f_close * Pyy_ph1_close, label='PH1', color=ph1_colour)
        ax[0].semilogx(f_close, f_close * Pyy_ph2_close, label='PH2', color=ph2_colour)
        ax[0].axvline(f_cut, color='red', linestyle='--')

        ax[0].set_xlabel("$f$ [Hz]")
        ax[1].semilogx(T_plus, f_close * Pyy_nkd_close, label='NC', color=nkd_colour)
        ax[1].semilogx(T_plus, f_close * Pyy_ph1_close, label='PH1', color=ph1_colour)
        ax[1].semilogx(T_plus, f_close * Pyy_ph2_close, label='PH2', color=ph2_colour)
        ax[1].axvline(T_plus_fcut, color='red', linestyle='--')
        
        ax[1].set_xlabel("$T^+$")
        ax[0].set_ylabel(r"$f \phi_{pp}$")
        # ax[1].set_ylabel(r"$f \phi_{pp}$")

        ax[0].set_ylim(0, 1e-2)
        ax[1].set_ylim(0, 1e-2)
        ax[0].set_xlim(1e0, 1e4)
        ax[1].set_xlim(1e0, 1e4)

        ax[0].legend()
        ax[1].legend()
        fig.savefig(f"{TF_CORRECTED_OUT}/700_100psi_raw_spec_close.png", dpi=410)

    # Load and apply tf
    f1_raw = np.load(f"{CAL_DIR_CLOSE}/f1_700_100psi.npy")
    H1_raw = np.load(f"{CAL_DIR_CLOSE}/H1_700_100psi.npy")
    f2_raw = np.load(f"{CAL_DIR_CLOSE}/f2_700_100psi.npy")
    H2_raw = np.load(f"{CAL_DIR_CLOSE}/H2_700_100psi.npy")

    ph1_close_nkd_raw = apply_frf(ph1_close, FS, f1_raw, H1_raw)
    ph2_close_nkd_raw = apply_frf(ph2_close, FS, f2_raw, H2_raw)
    # Compute the spectra of the TF-corrected signals
    f, Pyy_ph1_close_nkd_raw = compute_spec(FS, ph1_close_nkd_raw)
    f, Pyy_ph2_close_nkd_raw = compute_spec(FS, ph2_close_nkd_raw)

    if 2 in plot:

        # plot the TF-corrected spectra as T^+
        fig, ax = plt.subplots(1, 2, figsize=(5, 2.), sharex=True)
        T_plus = 1/f * (u_tau**2)/nu

        ax[0].semilogx(f, f * Pyy_nkd_close, label='NC', color=nkd_colour)
        ax[0].semilogx(f, f * Pyy_ph1_close_nkd_raw, label='PH1 (TF-corrected)', color=ph1_colour)
        ax[0].semilogx(f, f * Pyy_ph2_close_nkd_raw, label='PH2 (TF-corrected)', color=ph2_colour)
        ax[0].set_xlabel("$f$ [Hz]")

        ax[1].semilogx(T_plus, f * Pyy_nkd_close, label='NC', color=nkd_colour)
        ax[1].semilogx(T_plus, f * Pyy_ph1_close_nkd_raw, label='PH1 (TF-corrected)', color=ph1_colour)
        ax[1].semilogx(T_plus, f * Pyy_ph2_close_nkd_raw, label='PH2 (TF-corrected)', color=ph2_colour)
        ax[1].set_xlabel("$T^+$")

        ax[0].set_ylabel(r"$f \phi_{pp}$")

        fig.savefig(f"{TF_CORRECTED_OUT}/700_100psi_tf_spec_close.png", dpi=410)

    # load and apply fused anechoic tfs
    f1_fused = np.load(f"{CAL_DIR_COMB}/700_100psi_fused_anechoic_f1.npy")
    H1_fused = np.load(f"{CAL_DIR_COMB}/700_100psi_fused_anechoic_H1.npy")
    f2_fused = np.load(f"{CAL_DIR_COMB}/700_100psi_fused_anechoic_f2.npy")
    H2_fused = np.load(f"{CAL_DIR_COMB}/700_100psi_fused_anechoic_H2.npy")

    ph1_close_nkd_fused = apply_frf(ph1_close, FS, f1_fused, H1_fused)
    ph2_close_nkd_fused = apply_frf(ph2_close, FS, f2_fused, H2_fused)
    # Compute the spectra of the TF-corrected signals
    f, Pyy_ph1_close_nkd_fused = compute_spec(FS, ph1_close_nkd_fused)
    f, Pyy_ph2_close_nkd_fused = compute_spec(FS, ph2_close_nkd_fused)

    if 3 in plot:
        
        # plot the TF-corrected spectra as T^+
        fig, ax = plt.subplots(1, 2, figsize=(5, 2.), sharex=True)

        T_plus = 1/f * (u_tau**2)/nu

        ax[0].semilogx(f, f * Pyy_nkd_close, label='NC', color=nkd_colour)
        ax[0].semilogx(f, f * Pyy_ph1_close_nkd_fused, label='PH1 (Fused Anechoic TF)', color=ph1_colour)
        ax[0].semilogx(f, f * Pyy_ph2_close_nkd_fused, label='PH2 (Fused Anechoic TF)', color=ph2_colour)
        ax[0].set_xlabel("$f$ [Hz]")

        ax[1].semilogx(T_plus, f * Pyy_nkd_close, label='NC', color=nkd_colour)
        ax[1].semilogx(T_plus, f * Pyy_ph1_close_nkd_fused, label='PH1 (Fused Anechoic TF)', color=ph1_colour)
        ax[1].semilogx(T_plus, f * Pyy_ph2_close_nkd_fused, label='PH2 (Fused Anechoic TF)', color=ph2_colour)
        ax[1].set_xlabel("$T^+$")

        ax[0].set_ylabel(r"$f \phi_{pp}$")

        fig.savefig(f"{TF_CORRECTED_OUT}/700_100psi_tf_fused_anechoic_spec_close.png", dpi=410)

    # load and apply fused in-situ tfs
    f1_fused_insitu = np.load(f"{CAL_DIR_COMB}/700_100psi_fused_insitu_f1.npy")
    H1_fused_insitu = np.load(f"{CAL_DIR_COMB}/700_100psi_fused_insitu_H1.npy")
    f2_fused_insitu = np.load(f"{CAL_DIR_COMB}/700_100psi_fused_insitu_f2.npy")
    H2_fused_insitu = np.load(f"{CAL_DIR_COMB}/700_100psi_fused_insitu_H2.npy")

    ph1_close_nkd_fused_insitu = apply_frf(ph1_close, FS, f1_fused_insitu, H1_fused_insitu)
    ph2_close_nkd_fused_insitu = apply_frf(ph2_close, FS, f2_fused_insitu, H2_fused_insitu)
    # Compute the spectra of the TF-corrected signals
    f, Pyy_ph1_close_nkd_fused_insitu = compute_spec(FS, ph1_close_nkd_fused_insitu)
    f, Pyy_ph2_close_nkd_fused_insitu = compute_spec(FS, ph2_close_nkd_fused_insitu)

    if 4 in plot:
        
        # plot the TF-corrected spectra as T^+
        fig, ax = plt.subplots(1, 2, figsize=(5, 2.), sharey=True, sharex=True)

        T_plus = 1/f * (u_tau**2)/nu

        ax[0].semilogx(f, f * Pyy_nkd_close, label='NC', color=nkd_colour)
        ax[0].semilogx(f, f * Pyy_ph1_close_nkd_fused_insitu, label='PH1 (Fused In-situ TF)', color=ph1_colour)
        ax[0].semilogx(f, f * Pyy_ph2_close_nkd_fused_insitu, label='PH2 (Fused In-situ TF)', color=ph2_colour)
        ax[0].set_xlabel("$f$ [Hz]")

        ax[0].set_ylim(0, 1e-2)

        ax[1].semilogx(T_plus, f * Pyy_nkd_close, label='NC', color=nkd_colour)
        ax[1].semilogx(T_plus, f * Pyy_ph1_close_nkd_fused_insitu, label='PH1 (Fused In-situ TF)', color=ph1_colour)
        ax[1].semilogx(T_plus, f * Pyy_ph2_close_nkd_fused_insitu, label='PH2 (Fused In-situ TF)', color=ph2_colour)
        ax[1].set_xlabel("$T^+$")

        ax[0].set_ylabel(r"$f \phi_{pp}$")

        fig.savefig(f"{TF_CORRECTED_OUT}/700_100psi_tf_fused_insitu_spec_close.png", dpi=410)


if __name__ == "__main__":
    d = np.array([2300e-6, 700e-6, 400e-6])
    nu_utau = np.array([27e-6, 7.5e-6, 3.7e-6])
    dplus = d / nu_utau
    ic(dplus)
    # calibration()
    # calibration_700_atm()
    # calibration_700_50psi()
    calibration_700_100psi(plot=[4])
    # ic(np.exp(-1)/1)

    # flow_tests()
