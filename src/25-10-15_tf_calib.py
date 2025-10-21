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
    # plot_transfer_nc,
    plot_transfer_PH,
    plot_transfer_NC,
    # plot_corrected_trace_nc,
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
nc_colour = '#2ca02c' # matplotlib default green

# -------------------------------------------------------------------------
# >>> Units & column layout for loaded time-series (per MATLAB key) <<<
# Keys (column order):
#   - channelData_300_plug : col1=PH (pinhole), col2=nc (naked)   [calibration sweep]
#   - channelData_300_nose : col1=nc,            col2=NC          [calibration sweep]
#   - channelData_300      : col1=NC,             col2=PH          [real flow data]
DEFAULT_UNITS = {
    'channelData_300_plug': ('Pa', 'Pa'),  # PH, nc
    'channelData_300_nose': ('Pa', 'Pa'),  # nc, NC
    'channelData_300':      ('Pa', 'Pa'),  # NC,  PH
}

# If using volts, specify sensitivities (V/Pa) and preamp gains here:
SENSITIVITIES_V_PER_PA = {  # leave empty if not using 'V'
    # 'nc': 0.05,
    # 'PH':  0.05,
    # 'NC':  0.05,
}
PREAMP_GAIN = {  # linear gain; leave 1.0 if unknown
    'nc': 1.0,
    'PH':  1.0,
    'NC':  1.0,
}
# -------------------------------------------------------------------------
DATA_LAYOUT = {
    'channelData_300_plug': ('PH', 'nc'),  # col1, col2
    'channelData_300_nose': ('NC', 'nc'),
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

import numpy as np

def combine_anechoic_calibrations(
    f1, H1, g2_1,
    f2, H2, g2_2,
    *,
    gmin: float = 0.3,
    smooth_oct: float | None = 1/6,
    points_per_oct: int = 48,
    eps: float = 1e-12,
):
    """
    Fuse two anechoic FRF estimates (x->y) into a single broadband anchor.

    Weights are inverse-variance-like: w = g2 / (1 - g2), gated below gmin.
    Complex averaging is done on a common frequency grid, followed by optional
    complex-domain smoothing on a log-frequency grid.

    Parameters
    ----------
    f1, f2 : array_like
        Frequency vectors [Hz].
    H1, H2 : array_like (complex)
        H1 FRFs (x->y) from each anechoic run.
    g2_1, g2_2 : array_like in [0, 1]
        Magnitude-squared coherences from each run.
    gmin : float
        Coherence gate; bins with gamma^2 < gmin get near-zero weight.
    smooth_oct : float or None
        Optional complex smoothing span in octaves (e.g., 1/6). None = no smoothing.
    points_per_oct : int
        Resolution for log-frequency smoothing grid.
    eps : float
        Numerical floor.

    Returns
    -------
    f : ndarray
        Output frequency grid (chosen as the denser input grid).
    H_lab : ndarray (complex)
        Fused anechoic FRF (x->y).
    g2_lab : ndarray
        Coherence-like quality metric (weighted blend of inputs).
    """
    f1 = np.asarray(f1); f2 = np.asarray(f2)
    H1 = np.asarray(H1); H2 = np.asarray(H2)
    g2_1 = np.clip(np.asarray(g2_1), 0.0, 1.0)
    g2_2 = np.clip(np.asarray(g2_2), 0.0, 1.0)

    # Choose the denser frequency grid as the target
    f = f1 if f1.size >= f2.size else f2

    # Interpolate to common grid (linear on real/imag)
    def _interp_complex(f_src, z_src, f_tgt):
        z_src = np.asarray(z_src)
        re = np.interp(f_tgt, f_src, np.real(z_src), left=np.real(z_src[0]), right=np.real(z_src[-1]))
        im = np.interp(f_tgt, f_src, np.imag(z_src), left=np.imag(z_src[0]), right=np.imag(z_src[-1]))
        return re + 1j*im

    H1i = _interp_complex(f1, H1, f)
    H2i = _interp_complex(f2, H2, f)
    g2_1i = np.interp(f, f1, g2_1, left=g2_1[0], right=g2_1[-1])
    g2_2i = np.interp(f, f2, g2_2, left=g2_2[0], right=g2_2[-1])

    # Coherence-gated inverse-variance weights
    def _weights(g2):
        g2c = np.clip(g2, 0.0, 1.0 - 1e-9)
        w = g2c / (1.0 - g2c + eps)
        w = np.where(g2c >= gmin, w, 0.0)
        return w

    w1 = _weights(g2_1i)
    w2 = _weights(g2_2i)
    wsum = w1 + w2 + eps

    # Complex weighted average
    H_lab = (w1*H1i + w2*H2i) / wsum

    # Quality metric (for reference/plots)
    g2_lab = (w1*g2_1i + w2*g2_2i) / wsum
    g2_lab = np.clip(g2_lab, 0.0, 1.0)

    # Optional complex smoothing on a log-frequency grid
    if smooth_oct is not None and smooth_oct > 0:
        H_lab = _complex_smooth_logfreq(f, H_lab, span_oct=smooth_oct, points_per_oct=points_per_oct)

    return f, H_lab, g2_lab


def incorporate_insitu_calibration(
    f_lab, H_lab, g2_lab,
    f_ins, H_ins, g2_ins,
    *,
    gmin: float = 0.3,
    gmax: float = 0.8,
    ratio_smooth_oct: float = 1/6,
    post_smooth_oct: float | None = 1/6,
    points_per_oct: int = 48,
    eps: float = 1e-12,
):
    """
    Correct the fused anechoic FRF with in-situ data where the in-situ coherence is good.
    Uses a *complex ratio* R = H_ins / H_lab, smoothed on a log-frequency grid,
    and a confidence map C(f) derived from the in-situ coherence to blend magnitudes
    (in log-space) and phases (via unwrapped phase difference).

    Parameters
    ----------
    f_lab, H_lab, g2_lab : arrays
        Fused anechoic frequency grid, FRF, and coherence-like metric.
    f_ins, H_ins, g2_ins : arrays
        In-situ frequency grid, FRF, and coherence.
    gmin, gmax : floats
        Coherence-to-confidence mapping: gamma^2 <= gmin -> C=0, gamma^2 >= gmax -> C=1.
    ratio_smooth_oct : float
        Complex smoothing span (octaves) applied to the ratio R before blending.
    post_smooth_oct : float or None
        Optional final complex smoothing on H_hat.
    points_per_oct : int
        Resolution for log-frequency smoothing grid.
    eps : float
        Numerical floor.

    Returns
    -------
    f : ndarray
        Frequency grid (same as f_lab).
    H_hat : ndarray (complex)
        FRF after incorporating in-situ behavior at low frequencies.
    C : ndarray in [0,1]
        Confidence map derived from in-situ coherence.
    """
    f_lab = np.asarray(f_lab); H_lab = np.asarray(H_lab); g2_lab = np.asarray(g2_lab)
    f_ins = np.asarray(f_ins); H_ins = np.asarray(H_ins); g2_ins = np.clip(np.asarray(g2_ins), 0.0, 1.0)

    # Interpolate in-situ data to the lab grid
    def _interp_complex(f_src, z_src, f_tgt):
        z_src = np.asarray(z_src)
        re = np.interp(f_tgt, f_src, np.real(z_src), left=np.real(z_src[0]), right=np.real(z_src[-1]))
        im = np.interp(f_tgt, f_src, np.imag(z_src), left=np.imag(z_src[0]), right=np.imag(z_src[-1]))
        return re + 1j*im

    H_ins_i = _interp_complex(f_ins, H_ins, f_lab)
    g2_ins_i = np.interp(f_lab, f_ins, g2_ins, left=g2_ins[0], right=g2_ins[-1])

    # Complex ratio and smoothing (avoid log/phase wrap by smoothing R directly)
    R = H_ins_i / (H_lab + eps)
    R = _complex_smooth_logfreq(f_lab, R, span_oct=ratio_smooth_oct, points_per_oct=points_per_oct)

    # Confidence from in-situ coherence
    C = (g2_ins_i - gmin) / (gmax - gmin + eps)
    C = np.clip(C, 0.0, 1.0)

    # Blend magnitude in log-domain; blend phase via unwrapped phase difference
    mag_lab = np.abs(H_lab) + eps
    mag_ins = np.abs(H_lab * R) + eps  # smoothed in-situ-consistent magnitude
    log_mag_hat = (1.0 - C) * np.log(mag_lab) + C * np.log(mag_ins)
    mag_hat = np.exp(log_mag_hat)

    # Phase: unwrap the phase difference from the ratio
    dphi = np.unwrap(np.angle(R))
    phi_lab = np.unwrap(np.angle(H_lab))
    phi_hat = phi_lab + C * dphi

    H_hat = mag_hat * np.exp(1j * phi_hat)

    # Optional final complex smoothing
    if post_smooth_oct is not None and post_smooth_oct > 0:
        H_hat = _complex_smooth_logfreq(f_lab, H_hat, span_oct=post_smooth_oct, points_per_oct=points_per_oct)

    return f_lab, H_hat, C


# ---------- small internal helper ----------

def _complex_smooth_logfreq(
    f, z, *,
    span_oct: float = 1/6,
    points_per_oct: int = 48,
    eps: float = 1e-20
):
    """
    Complex moving-average smoothing with a *constant span in octaves*.
    Implemented by resampling to a log-frequency grid, smoothing there,
    then resampling back. Operates on real & imag parts (not magnitude-only).

    Notes
    -----
    - DC (f == 0) is passed through without smoothing.
    - Choose points_per_oct so that span_oct * points_per_oct >= ~5 for a stable window.
    """
    f = np.asarray(f); z = np.asarray(z)
    assert f.ndim == 1 and z.ndim == 1 and f.size == z.size
    # Pass through if span is tiny or there are too few positive bins
    pos = f > 0
    if span_oct <= 0 or pos.sum() < 8:
        return z.copy()

    fpos = f[pos]
    zpos = z[pos]
    # Build log-frequency grid
    lo, hi = fpos[0], fpos[-1]
    n_oct = np.log2(hi / max(lo, eps))
    n_pts = max(int(np.ceil(n_oct * points_per_oct)), 8)
    flog = np.linspace(np.log2(max(lo, eps)), np.log2(hi), n_pts)
    fgrid = np.power(2.0, flog)

    # Interp to log grid
    def _interp_complex(f_src, z_src, f_tgt):
        re = np.interp(f_tgt, f_src, np.real(z_src))
        im = np.interp(f_tgt, f_src, np.imag(z_src))
        return re + 1j*im

    zlog = _interp_complex(fpos, zpos, fgrid)

    # Moving average window length in samples on log grid
    wlen = max(int(round(span_oct * points_per_oct)), 1)
    # Force odd length to center the window
    wlen = wlen + (wlen % 2 == 0)
    half = wlen // 2

    # Convolve (boxcar) on real & imag separately
    box = np.ones(wlen) / wlen
    re_s = np.convolve(np.real(zlog), box, mode='same')
    im_s = np.convolve(np.imag(zlog), box, mode='same')
    zlog_s = re_s + 1j*im_s

    # Resample back to original positive f, then reinsert DC if present
    z_s_pos = _interp_complex(fgrid, zlog_s, fpos)
    z_s = z.copy()
    z_s[pos] = z_s_pos
    # Leave DC untouched
    return z_s


def wiener_forward(x, fs, f, H, gamma2, nfft_pow=0, demean=True, zero_dc=True, taper_hz=0.0):
    """
    Forward FRF application: given x (PH) and H_{PH->nc}, synthesize ŷ ≈ nc.
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


# --------- Robust forward equaliser for PH→nc ---------
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
    Produce a forward PH→nc equaliser H_stab that:
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
    root = 'data/20251014/tf_calib'
    fn1 = f'{root}/0psi_lp_16khz_ph1.mat'
    fn2 = f'{root}/0psi_lp_16khz_ph2.mat'
    OUTPUT_DIR = "figures/tf_calib"
    CAL_DIR = os.path.join(root, "tf_data")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CAL_DIR, exist_ok=True)

    u_tau = 0.58
    nu_utau = 27e-6
    nu = nu_utau * u_tau
    Re_tau = u_tau *0.035 / nu
    T10 = 0.1 * (u_tau**2)/nu
    T10000 = 1e-4 * (u_tau**2)/nu
    ic(nu, Re_tau, T10, T10000)

    f_cut = 2_100
    T_plus_fcut = 1/f_cut * (u_tau**2)/nu
    ic(T_plus_fcut)


    dat1 = sio.loadmat(fn1) # options are channelData_LP, channelData_is
    dat2 = sio.loadmat(fn2) # options are channelData_LP, channelData_is
    ic(dat1.keys(), dat2.keys())
    nc1 = dat1['channelData_LP'][:,2]
    nc2 = dat2['channelData_LP'][:,2]
    ph1 = dat1['channelData_LP'][:,0]
    ph2 = dat2['channelData_LP'][:,1]
    f, Pyy_nc1 = compute_spec(FS, nc1)
    f, Pyy_nc2 = compute_spec(FS, nc2)
    f, Pyy_ph1 = compute_spec(FS, ph1)
    f, Pyy_ph2 = compute_spec(FS, ph2)

    # plot the raw spectra as T^+
    fig, ax = plt.subplots(1, 1, figsize=(5, 2.), sharex=True)
    T_plus = f #* (u_tau**2)/nu
    ax.loglog(T_plus, f * Pyy_nc1, label='NC$_{R1}$', color=nc_colour, lw=0.5)
    ax.loglog(T_plus, f * Pyy_nc2, label='NC$_{R2}$', color=nc_colour, lw=0.5)
    ax.loglog(T_plus, f * Pyy_ph1, label='PH1', color=ph1_colour, lw=0.5)
    ax.loglog(T_plus, f * Pyy_ph2, label='PH2', color=ph2_colour, lw=0.5)
    ax.axvline(f_cut, color='red', linestyle='--', lw=1)
    ax.set_xlabel("$T^+$")
    ax.set_xlabel("$f$")
    ax.set_ylabel(r"$f \phi_{pp}$")

    # ax.set_ylim(1e-10, 1e-2)
    # ax.set_xlim(1e0, 1e4)

    ax.legend()
    fig.savefig(f"{OUTPUT_DIR}/700_atm_calib_spec_a2.png", dpi=410)

    f1, H1, gamma1 = estimate_frf(ph1, nc1, FS, npsg=2**14)
    np.save(f"{CAL_DIR}/H1_700_atm.npy", H1)
    np.save(f"{CAL_DIR}/gamma1_700_atm.npy", gamma1)
    np.save(f"{CAL_DIR}/f1_700_atm.npy", f)
    f2, H2, gamma2 = estimate_frf(ph2, nc2, FS, npsg=2**14)
    np.save(f"{CAL_DIR}/H2_700_atm.npy", H2)
    np.save(f"{CAL_DIR}/gamma2_700_atm.npy", gamma2)
    np.save(f"{CAL_DIR}/f2_700_atm.npy", f2)
    
    # Now plot the TF with cutoff, annotated
    mag1 = np.abs(H1); phase1 = np.unwrap(np.angle(H1))
    mag2 = np.abs(H2); phase2 = np.unwrap(np.angle(H2))
    fig, (ax_mag, ax_ph) = plt.subplots(2, 1, sharex=True, figsize=(6, 3), dpi=600)
    ax_mag.set_title(r'$H_{\mathrm{PH-NC}}$ ($700\mu m$, atm), with suggested cutoffs')
    ax_mag.loglog(f1, mag1, lw=1, color='k')
    ax_mag.loglog(f2, mag2, lw=1, color='k', ls='--')
    ax_mag.set_ylabel(r'$|H_{\mathrm{PH-NC}}(f)|$')
    ax_mag.set_ylim(0.1, 100)
    ax_mag.axvline(f_cut, color='red', linestyle='--', lw=1)
    ax_mag.text(f_cut, 10, fr'$T^+ \approx {T_plus_fcut:.1f}$', color='red', va='center', ha='right', rotation=90)

    ax_ph.semilogx(f1, phase1, lw=1, color='k')
    ax_ph.semilogx(f2, phase2, lw=1, color='k', ls='--')
    ax_ph.set_ylabel(r'$\angle H_{\mathrm{PH-NC}}(f)\,[\mathrm{rad}]$')
    ax_ph.set_xlabel(r'$f\ \mathrm{[Hz]}$')
    ax_ph.set_ylim(0, 7)
    ax_ph.axvline(f_cut, color='red', linestyle='--', lw=1)

    fig.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/700_atm_H_a2.pdf", dpi=600)
    plt.close()

def calibration_700_50psi():
    root = 'data/20251014/tf_calib'
    fn1 = f'{root}/50psi_lp_16khz_ph1.mat'
    fn2 = f'{root}/50psi_lp_16khz_ph2.mat'
    OUTPUT_DIR = "figures/tf_calib"
    CAL_DIR = os.path.join(root, "tf_data")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CAL_DIR, exist_ok=True)

    u_tau = 0.47
    nu_utau = 7.5e-6
    nu = nu_utau * u_tau
    
    f_cut = 4_700
    T_plus_fcut = 1/f_cut * (u_tau**2)/nu
    ic(T_plus_fcut)


    dat1 = sio.loadmat(fn1) # options are channelData_LP, channelData_is
    dat2 = sio.loadmat(fn2) # options are channelData_LP, channelData_is
    ic(dat1.keys(), dat2.keys())
    nc1 = dat1['channelData_LP'][:,2]
    nc2 = dat2['channelData_LP'][:,2]
    ph1 = dat1['channelData_LP'][:,0]
    ph2 = dat2['channelData_LP'][:,1]
    f, Pyy_nc1 = compute_spec(FS, nc1)
    f, Pyy_nc2 = compute_spec(FS, nc2)
    f, Pyy_ph1 = compute_spec(FS, ph1)
    f, Pyy_ph2 = compute_spec(FS, ph2)

    # plot the raw spectra as T^+
    fig, ax = plt.subplots(1, 1, figsize=(5, 2.), sharex=True)
    T_plus = f #* (u_tau**2)/nu
    ax.loglog(T_plus, f * Pyy_nc1, label='NC$_{R1}$', color=nc_colour, lw=0.5)
    ax.loglog(T_plus, f * Pyy_nc2, label='NC$_{R2}$', color=nc_colour, lw=0.5)
    ax.loglog(T_plus, f * Pyy_ph1, label='PH1', color=ph1_colour, lw=0.5)
    ax.loglog(T_plus, f * Pyy_ph2, label='PH2', color=ph2_colour, lw=0.5)
    ax.axvline(f_cut, color='red', linestyle='--', lw=1)
    ax.set_xlabel("$T^+$")
    ax.set_xlabel("$f$")
    ax.set_ylabel(r"$f \phi_{pp}$")

    ax.set_ylim(1e-10, 1e-2)
    # ax.set_xlim(1e0, 1e4)

    ax.legend()
    fig.savefig(f"{OUTPUT_DIR}/700_50psi_calib_spec_a2.png", dpi=410)

    f1, H1, gamma1 = estimate_frf(ph1, nc1, FS, npsg=2**10)
    np.save(f"{CAL_DIR}/H1_700_50psi.npy", H1)
    np.save(f"{CAL_DIR}/gamma1_700_50psi.npy", gamma1)
    np.save(f"{CAL_DIR}/f1_700_50psi.npy", f1)
    f2, H2, gamma2 = estimate_frf(ph2, nc2, FS, npsg=2**10)
    np.save(f"{CAL_DIR}/H2_700_50psi.npy", H2)
    np.save(f"{CAL_DIR}/gamma2_700_50psi.npy", gamma2)
    np.save(f"{CAL_DIR}/f2_700_50psi.npy", f2)
    
    # Now plot the TF with cutoff, annotated
    mag1 = np.abs(H1); phase1 = np.unwrap(np.angle(H1))
    mag2 = np.abs(H2); phase2 = np.unwrap(np.angle(H2))
    fig, (ax_mag, ax_ph) = plt.subplots(2, 1, sharex=True, figsize=(6, 3), dpi=600)
    ax_mag.set_title(r'$H_{\mathrm{PH-NC}}$ ($700\mu m$, 50psi), with suggested cutoffs')
    ax_mag.loglog(f1, mag1, lw=1, color='k')
    ax_mag.loglog(f2, mag2, lw=1, color='k', ls='--')
    ax_mag.set_ylabel(r'$|H_{\mathrm{PH-NC}}(f)|$')
    ax_mag.set_ylim(0.1, 100)
    ax_mag.axvline(f_cut, color='red', linestyle='--', lw=1)
    ax_mag.text(f_cut, 10, fr'$T^+ \approx {T_plus_fcut:.1f}$', color='red', va='center', ha='right', rotation=90)

    ax_ph.semilogx(f1, phase1, lw=1, color='k')
    ax_ph.semilogx(f2, phase2, lw=1, color='k', ls='--')
    ax_ph.set_ylabel(r'$\angle H_{\mathrm{PH-NC}}(f)\,[\mathrm{rad}]$')
    ax_ph.set_xlabel(r'$f\ \mathrm{[Hz]}$')
    # ax_ph.set_ylim(0, 7)
    ax_ph.axvline(f_cut, color='red', linestyle='--', lw=1)

    fig.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/700_50psi_H_a2.pdf", dpi=600)
    plt.close()

def calibration_700_100psi(plot=[0,1]):
    OUTPUT_DIR = "figures/tf_calib"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    root = 'data/20251014/tf_calib'
    fn1_far = f'{root}/100psi_lp_16khz_ph1.mat'
    fn2_far = f'{root}/100psi_lp_16khz_ph2.mat'
    CAL_DIR_FAR = os.path.join(root, "tf_data")
    os.makedirs(CAL_DIR_FAR, exist_ok=True)
    root = 'data/20251016/tf_calib'
    fn1_close = f'{root}/100psi_lp_16khz_ph1.mat'
    fn2_close = f'{root}/100psi_lp_16khz_ph2.mat'
    CAL_DIR_CLOSE = os.path.join(root, "tf_data")
    os.makedirs(CAL_DIR_CLOSE, exist_ok=True)

    u_tau = 0.52
    nu_utau = 3.7e-6
    nu = nu_utau * u_tau
    Re_tau = u_tau *0.035 / nu
    T10 = 0.1 * (u_tau**2)/nu
    ic(nu, Re_tau, T10)

    f_cut = 14_100
    T_plus_fcut = 1/f_cut * (u_tau**2)/nu
    ic(T_plus_fcut)


    dat1_far = sio.loadmat(fn1_far)
    dat2_far = sio.loadmat(fn2_far)
    ic(dat1_far.keys(), dat2_far.keys())
    nc1_far = dat1_far['channelData_LP'][:,2]
    nc2_far = dat2_far['channelData_LP'][:,2]
    ph1_far = dat1_far['channelData_LP'][:,0]
    ph2_far = dat2_far['channelData_LP'][:,1]

    dat1_close = sio.loadmat(fn1_close)
    dat2_close = sio.loadmat(fn2_close)
    ic(dat1_close.keys(), dat2_close.keys())
    nc1_close = dat1_close['channelData_LP'][:,2]
    nc2_close = dat2_close['channelData_LP'][:,2]
    ph1_close = dat1_close['channelData_LP'][:,0]
    ph2_close = dat2_close['channelData_LP'][:,1]

    f1_far, H1_far, gamma1_far = estimate_frf(ph1_far, nc1_far, FS, npsg=2**10)
    np.save(f"{CAL_DIR_FAR}/H1_700_100psi.npy", H1_far)
    np.save(f"{CAL_DIR_FAR}/gamma1_700_100psi.npy", gamma1_far)
    np.save(f"{CAL_DIR_FAR}/f1_700_100psi.npy", f1_far)
    f2_far, H2_far, gamma2_far = estimate_frf(ph2_far, nc2_far, FS, npsg=2**10)
    np.save(f"{CAL_DIR_FAR}/H2_700_100psi.npy", H2_far)
    np.save(f"{CAL_DIR_FAR}/gamma2_700_100psi.npy", gamma2_far)
    np.save(f"{CAL_DIR_FAR}/f2_700_100psi.npy", f2_far)

    f1_close, H1_close, gamma1_close = estimate_frf(ph1_close, nc1_close, FS, npsg=2**10)
    np.save(f"{CAL_DIR_CLOSE}/H1_700_100psi.npy", H1_close)
    np.save(f"{CAL_DIR_CLOSE}/gamma1_700_100psi.npy", gamma1_close)
    np.save(f"{CAL_DIR_CLOSE}/f1_700_100psi.npy", f1_close)
    f2_close, H2_close, gamma2_close = estimate_frf(ph2_close, nc2_close, FS, npsg=2**10)
    np.save(f"{CAL_DIR_CLOSE}/H2_700_100psi.npy", H2_close)
    np.save(f"{CAL_DIR_CLOSE}/gamma2_700_100psi.npy", gamma2_close)
    np.save(f"{CAL_DIR_CLOSE}/f2_700_100psi.npy", f2_close)


    # Add in the no-flow calibration data
    root = 'data/20251014/flow_data/far'
    fn_far = f'{root}/100psi.mat'
    CAL_DIR_FAR = os.path.join('data/20251014/tf_calib', "tf_data")
    
    dat = sio.loadmat(fn_far)
    ic(dat.keys())
    nc_is_far = dat['channelData_noflow'][:,2]
    ph1_is_far = dat['channelData_noflow'][:,0]
    ph2_is_far = dat['channelData_noflow'][:,1]

    root = 'data/20251016/flow_data/close'
    fn_close = f'{root}/100psi.mat'
    CAL_DIR_CLOSE = os.path.join('data/20251016/tf_calib', "tf_data")

    dat = sio.loadmat(fn_close)
    ic(dat.keys())
    nc_is_close = dat['channelData_noflow'][:,2]
    ph1_is_close = dat['channelData_noflow'][:,0]
    ph2_is_close = dat['channelData_noflow'][:,1]
    
    f1_is_far, H1_is_far, gamma1_is_far = estimate_frf(ph1_is_far, nc_is_far, FS, npsg=2**10)
    np.save(f"{CAL_DIR_FAR}/H1_700_100psi_is.npy", H1_is_far)
    np.save(f"{CAL_DIR_FAR}/gamma1_700_100psi_is.npy", gamma1_is_far)
    np.save(f"{CAL_DIR_FAR}/f1_700_100psi_is.npy", f1_is_far)
    f2_is_far, H2_is_far, gamma2_is_far = estimate_frf(ph2_is_far, nc_is_far, FS, npsg=2**10)
    np.save(f"{CAL_DIR_FAR}/H2_700_100psi_is.npy", H2_is_far)
    np.save(f"{CAL_DIR_FAR}/gamma2_700_100psi_is.npy", gamma2_is_far)
    np.save(f"{CAL_DIR_FAR}/f2_700_100psi_is.npy", f2_is_far)
    f1_is_close, H1_is_close, gamma1_is_close = estimate_frf(ph1_is_close, nc_is_close, FS, npsg=2**10)
    np.save(f"{CAL_DIR_CLOSE}/H1_700_100psi_is.npy", H1_is_close)
    np.save(f"{CAL_DIR_CLOSE}/gamma1_700_100psi_is.npy", gamma1_is_close)
    np.save(f"{CAL_DIR_CLOSE}/f1_700_100psi_is.npy", f1_is_close)
    f2_is_close, H2_is_close, gamma2_is_close = estimate_frf(ph2_is_close, nc_is_close, FS, npsg=2**10)
    np.save(f"{CAL_DIR_CLOSE}/H2_700_100psi_is.npy", H2_is_close)
    np.save(f"{CAL_DIR_CLOSE}/gamma2_700_100psi_is.npy", gamma2_is_close)
    np.save(f"{CAL_DIR_CLOSE}/f2_700_100psi_is.npy", f2_is_close)

    if 0 in plot:
        # Now plot the TF with cutoff, annotated
        mag1_far = np.abs(H1_far); phase1_far = np.unwrap(np.angle(H1_far))
        mag2_far = np.abs(H2_far); phase2_far = np.unwrap(np.angle(H2_far))
        mag1_close = np.abs(H1_close); phase1_close = np.unwrap(np.angle(H1_close))
        mag2_close = np.abs(H2_close); phase2_close = np.unwrap(np.angle(H2_close))

        mag1_is_far = np.abs(H1_is_far); phase1_is_far = np.unwrap(np.angle(H1_is_far))
        mag2_is_far = np.abs(H2_is_far); phase2_is_far = np.unwrap(np.angle(H2_is_far))
        mag1_is_close = np.abs(H1_is_close); phase1_is_close = np.unwrap(np.angle(H1_is_close))
        mag2_is_close = np.abs(H2_is_close); phase2_is_close = np.unwrap(np.angle(H2_is_close))

        # mag1_is = np.abs(H1_is); phase1_is = np.unwrap(np.angle(H1_is))
        # mag2_is = np.abs(H2_is); phase2_is = np.unwrap(np.angle(H2_is))
        fig, (ax_mag, ax_ph) = plt.subplots(2, 1, sharex=True, figsize=(9, 4.5), dpi=600)
        ax_mag.set_title(r'$H_{\mathrm{PH-NC}}$ ($700\mu m$, 100psi), with suggested cutoffs')
        ax_mag.loglog(f1_far, mag1_far, lw=1, color=ph1_colour, label='PH1 far')
        ax_mag.loglog(f2_far, mag2_far, lw=1, color=ph2_colour, label='PH2 far')
        ax_mag.loglog(f1_close, mag1_close, lw=1, color=ph1_colour, ls='--', label='PH1 close')
        ax_mag.loglog(f2_close, mag2_close, lw=1, color=ph2_colour, ls='--', label='PH2 close')
        ax_mag.loglog(f1_is_far, mag1_is_far, lw=1, color='k', ls=':', label='PH1 in-situ far')
        ax_mag.loglog(f2_is_far, mag2_is_far, lw=1, color='k', ls='-.', label='PH2 in-situ far')
        ax_mag.loglog(f1_is_close, mag1_is_close, lw=1, color='k', ls=':', label='PH1 in-situ close')
        ax_mag.loglog(f2_is_close, mag2_is_close, lw=1, color='k', ls='-.', label='PH2 in-situ close')

        ax_mag.legend(fontsize=8, ncol=2)

        ax_mag.set_ylabel(r'$|H_{\mathrm{PH-NC}}(f)|$')
        # ax_mag.set_ylim(0.1, 100)
        ax_mag.axvline(f_cut, color='red', linestyle='--', lw=1)
        ax_mag.text(f_cut, 10, fr'$T^+ \approx {T_plus_fcut:.1f}$', color='red', va='center', ha='right', rotation=90)

        ax_ph.semilogx(f1_far, phase1_far, lw=1, color=ph1_colour)
        ax_ph.semilogx(f2_far, phase2_far, lw=1, color=ph2_colour)
        ax_ph.semilogx(f1_close, phase1_close, lw=1, color=ph1_colour, ls='--')
        ax_ph.semilogx(f2_close, phase2_close, lw=1, color=ph2_colour, ls='--')
        ax_ph.semilogx(f1_is_far, phase1_is_far, lw=1, color='k', ls=':')
        ax_ph.semilogx(f2_is_far, phase2_is_far, lw=1, color='k', ls='-.')
        ax_ph.semilogx(f1_is_close, phase1_is_close, lw=1, color='k', ls=':')
        ax_ph.semilogx(f2_is_close, phase2_is_close, lw=1, color='k', ls='-.')
        ax_ph.set_ylabel(r'$\angle H_{\mathrm{PH-NC}}(f)\,[\mathrm{rad}]$')
        ax_ph.set_xlabel(r'$f\ \mathrm{[Hz]}$')
        # ax_ph.set_ylim(0, 7)
        ax_ph.axvline(f_cut, color='red', linestyle='--', lw=1)

        fig.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/700_100psi_H_2cal.png", dpi=410)
        plt.close()

    # fuse the anechoic transfer functions
    f1_lab, H1_lab, g1_lab = combine_anechoic_calibrations(f1_far, H1_far, gamma1_far, f1_close, H1_close, gamma1_close)
    f2_lab, H2_lab, g2_lab = combine_anechoic_calibrations(f2_far, H2_far, gamma2_far, f2_close, H2_close, gamma2_close)

    if 1 in plot:
        # plot the fused TFs
        mag1 = np.abs(H1_lab); phase1 = np.unwrap(np.angle(H1_lab))
        mag2 = np.abs(H2_lab); phase2 = np.unwrap(np.angle(H2_lab))
        fig, (ax_mag, ax_ph) = plt.subplots(2, 1, sharex=True, figsize=(9, 4.5), dpi=600)
        ax_mag.set_title(r'Fused $H_{\mathrm{PH-NC}}$ ($700\mu m$, 100psi), with suggested cutoffs')
        ax_mag.loglog(f1_lab, mag1, lw=1, color=ph1_colour)
        ax_mag.loglog(f2_lab, mag2, lw=1, color=ph2_colour)
        ax_mag.set_ylabel(r'$|H_{\mathrm{PH-NC}}(f)|$')
        # ax_mag.set_ylim(0.1, 100)
        ax_mag.axvline(f_cut, color='red', linestyle='--', lw=1)
        ax_mag.text(f_cut, 10, fr'$T^+ \approx {T_plus_fcut:.1f}$', color='red', va='center', ha='right', rotation=90)

        ax_ph.semilogx(f1_lab, phase1, lw=1, color=ph1_colour)
        ax_ph.semilogx(f2_lab, phase2, lw=1, color=ph2_colour)
        ax_ph.set_ylabel(r'$\angle H_{\mathrm{PH-NC}}(f)\,[\mathrm{rad}]$')
        ax_ph.set_xlabel(r'$f\ \mathrm{[Hz]}$')
        # ax_ph.set_ylim(0, 7)
        ax_ph.axvline(f_cut, color='red', linestyle='--', lw=1)

        fig.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/700_100psi_H_anechoic_fused.pdf", dpi=600)
        plt.close()

    # Incorporate the in-situ data to make final lab calibration
    f1_hat, H1_hat, C1 = incorporate_insitu_calibration(f1_lab, H1_lab, g1_lab,
                                                        f1_is_far, H1_is_far, gamma1_is_far)
    f1_hat, H1_hat, C1 = incorporate_insitu_calibration(f1_hat, H1_hat, g1_lab,
                                                        f1_is_close, H1_is_close, gamma1_is_close)
    f2_hat, H2_hat, C2 = incorporate_insitu_calibration(f2_lab, H2_lab, g2_lab,
                                                        f2_is_far, H2_is_far, gamma2_is_far)
    f2_hat, H2_hat, C2 = incorporate_insitu_calibration(f2_hat, H2_hat, g2_lab,
                                                        f2_is_close, H2_is_close, gamma2_is_close)
    if 2 in plot:
        # plot the final lab TFs
        mag1 = np.abs(H1_hat); phase1 = np.unwrap(np.angle(H1_hat))
        mag2 = np.abs(H2_hat); phase2 = np.unwrap(np.angle(H2_hat))
        fig, (ax_mag, ax_ph) = plt.subplots(2, 1, sharex=True, figsize=(9, 4.5), dpi=600)
        ax_mag.set_title(r'Final $H_{\mathrm{PH-NC}}$ ($700\mu m$, 100psi), with suggested cutoffs')
        ax_mag.loglog(f1_hat, mag1, lw=1, color=ph1_colour)
        ax_mag.loglog(f2_hat, mag2, lw=1, color=ph2_colour)
        ax_mag.set_ylabel(r'$|H_{\mathrm{PH-NC}}(f)|$')
        ax_mag.set_ylim(1e-3, 100)
        ax_mag.axvline(f_cut, color='red', linestyle='--', lw=1)
        ax_mag.text(f_cut, 10, fr'$T^+ \approx {T_plus_fcut:.1f}$', color='red', va='center', ha='right', rotation=90)

        ax_ph.semilogx(f1_hat, phase1, lw=1, color=ph1_colour)
        ax_ph.semilogx(f2_hat, phase2, lw=1, color=ph2_colour)
        ax_ph.set_ylabel(r'$\angle H_{\mathrm{PH-NC}}(f)\,[\mathrm{rad}]$')
        ax_ph.set_xlabel(r'$f\ \mathrm{[Hz]}$')
        # ax_ph.set_ylim(0, 7)
        ax_ph.axvline(f_cut, color='red', linestyle='--', lw=1)

        fig.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/700_100psi_H_fuse_situ.pdf", dpi=600)
        plt.close()

    
    # 4) Build a regularized inverse for pinhole->real (optional but recommended)
    # g2_bar = np.maximum(np.interp(f1_hat, f1_is_far, g1_is_far, left=g2_is_far[0], right=g2_is_far[-1]), g2_lab)
    # alpha = 1.0
    # lam = alpha * (1 - g2_bar) / (g2_bar + 1e-12) * (np.abs(H1_hat)**2)
    # F = np.conj(H1_hat) / (np.abs(H1_hat)**2 + lam + 1e-12)

if __name__ == "__main__":
    # calibration()
    # calibration_700_atm()
    # calibration_700_50psi()
    calibration_700_100psi(plot=[1, 2])

    # flow_tests()
