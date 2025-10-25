from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch, csd, get_window, iirnotch, sosfiltfilt
import matplotlib.pyplot as plt
from icecream import ic

# =============================================================================
# Constants & styling
# =============================================================================
FS: float = 50_000.0
NPERSEG: int = 2**14
WINDOW: str = "hann"

# --- POWER-AWARE CALIBRATION CONTROLS ---------------------------------------
# Input-noise de-biasing (H1 <- H1 / beta_safe, where beta = Sxx/(Sxx+Snx))
INPUT_NOISE_FRAC = 0.20       # fraction of lowest-RMS segments to estimate Snx
INPUT_NOISE_MIN_SEG = 8       # minimum segments used for noise PSD
INPUT_NOISE_BETA_MIN = 0.30   # floor on beta used for correction
INPUT_NOISE_GAIN_CLIP_DB = 6  # cap de-bias gain to avoid over-correction; None=disable

# Fusion weights: w ∝ (gamma^2/(1-gamma^2))^a * (Sxx/Snx)^b * m^c
FUSION_COH_GMIN = 0.40        # coherence gate
FUSION_EXP_A = 1.0            # exponent for coherence term
FUSION_EXP_B = 1.0            # exponent for input-SNR term
FUSION_EXP_M = 1.0            # exponent for #averages (Welch segments)

SMOOTH_OCT_ANE = 1/6          # complex smoothing after fusion (octave span)

# -----------------------------------------------------------------------------

# Colors
PH1_COLOR = "#c76713"  # orange
PH2_COLOR = "#9fda16"  # green-ish
NC_COLOR = "#2ca02c"   # matplotlib default green (kept for reference)

# =============================================================================
# Units & optional conversions (kept for compatibility with other workflows)
# =============================================================================
DEFAULT_UNITS = {
    "channelData_300_plug": ("Pa", "Pa"),  # PH, nc
    "channelData_300_nose": ("Pa", "Pa"),  # nc, NC
    "channelData_300": ("Pa", "Pa"),       # NC, PH
}

SENSITIVITIES_V_PER_PA: dict[str, float] = {
    # 'nc': 0.05,
    # 'PH': 0.05,
    # 'NC': 0.05,
}
PREAMP_GAIN: dict[str, float] = {"nc": 1.0, "PH": 1.0, "NC": 1.0}


def convert_to_pa(x: np.ndarray, units: str, *, channel_name: str = "unknown") -> np.ndarray:
    """
    Convert a pressure time series to Pa.
    Supported units: 'Pa', 'kPa', 'mbar', 'V' (requires sensitivities/gains above).
    """
    u = units.lower()
    x = np.asarray(x, float)
    if u == "pa":
        return x
    if u == "kpa":
        return x * 1e3
    if u == "mbar":
        return x * 100.0  # 1 mbar = 100 Pa
    if u in ("v", "volt", "volts"):
        if channel_name not in SENSITIVITIES_V_PER_PA or SENSITIVITIES_V_PER_PA[channel_name] is None:
            raise ValueError(
                f"Sensitivity (V/Pa) for channel '{channel_name}' not provided; cannot convert V→Pa."
            )
        sens = float(SENSITIVITIES_V_PER_PA[channel_name])  # V/Pa
        gain = float(PREAMP_GAIN.get(channel_name, 1.0))
        return x / (gain * sens)  # Pa = V / (gain * (V/Pa))
    raise ValueError(f"Unsupported units '{units}' for channel '{channel_name}'")


def load_mat_to_pa(path: str | Path, key: str, ch1_name: str, ch2_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load two channels and convert each to Pa using DEFAULT_UNITS[key].
    ch1_name/ch2_name are used only if units are 'V' (to look up sensitivity/gain).
    """
    path = Path(path)
    mat = loadmat(path, squeeze_me=True)
    if key not in mat:
        raise KeyError(f"Key '{key}' not found in {path}. Available: {list(mat.keys())}")
    data = np.asarray(mat[key])
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array under '{key}', got shape {data.shape} in {path}")
    if data.shape[1] == 2:
        x_r = data[:, 0].astype(float)
        y_r = data[:, 1].astype(float)
    elif data.shape[0] == 2:
        x_r = data[0, :].astype(float)
        y_r = data[1, :].astype(float)
    else:
        raise ValueError(f"Unsupported shape for '{key}': {data.shape} in {path}")

    units_pair = DEFAULT_UNITS.get(key, ("Pa", "Pa"))
    x_pa = convert_to_pa(x_r, units_pair[0], channel_name=ch1_name)
    y_pa = convert_to_pa(y_r, units_pair[1], channel_name=ch2_name)
    return x_pa, y_pa

# =============================================================================
# Spectral & FRF tools
# =============================================================================
def estimate_frf_with_psd(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    *,
    npsg: int = NPERSEG,
    window: str = WINDOW,
):
    """
    Welch-based H1 FRF and PSDs with effective number of averages m.
    Returns: f, H (complex), gamma2, Sxx, Syy, m
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    nseg = int(min(npsg, x.size, y.size))
    if nseg < 8:
        raise ValueError(f"Signal too short for FRF: n={min(x.size, y.size)}")
    nov = int(min(npsg // 2, nseg // 2))
    w = get_window(window, nseg, fftbins=True)
    step = nseg - nov
    m = 1 + max(0, (min(x.size, y.size) - nseg) // step)

    f, Sxx = welch(x, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)
    _, Syy = welch(y, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)
    _, Sxy = csd(x, y, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)

    H = np.conj(Sxy) / Sxx  # H1
    gamma2 = np.clip((np.abs(Sxy) ** 2) / (Sxx * Syy), 0.0, 1.0)
    return f, H, gamma2, Sxx, Syy, m


def estimate_frf(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    *,
    npsg: int = NPERSEG,
    window: str = WINDOW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate H1 FRF and magnitude-squared coherence using Welch/CSD.
    Returns f, H, gamma2
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
    _, Sxy = csd(x, y, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)  # x→y

    H = np.conj(Sxy) / Sxx               # H1 = Syx/Sxx = conj(Sxy)/Sxx
    gamma2 = (np.abs(Sxy) ** 2) / (Sxx * Syy)
    gamma2 = np.clip(gamma2.real, 0.0, 1.0)
    return f, H, gamma2


def compute_spec(fs: float, x: np.ndarray, npsg: int = NPERSEG) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD with sane defaults. Returns (f [Hz], Pxx [Pa^2/Hz])."""
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


def _interp_complex(f_src: np.ndarray, z_src: np.ndarray, f_tgt: np.ndarray) -> np.ndarray:
    z_src = np.asarray(z_src)
    re = np.interp(f_tgt, f_src, np.real(z_src), left=np.real(z_src[0]), right=np.real(z_src[-1]))
    im = np.interp(f_tgt, f_src, np.imag(z_src), left=np.imag(z_src[0]), right=np.imag(z_src[-1]))
    return re + 1j * im


def _complex_smooth_logfreq(
    f: np.ndarray,
    z: np.ndarray,
    *,
    span_oct: float = 1 / 6,
    points_per_oct: int = 48,
    eps: float = 1e-20,
) -> np.ndarray:
    """
    Complex moving-average smoothing with a constant span in octaves.
    Smoothing on a log-frequency grid; real & imag are smoothed separately.
    """
    f = np.asarray(f)
    z = np.asarray(z)
    assert f.ndim == 1 and z.ndim == 1 and f.size == z.size
    pos = f > 0
    if span_oct <= 0 or pos.sum() < 8:
        return z.copy()

    fpos = f[pos]
    zpos = z[pos]
    lo, hi = fpos[0], fpos[-1]
    n_oct = np.log2(hi / max(lo, eps))
    n_pts = max(int(np.ceil(n_oct * points_per_oct)), 8)
    flog = np.linspace(np.log2(max(lo, eps)), np.log2(hi), n_pts)
    fgrid = np.power(2.0, flog)

    zlog = _interp_complex(fpos, zpos, fgrid)

    wlen = max(int(round(span_oct * points_per_oct)), 1)
    if wlen % 2 == 0:
        wlen += 1
    box = np.ones(wlen) / wlen
    re_s = np.convolve(np.real(zlog), box, mode="same")
    im_s = np.convolve(np.imag(zlog), box, mode="same")
    zlog_s = re_s + 1j * im_s

    z_s_pos = _interp_complex(fgrid, zlog_s, fpos)
    z_s = z.copy()
    z_s[pos] = z_s_pos
    return z_s


# =============================================================================
# Input-noise handling (NEW)
# =============================================================================
def _periodogram_onesided(seg: np.ndarray, fs: float, w: np.ndarray) -> np.ndarray:
    """
    One-sided, window-corrected periodogram density for a single segment.
    """
    xw = seg * w
    X = np.fft.rfft(xw, n=len(xw))
    scale = (fs * np.sum(w**2))
    P = (np.abs(X) ** 2) / max(scale, 1e-20)
    # one-sided doubling except DC and Nyquist (if present)
    if len(xw) % 2 == 0:
        P[1:-1] *= 2.0
    else:
        P[1:] *= 2.0
    return P


def estimate_noise_psd_low_rms(
    x: np.ndarray,
    fs: float,
    *,
    nperseg: int,
    noverlap: int,
    window: str = WINDOW,
    frac_low_rms: float = INPUT_NOISE_FRAC,
    min_segments: int = INPUT_NOISE_MIN_SEG,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate input-channel noise PSD Snx by averaging the lowest-RMS segments.
    Robust when no dedicated driver-off capture exists.
    """
    x = np.asarray(x, float)
    if x.size < nperseg:
        # Fallback: standard Welch (will overestimate Snx if excitation always on)
        return compute_spec(fs, x, npsg=nperseg)

    w = get_window(window, nperseg, fftbins=True)
    step = nperseg - noverlap
    starts = np.arange(0, x.size - nperseg + 1, step, dtype=int)
    if starts.size == 0:
        return compute_spec(fs, x, npsg=nperseg)

    # Per-segment RMS
    rms = np.array([np.sqrt(np.mean(x[s:s + nperseg] ** 2)) for s in starts])
    k = max(min_segments, int(np.ceil(frac_low_rms * len(starts))))
    idx = np.argsort(rms)[:k]

    # Average periodograms of the selected low-RMS segments
    P_accum = None
    for s in starts[idx]:
        seg = x[s:s + nperseg]
        P = _periodogram_onesided(seg, fs, w)
        if P_accum is None:
            P_accum = P
        else:
            P_accum += P
    Pmean = P_accum / float(k)
    f = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return f, Pmean


def debias_H1_for_input_noise(
    H1: np.ndarray,
    Sxx_meas: np.ndarray,
    Snx: np.ndarray,
    *,
    beta_min: float = INPUT_NOISE_BETA_MIN,
    gain_clip_db: float | None = INPUT_NOISE_GAIN_CLIP_DB,
    eps: float = 1e-18,
):
    """
    De-bias H1 for input noise: H_unb = H1 / beta_safe, with optional gain clip.
    Returns H_unbiased, beta
    """
    ratio = np.clip(Snx / np.maximum(Sxx_meas, eps), 0.0, 1.0)
    beta = 1.0 - ratio
    beta_safe = np.clip(beta, beta_min, 1.0)
    gain = 1.0 / np.maximum(beta_safe, eps)
    if gain_clip_db is not None:
        gain = np.minimum(gain, 10 ** (gain_clip_db / 20.0))
    H_unbiased = H1 * gain
    return H_unbiased, beta


def combine_anechoic_calibrations_poweraware(
    f1, H1, g2_1, Sxx1, Snx1, m1,
    f2, H2, g2_2, Sxx2, Snx2, m2,
    *,
    gmin: float = FUSION_COH_GMIN,
    a: float = FUSION_EXP_A,   # coherence exponent
    b: float = FUSION_EXP_B,   # input-SNR exponent
    c_mexp: float = FUSION_EXP_M,  # averages exponent
    points_per_oct: int = 32,
    smooth_oct: float | None = SMOOTH_OCT_ANE,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Power-aware fusion: complex average with weights including coherence, input SNR, and #averages.
    """
    f1 = np.asarray(f1); f2 = np.asarray(f2)
    H1 = np.asarray(H1); H2 = np.asarray(H2)
    g2_1 = np.clip(np.asarray(g2_1), 0.0, 1.0)
    g2_2 = np.clip(np.asarray(g2_2), 0.0, 1.0)
    Sxx1 = np.asarray(Sxx1); Sxx2 = np.asarray(Sxx2)
    Snx1 = np.asarray(Snx1); Snx2 = np.asarray(Snx2)

    # Target grid: the denser one
    f = f1 if f1.size >= f2.size else f2
    def _interp(z, fx): return np.interp(f, fx, z)

    H1i = _interp(np.real(H1), f1) + 1j * _interp(np.imag(H1), f1)
    H2i = _interp(np.real(H2), f2) + 1j * _interp(np.imag(H2), f2)
    g2_1i = _interp(g2_1, f1); g2_2i = _interp(g2_2, f2)

    Sxx1i = _interp(Sxx1, f1); Sxx2i = _interp(Sxx2, f2)
    Snx1i = _interp(Snx1, f1); Snx2i = _interp(Snx2, f2)

    # Coherence weights with gating
    wγ1 = np.where(g2_1i >= gmin, (g2_1i / np.maximum(1 - g2_1i, eps)) ** a, 0.0)
    wγ2 = np.where(g2_2i >= gmin, (g2_2i / np.maximum(1 - g2_2i, eps)) ** a, 0.0)

    # Input SNR weights: Sxx/Snx
    wβ1 = np.maximum(Sxx1i / np.maximum(Snx1i, eps), 0.0) ** b
    wβ2 = np.maximum(Sxx2i / np.maximum(Snx2i, eps), 0.0) ** b

    # Averages weights
    wm1 = float(m1) ** c_mexp
    wm2 = float(m2) ** c_mexp

    w1 = wγ1 * wβ1 * wm1
    w2 = wγ2 * wβ2 * wm2
    wsum = w1 + w2 + eps

    H_lab = (w1 * H1i + w2 * H2i) / wsum
    g2_lab = np.clip((w1 * g2_1i + w2 * g2_2i) / wsum, 0.0, 1.0)

    if smooth_oct is not None and smooth_oct > 0:
        H_lab = _complex_smooth_logfreq(f, H_lab, span_oct=smooth_oct, points_per_oct=points_per_oct)
    return f, H_lab, g2_lab


def combine_anechoic_calibrations(
    f1, H1, g2_1,
    f2, H2, g2_2,
    *,
    gmin: float = 0.4,
    smooth_oct: float | None = 1 / 6,
    points_per_oct: int = 32,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Legacy fusion (coherence-only). Kept for reference / fallback.
    """
    f1 = np.asarray(f1); f2 = np.asarray(f2)
    H1 = np.asarray(H1); H2 = np.asarray(H2)
    g2_1 = np.clip(np.asarray(g2_1), 0.0, 1.0)
    g2_2 = np.clip(np.asarray(g2_2), 0.0, 1.0)

    # Choose the denser frequency grid as the target
    f = f1 if f1.size >= f2.size else f2

    H1i = _interp_complex(f1, H1, f)
    H2i = _interp_complex(f2, H2, f)
    g2_1i = np.interp(f, f1, g2_1, left=g2_1[0], right=g2_1[-1])
    g2_2i = np.interp(f, f2, g2_2, left=g2_2[0], right=g2_2[-1])

    def _weights(g2):
        g2c = np.clip(g2, 0.0, 1.0 - 1e-9)
        w = g2c / (1.0 - g2c + eps)
        return np.where(g2c >= gmin, w, 0.0)

    w1 = _weights(g2_1i)
    w2 = _weights(g2_2i)
    wsum = w1 + w2 + eps

    H_lab = (w1 * H1i + w2 * H2i) / wsum
    g2_lab = np.clip((w1 * g2_1i + w2 * g2_2i) / wsum, 0.0, 1.0)

    if smooth_oct is not None and smooth_oct > 0:
        H_lab = _complex_smooth_logfreq(f, H_lab, span_oct=smooth_oct, points_per_oct=points_per_oct)

    return f, H_lab, g2_lab


def incorporate_insitu_calibration(
    f_lab, H_lab, g2_lab,
    f_ins, H_ins, g2_ins,
    *,
    gmin: float = 0.3,
    gmax: float = 0.8,
    ratio_smooth_oct: float = 1 / 6,
    post_smooth_oct: float | None = 1 / 6,
    points_per_oct: int = 48,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Correct the fused anechoic FRF with in-situ data where the in-situ coherence is good.
    """
    f_lab = np.asarray(f_lab); H_lab = np.asarray(H_lab); g2_lab = np.asarray(g2_lab)
    f_ins = np.asarray(f_ins); g2_ins = np.clip(np.asarray(g2_ins), 0.0, 1.0)

    H_ins_i = _interp_complex(f_ins, H_ins, f_lab)
    g2_ins_i = np.interp(f_lab, f_ins, g2_ins, left=g2_ins[0], right=g2_ins[-1])

    R = H_ins_i / (H_lab + eps)
    R = _complex_smooth_logfreq(f_lab, R, span_oct=ratio_smooth_oct, points_per_oct=points_per_oct)

    C = (g2_ins_i - gmin) / (gmax - gmin + eps)
    C = np.clip(C, 0.0, 1.0)

    mag_lab = np.abs(H_lab) + eps
    mag_ins = np.abs(H_lab * R) + eps
    log_mag_hat = (1.0 - C) * np.log(mag_lab) + C * np.log(mag_ins)
    mag_hat = np.exp(log_mag_hat)

    dphi = np.unwrap(np.angle(R))
    phi_lab = np.unwrap(np.angle(H_lab))
    phi_hat = phi_lab + C * dphi

    H_hat = mag_hat * np.exp(1j * phi_hat)

    if post_smooth_oct is not None and post_smooth_oct > 0:
        H_hat = _complex_smooth_logfreq(f_lab, H_hat, span_oct=post_smooth_oct, points_per_oct=points_per_oct)

    return f_lab, H_hat, C


# =============================================================================
# FRF application & simple filtering helpers (unchanged logic)
# =============================================================================
def wiener_forward(
    x: np.ndarray,
    fs: float,
    f: np.ndarray,
    H: np.ndarray,
    gamma2: np.ndarray,
    nfft_pow: int = 0,
    demean: bool = True,
    zero_dc: bool = True,
    taper_hz: float = 0.0,
) -> np.ndarray:
    """
    Forward FRF application: given x (PH) and H_{PH->nc}, synthesize ŷ ≈ nc.
    Uses coherence-weighted magnitude (sqrt(gamma2)) and optional in-band taper.
    """
    x = np.asarray(x, float)
    f = np.asarray(f, float)
    H = np.asarray(H)
    gamma2 = np.asarray(gamma2, float)

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
    if f[0] < 0 or f[-1] > fs / 2 + 1e-9:
        raise ValueError("f must lie within [0, fs/2]")

    if demean:
        x = x - x.mean()

    N = x.size
    min_pow = int(np.ceil(np.log2(max(1, N))))
    if nfft_pow and nfft_pow > 0:
        Nfft = 2 ** max(nfft_pow, min_pow)
    else:
        Nfft = 2 ** min_pow

    X = np.fft.rfft(x, n=Nfft)
    fr = np.fft.rfftfreq(Nfft, d=1.0 / fs)

    Hr = np.interp(fr, f, np.real(H), left=0.0, right=0.0)
    Hi = np.interp(fr, f, np.imag(H), left=0.0, right=0.0)
    H_i = Hr + 1j * Hi

    g2_i = np.clip(np.interp(fr, f, gamma2, left=0.0, right=0.0), 0.0, 1.0)
    W = np.sqrt(g2_i)

    if taper_hz and taper_hz > 0.0:
        band_lo = f[0]
        band_hi = f[-1]
        lo_edge = np.where((fr >= band_lo) & (fr < band_lo + taper_hz))[0]
        hi_edge = np.where((fr <= band_hi) & (fr > band_hi - taper_hz))[0]
        if lo_edge.size > 0:
            t = (fr[lo_edge] - band_lo) / taper_hz
            W[lo_edge] *= 0.5 * (1 - np.cos(np.pi * t))
        if hi_edge.size > 0:
            t = (band_hi - fr[hi_edge]) / taper_hz
            W[hi_edge] *= 0.5 * (1 - np.cos(np.pi * t))

    Y = W * H_i * X
    if zero_dc and Y.size > 0:
        Y[0] = 0.0

    y_hat = np.fft.irfft(Y, n=Nfft)[:N]
    return y_hat


def wiener_inverse(
    y_r: np.ndarray,
    fs: float,
    f: np.ndarray,
    H: np.ndarray,
    gamma2: np.ndarray,
    demean: bool = True,
    zero_dc: bool = True,
) -> np.ndarray:
    """
    Coherence-weighted inverse filter: H_inv = gamma^2 * H* / |H|^2.
    """
    y = np.asarray(y_r, float)
    if demean:
        y = y - y.mean()
    N = y.size
    Nfft = int(2 ** np.ceil(np.log2(N)))
    Yr = np.fft.rfft(y, n=Nfft)
    fr = np.fft.rfftfreq(Nfft, d=1.0 / fs)

    mag = np.abs(H)
    phi = np.unwrap(np.angle(H))
    mag_i = np.interp(fr, f, mag, left=1.0, right=1.0)
    phi_i = np.interp(fr, f, phi, left=phi[0], right=phi[-1])
    Hi = mag_i * np.exp(1j * phi_i)

    g2_i = np.clip(np.interp(fr, f, gamma2, left=0.0, right=0.0), 0.0, 1.0)

    eps = np.finfo(float).eps
    Hinv = g2_i * np.conj(Hi) / np.maximum(mag_i**2, eps)
    if zero_dc:
        Hinv[0] = 0.0
        if Nfft % 2 == 0:
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
) -> np.ndarray:
    """
    Apply a measured FRF H (x→y) to a time series x to synthesise y.
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
    mag_i = np.interp(fr, f, mag, left=1.0, right=1.0)
    phi_i = np.interp(fr, f, phi, left=phi[0], right=phi[-1])
    Hi = mag_i * np.exp(1j * phi_i)

    if zero_dc:
        Hi[0] = 0.0
        if Nfft % 2 == 0:
            Hi[-1] = 0.0

    y = np.fft.irfft(X * Hi, n=Nfft)[:N]
    return y


def design_notches(fs: float, freqs: Iterable[float], Q: float = 30.0) -> np.ndarray | None:
    """
    Make a cascade of IIR notch filters (as SOS).
    """
    sos_list: list[np.ndarray] = []
    for f0 in freqs:
        w0 = f0 / (fs / 2.0)  # normalized (Nyquist=1)
        b, a = iirnotch(w0, Q)
        sos_list.append(np.hstack([b, a]))
    if not sos_list:
        return None
    return np.vstack(sos_list)


def apply_notches(x: np.ndarray, sos: np.ndarray | None) -> np.ndarray:
    """Apply zero-phase notch filtering to signal x."""
    if sos is None:
        return x
    return sosfiltfilt(sos, x)


# =============================================================================
# Calibration pipeline (DRY version of 0/50/100 psi blocks)
# =============================================================================
@dataclass(frozen=True)
class CalibCase:
    """Configuration for a single calibration case."""
    psi_label: str    # e.g., "0psi", "50psi", "100psi"
    u_tau: float
    nu_utau: float
    f_cut: float

    @property
    def nu(self) -> float:
        return self.u_tau * self.nu_utau

    @property
    def Tplus_at_fcut(self) -> float:
        return (self.u_tau**2) / (self.nu * self.f_cut)

    @property
    def psi_g(self) -> str:
        # Used in filenames that previously used 'psig'
        return self.psi_label.replace("psi", "psig")

    @property
    def title_suffix(self) -> str:
        return fr"($700\mu m$, {self.psi_label})"


# Fixed data roots per your original script
BASE_FAR = Path("data/20251014")
BASE_CLOSE = Path("data/20251016")
OUTPUT_DIR = Path("figures/tf_calib")
CAL_DIR_FAR = BASE_FAR / "tf_calib" / "tf_data"
CAL_DIR_CLOSE = BASE_CLOSE / "tf_calib" / "tf_data"
CAL_DIR_COMB = BASE_CLOSE / "flow_data" / "tf_combined"


def _ensure_dirs() -> None:
    for p in (OUTPUT_DIR, CAL_DIR_FAR, CAL_DIR_CLOSE, CAL_DIR_COMB):
        p.mkdir(parents=True, exist_ok=True)


def _save_frf_triplet(out_dir: Path, tag: str, f: np.ndarray, H: np.ndarray, g2: np.ndarray) -> None:
    """
    Save FRF components with the same filenames as in the original script.
    Example tag: '1_700_0psig', '1_700_0psi_is', '700_0psi_fused_anechoic_f1', etc.
    """
    np.save(out_dir / f"f{tag}.npy", f)
    np.save(out_dir / f"H{tag}.npy", H)
    np.save(out_dir / f"gamma{tag}.npy", g2)


def _plot_runs_overlay(
    title_suffix: str,
    f_cut: float,
    Tplus_fcut: float,
    runs: list[tuple[str, np.ndarray, np.ndarray]],
    outfile: Path,
) -> None:
    """
    Plot |H| and phase for a list of runs.
    runs: list of (label, f, H)
    """
    fig, (ax_mag, ax_ph) = plt.subplots(2, 1, sharex=True, figsize=(9, 3), dpi=600)
    ax_mag.set_title(r"$H_{\mathrm{PH-NC}}$ " + title_suffix + ", with suggested cutoffs")
    for label, f, H in runs:
        color = PH1_COLOR if "PH1" in label else (PH2_COLOR if "PH2" in label else "k")
        ls = "--" if "close" in label else (":" if "in-situ" in label and "PH1" in label else "-.")
        lw = 1
        mag = np.abs(H)
        phase = np.unwrap(np.angle(H))
        ax_mag.loglog(f, mag, lw=lw, color=color, ls=ls, label=label)
        ax_ph.semilogx(f, phase, lw=lw, color=color, ls=ls)

    ax_mag.legend(fontsize=8, ncol=2)
    ax_mag.set_ylabel(r"$|H_{\mathrm{PH-NC}}(f)|$")
    ax_mag.axvline(f_cut, color="red", linestyle="--", lw=1)
    ax_mag.text(f_cut, 10, fr"$T^+ \approx {Tplus_fcut:.1f}$", color="red", va="center", ha="right", rotation=90)

    ax_ph.set_ylabel(r"$\angle H_{\mathrm{PH-NC}}(f)\,[\mathrm{rad}]$")
    ax_ph.set_xlabel(r"$f\ \mathrm{[Hz]}$")
    ax_ph.axvline(f_cut, color="red", linestyle="--", lw=1)

    fig.tight_layout()
    plt.savefig(outfile, dpi=600)
    plt.close(fig)


def _plot_two_curves(
    title_suffix: str,
    f_cut: float,
    Tplus_fcut: float,
    curves: list[tuple[np.ndarray, np.ndarray, str]],
    outfile: Path,
    ymag_limits: tuple[float, float] | None = None,
) -> None:
    """
    Plot |H| and phase for exactly two curves (PH1, PH2).
    curves: list of (f, H, label_suffix)
    """
    fig, (ax_mag, ax_ph) = plt.subplots(2, 1, sharex=True, figsize=(9, 3), dpi=600)
    ax_mag.set_title(r"Fused $H_{\mathrm{PH-NC}}$ " + title_suffix + ", with suggested cutoffs")
    for f, H, suffix in curves:
        color = PH1_COLOR if "PH1" in suffix else PH2_COLOR
        mag = np.abs(H)
        phase = np.unwrap(np.angle(H))
        ax_mag.loglog(f, mag, lw=1, color=color)
        ax_ph.semilogx(f, phase, lw=1, color=color)

    ax_mag.set_ylabel(r"$|H_{\mathrm{PH-NC}}(f)|$")
    if ymag_limits is not None:
        ax_mag.set_ylim(*ymag_limits)
    ax_mag.axvline(f_cut, color="red", linestyle="--", lw=1)
    ax_mag.text(f_cut, 10, fr"$T^+ \approx {Tplus_fcut:.1f}$", color="red", va="center", ha="right", rotation=90)

    ax_ph.set_ylabel(r"$\angle H_{\mathrm{PH-NC}}(f)\,[\mathrm{rad}]$")
    ax_ph.set_xlabel(r"$f\ \mathrm{[Hz]}$")
    ax_ph.axvline(f_cut, color="red", linestyle="--", lw=1)

    fig.tight_layout()
    plt.savefig(outfile, dpi=600)
    plt.close(fig)


def _plot_gamma(
    title_suffix: str,
    curves: list[tuple[np.ndarray, np.ndarray, str]],
    outfile: Path,
) -> None:
    """Plot gamma^2 curves."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 2), dpi=600)
    ax.set_title(r"Coherence $\gamma^2$ used in calibration fusion " + title_suffix)
    for f, g2, label in curves:
        color = PH1_COLOR if "PH1" in label else PH2_COLOR
        ax.semilogx(f, g2, lw=1, color=color, label=label)
    ax.set_ylabel(r"$\gamma^2_{\mathrm{PH-NC}}(f)$")
    ax.set_xlabel(r"$f\ \mathrm{[Hz]}$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    plt.savefig(outfile, dpi=600)
    plt.close(fig)


def _load_lp_pair(fn_ph1: Path, fn_ph2: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load 'channelData_LP' from two .mat files (ph1-file, ph2-file), returning:
    (ph1, nc1, ph2, nc2)
    """
    d1 = loadmat(fn_ph1)
    d2 = loadmat(fn_ph2)
    # per your original column usage: [PH1, PH2, NC] in 'channelData_LP'
    ph1 = d1["channelData_LP"][:, 0].astype(float)
    nc1 = d1["channelData_LP"][:, 2].astype(float)
    ph2 = d2["channelData_LP"][:, 1].astype(float)
    nc2 = d2["channelData_LP"][:, 2].astype(float)
    return ph1, nc1, ph2, nc2


def _load_insitu_pair(fn: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load 'channelData_noflow' from a .mat, returning (ph1, ph2, nc).
    """
    d = loadmat(fn)
    ph1 = d["channelData_noflow"][:, 0].astype(float)
    ph2 = d["channelData_noflow"][:, 1].astype(float)
    nc = d["channelData_noflow"][:, 2].astype(float)
    return ph1, ph2, nc


def calibrate_700_case(case: CalibCase, plot: Iterable[int] = (0, 1, 2, 3)) -> None:
    """
    One-stop calibration runner for 700 μm pinholes at a given pressure case.
    Implements power-aware de-biased fusion + in-situ incorporation.
    """
    _ensure_dirs()

    # --- Diagnostics (kept: matches original debug intent)
    nu = case.nu
    if case.psi_label == "0psi":
        Re_tau = case.u_tau * 0.035 / nu
        T10 = 0.1 * (case.u_tau**2) / nu
        T10000 = 1e-4 * (case.u_tau**2) / nu
        ic(nu, Re_tau, T10, T10000)
    ic("T+ at f_cut:", case.Tplus_at_fcut)

    # --- File layout (matches originals)
    far_tf_dir = CAL_DIR_FAR
    close_tf_dir = CAL_DIR_CLOSE
    comb_dir = CAL_DIR_COMB

    far_tf_root = BASE_FAR / "tf_calib"
    close_tf_root = BASE_CLOSE / "tf_calib"

    far_flow = BASE_FAR / "flow_data" / "far" / f"{case.psi_label}.mat"
    close_flow = BASE_CLOSE / "flow_data" / "close" / f"{case.psi_label}.mat"

    # --- Load anechoic (LP) data
    fn1_far = far_tf_root / f"{case.psi_label}_lp_16khz_ph1.mat"
    fn2_far = far_tf_root / f"{case.psi_label}_lp_16khz_ph2.mat"
    fn1_close = close_tf_root / f"{case.psi_label}_lp_16khz_ph1.mat"
    fn2_close = close_tf_root / f"{case.psi_label}_lp_16khz_ph2.mat"

    ph1_far, nc1_far, ph2_far, nc2_far = _load_lp_pair(fn1_far, fn2_far)
    ph1_close, nc1_close, ph2_close, nc2_close = _load_lp_pair(fn1_close, fn2_close)

    # --- FRF+PSD for each anechoic run (PH1/PH2 × far/close)
    npsg = 2**9
    nov = npsg // 2

    f1_far,  H1_far,  g1_far,  Sxx1_far,  _, m1_far  = estimate_frf_with_psd(ph1_far,  nc1_far,  FS, npsg=npsg)
    f1_close, H1_close, g1_close, Sxx1_close, _, m1_close = estimate_frf_with_psd(ph1_close, nc1_close, FS, npsg=npsg)
    f2_far,  H2_far,  g2_far,  Sxx2_far,  _, m2_far  = estimate_frf_with_psd(ph2_far,  nc2_far,  FS, npsg=npsg)
    f2_close, H2_close, g2_close, Sxx2_close, _, m2_close = estimate_frf_with_psd(ph2_close, nc2_close, FS, npsg=npsg)

    # --- Estimate input-noise PSD per PH and run (low-RMS frames)
    fN1_far,  Snx1_far  = estimate_noise_psd_low_rms(ph1_far,  FS, nperseg=npsg, noverlap=nov)
    fN1_close, Snx1_close = estimate_noise_psd_low_rms(ph1_close, FS, nperseg=npsg, noverlap=nov)
    fN2_far,  Snx2_far  = estimate_noise_psd_low_rms(ph2_far,  FS, nperseg=npsg, noverlap=nov)
    fN2_close, Snx2_close = estimate_noise_psd_low_rms(ph2_close, FS, nperseg=npsg, noverlap=nov)

    # Resample noise PSDs to FRF frequency grids
    Snx1_far_on_f1_far     = np.interp(f1_far,  fN1_far,  Snx1_far)
    Snx1_close_on_f1_close = np.interp(f1_close, fN1_close, Snx1_close)
    Snx2_far_on_f2_far     = np.interp(f2_far,  fN2_far,  Snx2_far)
    Snx2_close_on_f2_close = np.interp(f2_close, fN2_close, Snx2_close)

    # --- De-bias H1 per run using input-noise estimate
    H1_far_unb,   beta1_far   = debias_H1_for_input_noise(H1_far,  Sxx1_far,  Snx1_far_on_f1_far)
    H1_close_unb, beta1_close = debias_H1_for_input_noise(H1_close, Sxx1_close, Snx1_close_on_f1_close)
    H2_far_unb,   beta2_far   = debias_H1_for_input_noise(H2_far,  Sxx2_far,  Snx2_far_on_f2_far)
    H2_close_unb, beta2_close = debias_H1_for_input_noise(H2_close, Sxx2_close, Snx2_close_on_f2_close)

    # --- Save per-run FRFs (de-biased) for backward compatibility with filenames
    _save_frf_triplet(far_tf_dir,  f"1_700_{case.psi_g}", f1_far,  H1_far_unb,  g1_far)
    _save_frf_triplet(far_tf_dir,  f"2_700_{case.psi_g}", f2_far,  H2_far_unb,  g2_far)
    _save_frf_triplet(close_tf_dir, f"1_700_{case.psi_g}", f1_close, H1_close_unb, g1_close)
    _save_frf_triplet(close_tf_dir, f"2_700_{case.psi_g}", f2_close, H2_close_unb, g2_close)

    # --- In-situ (no-flow) data inside "flow_data/*/{psi}.mat" (unchanged)
    ph1_is_far,  ph2_is_far,  nc_is_far  = _load_insitu_pair(far_flow)
    ph1_is_close, ph2_is_close, nc_is_close = _load_insitu_pair(close_flow)

    f1_is_far,  H1_is_far,  g1_is_far  = estimate_frf(ph1_is_far,  nc_is_far,  FS, npsg=npsg)
    f2_is_far,  H2_is_far,  g2_is_far  = estimate_frf(ph2_is_far,  nc_is_far,  FS, npsg=npsg)
    f1_is_close, H1_is_close, g1_is_close = estimate_frf(ph1_is_close, nc_is_close, FS, npsg=npsg)
    f2_is_close, H2_is_close, g2_is_close = estimate_frf(ph2_is_close, nc_is_close, FS, npsg=npsg)

    _save_frf_triplet(far_tf_dir,   f"1_700_{case.psi_label}_is", f1_is_far,  H1_is_far,  g1_is_far)
    _save_frf_triplet(far_tf_dir,   f"2_700_{case.psi_label}_is", f2_is_far,  H2_is_far,  g2_is_far)
    _save_frf_triplet(close_tf_dir, f"1_700_{case.psi_label}_is", f1_is_close, H1_is_close, g1_is_close)
    _save_frf_triplet(close_tf_dir, f"2_700_{case.psi_label}_is", f2_is_close, H2_is_close, g2_is_close)

    # --- Plot overlay of all (if requested) — use de-biased H for anechoic runs
    if 0 in plot:
        runs_ovl = [
            ("PH1 far",            f1_far,   H1_far_unb),
            ("PH2 far",            f2_far,   H2_far_unb),
            ("PH1 close",          f1_close, H1_close_unb),
            ("PH2 close",          f2_close, H2_close_unb),
            ("PH1 in-situ far",    f1_is_far,  H1_is_far),
            ("PH2 in-situ far",    f2_is_far,  H2_is_far),
            ("PH1 in-situ close",  f1_is_close, H1_is_close),
            ("PH2 in-situ close",  f2_is_close, H2_is_close),
        ]
        _plot_runs_overlay(
            case.title_suffix, case.f_cut, case.Tplus_at_fcut, runs_ovl,
            OUTPUT_DIR / f"700_{case.psi_label}_H_2cal.png"
        )

    # --- Fuse anechoic (power-aware)
    f1_lab, H1_lab, g1_lab = combine_anechoic_calibrations_poweraware(
        f1_far,  H1_far_unb,  g1_far,  Sxx1_far,  Snx1_far_on_f1_far,     m1_far,
        f1_close, H1_close_unb, g1_close, Sxx1_close, Snx1_close_on_f1_close, m1_close,
        gmin=FUSION_COH_GMIN, a=FUSION_EXP_A, b=FUSION_EXP_B, c_mexp=FUSION_EXP_M,
        smooth_oct=SMOOTH_OCT_ANE,
    )
    f2_lab, H2_lab, g2_lab = combine_anechoic_calibrations_poweraware(
        f2_far,  H2_far_unb,  g2_far,  Sxx2_far,  Snx2_far_on_f2_far,     m2_far,
        f2_close, H2_close_unb, g2_close, Sxx2_close, Snx2_close_on_f2_close, m2_close,
        gmin=FUSION_COH_GMIN, a=FUSION_EXP_A, b=FUSION_EXP_B, c_mexp=FUSION_EXP_M,
        smooth_oct=SMOOTH_OCT_ANE,
    )

    # Save anechoic fused (original filenames used 'psi' here)
    np.save(comb_dir / f"700_{case.psi_label}_fused_anechoic_f1.npy", f1_lab)
    np.save(comb_dir / f"700_{case.psi_label}_fused_anechoic_H1.npy", H1_lab)
    np.save(comb_dir / f"700_{case.psi_label}_fused_anechoic_f2.npy", f2_lab)
    np.save(comb_dir / f"700_{case.psi_label}_fused_anechoic_H2.npy", H2_lab)

    if 1 in plot:
        _plot_two_curves(
            case.title_suffix,
            case.f_cut,
            case.Tplus_at_fcut,
            [
                (f1_lab, H1_lab, "PH1 lab fused"),
                (f2_lab, H2_lab, "PH2 lab fused"),
            ],
            OUTPUT_DIR / f"700_{case.psi_label}_H_anechoic_fused.png",
        )

    # --- Incorporate in-situ (PH1 & PH2 each, far then close) — unchanged logic
    f1_hat, H1_hat, C1 = incorporate_insitu_calibration(f1_lab, H1_lab, g1_lab, f1_is_far,  H1_is_far,  g1_is_far)
    f1_hat, H1_hat, C1 = incorporate_insitu_calibration(f1_hat, H1_hat, g1_lab, f1_is_close, H1_is_close, g1_is_close)
    f2_hat, H2_hat, C2 = incorporate_insitu_calibration(f2_lab, H2_lab, g2_lab, f2_is_far,  H2_is_far,  g2_is_far)
    f2_hat, H2_hat, C2 = incorporate_insitu_calibration(f2_hat, H2_hat, g2_lab, f2_is_close, H2_is_close, g2_is_close)

    # Save in-situ fused (original filenames used 'psig' here)
    np.save(comb_dir / f"700_{case.psi_g}_fused_insitu_f1.npy", f1_hat)
    np.save(comb_dir / f"700_{case.psi_g}_fused_insitu_H1.npy", H1_hat)
    np.save(comb_dir / f"700_{case.psi_g}_fused_insitu_f2.npy", f2_hat)
    np.save(comb_dir / f"700_{case.psi_g}_fused_insitu_H2.npy", H2_hat)

    if 2 in plot:
        _plot_two_curves(
            case.title_suffix,
            case.f_cut,
            case.Tplus_at_fcut,
            [
                (f1_hat, H1_hat, "PH1 final"),
                (f2_hat, H2_hat, "PH2 final"),
            ],
            OUTPUT_DIR / f"700_{case.psi_label}_H_fuse_situ.png",
            ymag_limits=(1e-3, 100),
        )

    if 3 in plot:
        _plot_gamma(
            case.title_suffix,
            [
                (f1_lab, g1_lab, "PH1 lab fused"),
                (f2_lab, g2_lab, "PH2 lab fused"),
            ],
            OUTPUT_DIR / f"700_{case.psi_label}_gamma_fuse.png",
        )


# =============================================================================
# Entrypoint
# =============================================================================
def run_all_calibrations() -> None:
    """
    Replaces:
      calibration_700_0psi()
      calibration_700_50psi()
      calibration_700_100psi()
    """
    cases = [
        CalibCase("0psi",   u_tau=0.58, nu_utau=27e-6,  f_cut=2_100.0),
        CalibCase("50psi",  u_tau=0.47, nu_utau=7.5e-6, f_cut=4_700.0),
        CalibCase("100psi", u_tau=0.52, nu_utau=3.7e-6, f_cut=14_100.0),
    ]
    for c in cases:
        calibrate_700_case(c, plot=(0, 1, 2, 3))


if __name__ == "__main__":
    run_all_calibrations()
