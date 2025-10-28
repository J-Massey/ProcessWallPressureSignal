# tf_compute.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple
import inspect

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch, csd, get_window, iirnotch, sosfiltfilt

try:
    from icecream import ic  # lightweight, optional
except Exception:  # pragma: no cover
    def ic(*_a, **_k):  # type: ignore
        return None

# =============================================================================
# Constants & styling (exported so tf_plot.py can import them)
# =============================================================================
FS: float = 50_000.0
NPERSEG: int = 2**14
WINDOW: str = "hann"

# Colors (exported for plotting)
PH1_COLOR = "#c76713"  # orange
PH2_COLOR = "#9fda16"  # green-ish
NC_COLOR = "#2ca02c"   # matplotlib default green (kept for reference)

# =============================================================================
# Units & optional conversions (kept for compatibility with other workflows)
# =============================================================================
DEFAULT_UNITS = {
    "channelData_300_plug": ("Pa", "Pa"),
    "channelData_300_nose": ("Pa", "Pa"),
    "channelData_300": ("Pa", "Pa"),
}

SENSITIVITIES_V_PER_PA: dict[str, float] = {
    # 'nc': 0.05,
    # 'PH': 0.05,
    # 'NC': 0.05,
}
PREAMP_GAIN: dict[str, float] = {"nc": 1.0, "PH": 1.0, "NC": 1.0}


def convert_to_pa(x: np.ndarray, units: str, *, channel_name: str = "unknown") -> np.ndarray:
    """Convert a pressure time series to Pa."""
    u = units.lower()
    x = np.asarray(x, float)
    if u == "pa":
        return x
    if u == "kpa":
        return x * 1e3
    if u == "mbar":
        return x * 100.0
    if u in ("v", "volt", "volts"):
        if channel_name not in SENSITIVITIES_V_PER_PA or SENSITIVITIES_V_PER_PA[channel_name] is None:
            raise ValueError(
                f"Sensitivity (V/Pa) for channel '{channel_name}' not provided; cannot convert V→Pa."
            )
        sens = float(SENSITIVITIES_V_PER_PA[channel_name])  # V/Pa
        gain = float(PREAMP_GAIN.get(channel_name, 1.0))
        return x / (gain * sens)
    raise ValueError(f"Unsupported units '{units}' for channel '{channel_name}'")


def load_mat_to_pa(path: str | Path, key: str, ch1_name: str, ch2_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load two channels and convert each to Pa using DEFAULT_UNITS[key]."""
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
# Spectral tools
# =============================================================================
@dataclass
class FRFResult:
    f: np.ndarray
    H: np.ndarray
    gamma2: np.ndarray
    Sxx: Optional[np.ndarray] = None
    Syy: Optional[np.ndarray] = None
    Sxy: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)


def _welch_triplet(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    *,
    npsg: int = NPERSEG,
    window: str = WINDOW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    nseg = int(min(npsg, x.size, y.size))
    if nseg < 8:
        raise ValueError(f"Signal too short for FRF: n={min(x.size, y.size)}")
    nov = int(min(npsg // 2, nseg // 2))
    w = get_window(window, nseg, fftbins=True)

    f, Sxx = welch(x, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)
    _, Syy = welch(y, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)
    # SciPy: csd(x, y) = E{ X * conj(Y) }
    _, Sxy = csd(x, y, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)
    return f, Sxx, Syy, Sxy


def compute_spec(fs: float, x: np.ndarray, npsg: int = NPERSEG) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD with sane defaults. Returns (f [Hz], Pxx [Pa^2/Hz])."""
    x = np.asarray(x, float)
    nseg = int(min(npsg, x.size))
    nov = nseg // 2
    w = get_window(WINDOW, nseg, fftbins=True)
    f, Pxx = welch(
        x, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend="constant", scaling="density"
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


# ---- FRF method registry -----------------------------------------------------
FRF_REGISTRY: Dict[str, Callable[..., FRFResult]] = {}


def register_frf(name: str) -> Callable[[Callable[..., FRFResult]], Callable[..., FRFResult]]:
    """Decorator to register new FRF backends by name."""
    def _decorator(fn: Callable[..., FRFResult]) -> Callable[..., FRFResult]:
        FRF_REGISTRY[name] = fn
        return fn
    return _decorator


def _call_frf_backend(
    frf_fn: Callable[..., FRFResult],
    x: np.ndarray, y: np.ndarray, fs: float, *,
    npsg: int = NPERSEG, window: str = WINDOW,
    noise_psd: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> FRFResult:
    params = inspect.signature(frf_fn).parameters
    kwargs = {"npsg": npsg, "window": window}
    if "noise_psd" in params:
        kwargs["noise_psd"] = noise_psd
    return frf_fn(x, y, fs, **kwargs)



@register_frf("h1")
def frf_h1(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    *,
    npsg: int = NPERSEG,
    window: str = WINDOW,
    noise_psd: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> FRFResult:
    """Standard H1: H = conj(Sxy)/Sxx."""
    f, Sxx, Syy, Sxy = _welch_triplet(x, y, fs, npsg=npsg, window=window)
    H = np.conj(Sxy) / Sxx
    gamma2 = np.clip((np.abs(Sxy) ** 2) / (Sxx * Syy), 0.0, 1.0)
    return FRFResult(f=f, H=H, gamma2=gamma2, Sxx=Sxx, Syy=Syy, Sxy=Sxy, meta={"backend": "h1"})


@register_frf("h1_psd")
def frf_h1_psd(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    *,
    npsg: int = NPERSEG,
    window: str = WINDOW,
    noise_psd: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> FRFResult:
    """H1 and PSDs; meta includes estimated number of averages m."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    nseg = int(min(npsg, x.size, y.size))
    nov = int(min(npsg // 2, nseg // 2))
    step = nseg - nov
    m = 1 + max(0, (min(x.size, y.size) - nseg) // step)
    f, Sxx, Syy, Sxy = _welch_triplet(x, y, fs, npsg=npsg, window=window)
    H = np.conj(Sxy) / Sxx
    gamma2 = np.clip((np.abs(Sxy) ** 2) / (Sxx * Syy), 0.0, 1.0)
    return FRFResult(f=f, H=H, gamma2=gamma2, Sxx=Sxx, Syy=Syy, Sxy=Sxy, meta={"backend": "h1_psd", "nseg": nseg, "noverlap": nov, "m": m})


def debias_H1_for_input_noise(
    H1: np.ndarray,
    Sxx_meas: np.ndarray,
    Snx: np.ndarray,
    *,
    beta_min: float = 0.3,
    eps: float = 1e-18,
) -> tuple[np.ndarray, np.ndarray]:
    """Debias H1 using input-noise PSD Snx (interpolated to Sxx grid)."""
    ratio = np.clip(Snx / np.maximum(Sxx_meas, eps), 0.0, 1.0)
    beta = 1.0 - ratio
    beta_safe = np.clip(beta, beta_min, 1.0)
    H_unbiased = H1 / beta_safe
    return H_unbiased, beta


@register_frf("h1_debiased")
def frf_h1_debiased(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    *,
    npsg: int = NPERSEG,
    window: str = WINDOW,
    noise_psd: Optional[Tuple[np.ndarray, np.ndarray]] = None,  # (f, Snx)
) -> FRFResult:
    """
    H1, then debias for input noise using Snx (noise PSD for x).
    If noise_psd is None, falls back to plain H1.
    """
    base = frf_h1_psd(x, y, fs, npsg=npsg, window=window)
    if noise_psd is None or base.Sxx is None:
        return base
    f_n, Snx = noise_psd
    Snx_i = np.interp(base.f, f_n, Snx, left=Snx[0], right=Snx[-1])
    H_unb, beta = debias_H1_for_input_noise(base.H, base.Sxx, Snx_i)
    meta = dict(base.meta)
    meta.update({"backend": "h1_debiased"})
    return FRFResult(f=base.f, H=H_unb, gamma2=base.gamma2, Sxx=base.Sxx, Syy=base.Syy, Sxy=base.Sxy, meta=meta)


def combine_anechoic_calibrations(
    f1, H1, g2_1,
    f2, H2, g2_2,
    *,
    gmin: float = 0.4,
    smooth_oct: float | None = 1 / 6,
    points_per_oct: int = 32,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse two anechoic FRF estimates into a single broadband anchor."""
    f1 = np.asarray(f1); f2 = np.asarray(f2)
    H1 = np.asarray(H1); H2 = np.asarray(H2)
    g2_1 = np.clip(np.asarray(g2_1), 0.0, 1.0); g2_2 = np.clip(np.asarray(g2_2), 0.0, 1.0)
    f = f1 if f1.size >= f2.size else f2
    H1i = _interp_complex(f1, H1, f)
    H2i = _interp_complex(f2, H2, f)
    g2_1i = np.interp(f, f1, g2_1, left=g2_1[0], right=g2_1[-1])
    g2_2i = np.interp(f, f2, g2_2, left=g2_2[0], right=g2_2[-1])

    def _weights(g2):
        g2c = np.clip(g2, 0.0, 1.0 - 1e-9)
        w = g2c / (1.0 - g2c + eps)
        return np.where(g2c >= gmin, w, 0.0)

    w1 = _weights(g2_1i); w2 = _weights(g2_2i)
    wsum = w1 + w2 + eps
    H_lab = (w1 * H1i + w2 * H2i) / wsum
    g2_lab = np.clip((w1 * g2_1i + w2 * g2_2i) / wsum, 0.0, 1.0)
    if smooth_oct is not None and smooth_oct > 0:
        H_lab = _complex_smooth_logfreq(f, H_lab, span_oct=smooth_oct, points_per_oct=points_per_oct)
    return f, H_lab, g2_lab


def combine_anechoic_calibrations_poweraware(
    f1, H1, g2_1, Sxx1, Snx1,
    f2, H2, g2_2, Sxx2, Snx2,
    *,
    gmin: float = 0.4,
    a: float = 1.0,  # coherence exponent
    b: float = 1.0,  # input-SNR exponent
    points_per_oct: int = 32,
    smooth_oct: float | None = 1/6,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Power-aware fusion (coherence × input SNR)."""
    f1 = np.asarray(f1); f2 = np.asarray(f2)
    H1 = np.asarray(H1); H2 = np.asarray(H2)
    g2_1 = np.clip(np.asarray(g2_1), 0.0, 1.0); g2_2 = np.clip(np.asarray(g2_2), 0.0, 1.0)
    f = f1 if f1.size >= f2.size else f2

    def _interp(z_src, f_src, f_tgt): return np.interp(f_tgt, f_src, z_src)
    H1i = _interp(np.real(H1), f1, f) + 1j * _interp(np.imag(H1), f1, f)
    H2i = _interp(np.real(H2), f2, f) + 1j * _interp(np.imag(H2), f2, f)
    g2_1i = _interp(g2_1, f1, f); g2_2i = _interp(g2_2, f2, f)
    Sxx1i = _interp(Sxx1, f1, f); Sxx2i = _interp(Sxx2, f2, f)
    Snx1i = _interp(Snx1, f1, f); Snx2i = _interp(Snx2, f2, f)

    wγ1 = np.where(g2_1i >= gmin, (g2_1i / np.maximum(1 - g2_1i, eps))**a, 0.0)
    wγ2 = np.where(g2_2i >= gmin, (g2_2i / np.maximum(1 - g2_2i, eps))**a, 0.0)
    wβ1 = np.maximum((Sxx1i - 0.0) / np.maximum(Snx1i, eps), 0.0)**b
    wβ2 = np.maximum((Sxx2i - 0.0) / np.maximum(Snx2i, eps), 0.0)**b

    w1 = wγ1 * wβ1
    w2 = wγ2 * wβ2
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
    """Correct the fused anechoic FRF with in-situ data where in-situ coherence is good."""
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


def apply_frf(
    x: np.ndarray,
    fs: float,
    f: np.ndarray,
    H: np.ndarray,
    demean: bool = True,
    zero_dc: bool = True,
) -> np.ndarray:
    """Apply a measured FRF H (x→y) to a time series x to synthesise y."""
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
    """Make a cascade of IIR notch filters (as SOS)."""
    sos_list: list[np.ndarray] = []
    for f0 in freqs:
        w0 = f0 / (fs / 2.0)
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
# Calibration pipeline (compute-only; no plotting)
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
        return self.psi_label.replace("psi", "psig")

    @property
    def title_suffix(self) -> str:
        return fr"($700\mu m$, {self.psi_label})"


# Fixed data roots (unchanged from your script)
BASE_FAR = Path("data/20251014")
BASE_CLOSE = Path("data/20251016")

# Output locations (same names so plotting finds them)
OUTPUT_DIR = Path("figures/tf_calib")  # (used by tf_plot.py)
CAL_DIR_FAR = BASE_FAR / "tf_calib" / "tf_data"
CAL_DIR_CLOSE = BASE_CLOSE / "tf_calib" / "tf_data"
CAL_DIR_COMB = BASE_CLOSE / "flow_data" / "tf_combined"


def _ensure_dirs() -> None:
    for p in (CAL_DIR_FAR, CAL_DIR_CLOSE, CAL_DIR_COMB, OUTPUT_DIR):
        p.mkdir(parents=True, exist_ok=True)


def _save_frf_triplet(out_dir: Path, tag: str, f: np.ndarray, H: np.ndarray, g2: np.ndarray) -> None:
    """Save FRF components with the same filenames used in your original code."""
    np.save(out_dir / f"f{tag}.npy", f)
    np.save(out_dir / f"H{tag}.npy", H)
    np.save(out_dir / f"gamma{tag}.npy", g2)


def _save_array(out_dir: Path, name: str, arr: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}.npy", arr)


def _load_lp_pair(fn_ph1: Path, fn_ph2: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load 'channelData_LP' from two .mat files (ph1-file, ph2-file): returns (ph1, nc1, ph2, nc2)."""
    d1 = loadmat(fn_ph1)
    d2 = loadmat(fn_ph2)
    ph1 = d1["channelData_LP"][:, 0].astype(float)
    nc1 = d1["channelData_LP"][:, 2].astype(float)
    ph2 = d2["channelData_LP"][:, 1].astype(float)
    nc2 = d2["channelData_LP"][:, 2].astype(float)
    return ph1, nc1, ph2, nc2


def _load_insitu_pair(fn: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load 'channelData_noflow' from a .mat, returning (ph1, ph2, nc)."""
    d = loadmat(fn)
    ph1 = d["channelData_noflow"][:, 0].astype(float)
    ph2 = d["channelData_noflow"][:, 1].astype(float)
    nc = d["channelData_noflow"][:, 2].astype(float)
    return ph1, ph2, nc


def _maybe_noise_psd_for_ph(x_insitu: Optional[np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Helper: use in-situ (no-flow) PH as an estimate of input noise PSD, if available."""
    if x_insitu is None:
        return None
    fN, Snx = compute_spec(FS, x_insitu, npsg=2**9)
    return fN, Snx


def calibrate_700_case(
    case: CalibCase,
    *,
    frf_backend: str = "h1",
    npsg: int = 2**9,
    poweraware: bool = False,
) -> None:
    """
    Compute-only calibration runner for 700 μm pinholes at a given pressure case.
    """
    _ensure_dirs()
    ic("T+ at f_cut:", case.Tplus_at_fcut)

    far_tf_root = BASE_FAR / "tf_calib"
    close_tf_root = BASE_CLOSE / "tf_calib"
    far_flow = BASE_FAR / "flow_data" / "far" / f"{case.psi_label}.mat"
    close_flow = BASE_CLOSE / "flow_data" / "close" / f"{case.psi_label}.mat"

    fn1_far = far_tf_root / f"{case.psi_label}_lp_16khz_ph1.mat"
    fn2_far = far_tf_root / f"{case.psi_label}_lp_16khz_ph2.mat"
    fn1_close = close_tf_root / f"{case.psi_label}_lp_16khz_ph1.mat"
    fn2_close = close_tf_root / f"{case.psi_label}_lp_16khz_ph2.mat"

    # Load anechoic (LP)
    ph1_far, nc1_far, ph2_far, nc2_far = _load_lp_pair(fn1_far, fn2_far)
    ph1_close, nc1_close, ph2_close, nc2_close = _load_lp_pair(fn1_close, fn2_close)

    # Load in-situ (no-flow)
    ph1_is_far, ph2_is_far, nc_is_far = _load_insitu_pair(far_flow)
    ph1_is_close, ph2_is_close, nc_is_close = _load_insitu_pair(close_flow)

    # Optional noise PSDs for debiasing backend
    noise_psd_ph1_far = _maybe_noise_psd_for_ph(ph1_is_far)
    noise_psd_ph2_far = _maybe_noise_psd_for_ph(ph2_is_far)
    noise_psd_ph1_close = _maybe_noise_psd_for_ph(ph1_is_close)
    noise_psd_ph2_close = _maybe_noise_psd_for_ph(ph2_is_close)

    # Resolve chosen FRF backend
    if frf_backend not in FRF_REGISTRY:
        raise KeyError(f"Unknown FRF backend '{frf_backend}'. Known: {list(FRF_REGISTRY)}")
    frf_fn = FRF_REGISTRY[frf_backend]

    # Anechoic FRFs (PH1, PH2 × far/close)
    r1_far   = _call_frf_backend(frf_fn, ph1_far,   nc1_far,   FS, npsg=npsg, window=WINDOW, noise_psd=noise_psd_ph1_far)
    r2_far   = _call_frf_backend(frf_fn, ph2_far,   nc2_far,   FS, npsg=npsg, window=WINDOW, noise_psd=noise_psd_ph2_far)
    r1_close = _call_frf_backend(frf_fn, ph1_close, nc1_close, FS, npsg=npsg, window=WINDOW, noise_psd=noise_psd_ph1_close)
    r2_close = _call_frf_backend(frf_fn, ph2_close, nc2_close, FS, npsg=npsg, window=WINDOW, noise_psd=noise_psd_ph2_close)


    _save_frf_triplet(CAL_DIR_FAR, f"1_700_{case.psi_g}", r1_far.f, r1_far.H, r1_far.gamma2)
    _save_frf_triplet(CAL_DIR_FAR, f"2_700_{case.psi_g}", r2_far.f, r2_far.H, r2_far.gamma2)
    _save_frf_triplet(CAL_DIR_CLOSE, f"1_700_{case.psi_g}", r1_close.f, r1_close.H, r1_close.gamma2)
    _save_frf_triplet(CAL_DIR_CLOSE, f"2_700_{case.psi_g}", r2_close.f, r2_close.H, r2_close.gamma2)

    # In-situ FRFs (PH1, PH2 × far/close; all vs nc)
    r1_is_far = frf_fn(ph1_is_far, nc_is_far, FS, npsg=npsg, window=WINDOW)
    r2_is_far = frf_fn(ph2_is_far, nc_is_far, FS, npsg=npsg, window=WINDOW)
    r1_is_close = frf_fn(ph1_is_close, nc_is_close, FS, npsg=npsg, window=WINDOW)
    r2_is_close = frf_fn(ph2_is_close, nc_is_close, FS, npsg=npsg, window=WINDOW)

    _save_frf_triplet(CAL_DIR_FAR, f"1_700_{case.psi_label}_is", r1_is_far.f, r1_is_far.H, r1_is_far.gamma2)
    _save_frf_triplet(CAL_DIR_FAR, f"2_700_{case.psi_label}_is", r2_is_far.f, r2_is_far.H, r2_is_far.gamma2)
    _save_frf_triplet(CAL_DIR_CLOSE, f"1_700_{case.psi_label}_is", r1_is_close.f, r1_is_close.H, r1_is_close.gamma2)
    _save_frf_triplet(CAL_DIR_CLOSE, f"2_700_{case.psi_label}_is", r2_is_close.f, r2_is_close.H, r2_is_close.gamma2)

    # Fuse anechoic FRFs (PH1 and PH2 separately)
    if poweraware and (r1_far.Sxx is not None) and (r1_close.Sxx is not None):
        # Build crude noise PSDs from in-situ PH channels (already computed)
        f1_lab, H1_lab, g1_lab = combine_anechoic_calibrations_poweraware(
            r1_far.f, r1_far.H, r1_far.gamma2, r1_far.Sxx, np.interp(r1_far.f, *noise_psd_ph1_far) if noise_psd_ph1_far else np.ones_like(r1_far.f),
            r1_close.f, r1_close.H, r1_close.gamma2, r1_close.Sxx, np.interp(r1_close.f, *noise_psd_ph1_close) if noise_psd_ph1_close else np.ones_like(r1_close.f)
        )
        f2_lab, H2_lab, g2_lab = combine_anechoic_calibrations_poweraware(
            r2_far.f, r2_far.H, r2_far.gamma2, r2_far.Sxx, np.interp(r2_far.f, *noise_psd_ph2_far) if noise_psd_ph2_far else np.ones_like(r2_far.f),
            r2_close.f, r2_close.H, r2_close.gamma2, r2_close.Sxx, np.interp(r2_close.f, *noise_psd_ph2_close) if noise_psd_ph2_close else np.ones_like(r2_close.f)
        )
    else:
        f1_lab, H1_lab, g1_lab = combine_anechoic_calibrations(r1_far.f, r1_far.H, r1_far.gamma2, r1_close.f, r1_close.H, r1_close.gamma2)
        f2_lab, H2_lab, g2_lab = combine_anechoic_calibrations(r2_far.f, r2_far.H, r2_far.gamma2, r2_close.f, r2_close.H, r2_close.gamma2)

    # Save anechoic fused (now also save γ² for plotting)
    _save_array(CAL_DIR_COMB, f"700_{case.psi_label}_fused_anechoic_f1", f1_lab)
    _save_array(CAL_DIR_COMB, f"700_{case.psi_label}_fused_anechoic_H1", H1_lab)
    _save_array(CAL_DIR_COMB, f"700_{case.psi_label}_fused_anechoic_gamma1", g1_lab)
    _save_array(CAL_DIR_COMB, f"700_{case.psi_label}_fused_anechoic_f2", f2_lab)
    _save_array(CAL_DIR_COMB, f"700_{case.psi_label}_fused_anechoic_H2", H2_lab)
    _save_array(CAL_DIR_COMB, f"700_{case.psi_label}_fused_anechoic_gamma2", g2_lab)

    # Incorporate in-situ (PH1 & PH2 each, far then close)
    f1_hat, H1_hat, C1 = incorporate_insitu_calibration(f1_lab, H1_lab, g1_lab, r1_is_far.f, r1_is_far.H, r1_is_far.gamma2)
    f1_hat, H1_hat, C1 = incorporate_insitu_calibration(f1_hat, H1_hat, g1_lab, r1_is_close.f, r1_is_close.H, r1_is_close.gamma2)
    f2_hat, H2_hat, C2 = incorporate_insitu_calibration(f2_lab, H2_lab, g2_lab, r2_is_far.f, r2_is_far.H, r2_is_far.gamma2)
    f2_hat, H2_hat, C2 = incorporate_insitu_calibration(f2_hat, H2_hat, g2_lab, r2_is_close.f, r2_is_close.H, r2_is_close.gamma2)

    # Save in-situ fused (and the blend weights for diagnostics)
    _save_array(CAL_DIR_COMB, f"700_{case.psi_g}_fused_insitu_f1", f1_hat)
    _save_array(CAL_DIR_COMB, f"700_{case.psi_g}_fused_insitu_H1", H1_hat)
    _save_array(CAL_DIR_COMB, f"700_{case.psi_g}_fused_insitu_C1", C1)
    _save_array(CAL_DIR_COMB, f"700_{case.psi_g}_fused_insitu_f2", f2_hat)
    _save_array(CAL_DIR_COMB, f"700_{case.psi_g}_fused_insitu_H2", H2_hat)
    _save_array(CAL_DIR_COMB, f"700_{case.psi_g}_fused_insitu_C2", C2)


def run_all_calibrations(
    *,
    cases: Optional[Iterable[str]] = None,
    frf_backend: str = "h1",
    npsg: int = 2**9,
    poweraware: bool = False,
) -> None:
    """
    Compute all requested cases.
    """
    all_cases = [
        CalibCase("0psi",   u_tau=0.58, nu_utau=27e-6,  f_cut=2_100.0),
        CalibCase("50psi",  u_tau=0.47, nu_utau=7.5e-6, f_cut=4_700.0),
        CalibCase("100psi", u_tau=0.52, nu_utau=3.7e-6, f_cut=14_100.0),
    ]
    if cases is not None:
        wanted = set(cases)
        all_cases = [c for c in all_cases if c.psi_label in wanted]

    for c in all_cases:
        ic(f"Computing case: {c.psi_label} with backend={frf_backend}, poweraware={poweraware}")
        calibrate_700_case(c, frf_backend=frf_backend, npsg=npsg, poweraware=poweraware)


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute-only calibration pipeline (no plotting).")
    parser.add_argument(
        "--cases",
        type=str,
        default="0psi,50psi,100psi",
        help="Comma-separated list (e.g. 0psi,100psi) or 'all'",
    )
    parser.add_argument(
        "--frf-backend",
        type=str,
        default="h1",
        help=f"FRF backend to use. See --list-frf. Default: h1",
    )
    parser.add_argument(
        "--npsg",
        type=int,
        default=2**9,
        help="nperseg for Welch/CSD (default: 512).",
    )
    parser.add_argument(
        "--poweraware",
        action="store_true",
        help="Use power-aware coherence×SNR fusion for anechoic combination.",
    )
    parser.add_argument(
        "--list-frf",
        action="store_true",
        help="List available FRF backends and exit.",
    )
    args = parser.parse_args()

    if args.list_frf:
        print("Available FRF backends:")
        for k in FRF_REGISTRY:
            print(f"  - {k}")
        raise SystemExit(0)

    cases = None if args.cases.strip().lower() == "all" else [s.strip() for s in args.cases.split(",") if s.strip()]
    run_all_calibrations(cases=cases, frf_backend=args.frf_backend, npsg=args.npsg, poweraware=args.poweraware)
