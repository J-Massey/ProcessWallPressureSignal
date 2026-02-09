from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import csd, get_window, welch


def _interp_complex(f_src: np.ndarray, z_src: np.ndarray, f_tgt: np.ndarray) -> np.ndarray:
    """Linear interpolate complex spectrum to target frequencies (separate real/imag)."""
    z_src = np.asarray(z_src)
    re = np.interp(f_tgt, f_src, z_src.real, left=z_src.real[0], right=z_src.real[-1])
    im = np.interp(f_tgt, f_src, z_src.imag, left=z_src.imag[0], right=z_src.imag[-1])
    return re + 1j * im


def _complex_smooth_logfreq(
    f: np.ndarray,
    z: np.ndarray,
    *,
    span_oct: float = 1 / 6,
    ppo: int = 48,
    eps: float = 1e-20,
) -> np.ndarray:
    """
    Complex moving-average smoothing on a log-frequency grid (constant span in octaves).
    Real/imag smoothed separately to avoid phase wrap; smoothing is gentle.
    """
    f = np.asarray(f)
    z = np.asarray(z)
    if f.ndim != 1 or z.ndim != 1 or f.size != z.size:
        raise ValueError("f and z must be 1D arrays with matching lengths")
    pos = f > 0
    if span_oct <= 0 or pos.sum() < 8:
        return z.copy()
    fpos, zpos = f[pos], z[pos]
    f_lo = max(fpos[0], eps)
    f_hi = fpos[-1]
    n_oct = np.log2(f_hi / f_lo)
    n_pts = max(int(np.ceil(n_oct * ppo)), 8)
    flog = np.linspace(np.log2(f_lo), np.log2(f_hi), n_pts)
    fgrid = 2.0 ** flog
    zgrid = _interp_complex(fpos, zpos, fgrid)

    wlen = max(int(round(span_oct * ppo)), 1)
    if wlen % 2 == 0:
        wlen += 1
    ker = np.ones(wlen) / wlen
    re_s = np.convolve(zgrid.real, ker, mode="same")
    im_s = np.convolve(zgrid.imag, ker, mode="same")
    z_s_grid = re_s + 1j * im_s

    z_sm_pos = _interp_complex(fgrid, z_s_grid, fpos)
    z_out = z.copy()
    z_out[pos] = z_sm_pos
    return z_out


def estimate_frf(
    x: np.ndarray,
    y: np.ndarray,
    *,
    fs: float,
    window: str = "hann",
    nperseg: int = 2**10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    H1 frequency response and coherence, **x to y orientation**:
      SciPy: csd(x,y) = E{ X * conj(Y) }, so H1(x to y) = conj(S_xy) / S_xx
    Returns: (f [Hz], H (complex), gamma2 [0..1])
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = get_window(window, nperseg, fftbins=True)

    f, Sxx = welch(x, fs=fs, window=w, nperseg=nperseg, detrend="constant")
    _, Syy = welch(y, fs=fs, window=w, nperseg=nperseg, detrend="constant")
    _, Sxy = csd(x, y, fs=fs, window=w, nperseg=nperseg, detrend="constant")

    H = np.conjugate(Sxy) / Sxx
    g2 = np.clip((np.abs(Sxy) ** 2) / (Sxx * Syy), 0.0, 1.0)
    return f, H, g2


def combine_anechoic_calibrations(
    f1: np.ndarray,
    H1: np.ndarray,
    g2_1: np.ndarray,
    f2: np.ndarray,
    H2: np.ndarray,
    g2_2: np.ndarray,
    *,
    gmin: float = 0.4,
    smooth_oct: float | None = 1 / 6,
    points_per_oct: int = 32,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Coherence-weighted complex fusion of two FRFs onto a **common grid**, with optional
    gentle log-frequency smoothing to suppress ripple while preserving true features.
    """
    f1 = np.asarray(f1)
    f2 = np.asarray(f2)
    H1 = np.asarray(H1)
    H2 = np.asarray(H2)
    g2_1 = np.clip(np.asarray(g2_1), 0, 1)
    g2_2 = np.clip(np.asarray(g2_2), 0, 1)

    f_tgt = f1 if f1.size >= f2.size else f2
    H1_i = _interp_complex(f1, H1, f_tgt)
    H2_i = _interp_complex(f2, H2, f_tgt)
    g1_i = np.interp(f_tgt, f1, g2_1, left=g2_1[0], right=g2_1[-1])
    g2_i = np.interp(f_tgt, f2, g2_2, left=g2_2[0], right=g2_2[-1])

    def w_from_gamma2(g: np.ndarray) -> np.ndarray:
        g = np.clip(g, 0.0, 1.0 - 1e-9)
        w = g / (1.0 - g + eps)
        return np.where(g >= gmin, w, 0.0)

    w1 = w_from_gamma2(g1_i)
    w2 = w_from_gamma2(g2_i)
    wsum = w1 + w2 + eps
    H_fused = (w1 * H1_i + w2 * H2_i) / wsum
    g2_fused = np.clip((w1 * g1_i + w2 * g2_i) / wsum, 0.0, 1.0)

    if smooth_oct and smooth_oct > 0:
        H_fused = _complex_smooth_logfreq(
            f_tgt, H_fused, span_oct=smooth_oct, ppo=points_per_oct, eps=eps
        )

    return f_tgt, H_fused, g2_fused
