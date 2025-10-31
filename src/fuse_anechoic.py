import numpy as np

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
