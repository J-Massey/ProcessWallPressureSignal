# phys_tf_calib.py  — pressure-dependent physical FRFs (PH→NC) from semi-anechoic runs
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Dict
from icecream import ic

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.signal import csd, get_window, welch

from clean_raw_data import volts_to_pa  # expects volts→Pa conversion by channel

PSI_TO_PA = 6_894.76  # Pa per psi

# ----------------- complex helpers -----------------
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
    span_oct: float = 1/6,
    ppo: int = 48,
    eps: float = 1e-20
) -> np.ndarray:
    """
    Complex moving-average smoothing on a log-frequency grid (constant span in octaves).
    Real/imag smoothed separately to avoid phase wrap; smoothing is gentle.
    """
    f = np.asarray(f); z = np.asarray(z)
    assert f.ndim == 1 and z.ndim == 1 and f.size == z.size
    pos = f > 0
    if span_oct <= 0 or pos.sum() < 8:
        return z.copy()
    fpos, zpos = f[pos], z[pos]
    f_lo = max(fpos[0], eps); f_hi = fpos[-1]
    n_oct = np.log2(f_hi / f_lo)
    n_pts = max(int(np.ceil(n_oct * ppo)), 8)
    flog = np.linspace(np.log2(f_lo), np.log2(f_hi), n_pts)
    fgrid = 2.0 ** flog
    zgrid = _interp_for_smooth = _interp_complex(fpos, zpos, fgrid)

    # simple boxcar on real/imag in log domain
    wlen = max(int(round(span_high := span_oct * ppo)), 1)
    if wlen % 2 == 0:
        wlen += 1
    ker = np.ones(wlen) / wlen
    re_s = np.convolve(_interp_for_smooth.real, ker, mode="same")
    im_s = np.convolve(_interp_for_smooth.imag, ker, mode="same")
    z_s_grid = re_s + 1j * im_s
    # back-interpolate to original positive freq grid
    z_sm_pos = _interp_complex(fgrid, z_s_grid, fpos)
    z_out = z.copy()
    z_out[pos] = z_sm_pos
    return z_out

# ----------------- FRF (H1) -----------------
def _estimate_frf(
    x: np.ndarray,
    y: np.ndarray,
    *,
    fs: float,
    window: str = "hann",
    nperseg: int = 2**10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    H1 frequency response and coherence, **PH→NC orientation**:
      x:= PH (input), y:= NC (output)
      SciPy: csd(x,y) = E{ X * conj(Y) }, so H1(x→y) = conj(S_xy) / S_xx
    Returns: (f [Hz], H (complex), gamma2 [0..1])
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    w = get_window(window, nperseg, fftbins=True)

    f, Sxx = welch(x, fs=fs, window=w, nperseg=nperseg, detrend='constant')
    _, Syy = welch(y, fs=fs, window=w, nperseg=nperseg, detrend='constant')
    _, Sxy = csd(x, y, fs=fs, window=w, nperseg=nperseg, detrend='constant')

    H = np.conjugate(Sxy) / Sxx   # H1 for x→y
    g2 = np.clip((np.abs(Sxy) ** 2) / (Sxx * Syy), 0.0, 1.0)
    return f, H, g2

# ----------------- FRF fusion from dual-position runs -----------------
def combine_anechoic_calibrations(
    f1, H1, g2_1,
    f2, H2, g2_2,
    *,
    gmin: float = 0.4,
    smooth_oct: float | None = 1/6,
    points_per_oct: int = 32,
    eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Coherence-weighted complex fusion of two FRFs onto a **common grid**, with **optional**
    gentle log-frequency smoothing to suppress ripple while preserving true features.
    """
    f1 = np.asarray(f1); f2 = np.asarray(f2)
    H1 = np.asarray(H1); H2 = np.asarray(H2)
    g2_1 = np.clip(np.asarray(g2_1), 0, 1); g2_2 = np.clip(np.asarray(g2_2), 0, 1)

    # choose the denser grid as target
    f_tgt = f1 if f1.size >= f2.size else f2
    H1_i = _interp_complex(f1, H1, f_tgt)
    H2_i = _interp_complex(f2, H2, f_tgt)
    g1_i = np.interp(f_tgt, f1, g2_1, left=g2_1[0], right=g2_1[-1])
    g2_i = np.interp(f_tgt, f2, g2_2, left=g2_2[0], right=g2_2[-1])

    def w_from_gamma2(g):
        g = np.clip(g, 0.0, 1.0 - 1e-9)
        w = g / (1.0 - g + eps)   # larger weight for higher coherence
        return np.where(g >= gmin, w, 0.0)

    w1 = w_from_gamma2(g1_i); w2 = w_from_gamma2(g2_i)
    wsum = w1 + w2 + eps
    H_fused = (w1 * H1_i + w2 * H2_i) / wsum
    g2_fused = np.clip((w1 * g1_i + w2 * g2_i) / wsum, 0.0, 1.0)

    if smooth_oct and smooth_oct > 0:
        H_fused = _complex_smooth_logfreq(f_tgt, H_fused, span_oct=smooth_oct, ppo=points_per_oct, eps=eps)

    return f_tgt, H_fused, g2_fused

# ----------------- Sensitivity compensation (dB/kPa) -----------------
def correct_pressure_sensitivity(p_pa: np.ndarray, psig: float, alpha_db_per_kpa: float = 0.01) -> np.ndarray:
    """
    Compensate the pressure-sensor sensitivity drift vs. gauge pressure.
    If sensitivity drops by ~0.01 dB/kPa, multiply the *pressure* signal by
      gain = 10^( + alpha_dB_per_kPa * p_kPa / 20 )
    so the corrected Pa reflects the same effective sensitivity at all psig.
    """
    p_corr = np.asarray(p_pa, float) * 10.0 ** ( (float(psig) * PSI_TO_PA / 1000.0) * (alpha_db_per_kpa / 20.0) )
    return p_corr

# ----------------- Main API: save per-pressure physical FRFs ---------- 
def save_calibs(
    pressures: Iterable[float | int | str],
    *,
    calib_base: str | Path,
    fs: float,
    f_cuts: Sequence[float],
    gmin: float = 0.4,
    smooth_oct: float = 1/6,
    points_per_oct: int = 32,
    eps: float = 1e-12
) -> None:
    """
    Build PH→NC H₁ FRF for each pressure from dual-position (…_1, …_2) semi-anechoic runs:
      - Convert both channels to Pa (volts_to_pa) and compensate mic sensitivity vs psig.
      - Estimate H1 with x=PH, y=NC, using welch/csd: H = conj(Sxy)/Sxx (SciPy's definition).
      - Fuse PH1 and PH2 FRFs on a **common frequency grid**, coherence-weighted, optionally smoothed.
      - Save **f_fused**, **H_fused (complex)**, optional raw H1/H2 and fused γ², and numeric psig.
    """
    base = Path(calib_base)
    pressures = [int(p) for p in pressures]
    if len(f_cuts) != len(pressures):
        raise ValueError("f_cuts length must match number of pressures")

    for p_si, fcut in zip(pressures, f_cuts):
        psig = float(p_si)
        # ---- run 1: PH1→NC
        m1 = loadmat(base / f"calib_{p_si}psig_1.mat")
        ph1_v, _, nc_v, *_ = m1["channelData_WN"].T
        ph1_pa = volts_to_pa(ph1_v, "PH1", fcut)
        nc1_pa = volts_to_pa(nc_v,  "NC",  fcut)
        # compensate sensor sensitivity vs psig (amplitude gain)
        ph1_pa = correct_pressure_sensitivity(ph1_pa, psig)
        nc1_pa =  correct_pressure_sensitivity(nc1_pa,  psig)
        f1, H1, g2_1 = _estimate_frf(ph1_pa, nc1_pa, fs=fs)  # x=PH1, y=NC  ⇒ H:=H_{PH→NC}

        # ---- run 2: PH2→NC
        m2 = loadmat(base / f"calib_{p_si}psig_2.mat")
        _, ph2_v, nc_v2, *_ = m2["channelData_WN"].T
        ph2_pa = volts_to_pa(ph2_v, "PH2", fcut)
        nc2_pa = volts_to_pa(nc_v2,  "NC",  fcut)
        ph2_pa = correct_pressure_sensitivity(ph2_pa, psig)
        nc2_pa =  correct_pressure_sensitivity(nc2_pa,  psig)
        f2, H2, g2_2 = _estimate_frf(ph2_pa, nc2_pa, fs=fs)  # x=PH2 → y=NC

        # ---- fuse to physical anchor on a **common grid** and optionally smooth
        f_fused, H_fused, g2_fused = combine_anechoic_calibrations(
            f1, H1, g2_1, f2, H2, g2_2,
            gmin=gmin, smooth_oct=smooth_oct, points_per_oct=points_per_oct, eps=eps
        )

        # ---- persist (note: save the **fused** frequency vector)
        out = Path(calib_base) / f"calibs_{p_si}.h5"   # or f"calibs_{p_si}psig.h5"
        with h5py.File(out, "w") as hf:
            hf.create_dataset("frequencies", data=f_fused)   # <— use fused grid
            hf.create_dataset("H1", data=H1)                 # optional raw
            hf.create_dataset("H2", data=H2)
            hf.create_dataset("H_fused", data=H_fused)       # complex, PH→NC
            hf.create_dataset("gamma2_fused", data=g2_fused)
            hf.attrs["psig"] = psig
            hf.attrs["orientation"] = "H = NC/PH (H1 = conj(Sxy)/Sxx with x=PH, y=NC)"
            hf.attrs["fs_Hz"] = fs
            hf.attrs["fcut_Hz"] = fcut

        print(f"[ok] {p_si:>3} psig → {out}")


def save_NC_NKD_calibs(
    pressures: Iterable[float | int | str],
    *,
    calib_base: str | Path,
    fs: float,
    f_cuts: Sequence[float],
) -> None:
    """
    Build PH→NC H₁ FRF for each pressure from dual-position (…_1, …_2) semi-anechoic runs:
      - Convert both channels to Pa (volts_to_pa) and compensate mic sensitivity vs psig.
      - Estimate H1 with x=PH, y=NC, using welch/csd: H = conj(Sxy)/Sxx (SciPy's definition).
      - Fuse PH1 and PH2 FRFs on a **common frequency grid**, coherence-weighted, optionally smoothed.
      - Save **f_fused**, **H_fused (complex)**, optional raw H1/H2 and fused γ², and numeric psig.
    """
    base = Path(calib_base)
    pressures = [int(p) for p in pressures]
    if len(f_cuts) != len(pressures):
        raise ValueError("f_cuts length must match number of pressures")

    for p_si, fcut in zip(pressures, f_cuts):
        psig = float(p_si)
        # ---- run 1: PH1→NC
        m1 = loadmat(base / f"{p_si}psig/nkd-ns_nofacilitynoise.mat")
        ic(m1.keys())
        if p_si == 100:
            nkd, nc = m1["channelData_nofacitynoise"].T
        else:
            nkd, nc = m1["channelData"].T

        f1, H1, g2_1 = _estimate_frf(nc, nkd, fs=fs)  # x=PH1, y=NC  ⇒ H:=H_{PH→NC}
        # ---- persist (note: save the **fused** frequency vector)
        out = Path(calib_base) / f"calibs_{p_si}.h5"   # or f"calibs_{p_si}psig.h5"
        with h5py.File(out, "w") as hf:
            hf.create_dataset("frequencies", data=f1)   # <— use fused grid
            hf.create_dataset("H_fused", data=H1)       # complex, PH→NC
            hf.create_dataset("gamma2_fused", data=g2_1)
            hf.attrs["psig"] = psig
            hf.attrs["orientation"] = "H = NC/PH (H1 = conj(Sxy)/Sxx with x=nc, y=nkd)"
            hf.attrs["fs_Hz"] = fs
            hf.attrs["fcut_Hz"] = fcut
        print(f"[ok] {p_si:>3} psig → {out}")

# --------------- example CLI ---------------------------------------------------
if __name__ == "__main__":
    pressures = [0, 50, 100]                       # psig
    f_cuts    = [1200.0, 4000.0, 10000.0]          # per-label anti-alias lowpass in Hz
    # save_calibs(pressures, calib_base="data/20250930", fs=50_000.0, f_cuts=f_cuts)
    save_NC_NKD_calibs(pressures, calib_base="data/20250930", fs=50_000.0, f_cuts=f_cuts)
