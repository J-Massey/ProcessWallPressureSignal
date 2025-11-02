from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.signal import csd, get_window, welch

from clean_raw_data import volts_to_pa
from fuse_anechoic import combine_anechoic_calibrations


def _estimate_frf(
    x: np.ndarray,
    y: np.ndarray,
    *,
    fs: float,
    window: str = "hann",
    nperseg: int = 2**10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the H1 frequency response function and coherence."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    nseg = int(min(nperseg, x.size, y.size))
    if nseg < 8:
        raise ValueError(f"Signal too short for FRF: n={min(x.size, y.size)}")
    nov = int(min(nperseg // 2, nseg // 2))
    w = get_window(window, nseg, fftbins=True)

    f, Sxx = welch(x, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)
    _, Syy = welch(y, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)
    _, Sxy = csd(x, y, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)

    H = np.conj(Sxy) / Sxx
    gamma2 = (np.abs(Sxy) ** 2) / (Sxx * Syy)
    gamma2 = np.clip(gamma2.real, 0.0, 1.0)
    return f, H, gamma2


def save_calibs(
    pressures: Iterable[str],
    *,
    calib_base: str | Path,
    fs: float,
    f_cuts: Sequence[float],
    gmin: float = 0.4,
    smooth_oct: float = 1 / 6,
    points_per_oct: int = 32,
    eps: float = 1e-12,
) -> None:
    """
    Fuse dual-position calibrations into a single FRF per pressure and persist to HDF5.
    """
    base = Path(calib_base)
    pressures = list(pressures)
    if len(f_cuts) < len(pressures):
        raise ValueError("Insufficient number of f_cuts for the provided pressures")

    for idx, pressure in enumerate(pressures):
        cutoff = f_cuts[idx]

        dat = loadmat(base / f"calib_{pressure}_1.mat")
        ph1, ph2, nc, _ = dat["channelData_WN"].T
        nc_pa = volts_to_pa(nc, "NC", cutoff)
        ph1_pa = volts_to_pa(ph1, "PH1", cutoff)
        f1, H1, g2_1 = _estimate_frf(ph1_pa, nc_pa, fs=fs)

        dat = loadmat(base / f"calib_{pressure}_2.mat")
        ph1, ph2, nc, _ = dat["channelData_WN"].T
        nc_pa = volts_to_pa(nc, "NC", cutoff)
        ph2_pa = volts_to_pa(ph2, "PH2", cutoff)
        f2, H2, g2_2 = _estimate_frf(ph2_pa, nc_pa, fs=fs)

        f_fused, H_fused, _ = combine_anechoic_calibrations(
            f1,
            H1,
            g2_1,
            f2,
            H2,
            g2_2,
            gmin=gmin,
            smooth_oct=smooth_oct,
            points_per_oct=points_per_oct,
            eps=eps,
        )

        with h5py.File(base / f"calibs_{pressure}.h5", "w") as hf:
            hf.create_dataset("frequencies", data=f1)
            hf.create_dataset("H1", data=H1)
            hf.create_dataset("H2", data=H2)
            hf.create_dataset("H_fused", data=H_fused)
            hf.attrs["psig"] = pressure
