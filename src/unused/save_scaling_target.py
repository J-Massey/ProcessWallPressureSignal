from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
from scipy.signal import get_window, welch

from models import bl_model
from matplotlib import pyplot as plt

WINDOW = "hann"
NPERSEG = 2**10

DEFAULT_LABELS: tuple[str, ...] = ("0psig", "50psig", "100psig")
DEFAULT_FILES: tuple[str, ...] = (
    "0psig_close_cleaned.h5",
    "50psig_close_cleaned.h5",
    "100psig_close_cleaned.h5",
)
DEFAULT_P_GAUGE: tuple[float, ...] = (0.0, 50.0, 100.0)
DEFAULT_TEMP_DEG_C: tuple[float, ...] = (18.0, 20.0, 22.0)


def compute_spec(
    fs: float,
    x: np.ndarray,
    *,
    nperseg: int = NPERSEG,
    window: str = WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD with shared defaults used across the workflow."""
    x = np.asarray(x, float)
    nseg = int(min(nperseg, x.size))
    if nseg < 8:
        raise ValueError(f"Signal too short for PSD estimate: n={x.size}")
    nov = nseg // 2
    w = get_window(window, nseg, fftbins=True)
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


def save_scaling_target(
    *,
    fs: float,
    cleaned_base: str | Path,
    target_base: str | Path,
    labels: Sequence[str] = DEFAULT_LABELS,
    filenames: Sequence[str] = DEFAULT_FILES,
    p_gauge: Sequence[float] = DEFAULT_P_GAUGE,
    temp_deg_c: Sequence[float] = DEFAULT_TEMP_DEG_C,
) -> None:
    """
    Compute and persist the model/data scaling ratios for each cleaned dataset.
    """
    cleaned_base = Path(cleaned_base)
    target_base = Path(target_base)

    if not (len(labels) == len(filenames) == len(p_gauge) == len(temp_deg_c)):
        raise ValueError("labels, filenames, p_gauge, and temp_deg_c must be the same length")

    plt.subplots(1, 1, figsize=(7, 3), tight_layout=True)

    for label, fn, psig, _temp_c in zip(labels, filenames, p_gauge, temp_deg_c):
        with h5py.File(cleaned_base / fn, "r") as hf:
            ph1_clean = hf["ph1_clean"][:]
            ph2_clean = hf["ph2_clean"][:]
            u_tau = float(hf.attrs["u_tau"])
            nu = float(hf.attrs["nu"])
            rho = float(hf.attrs["rho"])
            Re_tau = hf.attrs["Re_tau"]
            cf_2 = hf.attrs["cf_2"]

        f_clean, Pyy_ph1_clean = compute_spec(fs, ph1_clean)
        _, Pyy_ph2_clean = compute_spec(fs, ph2_clean)
        T_plus = (u_tau**2 / nu) / f_clean

        g1_b, g2_b, rv_b = bl_model(T_plus, Re_tau, cf_2)
        bl_fphipp_plus = rv_b * (g1_b + g2_b)

        mask = f_clean < 1_000
        bl_fphipp_plus = bl_fphipp_plus[mask]

        f_clean_tf, Pyy_ph1_clean_tf = compute_spec(fs, ph1_clean)
        _, Pyy_ph2_clean_tf = compute_spec(fs, ph2_clean)

        data_fphipp_plus1_tf = (f_clean_tf * Pyy_ph1_clean_tf) / (rho**2 * u_tau**4)
        data_fphipp_plus2_tf = (f_clean_tf * Pyy_ph2_clean_tf) / (rho**2 * u_tau**4)

        mask_tf = f_clean_tf < 1_000
        f_clean_final = f_clean_tf[mask_tf]
        data_fphipp_plus1_tf_m = data_fphipp_plus1_tf[mask_tf]
        data_fphipp_plus2_tf_m = data_fphipp_plus2_tf[mask_tf]

        model_data_ratio1 = np.sqrt(data_fphipp_plus1_tf_m / bl_fphipp_plus)
        model_data_ratio2 = np.sqrt(data_fphipp_plus2_tf_m / bl_fphipp_plus)

        model_ratio_avg = (model_data_ratio1 + model_data_ratio2) / 2

        with h5py.File(target_base / f"lumped_scaling_{label}.h5", "w") as hf:
            hf.create_dataset("frequencies", data=f_clean_final)
            hf.create_dataset("scaling_ratio", data=model_ratio_avg)
            hf.attrs["rho"] = rho
            hf.attrs["u_tau"] = u_tau
            hf.attrs["nu"] = nu
            hf.attrs["psig"] = psig
