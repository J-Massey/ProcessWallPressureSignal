# tf_compute.py
from __future__ import annotations

import numpy as np
import h5py
from scipy.io import loadmat
from scipy.signal import welch, csd, get_window

from icecream import ic

from models import bl_model
from clean_raw_data import volts_to_pa, air_props_from_gauge
from fuse_anechoic import combine_anechoic_calibrations

# =============================================================================
# Constants & styling (exported so tf_plot.py can import them)
# =============================================================================
FS: float = 50_000.0
NPERSEG: int = 2**10
WINDOW: str = "hann"

# Colors (exported for plotting)
PH1_COLOR = "#c76713"  # orange
PH2_COLOR = "#9fda16"  # green-ish
NC_COLOR = "#2ca02c"   # matplotlib default green (kept for reference)

# --- constants (keep once, top of file) ---
R = 287.05        # J/kg/K
PSI_TO_PA = 6_894.76
P_ATM = 101_325.0
DELTA = 0.035  # m, bl-height of 'channel'
TDEG = [18, 20, 22]

# =============================================================================
# Units & optional conversions (kept for compatibility with other workflows)
# =============================================================================

SENSITIVITIES_V_PER_PA: dict[str, float] = {
    'nc': 50e-3,
    'PH1': 50e-3,
    'PH2': 50e-3,
    'NC': 50e-3,
}
PREAMP_GAIN: dict[str, float] = {"nc": 1.0, "PH1": 1.0, "PH2": 1.0, "NC": 1.0}
CLEANED_BASE = "data/final_cleaned/"
TARGET_BASE = "data/final_target/"


def estimate_frf(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    window: str = WINDOW,
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

def save_scaling_target():
    fns = [
        '0psig_close_cleaned.h5',
        '0psig_far_cleaned.h5',
        '50psig_close_cleaned.h5',
        '50psig_far_cleaned.h5',
        '100psig_close_cleaned.h5',
        '100psig_far_cleaned.h5',
        ]
    labels = [
        '0psig_close',
        '0psig_far',
        '50psig_close',
        '50psig_far',
        '100psig_close',
        '100psig_far',
    ]

    pgs = [0, 0, 50, 50, 100, 100]
    for idxfn, fn in enumerate(fns):
        with h5py.File(CLEANED_BASE + f'{fn}', 'r') as hf:
            ph1_clean = hf['ph1_clean'][:]
            ph2_clean = hf['ph2_clean'][:]
            u_tau = hf.attrs['u_tau']
            nu = hf.attrs['nu']
            rho = hf.attrs['rho']
            Re_tau = hf.attrs['Re_tau']
            cf_2 = hf.attrs['cf_2']  # default if missing

        f_clean, Pyy_ph1_clean = compute_spec(FS, ph1_clean)
        T_plus = 1/f_clean * (u_tau**2)/nu

        g1_b, g2_b, rv_b = bl_model(T_plus, Re_tau, cf_2)
        bl_fphipp_plus = rv_b*(g1_b+g2_b)

        bl_fphipp_plus = bl_fphipp_plus[f_clean < 1_000]

        f_clean_tf, Pyy_ph1_clean_tf = compute_spec(FS, ph1_clean)
        f_clean_tf, Pyy_ph2_clean_tf = compute_spec(FS, ph2_clean)

        T_plus_tf = 1/f_clean_tf * (u_tau**2)/nu

        data_fphipp_plus1_tf = (f_clean_tf * Pyy_ph1_clean_tf)/(rho**2 * u_tau**4)
        data_fphipp_plus2_tf = (f_clean_tf * Pyy_ph2_clean_tf)/(rho**2 * u_tau**4)

        # clip at the helmholtz resonance
        f_clean = f_clean_tf[f_clean_tf < 1_000]
        data_fphipp_plus1_tf_m = data_fphipp_plus1_tf[f_clean_tf < 1_000]
        data_fphipp_plus2_tf_m = data_fphipp_plus2_tf[f_clean_tf < 1_000]

        model_data_ratio1 = np.sqrt(data_fphipp_plus1_tf_m / bl_fphipp_plus)
        model_data_ratio2 = np.sqrt(data_fphipp_plus2_tf_m / bl_fphipp_plus)

        model_ratio_avg = (model_data_ratio1 + model_data_ratio2) / 2
        with h5py.File(TARGET_BASE + f"target_{labels[idxfn]}.h5", 'w') as hf:
            hf.create_dataset('frequencies', data=f_clean)
            hf.create_dataset('scaling_ratio', data=model_ratio_avg)
            hf.attrs['rho'] = rho
            hf.attrs['u_tau'] = u_tau
            hf.attrs['nu'] = nu
            hf.attrs['psig'] = pgs[idxfn]


if __name__ == "__main__":
    save_scaling_target()