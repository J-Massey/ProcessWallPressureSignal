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
BASE = "data/final_calibration"


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


def save_calibs(pressures):
    f_cut = [2_100, 4_700, 14_100]
    for i, pressure in enumerate(pressures):
        frequencies = np.arange(100, 3100, 100)
        dat = loadmat(BASE + f"/calib_{pressure}_1.mat")
        ph1, ph2, nc, _ = dat["channelData_WN"].T
        nc_pa = volts_to_pa(nc, "NC", f_cut[i])
        ph1_pa = volts_to_pa(ph1, "PH1", f_cut[i])
        f1, H1, g2_1 = estimate_frf(ph1_pa, nc_pa, fs=FS)
        # add calibration for ph2
        dat = loadmat(BASE + f"/calib_{pressure}_2.mat")
        ph1, ph2, nc, _ = dat["channelData_WN"].T
        nc_pa = volts_to_pa(nc, "NC", f_cut[i])
        ph2_pa = volts_to_pa(ph2, "PH2", f_cut[i])
        f2, H2, g2_2 = estimate_frf(ph2_pa, nc_pa, fs=FS)
        # Fuse the two transfer functions
        f_fused, H_fused, g2_fused = combine_anechoic_calibrations(
            f1, H1, g2_1,
            f2, H2, g2_2,
            gmin=0.4,
            smooth_oct=1/6,
            points_per_oct=32,
            eps=1e-12
        )
        # save frf
        with h5py.File(BASE + f"/calibs_{pressure}.h5", 'w') as hf:
            hf.create_dataset('frequencies', data=f1)
            hf.create_dataset('H1', data=H1)
            hf.create_dataset('H2', data=H2)
            hf.create_dataset('H_fused', data=H_fused)
            hf.attrs['psig'] = pressure

if __name__ == "__main__":
    psigs = [0, 50, 100]
    labels = [f"{psig}psig" for psig in psigs]
    save_calibs(labels)