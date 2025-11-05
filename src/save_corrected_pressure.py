# tf_compute.py
from __future__ import annotations

import numpy as np
import h5py
from scipy.interpolate import UnivariateSpline

from icecream import ic
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import scienceplots
plt.style.use(["science", "grid"])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")

from apply_frf import apply_frf
from fit_speaker_scales import fit_speaker_scaling_from_files
from models import bl_model
from clean_raw_data import air_props_from_gauge
from save_calibs import save_calibs
from src.unused.save_scaling_target import compute_spec, save_scaling_target

# =============================================================================
# Constants & styling (exported so tf_plot.py can import them)
# =============================================================================
FS: float = 50_000.0
NPERSEG: int = 2**12
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
    'nc': 52.4e-3,
    'PH1': 50.9e-3,
    'PH2': 51.7e-3,
    'NC': 52.4e-3,
}

PREAMP_GAIN: dict[str, float] = {"nc": 1.0, "PH1": 1.0, "PH2": 1.0, "NC": 1.0}
TONAL_BASE = "data/2025-10-28/tonal/"
CALIB_BASE = "data/final_calibration/"
TARGET_BASE = "data/final_target/"
CLEANED_BASE = "data/final_cleaned/"

def correct_pressure_sensitivity(p, psig):
    """
    Correct pressure sensor sensitivity based on gauge pressure [psig].
    Returns corrected pressure signal [Pa].
    """
    alpha = 0.01 # dB/kPa
    p_corr = p * 10**(psig * PSI_TO_PA / 1000 * alpha / 20)
    return p_corr


def save_corrected_pressure():
    """
    Apply the (rho, f)-scaled calibration FRF to measured time series and plot
    pre-multiplied, normalized spectra:  f * Pyy / (rho^2 * u_tau^4).
    """
    # --- fit rho–f scaling once from your saved target + calibration ---
    labels = ['0psig', '50psig', '100psig']
    with h5py.File("data/final_pressure/SU_2pt_pressure.h5", 'w') as hf_out:

        #This is the raw data with the freestream noise removed, explain
        hf_out.create_group("raw_data")
        h_raw = hf_out["raw_data"]
        # This is where the data has been corrected using a LEM
        hf_out.create_group("corrected_data")
        h_corrected = hf_out["corrected_data"]

    
        u_tau_uncertainty = [0.2, 0.1, 0.05]
        # --- main loop over datasets ---
        for i, L in enumerate(labels):

            # Load cleaned signals and attributes
            with h5py.File(CLEANED_BASE +f'{L}_far_cleaned.h5', 'r') as hf:
                ph1_clean_far = hf['ph1_clean'][:]
                ph2_clean_far = hf['ph2_clean'][:]
                u_tau = float(hf.attrs['u_tau'])
                nu = float(hf.attrs['nu'])
                rho = float(hf.attrs['rho'])
                Re_tau = hf.attrs.get('Re_tau', np.nan)

            h_raw.create_dataset(f'{L}_far/ph1', data=ph1_clean_far)
            h_raw.create_dataset(f'{L}_far/ph2', data=ph2_clean_far)
            with h5py.File(CLEANED_BASE +f'{L}_close_cleaned.h5', 'r') as hf:
                ph1_clean_close = hf['ph1_clean'][:]
                ph2_clean_close = hf['ph2_clean'][:]
            
            h_raw.create_dataset(f'{L}_close/ph1', data=ph1_clean_close)
            h_raw.create_dataset(f'{L}_close/ph2', data=ph2_clean_close)
            

            psig = int(L.replace('psig', ''))
            with h5py.File(CALIB_BASE + f"calibs_{psig}.h5", "r") as hf:
                f_cal = np.asarray(hf["frequencies"][:], float)
                H_cal = np.asarray(hf["H_fused"][:], complex)

            # --- apply FRF with fitted rho–f magnitude scaling ---
            # (uses your updated apply_frf that accepts scale_fn and rho)
            ph1_filt_far = apply_frf(ph1_clean_far, FS, f_cal, H_cal)
            ph2_filt_far = apply_frf(ph2_clean_far, FS, f_cal, H_cal)
            ph1_filt_close = apply_frf(ph1_clean_close, FS, f_cal, H_cal)
            ph2_filt_close = apply_frf(ph2_clean_close, FS, f_cal, H_cal)

            h_corrected.create_dataset(f'{L}_far/ph1', data=ph1_filt_far)
            h_corrected.create_dataset(f'{L}_far/ph2', data=ph2_filt_far)

            h_corrected.create_dataset(f'{L}_close/ph1', data=ph1_filt_close)
            h_corrected.create_dataset(f'{L}_close/ph2', data=ph2_filt_close)




if __name__ == "__main__":
    save_corrected_pressure()