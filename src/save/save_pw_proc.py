# tf_compute.py
from __future__ import annotations

import numpy as np
import h5py
import scipy.io as sio

from icecream import ic
from pathlib import Path

from src.apply_frf import apply_frf

# =============================================================================
# Constants & styling (exported so tf_plot.py can import them)
# =============================================================================
FS: float = 50_000.0
NPERSEG: int = 2**12
WINDOW: str = "hann"

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
    'PH1': 50.9e-3,
    'PH2': 51.7e-3,
    'NC': 52.4e-3,
    'nkd': 50.9e-3,
}

PREAMP_GAIN: dict[str, float] = {"nc": 1.0, "PH1": 1.0, "PH2": 1.0, "NC": 1.0}
TONAL_BASE = "data/2025-10-28/tonal/"
CAL_BASE = "data/final_calibration/"
TARGET_BASE = "data/final_target/"
CLEANED_BASE = "data/final_cleaned/"
RAW_BASE = "data/20251031/"

def correct_pressure_sensitivity(p, psig, alpha: float = 0.01):
    """
    Correct pressure sensor sensitivity based on gauge pressure [psig].
    Returns corrected pressure signal [Pa].
    """
    p_corr = p * 10**(psig * PSI_TO_PA / 1000 * alpha / 20)
    return p_corr


def volts_to_pa(x_volts: np.ndarray, channel: str) -> np.ndarray:
    sens = SENSITIVITIES_V_PER_PA[channel]  # V/Pa
    return x_volts / sens


def air_props_from_gauge(psi_gauge: float, T_K: float):
    """
    Return rho [kg/m^3], mu [Pa·s], nu [m^2/s] from gauge pressure [psi] and temperature [K].
    Sutherland's law for mu; nu = mu/rho.
    """
    p_abs = P_ATM + psi_gauge * PSI_TO_PA
    # Sutherland's
    mu0, T0, S = 1.716e-5, 273.15, 110.4
    mu = mu0 * (T_K/T0)**1.5 * (T0 + S)/(T_K + S)
    rho = p_abs / (R * T_K)
    nu = mu / rho
    return rho, mu, nu


def save_corrected_pressure():
    """
    Apply the (rho, f)-scaled calibration FRF to measured time series and plot
    pre-multiplied, normalized spectra:  f * Pyy / (rho^2 * u_tau^4).
    """
    # --- fit rho–f scaling once from your saved target + calibration ---
    labels = ['0psig', '50psig', '100psig']
    psigs  = [0.0, 50.0, 100.0]
    u_tau  = [0.537, 0.522, 0.506]
    u_tau_unc = [0.2, 0.1, 0.05]
    Tdeg = [18, 20, 22]
    Tk   = [273.15 + t for t in Tdeg]
    FS = 50_000.0
    Ue = 14.0
    analog_LP_filter = [2100, 4700, 14100]

    ph_processed = 'data/final_pressure/G_wallp_SU_production.hdf5'

    with h5py.File(ph_processed, 'w') as hf:
        # --- file-level metadata ---
        hf.attrs['title'] = "Wall-pressure (pin-hole) - processed & FRF from calibration"
        hf.attrs['fs_Hz'] = FS
        hf.attrs['Ue_m_per_s'] = Ue
        hf.attrs['DAQ'] = "24-bit"
        hf.attrs['mic_details'] = "HB&K 1/2'' Type 4964"
        # gL.attrs['sensor_serial'] = sensor_serial[i % len(sensor_serial)]


        # --- helpful top-level description ---
        hf.attrs['description'] = (
            "Processed wall-pressure signals from pinhole treated microphone applying FRF from calibration. "
            "Two measurements per condition: close/far correspond to the "
            "pinhole spacings used in wall-pressure dataset G. "
            "Includes frequency response function from semi-anechoic calibration signal with a white noise "
            "source measuring the pinhole and nosecone treated mic simultaneously."
        )

        g_fs = hf.create_group("wallp_production")

        for i, L in enumerate(labels):
            gL = g_fs.create_group(L)
            # condition-level metadata (numeric + units separate)
            rho, mu, nu = air_props_from_gauge(psigs[i], Tk[i])
            gL.attrs['psig'] = psigs[i]          # unit: psi(g)
            gL.attrs['u_tau'] = u_tau[i]         # unit: m/s
            gL.attrs['nu'] = nu
            gL.attrs['rho'] = rho
            gL.attrs['mu'] = mu
            gL.attrs['Re_tau'] = u_tau[i]*DELTA / nu         # unit: m/s
            gL.attrs['u_tau_rel_unc'] = u_tau_unc[i]
            gL.attrs['T_K'] = Tk[i]
            gL.attrs['analog_LP_filter_Hz'] = analog_LP_filter[i]

            g_rejected = gL.create_group('fs_noise_rejected_signals')

            g_rejected_far = g_rejected.create_group('far')
            g_rejected_far.attrs['spacing_m'] = 3.2*DELTA
            g_rejected_close = g_rejected.create_group('close')
            g_rejected_close.attrs['spacing_m'] = 2.8*DELTA

            # Load cleaned signals and attributes
            with h5py.File(CLEANED_BASE +f'{L}_far_cleaned.h5', 'r') as hf:
                ph1_clean_far = hf['ph1_clean'][:]
                ph2_clean_far = hf['ph2_clean'][:]
            
            g_rejected_far.create_dataset('ph1', data=ph1_clean_far)
            g_rejected_far.create_dataset('ph2', data=ph2_clean_far)

            with h5py.File(CLEANED_BASE +f'{L}_close_cleaned.h5', 'r') as hf:
                ph1_clean_close = hf['ph1_clean'][:]
                ph2_clean_close = hf['ph2_clean'][:]

            g_rejected_close.create_dataset('ph1', data=ph1_clean_close)
            g_rejected_close.create_dataset('ph2', data=ph2_clean_close)

            with h5py.File(CAL_BASE + f"calibs_{int(psigs[i])}.h5", "r") as hf:
                f_cal = np.asarray(hf["frequencies"][:], float)
                H_cal = np.asarray(hf["H_fused"][:], complex)
            with h5py.File("data/20250930/" +f'calibs_{int(psigs[i])}.h5', 'r') as hf:
                f_cal_nkd = hf['frequencies'][:].squeeze().astype(float)
                H_fused_nkd = hf['H_fused'][:].squeeze().astype(complex)

            # --- apply FRF with fitted rho–f magnitude scaling ---
            # (uses your updated apply_frf that accepts scale_fn and rho)
            ph1_filt_far = apply_frf(ph1_clean_far, FS, f_cal, H_cal)
            ph2_filt_far = apply_frf(ph2_clean_far, FS, f_cal, H_cal)
            ph1_filt_close = apply_frf(ph1_clean_close, FS, f_cal, H_cal)
            ph2_filt_close = apply_frf(ph2_clean_close, FS, f_cal, H_cal)
            ph1_filt_far = apply_frf(ph1_clean_far, FS, f_cal_nkd, H_fused_nkd)
            ph2_filt_far = apply_frf(ph2_clean_far, FS, f_cal_nkd, H_fused_nkd)
            ph1_filt_close = apply_frf(ph1_clean_close, FS, f_cal_nkd, H_fused_nkd)
            ph2_filt_close = apply_frf(ph2_clean_close, FS, f_cal_nkd, H_fused_nkd)

            g_corrected = gL.create_group('frf_corrected_signals')
            g_corrected_far = g_corrected.create_group('far')
            g_corrected_close = g_corrected.create_group('close')

            g_corrected_far.create_dataset('ph1', data=ph1_filt_far)
            g_corrected_far.create_dataset('ph2', data=ph2_filt_far)
            g_corrected_far.attrs['spacing_m'] = 3.2*DELTA

            g_corrected_close.create_dataset('ph1', data=ph1_filt_close)
            g_corrected_close.create_dataset('ph2', data=ph2_filt_close)
            g_corrected_close.attrs['spacing_m'] = 2.8*DELTA

if __name__ == "__main__":
    save_corrected_pressure()