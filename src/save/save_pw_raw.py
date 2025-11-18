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


def save_raw_ph_pressure():
    labels = ['0psig', '50psig', '100psig']
    psigs  = [0.0, 50.0, 100.0]
    u_tau  = [0.537, 0.522, 0.506]
    u_tau_unc = [0.2, 0.1, 0.05]
    Tdeg = [18, 20, 22]
    Tk   = [273.15 + t for t in Tdeg]
    FS = 50_000.0
    Ue = 14.0
    analog_LP_filter = [2100, 4700, 14100]

    ph_raw = 'data/final_pressure/G_wallp_SU_raw.hdf5'

    with h5py.File(ph_raw, 'w') as hf:
        # --- file-level metadata ---
        hf.attrs['title'] = "Wall-pressure (pin-hole) - raw & calibration"
        hf.attrs['fs_Hz'] = FS
        hf.attrs['Ue_m_per_s'] = Ue
        hf.attrs['DAQ'] = "24-bit NI-USB-6363"
        hf.attrs['mic_details'] = "HB&K 1/2'' Type 4964"
        # gL.attrs['sensor_serial'] = sensor_serial[i % len(sensor_serial)]


        # --- helpful top-level description ---
        
        hf.attrs['description'] = (
            "Raw wall-pressure signals from pinhole treated microphone. "
            "Two measurements per condition: close/far correspond to the "
            "pinhole spacings used in wall-pressure dataset G. "
            "Includes semi-anechoic calibration signal with a white noise"
            "source measuring the pinhole and nosecone treated mic simultaneously."
        )

        g_fs = hf.create_group("wallp_raw")

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
            gL.attrs['units'] = ['psig: psi(g)', 'u_tau: m/s', 'nu: m^2/s', 'rho: kg/m^3', 'mu: Pa·s', 'T_K: K', 'analog_LP_filter_Hz: Hz']

            # ---- load raw (.mat) ----
            nr_mat = Path(RAW_BASE) / f'far/{L}.mat'
            fr_mat = Path(RAW_BASE) / f'close/{L}.mat'

            dat_far  = sio.loadmat(nr_mat)
            dat_close = sio.loadmat(fr_mat)   # <- fix the typo: not nr_mat again

            # Expect columns: 0=PH1,1=PH2,2=NC  (rename if your files differ)
            X_far   = np.asarray(dat_far['channelData'])
            X_close = np.asarray(dat_close['channelData'])
            ph1_far_V, ph2_far_V, NC_far_V       = X_far[:,0],  X_far[:,1],  X_far[:,2]
            ph1_close_V, ph2_close_V, NC_close_V = X_close[:,0],X_close[:,1],X_close[:,2]

            # --- convert + pressure sensitivity corrections (your funcs) ---
            ph1_far  = correct_pressure_sensitivity(volts_to_pa(ph1_far_V,  'PH1'), psigs[i])
            ph2_far  = correct_pressure_sensitivity(volts_to_pa(ph2_far_V,  'PH2'), psigs[i])

            ph1_close = correct_pressure_sensitivity(volts_to_pa(ph1_close_V,'PH1'), psigs[i])
            ph2_close = correct_pressure_sensitivity(volts_to_pa(ph2_close_V,'PH2'), psigs[i])

            # --- store raw arrays (chunked + compressed) ---
            # Close
            gC = gL.create_group('close')
            gC.attrs['spacing_m'] = 2.8*DELTA
            gC.attrs['x_PH1'] = 15e-3+0.2*DELTA
            gC.attrs['x_PH2'] = 15e-3+0.2*DELTA + 2.8*DELTA
            gC.attrs['spacing_m'] = 2.8*DELTA
            gC.create_dataset('PH1_Pa', data=ph1_close)
            gC.create_dataset('PH2_Pa', data=ph2_close)
            
            # Far
            gF = gL.create_group('far')
            gF.attrs['spacing_m'] = 3.2*DELTA
            gF.attrs['x_PH2'] = 15e-3
            gF.attrs['x_PH1'] = 15e-3 + 3.2*DELTA
            gF.create_dataset('PH1_Pa', data=ph1_far)
            gF.create_dataset('PH2_Pa', data=ph2_far)

            # ---- run 1: PH1→NC
            m1 = sio.loadmat(CAL_BASE + f"calib_{L}_1.mat")
            ph1_v, _, nc_v, *_ = m1["channelData_WN"].T
            ph1_pa = volts_to_pa(ph1_v, "PH1")
            nc1_pa = volts_to_pa(nc_v,  "NC")
            # compensate sensor sensitivity vs psig (amplitude gain)
            ph1_pa = correct_pressure_sensitivity(ph1_pa, psigs[i])
            nc1_pa =  correct_pressure_sensitivity(nc1_pa,  psigs[i]) # x=PH1, y=NC  ⇒ H:=H_{PH→NC}

            # ---- run 2: PH2→NC
            m2 = sio.loadmat(CAL_BASE + f"calib_{L}_2.mat")
            _, ph2_v, nc_v2, *_ = m2["channelData_WN"].T
            ph2_pa = volts_to_pa(ph2_v, "PH2")
            nc2_pa = volts_to_pa(nc_v2,  "NC")
            ph2_pa = correct_pressure_sensitivity(ph2_pa, psigs[i])
            nc2_pa =  correct_pressure_sensitivity(nc2_pa,  psigs[i])

            gFRF = gL.create_group('FRF_PH_to_NC')
            gFRF.attrs['from'] = 'NC'
            gFRF.attrs['to']   = 'nkd'
            gFRF.attrs['note'] = 'Semi-anechoic calibration mapping the pinhole mic to nosecone mic'
            gR1 = gFRF.create_group('Run1')
            gR1.create_dataset('PH1_Pa', data=ph1_pa)
            gR1.create_dataset('NC_Pa',  data=nc1_pa)
            gR2 = gFRF.create_group('Run2')
            gR2.create_dataset('PH2_Pa', data=ph2_pa)
            gR2.create_dataset('NC_Pa',  data=nc2_pa)


if __name__ == "__main__":
    save_raw_ph_pressure()