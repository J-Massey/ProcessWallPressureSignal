# tf_compute.py
from __future__ import annotations

import numpy as np
import h5py
import scipy.io as sio

from icecream import ic
from pathlib import Path

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


def save_raw_fs_pressure():
    labels = ['0psig', '50psig', '100psig']
    psigs  = [0.0, 50.0, 100.0]
    u_tau  = [0.537, 0.522, 0.506]
    u_tau_unc = [0.2, 0.1, 0.05]
    Tdeg = [18, 20, 22]
    Tk   = [273.15 + t for t in Tdeg]
    FS = 50_000.0
    Ue = 14.0
    # sensor_serial = [123]  # example
    analog_LP_filter = [2100, 4700, 14100]

    fs_raw = "data/final_pressure/F_freestreamp_SU_raw.hdf5"

    with h5py.File(fs_raw, 'w') as hf:
        # --- file-level metadata ---
        hf.attrs['title'] = "Freestream pressure (nose-cone) - raw & calibration"
        hf.attrs['fs_Hz'] = FS
        hf.attrs['Ue_m_per_s'] = Ue
        hf.attrs['DAQ'] = "24-bit"
        hf.attrs['mic_details'] = "HB&K 1/2'' Type 4964"
        # gL.attrs['sensor_serial'] = sensor_serial[i % len(sensor_serial)]

        # --- helpful top-level description ---
        hf.attrs['description'] = (
            "Raw freestream pressure signals from nose-cone microphone. "
            "Two measurements per condition: close/far correspond to the "
            "pinhole spacings used in wall-pressure dataset G. "
            "Includes the simultaneous semi-anechoic calibration signals used " \
            "to compute FRFs from NC to nkd microphones."
        )

        g_fs = hf.create_group("freestream_raw")

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


            # ---- load raw (.mat) ----
            nr_mat = Path(RAW_BASE) / f'far/{L}.mat'
            fr_mat = Path(RAW_BASE) / f'close/{L}.mat'

            dat_far  = sio.loadmat(nr_mat)
            dat_close = sio.loadmat(fr_mat)

            # Expect columns: 0=PH1,1=PH2,2=NC  (rename if your files differ)
            X_far   = np.asarray(dat_far['channelData'])
            X_close = np.asarray(dat_close['channelData'])
            NC_far_V       = X_far[:,2]
            NC_close_V = X_close[:,2]

            # --- convert + pressure sensitivity corrections (your funcs) ---
            NC_far   = correct_pressure_sensitivity(volts_to_pa(NC_far_V,   'NC'), psigs[i])
            NC_close  = correct_pressure_sensitivity(volts_to_pa(NC_close_V, 'NC'), psigs[i])

            # --- store raw arrays
            # Close
            gC = gL.create_group('close')
            gC.create_dataset('NC_Pa',  data=NC_close)
            # Far
            gF = gL.create_group('far')
            gF.create_dataset('NC_Pa',  data=NC_far)

            # --- load simultaneous semi-anechoic calibration data & store ---
            base = Path("data/20250930")
            m1 = sio.loadmat(base / f"{L}/nkd-ns_nofacilitynoise.mat")
            if L == '100psig':
                nkd_cal, nc_cal = m1["channelData_nofacitynoise"].T
            else:
                nkd_cal, nc_cal = m1["channelData"].T

            nc_cal = correct_pressure_sensitivity(volts_to_pa(nc_cal,   'NC'), psigs[i])
            nkd_cal = correct_pressure_sensitivity(volts_to_pa(nkd_cal,   'nkd'), psigs[i])
            

            gFRF = gL.create_group('FRF_NC_to_nkd')
            gFRF.create_dataset('NC_Pa', data=nc_cal)
            gFRF.create_dataset('nkd_Pa',    data=nkd_cal)
            gFRF.attrs['from'] = 'NC'
            gFRF.attrs['to']   = 'nkd'
            gFRF.attrs['note'] = 'Semi-anechoic calibration mapping'


if __name__ == "__main__":
    save_raw_fs_pressure()
    with h5py.File('data/final_pressure/F_freestreamp_SU_raw.hdf5', 'r') as hf:
        print(hf.keys())
        print(hf['freestream_raw'].attrs.keys())
        print(hf['freestream_raw']['0psig'].attrs.keys())
        print(hf['freestream_raw']['0psig']['close'].attrs.keys())
        print(hf['freestream_raw']['0psig']['close']['NC_Pa'].shape)