# tf_compute.py
from __future__ import annotations

import numpy as np
import h5py
import scipy.io as sio

from icecream import ic
from pathlib import Path

from src.config_params import Config

# Load the config parameters (file paths, constants, etc.) from a central location to ensure consistency
cfg = Config()

# =============================================================================
# Constants & styling (exported so tf_plot.py can import them)
# =============================================================================
FS = cfg.FS
NPERSEG = cfg.NPERSEG
WINDOW = cfg.WINDOW

# --- constants (keep once, top of file) ---
R = cfg.R
PSI_TO_PA = cfg.PSI_TO_PA
P_ATM = cfg.P_ATM
DELTA = cfg.DELTA
TDEG = cfg.TDEG

# =============================================================================
# Units & optional conversions (kept for compatibility with other workflows)
# =============================================================================
SENSITIVITIES_V_PER_PA = cfg.SENSITIVITIES_V_PER_PA
PREAMP_GAIN = cfg.PREAMP_GAIN
RAW_BASE = cfg.RAW_BASE

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
    Return rho [kg/m^3], mu [Pa*s], nu [m^2/s] from gauge pressure [psi] and temperature [K].
    Sutherland's law for mu; nu = mu/rho.
    """
    p_abs = P_ATM + psi_gauge * PSI_TO_PA
    # Sutherland's
    mu0, T0, S = 1.716e-5, 273.15, 110.4
    mu = mu0 * (T_K/T0)**1.5 * (T0 + S)/(T_K + S)
    rho = p_abs / (R * T_K)
    nu = mu / rho
    return rho, mu, nu


def save_raw_fs_pressure(
    *,
    spacings: tuple[str, ...] | None = None,
    include_nc_calib: bool | None = None,
):
    labels = cfg.LABELS
    psigs = cfg.PSIGS
    u_tau = cfg.U_TAU
    u_tau_unc = cfg.U_TAU_REL_UNC
    Tdeg = cfg.TDEG
    Tk = [273.15 + t for t in Tdeg]
    FS = cfg.FS
    Ue = cfg.U_E
    # sensor_serial = [123]  # example
    analog_LP_filter = cfg.ANALOG_LP_FILTER

    spacings = cfg.SPACINGS if spacings is None else spacings
    include_nc_calib = cfg.INCLUDE_NC_CALIB_RAW if include_nc_calib is None else include_nc_calib

    fs_raw = cfg.NKD_RAW_FILE

    with h5py.File(fs_raw, 'w') as hf:
        # --- file-level metadata ---
        hf.attrs['title'] = "Freestream pressure (nose-cone) - raw & calibration"
        hf.attrs['fs_Hz'] = FS
        hf.attrs['Ue_m_per_s'] = np.asarray(Ue, float)
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
            gL.attrs['Ue_m_per_s'] = float(Ue[i])


            seen_any = False
            for sp in spacings:
                mat_path = Path(RAW_BASE) / f"{sp}/{L}.mat"
                if not mat_path.exists():
                    print(f"[skip] missing raw mat file: {mat_path}")
                    continue
                dat = sio.loadmat(mat_path)
                X = np.asarray(dat["channelData"])
                NC_v = X[:, 2]
                NC = correct_pressure_sensitivity(volts_to_pa(NC_v, "NC"), psigs[i])
                gS = gL.create_group(sp)
                gS.create_dataset("NC_Pa", data=NC)
                seen_any = True
            if not seen_any:
                raise FileNotFoundError(f"No raw files found for {L} in {RAW_BASE}")

            if include_nc_calib:
                base = Path(cfg.RAW_CAL_BASE) / "NC"
                mat_path = base / f"{L}/nkd-ns_nofacilitynoise.mat"
                if not mat_path.exists():
                    print(f"[skip] missing NC calib file: {mat_path}")
                else:
                    m1 = sio.loadmat(mat_path)
                    if L == "100psig":
                        nkd_cal, nc_cal = m1["channelData_nofacitynoise"].T
                    else:
                        nkd_cal, nc_cal = m1["channelData"].T

                    nc_cal = correct_pressure_sensitivity(volts_to_pa(nc_cal, "NC"), psigs[i])
                    nkd_cal = correct_pressure_sensitivity(volts_to_pa(nkd_cal, "nkd"), psigs[i])

                    gFRF = gL.create_group("FRF_NC_to_nkd")
                    gFRF.create_dataset("NC_Pa", data=nc_cal)
                    gFRF.create_dataset("nkd_Pa", data=nkd_cal)
                    gFRF.attrs["from"] = "NC"
                    gFRF.attrs["to"] = "nkd"
                    gFRF.attrs["note"] = "Semi-anechoic calibration mapping"


if __name__ == "__main__":
    save_raw_fs_pressure()
    with h5py.File(cfg.NKD_RAW_FILE, "r") as hf:
        print(hf.keys())
        print(hf['freestream_raw'].attrs.keys())
        print(hf['freestream_raw']['0psig'].attrs.keys())
        print(hf['freestream_raw']['0psig']['close'].attrs.keys())
        print(hf['freestream_raw']['0psig']['close']['NC_Pa'].shape)
