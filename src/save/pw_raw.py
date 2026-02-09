"""
File GRP1: Transform the raw pressure signals from V to Pa, correct for sensitivity changes with pressure, and save the raw arrays in a structured HDF5 file with metadata. Also include the calibration runs mapping the pinhole mic to the nosecone mic.
"""
import os

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
CAL_BASE = Path(cfg.RAW_CAL_BASE) / "PH"
os.makedirs(CAL_BASE, exist_ok=True)
RAW_BASE = cfg.RAW_BASE
os.makedirs(Path(RAW_BASE), exist_ok=True)

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


def save_raw_ph_pressure(
    *,
    spacings: tuple[str, ...] | None = None,
):
    labels = cfg.LABELS
    psigs = cfg.PSIGS
    u_tau = cfg.U_TAU
    u_tau_unc = cfg.U_TAU_REL_UNC
    Tdeg = cfg.TDEG
    Tk = [273.15 + t for t in Tdeg]
    FS = cfg.FS
    Ue = cfg.U_E
    analog_LP_filter = cfg.ANALOG_LP_FILTER
    spacings = cfg.SPACINGS if spacings is None else spacings

    spacing_meta = {
        "close": {
            "spacing_m": 2.8 * DELTA,
            "x_PH1": 15e-3 + 0.2 * DELTA,
            "x_PH2": 15e-3 + 0.2 * DELTA + 2.8 * DELTA,
        },
        "far": {
            "spacing_m": 3.2 * DELTA,
            "x_PH2": 15e-3,
            "x_PH1": 15e-3 + 3.2 * DELTA,
        },
    }

    ph_raw = cfg.PH_RAW_FILE
    os.makedirs(Path(ph_raw).parent, exist_ok=True)

    with h5py.File(ph_raw, 'w') as hf:
        # --- file-level metadata ---
        hf.attrs['title'] = "Wall-pressure (pin-hole) - raw & calibration"
        hf.attrs['fs_Hz'] = FS
        hf.attrs['Ue_m_per_s'] = np.asarray(Ue, float)
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
            gL.attrs['Ue_m_per_s'] = float(Ue[i])
            gL.attrs['units'] = ['psig: psi(g)', 'u_tau: m/s', 'nu: m^2/s', 'rho: kg/m^3', 'mu: Pa*s', 'T_K: K', 'analog_LP_filter_Hz: Hz']

            seen_any = False
            for sp in spacings:
                mat_path = Path(RAW_BASE) / f"{sp}/{L}.mat"
                if not mat_path.exists():
                    print(f"[skip] missing raw mat file: {mat_path}")
                    continue
                dat = sio.loadmat(mat_path)
                X = np.asarray(dat["channelData"])
                ph1_v = X[:, 0]
                ph2_v = X[:, 1]

                ph1 = correct_pressure_sensitivity(volts_to_pa(ph1_v, "PH1"), psigs[i])
                ph2 = correct_pressure_sensitivity(volts_to_pa(ph2_v, "PH2"), psigs[i])

                gS = gL.create_group(sp)
                meta = spacing_meta.get(sp)
                if meta:
                    gS.attrs["spacing_m"] = meta["spacing_m"]
                    gS.attrs["x_PH1"] = meta["x_PH1"]
                    gS.attrs["x_PH2"] = meta["x_PH2"]
                gS.create_dataset("PH1_Pa", data=ph1)
                gS.create_dataset("PH2_Pa", data=ph2)
                seen_any = True
            if not seen_any:
                raise FileNotFoundError(f"No raw files found for {L} in {RAW_BASE}")

            # ---- run 1: PH1 to NC
            m1 = sio.loadmat(CAL_BASE / f"calib_{L}_1.mat")
            ph1_v, _, nc_v, *_ = m1["channelData_WN"].T
            ph1_pa = volts_to_pa(ph1_v, "PH1")
            nc1_pa = volts_to_pa(nc_v,  "NC")
            # compensate sensor sensitivity vs psig (amplitude gain)
            ph1_pa = correct_pressure_sensitivity(ph1_pa, psigs[i])
            nc1_pa =  correct_pressure_sensitivity(nc1_pa,  psigs[i]) # x=PH1, y=NC, H:=H_{PH_to_NC}

            # ---- run 2: PH2 to NC
            m2 = sio.loadmat(CAL_BASE / f"calib_{L}_2.mat")
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
