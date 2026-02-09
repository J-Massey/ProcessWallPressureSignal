# tf_compute.py
from __future__ import annotations

import numpy as np
import h5py
from scipy.signal import butter, sosfiltfilt

from icecream import ic
from pathlib import Path

from src.core.apply_frf import apply_frf
from src.core.wiener_filter_torch import wiener_cancel_background, cancel_background_freq, wiener_cancel_hybrid
from src.config_params import Config

cfg = Config()

# =============================================================================
# Constants & styling (exported so tf_plot.py can import them)
# =============================================================================
FS = cfg.FS
NPERSEG = cfg.NPERSEG
WINDOW = cfg.WINDOW

# --- constants (keep once, top of file) ---
R = cfg.R        # J/kg/K
PSI_TO_PA = cfg.PSI_TO_PA
P_ATM = cfg.P_ATM
DELTA = cfg.DELTA  # m, bl-height of 'channel'
TDEG = cfg.TDEG

# =============================================================================
# Units & optional conversions (kept for compatibility with other workflows)
# =============================================================================

SENSITIVITIES_V_PER_PA = cfg.SENSITIVITIES_V_PER_PA
PREAMP_GAIN = cfg.PREAMP_GAIN
CAL_BASE = cfg.TF_BASE
RAW_BASE = cfg.RAW_BASE


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
    labels = cfg.LABELS
    psigs = cfg.PSIGS
    u_tau = cfg.U_TAU
    u_tau_unc = cfg.U_TAU_REL_UNC
    Tdeg = cfg.TDEG
    Tk = [273.15 + t for t in Tdeg]
    FS = cfg.FS
    Ue = cfg.U_E
    analog_LP_filter = cfg.ANALOG_LP_FILTER

    ph_processed = cfg.PH_PROCESSED_FILE
    ph_raw = cfg.PH_RAW_FILE

    with h5py.File(ph_processed, 'w') as hf:
        # --- file-level metadata ---
        hf.attrs['title'] = "Wall-pressure (pin-hole) - processed & FRF from calibration"
        hf.attrs['fs_Hz'] = FS
        hf.attrs['Ue_m_per_s'] = np.asarray(Ue, float)
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
            gL.attrs['Ue_m_per_s'] = float(Ue[i])
            gL.attrs['units'] = ['psig: psi(g)', 'u_tau: m/s', 'nu: m^2/s', 'rho: kg/m^3', 'mu: Pa·s', 'T_K: K', 'analog_LP_filter_Hz: Hz']
            # --- load raw signals ---
            ph_raw = cfg.PH_RAW_FILE
            with h5py.File(ph_raw, 'r') as f_raw:
                ph1_raw_far = f_raw[f'wallp_raw/{L}/far/PH1_Pa'][:]
                ph2_raw_far = f_raw[f'wallp_raw/{L}/far/PH2_Pa'][:]
                ph1_raw_close = f_raw[f'wallp_raw/{L}/close/PH1_Pa'][:]
                ph2_raw_close = f_raw[f'wallp_raw/{L}/close/PH2_Pa'][:]

            with h5py.File(f"{CAL_BASE}/PH/calibs_{int(psigs[i])}.h5", "r") as hf:
                f_cal = np.asarray(hf["frequencies"][:], float)
                H_cal = np.asarray(hf["H_fused"][:], complex)
            with h5py.File(f"{CAL_BASE}/NC/calibs_{int(psigs[i])}.h5", "r") as hf:
                f_cal_nkd = hf['frequencies'][:].squeeze().astype(float)
                H_fused_nkd = hf['H_fused'][:].squeeze().astype(complex)

            # --- apply FRF with fitted rho–f magnitude scaling ---
            # (uses your updated apply_frf that accepts scale_fn and rho)
            ph1_filt_far = apply_frf(ph1_raw_far, FS, f_cal, H_cal)
            ph2_filt_far = apply_frf(ph2_raw_far, FS, f_cal, H_cal)
            ph1_filt_close = apply_frf(ph1_raw_close, FS, f_cal, H_cal)
            ph2_filt_close = apply_frf(ph2_raw_close, FS, f_cal, H_cal)

            ph1_filt_far = apply_frf(ph1_filt_far, FS, f_cal_nkd, H_fused_nkd)
            ph2_filt_far = apply_frf(ph2_filt_far, FS, f_cal_nkd, H_fused_nkd)
            ph1_filt_close = apply_frf(ph1_filt_close, FS, f_cal_nkd, H_fused_nkd)
            ph2_filt_close = apply_frf(ph2_filt_close, FS, f_cal_nkd, H_fused_nkd)

            g_corrected = gL.create_group('frf_corrected_signals')
            g_corrected_far = g_corrected.create_group('far')
            g_corrected_close = g_corrected.create_group('close')

            g_corrected_far.create_dataset('PH1_Pa', data=ph1_filt_far)
            g_corrected_far.create_dataset('PH2_Pa', data=ph2_filt_far)
            g_corrected_far.attrs['spacing_m'] = 3.2*DELTA
            g_corrected_far.attrs['x_PH2'] = 15e-3
            g_corrected_far.attrs['x_PH1'] = 15e-3 + 3.2*DELTA

            g_corrected_close.create_dataset('PH1_Pa', data=ph1_filt_close)
            g_corrected_close.create_dataset('PH2_Pa', data=ph2_filt_close)
            g_corrected_close.attrs['spacing_m'] = 2.8*DELTA
            g_corrected_close.attrs['x_PH2'] = 15e-3+0.2*DELTA
            g_corrected_close.attrs['x_PH1'] = 15e-3+0.2*DELTA + 2.8*DELTA
            
            g_rejected = gL.create_group('fs_noise_rejected_signals')

            g_rejected_far = g_rejected.create_group('far')
            g_rejected_far.attrs['spacing_m'] = 3.2*DELTA
            g_rejected_far.attrs['x_PH2'] = 15e-3
            g_rejected_far.attrs['x_PH1'] = 15e-3 + 3.2*DELTA

            g_rejected_close = g_rejected.create_group('close')
            g_rejected_close.attrs['spacing_m'] = 2.8*DELTA
            g_rejected_close.attrs['x_PH2'] = 15e-3+0.2*DELTA
            g_rejected_close.attrs['x_PH1'] = 15e-3+0.2*DELTA + 2.8*DELTA

            # --- reject the background noise
            nkd_raw = cfg.NKD_PROCESSED_FILE
            with h5py.File(nkd_raw, 'r') as f_nkd:
                nkd_far = f_nkd[f'freestream_production/{L}/far/NC_Pa'][:]
                nkd_close = f_nkd[f'freestream_production/{L}/close/NC_Pa'][:]

            # --- take out the mean ---
            ph1_filt_far -= np.mean(ph1_filt_far); ph1_filt_close -= np.mean(ph1_filt_close)
            ph2_filt_far -= np.mean(ph2_filt_far); ph2_filt_close -= np.mean(ph2_filt_close)
            nkd_far -= np.mean(nkd_far); nkd_close -= np.mean(nkd_close)

            # --- apply a band pass filter between 0.1 Hz and f_cut ---
            def bandpass_filter(data, fs, f_low, f_high, order=3):
                sos = butter(order, [f_low, f_high], btype='band', fs=fs, output='sos')
                filtered = sosfiltfilt(sos, data)
                filtered = np.nan_to_num(filtered, nan=0.0)
                return filtered
            
            ph1_filt_far = bandpass_filter(ph1_filt_far, FS, 1, analog_LP_filter[i])
            ph2_filt_far = bandpass_filter(ph2_filt_far, FS, 1, analog_LP_filter[i])
            nkd_far = bandpass_filter(nkd_far, FS, 1, analog_LP_filter[i])
            ph1_filt_close = bandpass_filter(ph1_filt_close, FS, 1, analog_LP_filter[i])
            ph2_filt_close = bandpass_filter(ph2_filt_close, FS, 1, analog_LP_filter[i])
            nkd_close = bandpass_filter(nkd_close, FS, 1, analog_LP_filter[i])

            # --- cancel background with Wiener filter ---
            ph1_clean_far = wiener_cancel_background(
                ph1_filt_far, nkd_far, FS)#.cpu().numpy()
            ph2_clean_far = wiener_cancel_background(
                ph2_filt_far, nkd_far, FS)#.cpu().numpy()
            ph1_clean_close = wiener_cancel_background(
                ph1_filt_close, nkd_close, FS)#.cpu().numpy()
            ph2_clean_close = wiener_cancel_background(
                ph2_filt_close, nkd_close, FS)#.cpu().numpy()
            
            g_rejected_far.create_dataset('PH1_Pa', data=ph1_clean_far)
            g_rejected_far.create_dataset('PH2_Pa', data=ph2_clean_far)
            g_rejected_close.create_dataset('PH1_Pa', data=ph1_clean_close)
            g_rejected_close.create_dataset('PH2_Pa', data=ph2_clean_close)


if __name__ == "__main__":
    save_corrected_pressure()
