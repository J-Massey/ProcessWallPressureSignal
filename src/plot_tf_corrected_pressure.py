import h5py
import numpy as np
from scipy.signal import welch, csd, get_window
import scipy.io as sio

from icecream import ic
from pathlib import Path

from scipy.signal import butter, sosfiltfilt
import torch
from tqdm import tqdm

from wiener_filter_torch import wiener_cancel_background_torch
from apply_frf import apply_frf

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import scienceplots
plt.style.use(["science", "grid"])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")

############################
# Constants & defaults
############################
FS = 50_000.0
NPERSEG = 2**12
WINDOW = "hann"
TRIM_CAL_SECS = 5  # seconds trimmed from the start of calibration runs (0 to disable)

nc_colour = '#1f77b4'  # matplotlib default blue
ph1_colour = "#c76713"  # matplotlib default orange
ph2_colour = "#9fda16"  # matplotlib default red
nkd_colour = '#2ca02c' # matplotlib default green

DEFAULT_UNITS = {
    'channelData_300_plug': ('Pa', 'Pa'),  # PH, NKD
    'channelData_300_nose': ('Pa', 'Pa'),  # NKD, NC
    'channelData_300':      ('Pa', 'Pa'),  # NC,  PH
}

# --- constants (keep once, top of file) ---
R = 287.05        # J/kg/K
PSI_TO_PA = 6_894.76
P_ATM = 101_325.0
DELTA = 0.035  # m, bl-height of 'channel'
TDEG = [18, 20, 22]

TPLUS_CUT = 10  # picked so that we cut at half the inner peak

CHAIN_SENS_V_PER_PA = {
    'ph1': 50.9e-3,  # V/Pa
    'ph2': 51.7e-3,  # V/Pa
    'nc':  52.4e-3,  # V/Pa
}

def bandpass_filter(data, fs, f_low, f_high, order=4):
    sos = butter(order, [f_low, f_high], btype='band', fs=fs, output='sos')
    filtered = sosfiltfilt(sos, data)
    # protect against NaNs and negatives
    filtered = np.nan_to_num(filtered, nan=0.0)
    # filtered[filtered < 0] = 0
    return filtered


def volts_to_pa(x_volts: np.ndarray, channel: str, f_cut: float) -> np.ndarray:
    sens = CHAIN_SENS_V_PER_PA[channel]  # V/Pa
    # Band pass filter 0.1-f_cut Hz
    x_volts = bandpass_filter(x_volts, FS, 0.1, f_cut)
    return x_volts / sens


def correct_pressure_sensitivity(p, psig, alpha: float = 0.012):
    """
    Correct pressure sensor sensitivity based on gauge pressure [psig].
    Returns corrected pressure signal [Pa].
    """
    p_corr = p * 10**(psig * PSI_TO_PA / 1000 * alpha / 20)
    return p_corr


def f_cut_from_Tplus(u_tau: float, nu: float, Tplus_cut: float = TPLUS_CUT):
    return (u_tau**2 / nu) / Tplus_cut


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


def bl_model(Tplus, Re_tau: float, cf_2: float) -> np.ndarray:
    A1 = 2.2
    sig1 = 3.9
    mean_Tplus = 20
    A2 = 1.4 * (np.log10(Re_tau) - 2.2)
    sig2 = 1.2
    mean_To = 0.82
    r1 = 0.5
    r2 = 7
    rv = np.exp(r1 * Tplus)/(np.exp(r1*r2) + np.exp(r1 * Tplus)) # correct
    rv = np.nan_to_num(rv, nan=1)  # replace NaNs with 0
    mean_To_plus = mean_To * Re_tau * np.sqrt(cf_2)
    g1 = A1 * np.exp(-sig1 * (np.log10(Tplus) - np.log10(mean_Tplus))**2)
    g2 = A2 * np.exp(-sig2 * (np.log10(Tplus) - np.log10(mean_To_plus))**2)
    return g1, g2, rv

def channel_model(Tplus, Re_tau: float, u_tau: float, u_cl) -> np.ndarray:
    A1 = 2.1*(1 - 100/Re_tau)
    sig1 = 4.4
    mean_Tplus = 12
    A2 = 0.9 * (np.log10(Re_tau) - 2.2)
    sig2 = 1.0
    mean_To = 0.6
    r1 = 0.5
    r2 = 3
    rv = np.exp(r1 * Tplus)/(np.exp(r1*r2) + np.exp(r1 * Tplus)) # correct
    rv = np.nan_to_num(rv, nan=1)  # replace NaNs with 0
    mean_To_plus = mean_To * Re_tau * u_tau/u_cl
    g1 = A1 * np.exp(-sig1 * (np.log10(Tplus) - np.log10(mean_Tplus))**2)
    g2 = A2 * np.exp(-sig2 * (np.log10(Tplus) - np.log10(mean_To_plus))**2)
    return g1, g2, rv


def plot_model_comparison():
    labels = ['0psig', '50psig', '100psig']
    colours = ['C0', 'C1', 'C2']
    psigs = [0, 50, 100]
    with h5py.File(f'data/final_pressure/SU_2pt_pressure.h5', 'r') as f:
        grp = f['raw_data']

    fig, ax = plt.subplots(2, 1, figsize=(9, 5), tight_layout=True)

    for idxfn, fn in enumerate(labels):
        with h5py.File("data/final_calibration/" +f'calibs_{fn}.h5', 'r') as hf:
            H_fused = hf['H_fused'][:].squeeze().astype(complex)
            f_cal = hf['frequencies'][:].squeeze().astype(float)
        ic(H_fused.shape, f_cal.shape)
            # Load cleaned signals and attributes
        with h5py.File("data/final_cleaned/" +f'{fn}_far_cleaned.h5', 'r') as hf:
            ph1_clean_far = hf['ph1_clean'][:]
            ph1_clean_far = correct_pressure_sensitivity(ph1_clean_far, psigs[idxfn])
            ph2_clean_far = hf['ph2_clean'][:]
            ph2_clean_far = correct_pressure_sensitivity(ph2_clean_far, psigs[idxfn])
            u_tau = float(hf.attrs['u_tau'])
            nu = float(hf.attrs['nu'])
            rho = float(hf.attrs['rho'])
            Re_tau = hf.attrs.get('Re_tau', np.nan)
            cf_2 = hf.attrs.get('cf_2', np.nan)
        ph1_clean_far = apply_frf(ph1_clean_far, FS, f_cal, H_fused)
        ph2_clean_far = apply_frf(ph2_clean_far, FS, f_cal, H_fused)
        f_clean, Pyy_ph1_clean = compute_spec(FS, ph1_clean_far)
        f_clean, Pyy_ph2_clean = compute_spec(FS, ph2_clean_far)
        T_plus = 1/f_clean * (u_tau**2)/nu

        g1_b, g2_b, rv_b = bl_model(T_plus, Re_tau, cf_2)
        bl_fphipp_plus = rv_b*(g1_b+g2_b)

        data_fphipp_plus1 = (f_clean * Pyy_ph1_clean)/(rho**2 * u_tau**4)
        data_fphipp_plus2 = (f_clean * Pyy_ph2_clean)/(rho**2 * u_tau**4)
        ax[0].semilogx(f_clean, data_fphipp_plus1, label=labels[idxfn], alpha=0.6, color=colours[idxfn], ls='-.')
        ax[0].semilogx(f_clean, data_fphipp_plus2, label=labels[idxfn], alpha=0.6, color=colours[idxfn])
        ax[0].semilogx(f_clean, bl_fphipp_plus, label=f'Model {labels[idxfn]}', linestyle='--', color=colours[idxfn])

        ax[1].semilogx(T_plus, data_fphipp_plus1, label=labels[idxfn], alpha=0.6, color=colours[idxfn], ls='-.')
        ax[1].semilogx(T_plus, data_fphipp_plus2, label=labels[idxfn], alpha=0.6, color=colours[idxfn])
        ax[1].semilogx(T_plus, bl_fphipp_plus, label=f'Model {labels[idxfn]}', linestyle='--', color=colours[idxfn])

        with h5py.File("data/final_cleaned/" +f'{fn}_close_cleaned.h5', 'r') as hf:
            ph1_clean_close = hf['ph1_clean'][:]
            ph1_clean_close = correct_pressure_sensitivity(ph1_clean_close, psigs[idxfn])
            ph2_clean_close = hf['ph2_clean'][:]
            ph2_clean_close = correct_pressure_sensitivity(ph2_clean_close, psigs[idxfn])
            u_tau = float(hf.attrs['u_tau'])
            nu = float(hf.attrs['nu'])
            rho = float(hf.attrs['rho'])
            Re_tau = hf.attrs.get('Re_tau', np.nan)
            cf_2 = hf.attrs.get('cf_2', np.nan)
        ph1_clean_close = apply_frf(ph1_clean_close, FS, f_cal, H_fused)
        ph2_clean_close = apply_frf(ph2_clean_close, FS, f_cal, H_fused)
        f_clean, Pyy_ph1_clean = compute_spec(FS, ph1_clean_close)
        f_clean, Pyy_ph2_clean = compute_spec(FS, ph2_clean_close)
        T_plus = 1/f_clean * (u_tau**2)/nu

        g1_b, g2_b, rv_b = bl_model(T_plus, Re_tau, cf_2)
        bl_fphipp_plus = rv_b*(g1_b+g2_b)

        data_fphipp_plus1 = (f_clean * Pyy_ph1_clean)/(rho**2 * u_tau**4)
        data_fphipp_plus2 = (f_clean * Pyy_ph2_clean)/(rho**2 * u_tau**4)
        ax[0].semilogx(f_clean, data_fphipp_plus1, label=labels[idxfn], alpha=0.6, color=colours[idxfn], ls='-.')
        ax[0].semilogx(f_clean, data_fphipp_plus2, label=labels[idxfn], alpha=0.6, color=colours[idxfn])
        ax[0].semilogx(f_clean, bl_fphipp_plus, label=f'Model {labels[idxfn]}', linestyle='--', color=colours[idxfn])

        ax[1].semilogx(T_plus, data_fphipp_plus1, label=labels[idxfn], alpha=0.6, color=colours[idxfn], ls='-.')
        ax[1].semilogx(T_plus, data_fphipp_plus2, label=labels[idxfn], alpha=0.6, color=colours[idxfn])
        ax[1].semilogx(T_plus, bl_fphipp_plus, label=f'Model {labels[idxfn]}', linestyle='--', color=colours[idxfn])

    ax[0].set_xlabel(r"$f$ [Hz]")
    ax[0].set_ylabel(r"${f \phi_{pp}}^+$")
    ax[0].set_xlim(1, 1e4)
    ax[0].set_ylim(0, 8)
    ax[0].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
    ax[0].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

    ax[1].set_xlabel(r"$T^+$")
    ax[1].set_ylabel(r"${f \phi_{pp}}^+$")
    ax[1].set_xlim(1, 1e4)
    ax[1].set_ylim(0, 8)
    ax[1].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
    ax[1].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

    labels_handles = ['1 000 PH2', '1 000 PH1',
                      '5 000 PH2', '5 000 PH1',
                      '9 000 PH2', '9 000 PH1']
    label_colours = ['C0', 'C0', 'C1', 'C1', 'C2', 'C2']
    label_styles = ['solid', '-.', 'solid', '-.', 'solid', '-.']
    custom_lines = [Line2D([0], [0], color=label_colours[i], linestyle=label_styles[i]) for i in range(len(labels_handles))]
    ax[0].legend(custom_lines, labels_handles, loc='upper right', fontsize=8)
    fig.savefig('figures/final/tf_corrected_spectra_comparison.png', dpi=410)

def plot_model_comparison_roi():
    labels = ['0psig', '50psig', '100psig']
    colours = ['C0', 'C1', 'C2']
    psigs = [0, 50, 100]

    fig, ax = plt.subplots(2, 1, figsize=(9, 5), tight_layout=True)
    f_cutl, f_cuth = 100, 1_000  # Hz
    u_tau_error = [0.2, 0.1, 0.05] #% uncertainty in u_tau for 0, 50, 100 psig

    for idxfn, fn in enumerate(labels):
        with h5py.File("data/final_calibration/" +f'calibs_{fn}.h5', 'r') as hf:
            H_fused = hf['H_fused'][:].squeeze().astype(complex)
            f_cal = hf['frequencies'][:].squeeze().astype(float)
            # Load cleaned signals and attributes
        with h5py.File("data/final_cleaned/" +f'{fn}_far_cleaned.h5', 'r') as hf:
            ph1_clean_far = hf['ph1_clean'][:]
            ph1_clean_far = correct_pressure_sensitivity(ph1_clean_far, psigs[idxfn])
            ph2_clean_far = hf['ph2_clean'][:]
            ph2_clean_far = correct_pressure_sensitivity(ph2_clean_far, psigs[idxfn])
            u_tau = float(hf.attrs['u_tau'])
            nu = float(hf.attrs['nu'])
            rho = float(hf.attrs['rho'])
            Re_tau = hf.attrs.get('Re_tau', np.nan)
            cf_2 = hf.attrs.get('cf_2', np.nan)
        ph1_clean_far = apply_frf(ph1_clean_far, FS, f_cal, H_fused)
        ph2_clean_far = apply_frf(ph2_clean_far, FS, f_cal, H_fused)
        f_clean, Pyy_ph1_clean = compute_spec(FS, ph1_clean_far)
        f_clean, Pyy_ph2_clean = compute_spec(FS, ph2_clean_far)
        T_plus = 1/f_clean * (u_tau**2)/nu

        g1_b, g2_b, rv_b = bl_model(T_plus, Re_tau, cf_2)
        bl_fphipp_plus = rv_b*(g1_b+g2_b)

        data_fphipp_plus1 = (f_clean * Pyy_ph1_clean)/(rho**2 * u_tau**4)
        data_fphipp_plus2 = (f_clean * Pyy_ph2_clean)/(rho**2 * u_tau**4)
        # ax[0].semilogx(f_clean, data_fphipp_plus1, label=labels[idxfn], alpha=0.6, color=colours[idxfn])
        ax[0].semilogx(f_clean[((f_clean>f_cutl) & (f_clean<f_cuth))], data_fphipp_plus2[((f_clean>f_cutl) & (f_clean<f_cuth))], label=labels[idxfn], alpha=0.6, color=colours[idxfn])
        ax[0].semilogx(f_clean, bl_fphipp_plus, label=f'Model {labels[idxfn]}', linestyle='--', color=colours[idxfn])

        # ax[1].semilogx(T_plus, data_fphipp_plus1, label=labels[idxfn], alpha=0.6, color=colours[idxfn])
        ax[1].semilogx(T_plus[((f_clean>f_cutl) & (f_clean<f_cuth))], data_fphipp_plus2[((f_clean>f_cutl) & (f_clean<f_cuth))], label=labels[idxfn], alpha=0.6, color=colours[idxfn])
        ax[1].semilogx(T_plus, bl_fphipp_plus, label=f'Model {labels[idxfn]}', linestyle='--', color=colours[idxfn])

        with h5py.File("data/final_cleaned/" +f'{fn}_close_cleaned.h5', 'r') as hf:
            ph1_clean_close = hf['ph1_clean'][:]
            ph1_clean_close = correct_pressure_sensitivity(ph1_clean_close, psigs[idxfn])
            ph2_clean_close = hf['ph2_clean'][:]
            ph2_clean_close = correct_pressure_sensitivity(ph2_clean_close, psigs[idxfn])
            u_tau = float(hf.attrs['u_tau'])
            nu = float(hf.attrs['nu'])
            rho = float(hf.attrs['rho'])
            Re_tau = hf.attrs.get('Re_tau', np.nan)
            cf_2 = hf.attrs.get('cf_2', np.nan)
        ph1_clean_close = apply_frf(ph1_clean_close, FS, f_cal, H_fused)
        ph2_clean_close = apply_frf(ph2_clean_close, FS, f_cal, H_fused)
        f_clean, Pyy_ph1_clean = compute_spec(FS, ph1_clean_close)
        f_clean, Pyy_ph2_clean = compute_spec(FS, ph2_clean_close)
        T_plus = 1/f_clean * (u_tau**2)/nu

        g1_b, g2_b, rv_b = bl_model(T_plus, Re_tau, cf_2)
        bl_fphipp_plus = rv_b*(g1_b+g2_b)

        data_fphipp_plus1 = (f_clean * Pyy_ph1_clean)/(rho**2 * u_tau**4)
        data_fphipp_plus2 = (f_clean * Pyy_ph2_clean)/(rho**2 * u_tau**4)
        # ax[0].semilogx(f_clean, data_fphipp_plus1, label=labels[idxfn], alpha=0.6, color=colours[idxfn])
        ax[0].semilogx(f_clean[((f_clean>f_cutl) & (f_clean<f_cuth))], data_fphipp_plus2[((f_clean>f_cutl) & (f_clean<f_cuth))], label=labels[idxfn], alpha=0.6, color=colours[idxfn])
        ax[0].semilogx(f_clean, bl_fphipp_plus, label=f'Model {labels[idxfn]}', linestyle='--', color=colours[idxfn])

        # ax[1].semilogx(T_plus, data_fphipp_plus1, label=labels[idxfn], alpha=0.6, color=colours[idxfn])
        ax[1].semilogx(T_plus[((f_clean>f_cutl) & (f_clean<f_cuth))], data_fphipp_plus2[((f_clean>f_cutl) & (f_clean<f_cuth))], label=labels[idxfn], alpha=0.6, color=colours[idxfn])
        # plot error bars due to u_tau uncertainty
        u_tau_low = u_tau * (1 - u_tau_error[idxfn])
        u_tau_high = u_tau * (1 + u_tau_error[idxfn])
        T_plus_low = 1/f_clean * (u_tau_low**2)/nu
        T_plus_high = 1/f_clean * (u_tau_high**2)/nu
        data_fphipp_plus2_low = (f_clean * Pyy_ph2_clean)/(rho**2 * u_tau_low**4)
        data_fphipp_plus2_high = (f_clean * Pyy_ph2_clean)/(rho**2 * u_tau_high**4)
        # ax[0].fill_between(f_clean[((f_clean>f_cutl) & (f_clean<f_cuth))],
        #                    data_fphipp_plus2_low[((f_clean>f_cutl) & (f_clean<f_cuth))],
        #                    data_fphipp_plus2_high[((f_clean>f_cutl) & (f_clean<f_cuth))],
        #                    color=colours[idxfn], alpha=0.2)
        ax[1].fill_between(T_plus[((f_clean>f_cutl) & (f_clean<f_cuth))],
                           data_fphipp_plus2_low[((f_clean>f_cutl) & (f_clean<f_cuth))],
                           data_fphipp_plus2_high[((f_clean>f_cutl) & (f_clean<f_cuth))],
                           color=colours[idxfn], alpha=0.2)
        ax[1].semilogx(T_plus, bl_fphipp_plus, label=f'Model {labels[idxfn]}', linestyle='--', color=colours[idxfn])

    ax[0].set_xlabel(r"$f$ [Hz]")
    ax[0].set_ylabel(r"${f \phi_{pp}}^+$")
    ax[0].set_xlim(50, 1e3)
    ax[0].set_ylim(0, 6)
    ax[0].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
    ax[0].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

    ax[1].set_xlabel(r"$T^+$")
    ax[1].set_ylabel(r"${f \phi_{pp}}^+$")
    ax[1].set_xlim(1, 3_000)
    ax[1].set_ylim(0, 6)
    ax[1].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
    ax[1].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

    labels_handles = ['1 000 PH2', '1 000 Model',
                      '5 000 PH2', '5 000 Model',
                      '9 000 PH2', '9 000 Model']
    label_colours = ['C0', 'C0', 'C1', 'C1', 'C2', 'C2']
    label_styles = ['solid', '--', 'solid', '--', 'solid', '--']
    custom_lines = [Line2D([0], [0], color=label_colours[i], linestyle=label_styles[i]) for i in range(len(labels_handles))]
    ax[0].legend(custom_lines, labels_handles, loc='upper left', fontsize=8)
    fig.savefig('figures/final/tf_corrected_spectra_roi.png', dpi=410)


if __name__ == "__main__":
    # plot_model_comparison()
    plot_model_comparison_roi()
    
