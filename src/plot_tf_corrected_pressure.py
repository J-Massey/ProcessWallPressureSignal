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
from matplotlib.colors import to_rgba
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

COLOURS = ("#1e8ad8", "#ff7f0e", "#26bd26")  # hex equivalents of C0, C1, C2


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


def plot_model_comparison_roi():
    labels = ['0psig', '50psig', '100psig']
    psigs = [0, 50, 100]

    fig, axs = plt.subplots(1, 3, figsize=(8, 3), tight_layout=True, sharey=True, sharex=True)
    f_cutl, f_cuth = 100, 1_000  # Hz
    u_tau_error = [0.2, 0.1, 0.05] #% uncertainty in u_tau for 0, 50, 100 psig

    for idxfn, fn in enumerate(labels):
        ax = axs[idxfn]
        with h5py.File("data/final_calibration/" +f'calibs_{psigs[idxfn]}.h5', 'r') as hf:
            H_fused = hf['H_fused'][:].squeeze().astype(complex)
            f_cal = hf['frequencies'][:].squeeze().astype(float)
        with h5py.File("data/20250930/" +f'calibs_{psigs[idxfn]}.h5', 'r') as hf:
            H_fused_nkd = hf['H_fused'][:].squeeze().astype(complex)
            f_cal_nkd = hf['frequencies'][:].squeeze().astype(float)
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
        ph1_clean_far = apply_frf(ph1_clean_far, FS, f_cal_nkd, H_fused_nkd)
        ph2_clean_far = apply_frf(ph2_clean_far, FS, f_cal_nkd, H_fused_nkd)
        f_clean, Pyy_ph1_clean = compute_spec(FS, ph1_clean_far)
        f_clean, Pyy_ph2_clean = compute_spec(FS, ph2_clean_far)
        T_plus = 1/f_clean * (u_tau**2)/nu

        g1_b, g2_b, rv_b = bl_model(T_plus, Re_tau, cf_2)
        g1_c, g2_c, rv_c = channel_model(T_plus, Re_tau, u_tau, u_cl=14)  # u_cl ~ 15 m/s
        bl_fphipp_plus = rv_b*(g1_b+g2_b)
        channel_fphipp_plus = rv_c*(g1_c+g2_c)

        data_fphipp_plus1 = (f_clean * Pyy_ph1_clean)/(rho**2 * u_tau**4)
        data_fphipp_plus2 = (f_clean * Pyy_ph2_clean)/(rho**2 * u_tau**4)
        ax.semilogx(T_plus, bl_fphipp_plus, label=f'Model {labels[idxfn]}', linestyle='--', color=COLOURS[idxfn], lw=0.7)
        ax.semilogx(T_plus, channel_fphipp_plus, label=f'Model {labels[idxfn]}', linestyle='-.', color=COLOURS[idxfn], lw=0.7)

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
        ph1_clean_close = apply_frf(ph1_clean_close, FS, f_cal_nkd, H_fused_nkd)
        ph2_clean_close = apply_frf(ph2_clean_close, FS, f_cal_nkd, H_fused_nkd)

        f_clean, Pyy_ph1_clean = compute_spec(FS, ph1_clean_close)
        f_clean, Pyy_ph2_clean = compute_spec(FS, ph2_clean_close)
        T_plus = 1/f_clean * (u_tau**2)/nu

        g1_b, g2_b, rv_b = bl_model(T_plus, Re_tau, cf_2)
        bl_fphipp_plus = rv_b*(g1_b+g2_b)

        data_fphipp_plus1 = (f_clean * Pyy_ph1_clean)/(rho**2 * u_tau**4)
        data_fphipp_plus2 = (f_clean * Pyy_ph2_clean)/(rho**2 * u_tau**4)
        # plot error bars due to u_tau uncertainty
        # mask the frequency window once
        # --- window & data ---
        mask = (f_clean > f_cutl) & (f_clean < f_cuth)
        f_m   = f_clean[mask]
        P_m   = Pyy_ph2_clean[mask]

        # --- u_tau range ---
        u_nom = u_tau
        u_lo  = u_nom * (1 - u_tau_error[idxfn])
        u_hi  = u_nom * (1 + u_tau_error[idxfn])

        # sampling across the range (odd so one is exactly nominal)
        n =  nine =  nine = 16  # keep small; adjust as desired
        u_grid = np.linspace(u_lo, u_hi, n)

        # linear fade to edges (alpha), peak at the centre
        mid = 0.5*(u_lo + u_hi)
        span = (u_hi - u_lo)
        def fade(u):
            w = 1 - abs(u - mid)/(0.5*span)          # 1 at centre, 0 at edges
            return 0.15 + 0.75*np.clip(w, 0, 1)      # avoid fully transparent

        base = 'gray' #COLOURS[idxfn]
        # draw edges first, centre last
        order = np.argsort(np.abs(u_grid - u_nom))[::-1]

        for j in order:
            u = u_grid[j]
            T = (u**2) / (nu * f_m)                                  # x (varies with u)
            Y = (f_m * P_m) / (rho**2 * u**4)                        # y (varies with u)
            a = fade(u)
            ax.semilogx(T, Y, color=to_rgba(base, a), linewidth=1)

        # nominal on top & labelled
        T_nom = (u_nom**2) / (nu * f_m)
        Y_nom = (f_m * P_m) / (rho**2 * u_nom**4)
        ax.semilogx(T_nom, Y_nom, color=COLOURS[idxfn], linewidth=1, label=labels[idxfn], zorder=10)

        ax.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

    axs[1].set_xlabel(r"$T^+$")
    axs[0].set_ylabel(r"$({f \phi_{pp}}^+)_{\mathrm{corr.}}$")
    ax.set_xlim(7, 7_000)
    ax.set_ylim(0, 6)

    labels_handles = ['1 000 PH2',
                      '5 000 PH2',
                      '9 000 PH2',]
    label_colours = COLOURS
    label_styles = ['-', '-', '-']
    custom_lines = [Line2D([0], [0], color=label_colours[i], linestyle=label_styles[i]) for i in range(len(labels_handles))]
    leg1 = ax.legend(custom_lines, labels_handles, loc='upper right', fontsize=8)
    ax.add_artist(leg1)
    labels_handles2 = ['BL model',
                       'Channel model']
    label_colours2 = ['black', 'black']
    label_styles2 = ['--', '-.']
    custom_lines2 = [Line2D([0], [0], color=label_colours2[i], linestyle=label_styles2[i]) for i in range(len(labels_handles2))]
    axs[1].legend(custom_lines2, labels_handles2, loc='upper center', fontsize=8)

    fig.savefig('figures/final/tf_corrected_spectra_roi.png', dpi=410)

def plot_model_comparison_p_sensitivity():
    labels = ['0psig', '50psig', '100psig']
    psigs = [0, 50, 100]

    fig, axs = plt.subplots(1, 3, figsize=(8, 3), tight_layout=True, sharey=True, sharex=True)
    f_cutl, f_cuth = 100, 1_000  # Hz
    u_tau_error = [0.2, 0.1, 0.05] #% uncertainty in u_tau for 0, 50, 100 psig

    for idxfn, fn in enumerate(labels):
        ax = axs[idxfn]
        with h5py.File("data/final_calibration/" +f'calibs_{psigs[idxfn]}.h5', 'r') as hf:
            H_fused = hf['H_fused'][:].squeeze().astype(complex)
            f_cal = hf['frequencies'][:].squeeze().astype(float)
        with h5py.File("data/20250930/" +f'calibs_{psigs[idxfn]}.h5', 'r') as hf:
            H_fused_nkd = hf['H_fused'][:].squeeze().astype(complex)
            f_cal_nkd = hf['frequencies'][:].squeeze().astype(float)
            # Load cleaned signals and attributes
        with h5py.File("data/final_cleaned/" +f'{fn}_far_cleaned.h5', 'r') as hf:
            ph2_clean_far = hf['ph2_clean'][:]
            ph2_clean_far_l = correct_pressure_sensitivity(ph2_clean_far, psigs[idxfn], alpha=0.005)
            ph2_clean_far_u = correct_pressure_sensitivity(ph2_clean_far, psigs[idxfn], alpha=0.015)
            u_tau = float(hf.attrs['u_tau'])
            nu = float(hf.attrs['nu'])
            rho = float(hf.attrs['rho'])
            Re_tau = hf.attrs.get('Re_tau', np.nan)
            cf_2 = hf.attrs.get('cf_2', np.nan)
        ph2_clean_far_l = apply_frf(ph2_clean_far_l, FS, f_cal, H_fused)
        ph2_clean_far_l = apply_frf(ph2_clean_far_l, FS, f_cal_nkd, H_fused_nkd)
        ph2_clean_far_u = apply_frf(ph2_clean_far_u, FS, f_cal, H_fused)
        ph2_clean_far_u = apply_frf(ph2_clean_far_u, FS, f_cal_nkd, H_fused_nkd)
        f_clean, Pyy_ph2_clean_u = compute_spec(FS, ph2_clean_far_u)
        f_clean, Pyy_ph2_clean_l = compute_spec(FS, ph2_clean_far_l)
        T_plus = 1/f_clean * (u_tau**2)/nu

        g1_b, g2_b, rv_b = bl_model(T_plus, Re_tau, cf_2)
        g1_c, g2_c, rv_c = channel_model(T_plus, Re_tau, u_tau, u_cl=14)  # u_cl ~ 15 m/s
        bl_fphipp_plus = rv_b*(g1_b+g2_b)
        channel_fphipp_plus = rv_c*(g1_c+g2_c)
        
        ax.semilogx(T_plus, bl_fphipp_plus, label=f'Model {labels[idxfn]}', linestyle='--', color=COLOURS[idxfn], lw=0.7)
        ax.semilogx(T_plus, channel_fphipp_plus, label=f'Model {labels[idxfn]}', linestyle='-.', color=COLOURS[idxfn], lw=0.7)

        # plot error bars due to u_tau uncertainty
        # mask the frequency window once
        # --- window & data ---
        mask = (f_clean > f_cutl) & (f_clean < f_cuth)
        f_m   = f_clean[mask]
        P_m_u   = Pyy_ph2_clean_u[mask]
        P_m_l   = Pyy_ph2_clean_l[mask]

        # --- u_tau range ---
        u_nom = u_tau

        # nominal on top & labelled
        T_nom = (u_nom**2) / (nu * f_m)
        Y_nom_u = (f_m * P_m_u) / (rho**2 * u_nom**4)
        Y_nom_l = (f_m * P_m_l) / (rho**2 * u_nom**4)
        ax.semilogx(T_nom, Y_nom_u, color=COLOURS[idxfn], linewidth=1, label=labels[idxfn], zorder=10)
        ax.semilogx(T_nom, Y_nom_l, color=COLOURS[idxfn], linewidth=1, label=labels[idxfn], zorder=10)

        ax.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

    axs[1].set_xlabel(r"$T^+$")
    axs[0].set_ylabel(r"$({f \phi_{pp}}^+)_{\mathrm{corr.}}$")
    ax.set_xlim(7, 7_000)
    ax.set_ylim(0, 6)

    labels_handles = ['1 000 PH2',
                      '5 000 PH2',
                      '9 000 PH2',]
    label_colours = COLOURS
    label_styles = ['-', '-', '-']
    custom_lines = [Line2D([0], [0], color=label_colours[i], linestyle=label_styles[i]) for i in range(len(labels_handles))]
    leg1 = ax.legend(custom_lines, labels_handles, loc='upper right', fontsize=8)
    ax.add_artist(leg1)
    labels_handles2 = ['BL model',
                       'Channel model']
    label_colours2 = ['black', 'black']
    label_styles2 = ['--', '-.']
    custom_lines2 = [Line2D([0], [0], color=label_colours2[i], linestyle=label_styles2[i]) for i in range(len(labels_handles2))]
    axs[1].legend(custom_lines2, labels_handles2, loc='upper center', fontsize=8)

    fig.suptitle(r'Pressure sensitivity correction $\alpha\in[0.005, 0.015]dB kPa^{-1}$ variation')

    fig.savefig('figures/final/tf_corrected_spectra_p_sensitivity.png', dpi=410)


if __name__ == "__main__":
    # plot_model_comparison()
    # plot_model_comparison_roi()
    plot_model_comparison_p_sensitivity()
    
