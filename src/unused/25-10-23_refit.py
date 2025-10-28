import numpy as np
from scipy.signal import welch, csd, get_window, iirnotch, sosfiltfilt, find_peaks
import scipy.signal as signal
import scipy.io as sio
from matplotlib import pyplot as plt
from icecream import ic
import os

from tqdm import tqdm

from wiener_filter_gib import wiener_cancel_background
from wiener_filter_torch import wiener_cancel_background_torch
from stft_wiener import wiener_cancel_background_stft_torch

from plotting import (
    plot_spectrum,
    plot_raw_spectrum,
    plot_transfer_NKD,
    plot_transfer_PH,
    plot_transfer_NC,
    plot_corrected_trace_NKD,
    plot_corrected_trace_NC,
    plot_corrected_trace_PH,
    plot_time_series,
    plot_spectrum_pipeline,
)

plt.style.use(["science", "grid"])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")

############################
#Plan
#    \item Raw TS
#    \item Raw spectra
#    \item PH calibration
#    \item TF corrected TS
#    \item TF corrected spectra
#    \item Notched electrical facility noise
#    \item Notched FS and PH spectra
#    \item Coherence correction
#    \item Cleaned spectra
############################

############################
# Constants & defaults
############################
FS = 50_000.0
NPERSEG = 2**15
WINDOW = "hann"
TRIM_CAL_SECS = 5  # seconds trimmed from the start of calibration runs (0 to disable)

R = 287.0         # J/kg/K
T = 298.0         # K (adjust if you have per-case temps)
P_0psi = 101_325.0 # Pa
PSI_TO_PA = 6_894.76

nc_colour = '#1f77b4'  # matplotlib default blue
ph1_colour = "#c76713"  # matplotlib default orange
ph2_colour = "#9fda16"  # matplotlib default red
nkd_colour = '#2ca02c' # matplotlib default green

# -------------------------------------------------------------------------
# >>> Units & column layout for loaded time-series (per MATLAB key) <<<
# Keys (column order):
#   - channelData_300_plug : col1=PH (pinhole), col2=NKD (naked)   [calibration sweep]
#   - channelData_300_nose : col1=NKD,            col2=NC          [calibration sweep]
#   - channelData_300      : col1=NC,             col2=PH          [real flow data]
DEFAULT_UNITS = {
    'channelData_300_plug': ('Pa', 'Pa'),  # PH, NKD
    'channelData_300_nose': ('Pa', 'Pa'),  # NKD, NC
    'channelData_300':      ('Pa', 'Pa'),  # NC,  PH
}

# sensitivity = 20*1e-3 # V/Pa
sensitivity = 316e-3 * 50e-3 # V/Pa

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

def apply_frf(
    x: np.ndarray,
    fs: float,
    f: np.ndarray,
    H: np.ndarray,
    demean: bool = True,
    zero_dc: bool = True,
):
    """
    Apply a measured FRF H (x→y) to a time series x to synthesise y.
    This is the forward operation: Y = H · X in the frequency domain.
    """
    x = np.asarray(x, float)
    if demean:
        x = x - x.mean()

    N = x.size
    Nfft = int(2 ** np.ceil(np.log2(N)))
    X = np.fft.rfft(x, n=Nfft)
    fr = np.fft.rfftfreq(Nfft, d=1.0 / fs)

    mag = np.abs(H)
    phi = np.unwrap(np.angle(H))
    # Safer OOB behaviour: taper magnitude to zero outside measured band
    mag_i = np.interp(fr, f, mag, left=1.0, right=1.0)
    phi_i = np.interp(fr, f, phi, left=phi[0], right=phi[-1])
    Hi = mag_i * np.exp(1j * phi_i)

    if zero_dc:
        Hi[0] = 0.0
        if Nfft % 2 == 0:
            Hi[-1] = 0.0

    y = np.fft.irfft(X * Hi, n=Nfft)[:N]
    return y

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


def estimate_frf(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    window: str = WINDOW,
    detrend: str = "false",
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



def rerun(plot=[0,1,2,3,4]):
    u_tau = 0.58
    nu_utau = 27e-6
    nu = nu_utau * u_tau
    f_cut = 2_100
    T_plus_fcut = 1/f_cut * (u_tau**2)/nu
    rho = 1.2 # kg/m^3
    cf = 2*(u_tau**2)/14**2
    Re_tau = u_tau * 0.035 / nu
    ic(Re_tau)

    rrun = 'data/20251014/flow_data/far/0psi.mat'
    dat = sio.loadmat(rrun) # options are channelData_LP, channelData_NF
    ic(dat.keys())
    nkd_far = dat['channelData_flow'][:,2] * 1/sensitivity
    ph1_far = dat['channelData_flow'][:,0] * 1/sensitivity
    ph2_far = dat['channelData_flow'][:,1] * 1/sensitivity
    # Apply band pass at 0.1-f_cut Hz
    nkd_far = bandpass_filter(nkd_far, 0.1, f_cut, FS)
    ph1_far = bandpass_filter(ph1_far, 0.1, f_cut, FS)
    ph2_far = bandpass_filter(ph2_far, 0.1, f_cut, FS)

    f_far, Pyy_nkd_far = compute_spec(FS, nkd_far)
    f_far, Pyy_ph1_far = compute_spec(FS, ph1_far)
    f_far, Pyy_ph2_far = compute_spec(FS, ph2_far)

    if 0 in plot:
        fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, tight_layout=True)
        T_plus = 1/f_far * (u_tau**2)/nu
        g1, g2, rv = bl_model(T_plus, Re_tau, cf)
        g1_c, g2_c, rv_c = channel_model(T_plus, Re_tau, u_tau, 14)

        fig.suptitle(r"$Re_\tau\approx$ 1,300 (700$\mu$m) - Original")

        ax[0].semilogx(f_far, (f_far * Pyy_nkd_far)/(rho**2 * u_tau**4), label='NC', color=nkd_colour)
        ax[0].semilogx(f_far, (f_far * Pyy_ph1_far)/(rho**2 * u_tau**4), label='PH1', color=ph1_colour)
        ax[0].semilogx(f_far, (f_far * Pyy_ph2_far)/(rho**2 * u_tau**4), label='PH2', color=ph2_colour)
        ax[0].semilogx(f_far, rv*(g1+g2), label='BL Model', color='k', linestyle='--')
        ax[0].semilogx(f_far, rv_c*(g1_c+g2_c), label='Channel Model', color='k', linestyle='-.')
        ax[0].axvline(f_cut, color='red', linestyle='--')
        ax[0].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
        ax[0].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

        ax[0].set_xlabel("$f$ [Hz]")

        ax[1].semilogx(T_plus, (f_far * Pyy_nkd_far)/(rho**2 * u_tau**4), label='NC', color=nkd_colour)
        ax[1].semilogx(T_plus, (f_far * Pyy_ph1_far)/(rho**2 * u_tau**4), label='PH1', color=ph1_colour)
        ax[1].semilogx(T_plus, (f_far * Pyy_ph2_far)/(rho**2 * u_tau**4), label='PH2', color=ph2_colour)
        ax[1].semilogx(T_plus, rv*(g1+g2), label='BL Model', color='k', linestyle='--')
        ax[1].semilogx(T_plus, rv_c*(g1_c+g2_c), label='Channel Model', color='k', linestyle='-.')

        ax[1].axvline(T_plus_fcut, color='red', linestyle='--')
        ax[1].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
        ax[1].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

        ax[1].set_xlabel("$T^+$")
        ax[0].set_ylabel(r"${f \phi_{pp}}^+$")

        ax[0].set_xlim(1, 1e4)
        ax[0].set_ylim(0, 5)

        ax[0].legend()
        ax[1].legend()
        fig.savefig('figures/remount_test/original.png', dpi=410)

        ph1_clean = wiener_cancel_background_torch(ph1_far, nkd_far, FS).cpu().numpy()
        ph2_clean = wiener_cancel_background_torch(ph2_far, nkd_far, FS).cpu().numpy()
        f_clean, Pyy_ph1_clean = compute_spec(FS, ph1_clean)
        f_clean, Pyy_ph2_clean = compute_spec(FS, ph2_clean)

        fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, tight_layout=True)
        fig.suptitle(r"$Re_\tau\approx$ 1,300 (700$\mu$m) - Original")
        ax[0].semilogx(f_clean, (f_clean * Pyy_ph1_clean)/(rho**2 * u_tau**4), label='PH1 Cleaned', color=ph1_colour)
        ax[0].semilogx(f_far, (f_far * Pyy_ph1_far)/(rho**2 * u_tau**4), label='PH1 Original', color=ph1_colour, alpha=0.3)
        ax[0].semilogx(f_clean, (f_clean * Pyy_ph2_clean)/(rho**2 * u_tau**4), label='PH2 Cleaned', color=ph2_colour)
        ax[0].semilogx(f_far, (f_far * Pyy_ph2_far)/(rho**2 * u_tau**4), label='PH2 Original', color=ph2_colour, alpha=0.3)
        ax[0].semilogx(f_far, rv*(g1+g2), label='BL Model', color='k', linestyle='--')
        ax[0].semilogx(f_far, rv_c*(g1_c+g2_c), label='Channel Model', color='k', linestyle='-.')
        ax[0].axvline(f_cut, color='red', linestyle='--')
        ax[0].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
        ax[0].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

        ax[0].set_xlabel("$f$ [Hz]")

        T_plus_clean = 1/f_clean * (u_tau**2)/nu
        ax[1].semilogx(T_plus_clean, (f_clean * Pyy_ph1_clean)/(rho**2 * u_tau**4), label='PH1 Cleaned', color=ph1_colour)
        ax[1].semilogx(T_plus, (f_far * Pyy_ph1_far)/(rho**2 * u_tau**4), label='PH1 Original', color=ph1_colour, alpha=0.3)
        ax[1].semilogx(T_plus_clean, (f_clean * Pyy_ph2_clean)/(rho**2 * u_tau**4), label='PH2 Cleaned', color=ph2_colour)
        ax[1].semilogx(T_plus, (f_far * Pyy_ph2_far)/(rho**2 * u_tau**4), label='PH2 Original', color=ph2_colour, alpha=0.3)
        ax[1].semilogx(T_plus, rv*(g1+g2), label='BL Model', color='k', linestyle='--')
        ax[1].semilogx(T_plus, rv_c*(g1_c+g2_c), label='Channel Model', color='k', linestyle='-.')
        ax[1].axvline(T_plus_fcut, color='red', linestyle='--')
        ax[1].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
        ax[1].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

        ax[1].set_xlabel(r"$T^+$")

        ax[0].set_ylabel(r"${f \phi_{pp}}^+$")
        ax[0].set_xlim(1, 1e4)

        ax[0].set_ylim(0, 15)
        ax[0].legend()
        fig.savefig('figures/remount_test/original_cleaned.png', dpi=410)
    
    rrun = 'data/20251023/remount/reRun.mat'
    dat = sio.loadmat(rrun) # options are channelData_LP, channelData_NF
    ic(dat.keys())
    nkd_far = dat['channelData_LP'][:,2]* 1/sensitivity
    ph1_far = dat['channelData_LP'][:,0]* 1/sensitivity
    ph2_far = dat['channelData_LP'][:,1]* 1/sensitivity
    f_far, Pyy_nkd_far = compute_spec(FS, nkd_far)
    f_far, Pyy_ph1_far = compute_spec(FS, ph1_far)
    f_far, Pyy_ph2_far = compute_spec(FS, ph2_far)

    if 1 in plot:
        fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, tight_layout=True)
        T_plus = 1/f_far * (u_tau**2)/nu
        g1, g2, rv = bl_model(T_plus, Re_tau, cf)

        fig.suptitle(r"$Re_\tau\approx$ 1,300 (700$\mu$m) - Original Refit")

        ax[0].semilogx(f_far, (f_far * Pyy_nkd_far)/(rho**2 * u_tau**4), label='NC', color=nkd_colour)
        ax[0].semilogx(f_far, (f_far * Pyy_ph1_far)/(rho**2 * u_tau**4), label='PH1', color=ph1_colour)
        ax[0].semilogx(f_far, (f_far * Pyy_ph2_far)/(rho**2 * u_tau**4), label='PH2', color=ph2_colour)
        ax[0].semilogx(f_far, rv*(g1+g2), label='BL Model', color='k', linestyle='--')
        ax[0].semilogx(f_far, rv_c*(g1_c+g2_c), label='Channel Model', color='k', linestyle='-.')
        ax[0].axvline(f_cut, color='red', linestyle='--')
        ax[0].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
        ax[0].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)


        ax[0].set_xlabel("$f$ [Hz]")

        ax[1].semilogx(T_plus, (f_far * Pyy_nkd_far)/(rho**2 * u_tau**4), label='NC', color=nkd_colour)
        ax[1].semilogx(T_plus, (f_far * Pyy_ph1_far)/(rho**2 * u_tau**4), label='PH1', color=ph1_colour)
        ax[1].semilogx(T_plus, (f_far * Pyy_ph2_far)/(rho**2 * u_tau**4), label='PH2', color=ph2_colour)
        ax[1].semilogx(T_plus, rv*(g1+g2), label='BL Model', color='k', linestyle='--')
        ax[1].semilogx(T_plus, rv_c*(g1_c+g2_c), label='Channel Model', color='k', linestyle='-.')
        ax[1].axvline(T_plus_fcut, color='red', linestyle='--')
        ax[1].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
        ax[1].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

        ax[1].set_xlabel("$T^+$")
        ax[0].set_ylabel(r"${f \phi_{pp}}^+$")

        ax[0].set_xlim(1, 1e4)
        ax[0].set_ylim(0, 5)
        ax[0].legend()
        ax[1].legend()
        fig.savefig('figures/remount_test/refit_original.png', dpi=410)

        ph1_clean = wiener_cancel_background_torch(ph1_far, nkd_far, FS).cpu().numpy()
        ph2_clean = wiener_cancel_background_torch(ph2_far, nkd_far, FS).cpu().numpy()
        f_clean, Pyy_ph1_clean = compute_spec(FS, ph1_clean)
        f_clean, Pyy_ph2_clean = compute_spec(FS, ph2_clean)

        fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, tight_layout=True)
        ax[0].semilogx(f_clean, (f_clean * Pyy_ph1_clean)/(rho**2 * u_tau**4), label='PH1 Cleaned', color=ph1_colour)
        ax[0].semilogx(f_far, (f_far * Pyy_ph1_far)/(rho**2 * u_tau**4), label='PH1 Original', color=ph1_colour, alpha=0.3)
        ax[0].semilogx(f_clean, (f_clean * Pyy_ph2_clean)/(rho**2 * u_tau**4), label='PH2 Cleaned', color=ph2_colour)
        ax[0].semilogx(f_far, (f_far * Pyy_ph2_far)/(rho**2 * u_tau**4), label='PH2 Original', color=ph2_colour, alpha=0.3)
        ax[0].semilogx(f_far, rv*(g1+g2), label='BL Model', color='k', linestyle='--')
        ax[0].semilogx(f_far, rv_c*(g1_c+g2_c), label='Channel Model', color='k', linestyle='-.')
        ax[0].axvline(f_cut, color='red', linestyle='--')
        ax[0].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
        ax[0].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

        ax[0].set_xlabel("$f$ [Hz]")

        T_plus_clean = 1/f_clean * (u_tau**2)/nu
        ax[1].semilogx(T_plus_clean, (f_clean * Pyy_ph1_clean)/(rho**2 * u_tau**4), label='PH1 Cleaned', color=ph1_colour)
        ax[1].semilogx(T_plus, (f_far * Pyy_ph1_far)/(rho**2 * u_tau**4), label='PH1 Original', color=ph1_colour, alpha=0.3)
        ax[1].semilogx(T_plus_clean, (f_clean * Pyy_ph2_clean)/(rho**2 * u_tau**4), label='PH2 Cleaned', color=ph2_colour)
        ax[1].semilogx(T_plus, (f_far * Pyy_ph2_far)/(rho**2 * u_tau**4), label='PH2 Original', color=ph2_colour, alpha=0.3)
        ax[1].semilogx(T_plus, rv*(g1+g2), label='BL Model', color='k', linestyle='--')
        ax[1].semilogx(T_plus, rv_c*(g1_c+g2_c), label='Channel Model', color='k', linestyle='-.')
        ax[1].axvline(T_plus_fcut, color='red', linestyle='--')
        ax[1].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
        ax[1].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

        ax[1].set_xlabel(r"$T^+$")

        ax[0].set_ylabel(r"${f \phi_{pp}}^+$")
        ax[0].set_xlim(1, 1e4)

        ax[0].set_ylim(0, 15)
        ax[0].legend()
        fig.savefig('figures/remount_test/refit_original_cleaned.png', dpi=410)

    rrun = 'data/20251023/remount/swtichedNkdPH_run1.mat'
    dat = sio.loadmat(rrun) # options are channelData_LP, channelData_NF
    ic(dat.keys())
    nkd_far = dat['channelData_LP'][:,2]* 1/sensitivity
    ph1_far = dat['channelData_LP'][:,0]* 1/sensitivity
    ph2_far = dat['channelData_LP'][:,1]* 1/sensitivity
    f_far, Pyy_nkd_far = compute_spec(FS, nkd_far)
    f_far, Pyy_ph1_far = compute_spec(FS, ph1_far)
    f_far, Pyy_ph2_far = compute_spec(FS, ph2_far)

    if 0 in plot:
        fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, tight_layout=True)
        T_plus = 1/f_far * (u_tau**2)/nu
        g1, g2, rv = bl_model(T_plus, Re_tau, cf)

        fig.suptitle(r"$Re_\tau\approx$ 1,300 (700$\mu$m) - Switched PH1/NKD Run 1")
        
        ax[0].semilogx(f_far, (f_far * Pyy_nkd_far)/(rho**2 * u_tau**4), label='NC', color=nkd_colour)
        ax[0].semilogx(f_far, (f_far * Pyy_ph1_far)/(rho**2 * u_tau**4), label='PH1', color=ph1_colour)
        ax[0].semilogx(f_far, (f_far * Pyy_ph2_far)/(rho**2 * u_tau**4), label='PH2', color=ph2_colour)
        ax[0].semilogx(f_far, rv*(g1+g2), label='BL Model', color='k', linestyle='--')
        ax[0].semilogx(f_far, rv_c*(g1_c+g2_c), label='Channel Model', color='k', linestyle='-.')
        ax[0].axvline(f_cut, color='red', linestyle='--')
        ax[0].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
        ax[0].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

        ax[0].set_xlabel("$f$ [Hz]")

        ax[1].semilogx(T_plus, (f_far * Pyy_nkd_far)/(rho**2 * u_tau**4), label='NC', color=nkd_colour)
        ax[1].semilogx(T_plus, (f_far * Pyy_ph1_far)/(rho**2 * u_tau**4), label='PH1', color=ph1_colour)
        ax[1].semilogx(T_plus, (f_far * Pyy_ph2_far)/(rho**2 * u_tau**4), label='PH2', color=ph2_colour)
        ax[1].semilogx(T_plus, rv*(g1+g2), label='BL Model', color='k', linestyle='--')
        ax[1].semilogx(T_plus, rv_c*(g1_c+g2_c), label='Channel Model', color='k', linestyle='-.')
        ax[1].axvline(T_plus_fcut, color='red', linestyle='--')
        ax[1].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
        ax[1].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

        ax[1].set_xlabel("$T^+$")
        ax[0].set_ylabel(r"${f \phi_{pp}}^+$")

        ax[0].set_xlim(1, 1e4)
        ax[0].set_ylim(0, 5)

        ax[0].legend()
        ax[1].legend()
        fig.savefig('figures/remount_test/switched1.png', dpi=410)

        ph1_clean = wiener_cancel_background_torch(ph1_far, nkd_far, FS).cpu().numpy()
        ph2_clean = wiener_cancel_background_torch(ph2_far, nkd_far, FS).cpu().numpy()
        f_clean, Pyy_ph1_clean = compute_spec(FS, ph1_clean)
        f_clean, Pyy_ph2_clean = compute_spec(FS, ph2_clean)

        fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, tight_layout=True)
        ax[0].semilogx(f_clean, (f_clean * Pyy_ph1_clean)/(rho**2 * u_tau**4), label='PH1 Cleaned', color=ph1_colour)
        ax[0].semilogx(f_far, (f_far * Pyy_ph1_far)/(rho**2 * u_tau**4), label='PH1 Original', color=ph1_colour, alpha=0.3)
        ax[0].semilogx(f_clean, (f_clean * Pyy_ph2_clean)/(rho**2 * u_tau**4), label='PH2 Cleaned', color=ph2_colour)
        ax[0].semilogx(f_far, (f_far * Pyy_ph2_far)/(rho**2 * u_tau**4), label='PH2 Original', color=ph2_colour, alpha=0.3)
        ax[0].semilogx(f_far, rv*(g1+g2), label='BL Model', color='k', linestyle='--')
        ax[0].semilogx(f_far, rv_c*(g1_c+g2_c), label='Channel Model', color='k', linestyle='-.')
        ax[0].axvline(f_cut, color='red', linestyle='--')
        ax[0].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
        ax[0].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

        ax[0].set_xlabel("$f$ [Hz]")

        T_plus_clean = 1/f_clean * (u_tau**2)/nu
        ax[1].semilogx(T_plus_clean, (f_clean * Pyy_ph1_clean)/(rho**2 * u_tau**4), label='PH1 Cleaned', color=ph1_colour)
        ax[1].semilogx(T_plus, (f_far * Pyy_ph1_far)/(rho**2 * u_tau**4), label='PH1 Original', color=ph1_colour, alpha=0.3)
        ax[1].semilogx(T_plus_clean, (f_clean * Pyy_ph2_clean)/(rho**2 * u_tau**4), label='PH2 Cleaned', color=ph2_colour)
        ax[1].semilogx(T_plus, (f_far * Pyy_ph2_far)/(rho**2 * u_tau**4), label='PH2 Original', color=ph2_colour, alpha=0.3)
        ax[1].semilogx(T_plus, rv*(g1+g2), label='BL Model', color='k', linestyle='--')
        ax[1].semilogx(T_plus, rv_c*(g1_c+g2_c), label='Channel Model', color='k', linestyle='-.')
        ax[1].axvline(T_plus_fcut, color='red', linestyle='--')
        ax[1].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
        ax[1].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

        ax[1].set_xlabel(r"$T^+$")

        ax[0].set_ylabel(r"${f \phi_{pp}}^+$")
        ax[0].set_xlim(1, 1e4)

        ax[0].set_ylim(0, 5)
        ax[0].legend()
        fig.savefig('figures/remount_test/switched1_cleaned.png', dpi=410)

    rrun = 'data/20251023/remount/swtichedNkdPH_run2.mat'
    dat = sio.loadmat(rrun) # options are channelData_LP, channelData_NF
    ic(dat.keys())
    nkd_far = dat['channelData_LP'][:,2]* 1/sensitivity
    ph1_far = dat['channelData_LP'][:,0]* 1/sensitivity
    ph2_far = dat['channelData_LP'][:,1]* 1/sensitivity
    f_far, Pyy_nkd_far = compute_spec(FS, nkd_far)
    f_far, Pyy_ph1_far = compute_spec(FS, ph1_far)
    f_far, Pyy_ph2_far = compute_spec(FS, ph2_far)

    if 0 in plot:
        fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, tight_layout=True)
        T_plus = 1/f_far * (u_tau**2)/nu
        g1, g2, rv = bl_model(T_plus, Re_tau, cf)

        fig.suptitle(r"$Re_\tau\approx$ 1,300 (700$\mu$m) - Switched PH2/NKD Run 2")

        ax[0].semilogx(f_far, (f_far * Pyy_nkd_far)/(rho**2 * u_tau**4), label='NC', color=nkd_colour)
        ax[0].semilogx(f_far, (f_far * Pyy_ph1_far)/(rho**2 * u_tau**4), label='PH1', color=ph1_colour)
        ax[0].semilogx(f_far, (f_far * Pyy_ph2_far)/(rho**2 * u_tau**4), label='PH2', color=ph2_colour)
        ax[0].semilogx(f_far, rv*(g1+g2), label='BL Model', color='k', linestyle='--')
        ax[0].semilogx(f_far, rv_c*(g1_c+g2_c), label='Channel Model', color='k', linestyle='-.')
        ax[0].axvline(f_cut, color='red', linestyle='--')
        ax[0].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
        ax[0].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

        ax[0].set_xlabel("$f$ [Hz]")

        ax[1].semilogx(T_plus, (f_far * Pyy_nkd_far)/(rho**2 * u_tau**4), label='NC', color=nkd_colour)
        ax[1].semilogx(T_plus, (f_far * Pyy_ph1_far)/(rho**2 * u_tau**4), label='PH1', color=ph1_colour)
        ax[1].semilogx(T_plus, (f_far * Pyy_ph2_far)/(rho**2 * u_tau**4), label='PH2', color=ph2_colour)
        ax[1].semilogx(T_plus, rv*(g1+g2), label='BL Model', color='k', linestyle='--')
        ax[1].semilogx(T_plus, rv_c*(g1_c+g2_c), label='Channel Model', color='k', linestyle='-.')
        ax[1].axvline(T_plus_fcut, color='red', linestyle='--')
        ax[1].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
        ax[1].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

        ax[1].set_xlabel("$T^+$")
        ax[0].set_ylabel(r"${f \phi_{pp}}^+$")

        ax[0].set_xlim(1, 1e4)
        ax[0].set_ylim(0, 5)

        ax[0].legend()
        ax[1].legend()
        fig.savefig('figures/remount_test/switched2.png', dpi=410)

        ph1_clean = wiener_cancel_background_torch(ph1_far, nkd_far, FS).cpu().numpy()
        ph2_clean = wiener_cancel_background_torch(ph2_far, nkd_far, FS).cpu().numpy()
        f_clean, Pyy_ph1_clean = compute_spec(FS, ph1_clean)
        f_clean, Pyy_ph2_clean = compute_spec(FS, ph2_clean)

        fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, tight_layout=True)
        ax[0].semilogx(f_clean, (f_clean * Pyy_ph1_clean)/(rho**2 * u_tau**4), label='PH1 Cleaned', color=ph1_colour)
        ax[0].semilogx(f_far, (f_far * Pyy_ph1_far)/(rho**2 * u_tau**4), label='PH1 Original', color=ph1_colour, alpha=0.3)
        ax[0].semilogx(f_clean, (f_clean * Pyy_ph2_clean)/(rho**2 * u_tau**4), label='PH2 Cleaned', color=ph2_colour)
        ax[0].semilogx(f_far, (f_far * Pyy_ph2_far)/(rho**2 * u_tau**4), label='PH2 Original', color=ph2_colour, alpha=0.3)
        ax[0].semilogx(f_far, rv*(g1+g2), label='BL Model', color='k', linestyle='--')
        ax[0].semilogx(f_far, rv_c*(g1_c+g2_c), label='Channel Model', color='k', linestyle='-.')
        ax[0].axvline(f_cut, color='red', linestyle='--')
        ax[0].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
        ax[0].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

        ax[0].set_xlabel("$f$ [Hz]")

        T_plus_clean = 1/f_clean * (u_tau**2)/nu
        ax[1].semilogx(T_plus_clean, (f_clean * Pyy_ph1_clean)/(rho**2 * u_tau**4), label='PH1 Cleaned', color=ph1_colour)
        ax[1].semilogx(T_plus, (f_far * Pyy_ph1_far)/(rho**2 * u_tau**4), label='PH1 Original', color=ph1_colour, alpha=0.3)
        ax[1].semilogx(T_plus_clean, (f_clean * Pyy_ph2_clean)/(rho**2 * u_tau**4), label='PH2 Cleaned', color=ph2_colour)
        ax[1].semilogx(T_plus, (f_far * Pyy_ph2_far)/(rho**2 * u_tau**4), label='PH2 Original', color=ph2_colour, alpha=0.3)
        ax[1].semilogx(T_plus, rv*(g1+g2), label='BL Model', color='k', linestyle='--')
        ax[1].semilogx(T_plus, rv_c*(g1_c+g2_c), label='Channel Model', color='k', linestyle='-.')
        ax[1].axvline(T_plus_fcut, color='red', linestyle='--')
        ax[1].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
        ax[1].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

        ax[1].set_xlabel(r"$T^+$")

        ax[0].set_ylabel(r"${f \phi_{pp}}^+$")
        ax[0].set_xlim(1, 1e4)

        ax[0].set_ylim(0, 5)
        ax[0].legend()
        fig.savefig('figures/remount_test/switched2_cleaned.png', dpi=410)

def plot_remount_new(plot=[0]):

    u_tau = 0.58
    nu_utau = 27e-6
    nu = nu_utau * u_tau
    f_cut = 2_100
    T_plus_fcut = 1/f_cut * (u_tau**2)/nu
    rho = 1.2 # kg/m^3
    cf = 2*(u_tau**2)/14**2
    Re_tau = u_tau * 0.035 / nu
    ic(Re_tau)
    sensitivity = 316e-3 * 50e-3

    rrun = 'data/20251023/remount/newmounting.mat'
    dat = sio.loadmat(rrun) # options are channelData_LP, channelData_NF
    ic(dat.keys())
    nkd_far = dat['channelData_LP'][:,2] * 1/sensitivity
    ph1_far = dat['channelData_LP'][:,0] * 1/sensitivity
    ph2_far = dat['channelData_LP'][:,1] * 1/sensitivity
    f_far, Pyy_nkd_far = compute_spec(FS, nkd_far)
    f_far, Pyy_ph1_far = compute_spec(FS, ph1_far)
    f_far, Pyy_ph2_far = compute_spec(FS, ph2_far)

    if 0 in plot:
        fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, tight_layout=True)
        T_plus = 1/f_far * (u_tau**2)/nu
        g1, g2, rv = bl_model(T_plus, Re_tau, cf)
        g1_c, g2_c, rv_c = channel_model(T_plus, Re_tau, u_tau, 14)

    if 0 in plot:
        fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, tight_layout=True)
        T_plus = 1/f_far * (u_tau**2)/nu
        g1, g2, rv = bl_model(T_plus, Re_tau, cf)
        g1_c, g2_c, rv_c = channel_model(T_plus, Re_tau, u_tau, 14)


        fig.suptitle(r"$Re_\tau\approx$ 1,300 (700$\mu$m) - New Mounting")

        ax[0].semilogx(f_far, (f_far * Pyy_nkd_far)/(rho**2 * u_tau**4), label='NC', color=nkd_colour)
        ax[0].semilogx(f_far, (f_far * Pyy_ph1_far)/(rho**2 * u_tau**4), label='PH1', color=ph1_colour)
        ax[0].semilogx(f_far, (f_far * Pyy_ph2_far)/(rho**2 * u_tau**4), label='PH2', color=ph2_colour)
        ax[0].semilogx(f_far, rv*(g1+g2), label='BL Model', color='k', linestyle='--')
        ax[0].semilogx(f_far, rv_c*(g1_c+g2_c), label='Channel Model', color='k', linestyle='-.')
        ax[0].axvline(f_cut, color='red', linestyle='--')
        ax[0].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
        ax[0].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

        ax[0].set_xlabel("$f$ [Hz]")

        ax[1].semilogx(T_plus, (f_far * Pyy_nkd_far)/(rho**2 * u_tau**4), label='NC', color=nkd_colour)
        ax[1].semilogx(T_plus, (f_far * Pyy_ph1_far)/(rho**2 * u_tau**4), label='PH1', color=ph1_colour)
        ax[1].semilogx(T_plus, (f_far * Pyy_ph2_far)/(rho**2 * u_tau**4), label='PH2', color=ph2_colour)
        ax[1].semilogx(T_plus, rv*(g1+g2), label='BL Model', color='k', linestyle='--')
        ax[1].semilogx(T_plus, rv_c*(g1_c+g2_c), label='Channel Model', color='k', linestyle='-.')
        ax[1].axvline(T_plus_fcut, color='red', linestyle='--')
        ax[1].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
        ax[1].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

        ax[1].set_xlabel("$T^+$")
        ax[0].set_ylabel(r"${f \phi_{pp}}^+$")

        ax[0].set_xlim(1, 1e4)
        ax[0].set_ylim(0, 8)

        ax[0].legend()
        ax[1].legend()
        fig.savefig('figures/remount_test/new_mount.png', dpi=410)

        ph1_clean = wiener_cancel_background_torch(ph1_far, nkd_far, FS).cpu().numpy()
        ph2_clean = wiener_cancel_background_torch(ph2_far, nkd_far, FS).cpu().numpy()
        f_clean, Pyy_ph1_clean = compute_spec(FS, ph1_clean)
        f_clean, Pyy_ph2_clean = compute_spec(FS, ph2_clean)

        fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, tight_layout=True)
        fig.suptitle(r"$Re_\tau\approx$ 1,300 (700$\mu$m) - New Mounting")

        ax[0].semilogx(f_clean, (f_clean * Pyy_ph1_clean)/(rho**2 * u_tau**4), label='PH1 Cleaned', color=ph1_colour)
        ax[0].semilogx(f_far, (f_far * Pyy_ph1_far)/(rho**2 * u_tau**4), label='PH1 Original', color=ph1_colour, alpha=0.3)
        ax[0].semilogx(f_clean, (f_clean * Pyy_ph2_clean)/(rho**2 * u_tau**4), label='PH2 Cleaned', color=ph2_colour)
        ax[0].semilogx(f_far, (f_far * Pyy_ph2_far)/(rho**2 * u_tau**4), label='PH2 Original', color=ph2_colour, alpha=0.3)
        ax[0].semilogx(f_far, rv*(g1+g2), label='BL Model', color='k', linestyle='--')
        ax[0].semilogx(f_far, rv_c*(g1_c+g2_c), label='Channel Model', color='k', linestyle='-.')
        ax[0].axvline(f_cut, color='red', linestyle='--')
        ax[0].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
        ax[0].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

        ax[0].set_xlabel("$f$ [Hz]")

        T_plus_clean = 1/f_clean * (u_tau**2)/nu
        ax[1].semilogx(T_plus_clean, (f_clean * Pyy_ph1_clean)/(rho**2 * u_tau**4), label='PH1 Cleaned', color=ph1_colour)
        ax[1].semilogx(T_plus, (f_far * Pyy_ph1_far)/(rho**2 * u_tau**4), label='PH1 Original', color=ph1_colour, alpha=0.3)
        ax[1].semilogx(T_plus_clean, (f_clean * Pyy_ph2_clean)/(rho**2 * u_tau**4), label='PH2 Cleaned', color=ph2_colour)
        ax[1].semilogx(T_plus, (f_far * Pyy_ph2_far)/(rho**2 * u_tau**4), label='PH2 Original', color=ph2_colour, alpha=0.3)
        ax[1].semilogx(T_plus, rv*(g1+g2), label='BL Model', color='k', linestyle='--')
        ax[1].semilogx(T_plus, rv_c*(g1_c+g2_c), label='Channel Model', color='k', linestyle='-.')
        ax[1].axvline(T_plus_fcut, color='red', linestyle='--')
        ax[1].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
        ax[1].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

        ax[1].set_xlabel(r"$T^+$")

        ax[0].set_ylabel(r"${f \phi_{pp}}^+$")
        ax[0].set_xlim(1, 1e4)

        ax[0].set_ylim(0, 15)
        ax[0].legend()
        fig.savefig('figures/remount_test/new_mount_cleaned.png', dpi=410)

def change_sensitivity():
    u_tau = 0.58
    nu_utau = 27e-6
    nu = nu_utau * u_tau
    f_cut = 2_100
    T_plus_fcut = 1/f_cut * (u_tau**2)/nu
    rho = 1.2 # kg/m^3
    cf = 2*(u_tau**2)/14**2
    Re_tau = u_tau * 0.035 / nu
    ic(Re_tau)
    rrun = 'data/20251023/remount/1VperPa.mat'
    dat = sio.loadmat(rrun) # options are channelData_LP, channelData_NF
    sensitivity = 1/12 # V/Pa
    ic(dat.keys())
    nkd_far = dat['channelData_LP'][:,2]* 1/sensitivity
    ph1_far = dat['channelData_LP'][:,0]* 1/sensitivity
    ph2_far = dat['channelData_LP'][:,1]* 1/sensitivity
    f_far, Pyy_nkd_far = compute_spec(FS, nkd_far)
    f_far, Pyy_ph1_far = compute_spec(FS, ph1_far)
    f_far, Pyy_ph2_far = compute_spec(FS, ph2_far)

    fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, tight_layout=True)
    T_plus = 1/f_far * (u_tau**2)/nu
    g1, g2, rv = bl_model(T_plus, Re_tau, cf)
    g1_c, g2_c, rv_c = channel_model(T_plus, Re_tau, u_tau, 14)

    fig.suptitle(r"$Re_\tau\approx$ 1,300 (700$\mu$m) - New Mounting")

    ax[0].semilogx(f_far, (f_far * Pyy_nkd_far)/(rho**2 * u_tau**4), label='NC', color=nkd_colour)
    ax[0].semilogx(f_far, (f_far * Pyy_ph1_far)/(rho**2 * u_tau**4), label='PH1', color=ph1_colour)
    ax[0].semilogx(f_far, (f_far * Pyy_ph2_far)/(rho**2 * u_tau**4), label='PH2', color=ph2_colour)
    ax[0].semilogx(f_far, rv*(g1+g2), label='BL Model', color='k', linestyle='--')
    ax[0].semilogx(f_far, rv_c*(g1_c+g2_c), label='Channel Model', color='k', linestyle='-.')
    ax[0].axvline(f_cut, color='red', linestyle='--')
    ax[0].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
    ax[0].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

    ax[0].set_xlabel("$f$ [Hz]")

    ax[1].semilogx(T_plus, (f_far * Pyy_nkd_far)/(rho**2 * u_tau**4), label='NC', color=nkd_colour)
    ax[1].semilogx(T_plus, (f_far * Pyy_ph1_far)/(rho**2 * u_tau**4), label='PH1', color=ph1_colour)
    ax[1].semilogx(T_plus, (f_far * Pyy_ph2_far)/(rho**2 * u_tau**4), label='PH2', color=ph2_colour)
    ax[1].semilogx(T_plus, rv*(g1+g2), label='BL Model', color='k', linestyle='--')
    ax[1].semilogx(T_plus, rv_c*(g1_c+g2_c), label='Channel Model', color='k', linestyle='-.')
    ax[1].axvline(T_plus_fcut, color='red', linestyle='--')
    ax[1].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
    ax[1].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

    ax[1].set_xlabel("$T^+$")
    ax[0].set_ylabel(r"${f \phi_{pp}}^+$")

    ax[0].set_xlim(1, 1e4)
    ax[0].set_ylim(0, 5)

    ax[0].legend()
    ax[1].legend()
    fig.savefig('figures/remount_test/new_mount.png', dpi=410)

    ph1_clean = wiener_cancel_background_torch(ph1_far, nkd_far, FS).cpu().numpy()
    ph2_clean = wiener_cancel_background_torch(ph2_far, nkd_far, FS).cpu().numpy()
    f_clean, Pyy_ph1_clean = compute_spec(FS, ph1_clean)
    f_clean, Pyy_ph2_clean = compute_spec(FS, ph2_clean)

    fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, tight_layout=True)
    fig.suptitle(r"$Re_\tau\approx$ 1,300 (700$\mu$m) - New Mounting")

    ax[0].semilogx(f_clean, (f_clean * Pyy_ph1_clean)/(rho**2 * u_tau**4), label='PH1 Cleaned', color=ph1_colour)
    ax[0].semilogx(f_far, (f_far * Pyy_ph1_far)/(rho**2 * u_tau**4), label='PH1 Original', color=ph1_colour, alpha=0.3)
    ax[0].semilogx(f_clean, (f_clean * Pyy_ph2_clean)/(rho**2 * u_tau**4), label='PH2 Cleaned', color=ph2_colour)
    ax[0].semilogx(f_far, (f_far * Pyy_ph2_far)/(rho**2 * u_tau**4), label='PH2 Original', color=ph2_colour, alpha=0.3)
    ax[0].semilogx(f_far, rv*(g1+g2), label='BL Model', color='k', linestyle='--')
    ax[0].semilogx(f_far, rv_c*(g1_c+g2_c), label='Channel Model', color='k', linestyle='-.')
    ax[0].axvline(f_cut, color='red', linestyle='--')
    ax[0].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
    ax[0].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

    ax[0].set_xlabel("$f$ [Hz]")

    T_plus_clean = 1/f_clean * (u_tau**2)/nu
    ax[1].semilogx(T_plus_clean, (f_clean * Pyy_ph1_clean)/(rho**2 * u_tau**4), label='PH1 Cleaned', color=ph1_colour)
    ax[1].semilogx(T_plus, (f_far * Pyy_ph1_far)/(rho**2 * u_tau**4), label='PH1 Original', color=ph1_colour, alpha=0.3)
    ax[1].semilogx(T_plus_clean, (f_clean * Pyy_ph2_clean)/(rho**2 * u_tau**4), label='PH2 Cleaned', color=ph2_colour)
    ax[1].semilogx(T_plus, (f_far * Pyy_ph2_far)/(rho**2 * u_tau**4), label='PH2 Original', color=ph2_colour, alpha=0.3)
    ax[1].semilogx(T_plus, rv*(g1+g2), label='BL Model', color='k', linestyle='--')
    ax[1].semilogx(T_plus, rv_c*(g1_c+g2_c), label='Channel Model', color='k', linestyle='-.')
    ax[1].axvline(T_plus_fcut, color='red', linestyle='--')
    ax[1].grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
    ax[1].grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

    ax[1].set_xlabel(r"$T^+$")

    ax[0].set_ylabel(r"${f \phi_{pp}}^+$")
    ax[0].set_xlim(1, 1e4)

    ax[0].set_ylim(0, 5)
    ax[0].legend()
    fig.savefig('figures/remount_test/new_mount_cleaned_sens.png', dpi=410)


    



if __name__ == "__main__":
    rerun()
    # plot_remount_new(plot=[0])
    # change_sensitivity()
    # ic(np.exp(-1)/1)

    # flow_tests()
