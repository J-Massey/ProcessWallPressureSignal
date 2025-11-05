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

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import scienceplots
plt.style.use(["science", "grid"])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")

from models import channel_model, bl_model
from apply_frf import apply_frf
from save_calibs import _estimate_frf

############################
# Constants & defaults
############################
FS = 50_000.0
NPERSEG = 2**10
WINDOW = "hann"
TRIM_CAL_SECS = 5  # seconds trimmed from the start of calibration runs (0 to disable)

nc_colour = '#1f77b4'  # matplotlib default blue
ph1_colour = "#c76713"  # matplotlib default orange
ph2_colour = "#9fda16"  # matplotlib default red
nkd_colour = '#2ca02c' # matplotlib default green

# --- constants (keep once, top of file) ---
R = 287.05        # J/kg/K
PSI_TO_PA = 6_894.76
P_ATM = 101_325.0
DELTA = 0.035  # m, bl-height of 'channel'
TDEG = [18, 20, 22]

TPLUS_CUT = 10  # picked so that we cut at half the inner peak

CHAIN_SENS_V_PER_PA = {
    'PH1': 50.9e-3,  # V/Pa
    'PH2': 51.7e-3,  # V/Pa
    'NC':  52.4e-3,  # V/Pa
}


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

def correct_pressure_sensitivity(p, psig, alpha: float = 0.012):
    """
    Correct pressure sensor sensitivity based on gauge pressure [psig].
    Returns corrected pressure signal [Pa].
    """
    p_corr = p * 10**(psig * PSI_TO_PA / 1000 * alpha / 20)
    return p_corr




psigs = [0, 50, 100]
colors = ['C0', 'C1', 'C2']
RAW_MEAS_BASE = 'data/20251031/'
# with h5py.File("data/final_pressure/SU_2pt_pressure.h5", 'r') as h:
fig, ax = plt.subplots(1,1, figsize=(6,4), tight_layout=True, sharex=True)
for idx, psig in enumerate(psigs):
    # ax = axs[0]
    # Plot the wn and the TF in the calibration
    dat = sio.loadmat(f"data/final_calibration/calib_{psig}psig_1.mat")
    ph1_nf, _, nc1, _ = dat["channelData_WN"].T
    dat = sio.loadmat(f"data/final_calibration/calib_{psig}psig_2.mat")
    _, ph2_nf, nc2, _ = dat["channelData_WN"].T

    nc1 = correct_pressure_sensitivity(nc1, psig)
    nc2 = correct_pressure_sensitivity(nc2, psig)

    f, Pxx = compute_spec(FS, nc1)
    ax.semilogx(f, np.sqrt(Pxx), color=colors[idx])
    f, Pxx = compute_spec(FS, nc2)
    ax.semilogx(f, np.sqrt(Pxx), color=colors[idx], label=f'{psig} psig')

    # Plot the actual semi-anechoic transfer
    # f1, H1, _ = _estimate_frf(ph1_nf, nc1, fs=FS)
    # f2, H2, _ = _estimate_frf(ph2_nf, nc2, fs=FS)
    # ax = axs[1]
    # ax.semilogx(f1, np.abs(H1), label='PH1 FRF (calib)', color=ph1_colour, ls='--')
    # ax.semilogx(f2, np.abs(H2), label='PH2 FRF (calib)', color=ph2_colour, ls='--')

ax.set_xlim(50, 1000)
ax.set_ylim(5e-5, 3e-3)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Input signal PSD")
ax.legend()


plt.savefig(f"figures/tf_two_ways/input_sig.png", dpi=300)
plt.close(fig)
    