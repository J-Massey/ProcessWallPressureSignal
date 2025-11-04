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
NPERSEG = 2**11
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
GAMMA = 1.4  # air
R_AIR = 287.05
PSI_TO_PA = 6894.76
P_ATM = 101_325.0

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


def air_rho(psig, TdegC=20.0):
    """Ideal-gas density at gauge pressure psig and temperature TdegC."""
    T = 273.15 + float(TdegC)
    p_abs = P_ATM + float(psig) * PSI_TO_PA
    return p_abs / (R_AIR * T)

def air_c(TdegC=20.0):
    """Sound speed at TdegC (weak p-dependence ignored)."""
    T = 273.15 + float(TdegC)
    return np.sqrt(GAMMA * R_AIR * T)

def _finite_mask(*arrs):
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m

def resample_to_common_f(f_list, S_list, fband=(50.0, 1000.0)):
    """Common frequency grid over the true overlap; log‑interp each spectrum."""
    # densest grid as reference
    iref = int(np.argmax([len(fi) for fi in f_list]))
    fref = np.asarray(f_list[iref], float)

    # overlap band (drop DC if present)
    fmins = [fi[1] if fi[0] == 0 else fi[0] for fi in f_list]
    fmaxs = [fi[-1] for fi in f_list]
    fmin_all = max(fmins)
    fmax_all = min(fmaxs)
    if fband is not None:
        fmin_all = max(fmin_all, float(fband[0]))
        fmax_all = min(fmax_all, float(fband[1]))
    if not (np.isfinite(fmin_all) and np.isfinite(fmax_all) and fmax_all > fmin_all):
        raise RuntimeError("No overlapping frequency band across runs.")

    mref = (fref >= fmin_all) & (fref <= fmax_all)
    f_common = fref[mref]
    if f_common.size == 0:
        raise RuntimeError("Empty overlap on reference grid.")

    S_common = []
    for fi, Si in zip(f_list, S_list):
        fi = np.asarray(fi, float)
        Si = np.asarray(Si, float)
        Si = np.maximum(Si, 1e-30)          # keep >0 for log
        # since f_common ⊂ [fi.min, fi.max], no NaNs needed
        logSi_c = np.interp(f_common, fi, np.log(Si))
        S_common.append(np.exp(logSi_c))

    return f_common, S_common
def fit_pressure_invariant_source(
    f, S_runs, psigs, *, TdegC=20.0, use_power=False,
    z_model="rho_c", fref=200.0, fband=None, clip_floor=1e-18, ridge=1e-6
):
    """LEM fit with correct dB handling and ridge regularisation."""
    f = np.asarray(f, float)
    Nruns = len(S_runs); assert Nruns == len(psigs)

    rho = np.array([air_rho(p, TdegC) for p in psigs])
    c = air_c(TdegC)
    if z_model.lower() == "rho_c": z_run = rho * c
    elif z_model.lower() == "rho": z_run = rho
    else: raise ValueError("z_model must be 'rho_c' or 'rho'")

    zref = float(np.median(z_run)); fref = float(fref)
    F = np.tile(f, Nruns)
    Z = np.repeat(z_run, f.size)
    S = np.concatenate([np.asarray(Si, float) for Si in S_runs])
    S = np.maximum(S, clip_floor)

    m = np.isfinite(F) & np.isfinite(Z) & np.isfinite(S)
    if fband is not None:
        fmin, fmax = fband
        m &= (F >= fmin) & (F <= fmax)

    L = 10.0 if use_power else 20.0
    y = L * np.log10(S[m])

    # KEY FIX: keep X without L, then divide slopes by L after the fit
    X = np.column_stack([
        np.ones_like(F[m]),
        np.log10(F[m] / fref),
        np.log10(Z[m] / zref),
    ])

    rank = np.linalg.matrix_rank(X)
    if rank < 3:
        raise RuntimeError(f"Design matrix rank={rank} < 3. Need ≥2 pressures and varying f.")

    XtX = X.T @ X
    XtX.flat[::XtX.shape[0]+1] += ridge
    beta = np.linalg.solve(XtX, X.T @ y)
    c0 = float(beta[0])
    a  = float(beta[1] / L)   # slope per decade in linear (not dB) form
    b  = float(beta[2] / L)

    yhat = X @ beta
    resid = y - yhat
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) @ (y - y.mean())))
    R2  = 1.0 - sse/sst if sst > 0 else np.nan
    cond = float(np.linalg.cond(XtX))

    eps = np.finfo(float).tiny
    def predict_db(f_in, z_in, S_in=None):
        f_in = np.asarray(f_in, float); z_in = np.asarray(z_in, float)
        # correct prediction: put L outside the parentheses
        yhat = c0 + L * (a * np.log10(f_in / fref) + b * np.log10(z_in / zref))
        if S_in is None: return yhat
        return L * np.log10(np.maximum(np.asarray(S_in, float), eps)) - yhat

    def make_invariant(S_in, z_in):
        z_in = np.asarray(z_in, float)
        return np.asarray(S_in, float) / (z_in / zref) ** b

    return {
        "c0": c0, "a": a, "b": b,
        "fref": fref, "zref": zref,
        "z_model": z_model, "use_power": bool(use_power),
        "predict": predict_db, "make_invariant": make_invariant,
        "diagnostics": {"rank": rank, "R2": R2, "cond": cond, "ridge": ridge, "n_pts": int(X.shape[0])}
    }


def plot_pressure_invariant(
    f: np.ndarray,
    S_runs: list[np.ndarray],
    psigs: list[float],
    model: dict,
    *,
    TdegC: float = 20.0,
    colors: list[str] | None = None,
    save: str | None = None,
    xlim: tuple[float, float] | None = (50, 1000)
):
    """
    Plot raw source spectra vs pressure and the pressure-invariant collapse.

    f       : (Nf,) frequency grid (Hz)
    S_runs  : list of (Nf,) arrays (ASD if model['use_power']==False, PSD if True)
    psigs   : list of gauge pressures corresponding to S_runs
    model   : dict returned by fit_pressure_invariant_source(...)
    """
    if colors is None:
        colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    # Build z for each run
    rho = np.array([air_rho(p, TdegC) for p in psigs])
    if model["z_model"].lower() == "rho_c":
        z = rho * air_c(TdegC)
    elif model["z_model"].lower() == "rho":
        z = rho
    else:
        raise ValueError("model['z_model'] must be 'rho_c' or 'rho'")

    # Make invariant sources
    S_inv_runs = [model["make_invariant"](S, zi) for S, zi in zip(S_runs, z)]

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.4), tight_layout=True, sharex=True)
    # Left: raw
    for k, (psig, S) in enumerate(zip(psigs, S_runs)):
        ax[0].semilogx(f, S, color=colors[k % len(colors)], label=f"{psig} psig")
    ax[0].set_title("Raw source")
    ax[0].set_xlabel("Frequency [Hz]")
    ylabel = "ASD [arb.]" if not model["use_power"] else "PSD [arb.]"
    ax[0].set_ylabel(ylabel)
    ax[0].legend(frameon=True, fontsize=8)

    # Right: invariant
    for k, (psig, S_inv) in enumerate(zip(psigs, S_inv_runs)):
        ax[1].semilogx(f, S_inv, color=colors[k % len(colors)], label=f"{psig} psig")
    ax[1].set_title("Pressure-invariant source")
    ax[1].set_xlabel("Frequency [Hz]")
    if xlim is not None:
        for a in ax:
            a.set_xlim(*xlim)

    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=300)
        plt.close(fig)
    else:
        plt.show()

    return S_inv_runs


psigs = [0, 50, 100]
colors = ['C0', 'C1', 'C2']
RAW_MEAS_BASE = 'data/20251031/'

fig, ax = plt.subplots(1,1, figsize=(6,4), tight_layout=True, sharex=True)
# collect per-run f and ASD
# build per-run f,S
f_list, S_list = [], []
for psig in psigs:
    dat = sio.loadmat(f"data/final_calibration/calib_{psig}psig_1.mat")
    ph1_nf, _, nc1, _ = dat["channelData_WN"].T
    nc1 = correct_pressure_sensitivity(nc1, psig)
    f_i, Pxx_i = compute_spec(FS, nc1)
    f_list.append(f_i); S_list.append(np.sqrt(Pxx_i))

f, S_runs = resample_to_common_f(f_list, S_list, fband=(50.0, 1000.0))

model = fit_pressure_invariant_source(
    f, S_runs, psigs, TdegC=20.0,
    use_power=False, z_model="rho_c", fref=700.0,
    fband=None, ridge=1e-6
)
print(model["a"], model["b"], model["diagnostics"])

_ = plot_pressure_invariant(
    f, S_runs, psigs, model, TdegC=20.0,
    save="figures/tf_two_ways/source_invariant.png",
    xlim=(50, 1000)
)


