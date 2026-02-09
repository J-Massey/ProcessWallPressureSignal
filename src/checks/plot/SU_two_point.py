from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, get_window
import scienceplots
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from icecream import ic
from tqdm import tqdm

from src.config_params import Config

cfg = Config()

plt.style.use(["science", "grid"])
plt.rcParams["font.size"] = 10.5
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")

# -------------------- constants --------------------
FS = cfg.FS
NPERSEG = 2**12          # keep one value for all runs
WINDOW  = cfg.WINDOW

LABELS = ("0psig", "50psig", "100psig")
PSIGS  = (0.0, 50.0, 100.0)
COLOURS = ("#1e8ad8", "#ff7f0e", "#26bd26")  # hex equivalents of C0, C1, C2
FIG_DIR = Path("figures") / "from_data"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def compute_spec(x: np.ndarray, fs: float = FS, nperseg: int = NPERSEG):
    """Welch PSD with consistent settings. Returns f [Hz], Pxx [Pa^2/Hz]."""
    x = np.asarray(x, float)
    nseg = min(nperseg, x.size)
    if nseg < 16:
        raise ValueError(f"Signal too short for Welch: n={x.size}, nperseg={nperseg}")
    w = get_window(WINDOW, nseg, fftbins=True)
    f, Pxx = welch(
        x, fs=fs, window=w, nperseg=nseg, noverlap=nseg//2,
        detrend="constant", scaling="density", return_onesided=True,
    )
    return f, Pxx

def _get_ue(hf: h5py.File, gL: h5py.Group, idx: int, default: float = 14.0) -> float:
    if "Ue_m_per_s" in gL.attrs:
        return float(np.atleast_1d(gL.attrs["Ue_m_per_s"])[0])
    ue_attr = hf.attrs.get("Ue_m_per_s", default)
    ue_arr = np.atleast_1d(ue_attr)
    if ue_arr.size > idx:
        return float(ue_arr[idx])
    return float(ue_arr[0])

def plot_2pt_inner():
    labels = ['0psig', '50psig', '100psig']
    Re_nom = [1_500, 4_500, 9_000]

    f_cutl, f_cuth = 100.0, 1_000.0  # Hz

    fig, axs = plt.subplots(1, 3, figsize=(8, 3), tight_layout=True,
                            sharex=True, sharey=True)

    with h5py.File(cfg.PH_PROCESSED_FILE, "r") as hf:
        g_fs = hf["wallp_production"]
        # fall back to global FS if attribute is missing
        fs = float(hf.attrs.get("fs_Hz", FS))
        for i, L in enumerate(labels):
            ax = axs[i]
            gL = g_fs[L]
            Ue = _get_ue(hf, gL, i)

            # scalarise attrs in case h5py gives small arrays
            u_tau = float(np.atleast_1d(gL.attrs["u_tau"])[0])
            nu    = float(np.atleast_1d(gL.attrs["nu"])[0])
            rho   = float(np.atleast_1d(gL.attrs["rho"])[0])
            Re_tau = float(np.atleast_1d(gL.attrs["Re_tau"])[0])
            u_tau_rel_unc = float(
                np.atleast_1d(gL.attrs.get("u_tau_rel_unc", 0.0))[0]
            )

            g_corr = gL["frf_corrected_signals"]
            g_corr = gL["fs_noise_rejected_signals"]
            g_far   = g_corr["far"]
            g_close = g_corr["close"]

            # use PH2 for spectra / models as before
            ph1_far   = g_far["PH1_Pa"][:]
            ph2_far   = g_far["PH2_Pa"][:]
            ph1_close = g_close["PH1_Pa"][:]
            ph2_close = g_close["PH2_Pa"][:]

            nt = ph2_far.size
            mean_p2_far   = np.mean(ph1_far**2 + ph2_far**2) / 2.0
            mean_p2_close = np.mean(ph1_close**2 + ph2_close**2) / 2.0


            # log-spaced time lags between 0.01 and 1 s
            tmin, tmax = 0.001, 0.1 # seconds
            lags_sec = np.logspace(np.log10(tmin), np.log10(tmax), 64)
            lags_samp = np.unique(np.round(lags_sec * fs).astype(int))  # integer sample lags
            lags_sec = lags_samp / fs  # snap back to actual times

            R_norm_far   = np.empty(lags_samp.size, dtype=float)
            R_norm_close = np.empty(lags_samp.size, dtype=float)

            start = int(fs)  # discard first 1 s if you want

            for k, lag in tqdm(enumerate(lags_samp), total=lags_samp.size):
                # FAR
                p1 = ph1_far[start:nt-lag]
                p2 = ph2_far[start+lag:nt]
                R  = np.mean(p1 * p2)
                R_norm_far[k] = R / mean_p2_far

                # CLOSE (match trimming)
                p1 = ph1_close[start:nt-lag]
                p2 = ph2_close[start+lag:nt]
                R  = np.mean(p1 * p2)
                R_norm_close[k] = R / mean_p2_close

            # x-axis: convective distance (or just lags_sec if you want time)
            t_conv = lags_sec  # seconds
            t_conv_plus = t_conv * (u_tau**2)/nu

            ax.semilogx(t_conv_plus, R_norm_far,   linestyle="-",  color=COLOURS[i])
            ax.semilogx(t_conv_plus, R_norm_close, linestyle="--", color=COLOURS[i])
            re_labs = fr'$Re_\tau^{{nom}}={Re_nom[i]:,.0f}$; {labels[i]}'
        
            ax.set_title(f"{re_labs}")
            ax.set_xlabel(r"$\Delta t^+$")

        custom_lines = [
            Line2D([0], [0], color='gray', linestyle='-', lw=1),
            Line2D([0], [0], color='gray', linestyle='--', lw=1),
        ]
        axs[0].legend(custom_lines, ['far', 'close'], loc='upper right', fontsize=8)

    axs[0].set_ylabel(r"$R_{pp}(\Delta t) / R_{pp}(0)$")
            
    plt.savefig(FIG_DIR / "2pt_correlation_inner.png", dpi=600)

def compute_spec(x: np.ndarray, fs: float = FS, nperseg: int = NPERSEG):
    """Welch PSD with consistent settings. Returns f [Hz], Pxx [Pa^2/Hz]."""
    x = np.asarray(x, float)
    nseg = min(nperseg, x.size)
    if nseg < 16:
        raise ValueError(f"Signal too short for Welch: n={x.size}, nperseg={nperseg}")
    w = get_window(WINDOW, nseg, fftbins=True)
    f, Pxx = welch(
        x, fs=fs, window=w, nperseg=nseg, noverlap=nseg//2,
        detrend="constant", scaling="density", return_onesided=True,
    )
    return f, Pxx

def plot_2pt_outer():
    labels = ['0psig', '50psig', '100psig']
    Re_nom = [1_500, 4_500, 9_000]



    f_cutl, f_cuth = 100.0, 1_000.0  # Hz

    fig, axs = plt.subplots(1, 3, figsize=(8, 3), tight_layout=True,
                            sharex=True, sharey=True)

    with h5py.File(cfg.PH_PROCESSED_FILE, "r") as hf:
        g_fs = hf["wallp_production"]
        # fall back to global FS if attribute is missing
        fs = float(hf.attrs.get("fs_Hz", FS))
        for i, L in enumerate(labels):
            ax = axs[i]
            gL = g_fs[L]
            Ue = _get_ue(hf, gL, i)

            # scalarise attrs in case h5py gives small arrays
            u_tau = float(np.atleast_1d(gL.attrs["u_tau"])[0])
            nu    = float(np.atleast_1d(gL.attrs["nu"])[0])
            rho   = float(np.atleast_1d(gL.attrs["rho"])[0])
            Re_tau = float(np.atleast_1d(gL.attrs["Re_tau"])[0])
            u_tau_rel_unc = float(
                np.atleast_1d(gL.attrs.get("u_tau_rel_unc", 0.0))[0]
            )

            g_corr = gL["frf_corrected_signals"]
            g_corr = gL["fs_noise_rejected_signals"]
            g_far   = g_corr["far"]
            g_close = g_corr["close"]

            # use PH2 for spectra / models as before
            ph1_far   = g_far["PH1_Pa"][:]
            ph2_far   = g_far["PH2_Pa"][:]
            ph1_close = g_close["PH1_Pa"][:]
            ph2_close = g_close["PH2_Pa"][:]

            nt = ph2_far.size
            mean_p2_far   = np.mean(ph1_far**2 + ph2_far**2) / 2.0
            mean_p2_close = np.mean(ph1_close**2 + ph2_close**2) / 2.0


            # log-spaced time lags between 0.01 and 1 s
            tmin, tmax = 0.001, 0.1 # seconds
            lags_sec = np.logspace(np.log10(tmin), np.log10(tmax), 64)
            lags_samp = np.unique(np.round(lags_sec * fs).astype(int))  # integer sample lags
            lags_sec = lags_samp / fs  # snap back to actual times

            R_norm_far   = np.empty(lags_samp.size, dtype=float)
            R_norm_close = np.empty(lags_samp.size, dtype=float)

            start = int(fs)  # discard first 1 s if you want

            for k, lag in tqdm(enumerate(lags_samp), total=lags_samp.size):
                # FAR
                p1 = ph1_far[start:nt-lag]
                p2 = ph2_far[start+lag:nt]
                R  = np.mean(p1 * p2)
                R_norm_far[k] = R / mean_p2_far

                # CLOSE (match trimming)
                p1 = ph1_close[start:nt-lag]
                p2 = ph2_close[start+lag:nt]
                R  = np.mean(p1 * p2)
                R_norm_close[k] = R / mean_p2_close

            # x-axis: convective distance (or just lags_sec if you want time)
            t_conv = lags_sec  # seconds
            t_conv_outer = t_conv * Ue/0.035

            ax.semilogx(t_conv_outer, R_norm_far,   linestyle="-",  color=COLOURS[i])
            ax.semilogx(t_conv_outer, R_norm_close, linestyle="--", color=COLOURS[i])
            re_labs = fr'$Re_\tau^{{nom}}={Re_nom[i]:,.0f}$; {labels[i]}'
        
            ax.set_title(f"{re_labs}")
            ax.set_xlabel(r"$\Delta t^o$")

        custom_lines = [
            Line2D([0], [0], color='gray', linestyle='-', lw=1),
            Line2D([0], [0], color='gray', linestyle='--', lw=1),
        ]
        axs[0].legend(custom_lines, ['far', 'close'], loc='upper right', fontsize=8)

    axs[0].set_ylabel(r"$R_{pp}(\Delta t) / R_{pp}(0)$")
            
    plt.savefig(FIG_DIR / "2pt_correlation_outer.png", dpi=600)

def plot_2pt_speed_outer():
    labels = ['0psig', '50psig', '100psig']
    Re_nom = [1_500, 5_000, 9_000]




    f_cutl, f_cuth = 100.0, 1_000.0  # Hz

    fig, axs = plt.subplots(1, 3, figsize=(8, 3), tight_layout=True,
                            sharex=True, sharey=True)

    with h5py.File(cfg.PH_PROCESSED_FILE, "r") as hf:
        g_fs = hf["wallp_production"]
        # fall back to global FS if attribute is missing
        fs = float(hf.attrs.get("fs_Hz", FS))
        for i, L in enumerate(labels):
            ax = axs[i]
            gL = g_fs[L]
            Ue = _get_ue(hf, gL, i)

            # scalarise attrs in case h5py gives small arrays
            u_tau = float(np.atleast_1d(gL.attrs["u_tau"])[0])
            nu    = float(np.atleast_1d(gL.attrs["nu"])[0])
            rho   = float(np.atleast_1d(gL.attrs["rho"])[0])
            Re_tau = float(np.atleast_1d(gL.attrs["Re_tau"])[0])
            u_tau_rel_unc = float(
                np.atleast_1d(gL.attrs.get("u_tau_rel_unc", 0.0))[0]
            )

            g_corr = gL["fs_noise_rejected_signals"]
            g_far   = g_corr["far"]
            far_space = g_far.attrs['spacing_m']
            g_close = g_corr["close"]
            close_space = g_close.attrs['spacing_m']

            # use PH2 for spectra / models as before
            ph1_far   = g_far["PH1_Pa"][:]
            ph2_far   = g_far["PH2_Pa"][:]
            ph1_close = g_close["PH1_Pa"][:]
            ph2_close = g_close["PH2_Pa"][:]

            nt = ph2_far.size


            # log-spaced time lags between 0.01 and 1 s
            tmin, tmax = 0.001, 0.1 # seconds
            lags_sec = np.logspace(np.log10(tmin), np.log10(tmax), 128)
            lags_samp = np.unique(np.round(lags_sec * fs).astype(int))  # integer sample lags
            lags_sec = lags_samp / fs  # snap back to actual times

            R_norm_far   = np.empty(lags_samp.size, dtype=float)
            R_norm_close = np.empty(lags_samp.size, dtype=float)

            start = int(fs)  # discard first 1 s if you want

            for k, lag in tqdm(enumerate(lags_samp), total=lags_samp.size):
                # FAR
                p1 = ph1_far[start:nt-lag] - np.mean(ph1_far)
                p2 = ph2_far[start+lag:nt] - np.mean(ph2_far)
                R  = np.mean(p1 * p2)
                R_norm_far[k] = R / (np.sqrt(np.mean((ph1_far- np.mean(ph1_far))**2)) * np.sqrt(np.mean((ph2_far - np.mean(ph2_far))**2)))

                # CLOSE (match trimming)
                p1 = ph1_close[start:nt-lag] - np.mean(ph1_close)
                p2 = ph2_close[start+lag:nt] - np.mean(ph2_close)
                R  = np.mean(p1 * p2)
                R_norm_close[k] = R / (np.sqrt(np.mean((ph1_close- np.mean(ph1_close))**2)) * np.sqrt(np.mean((ph2_close - np.mean(ph2_close))**2)))

            # x-axis: convective distance (or just lags_sec if you want time)
            t_conv = lags_sec  # seconds
            speed_far_out = (far_space / t_conv)/Ue
            speed_close_out = (close_space / t_conv)/Ue

            ax.semilogx(speed_far_out, R_norm_far,   linestyle="-",  color=COLOURS[i])
            ax.semilogx(speed_close_out, R_norm_close, linestyle="--", color=COLOURS[i])

            re_labs = fr'$Re_\tau^{{nom}}={Re_nom[i]:,.0f}$; {labels[i]}'
            ax.set_title(f"{re_labs}")
            ax.set_xlabel(r"$U_c^o$")
            # ax.axvline(0.85, color='darkblue', linestyle=':', lw=0.8)
            # ax.axvline(0.8, color='darkblue', linestyle=':', lw=0.8)
            # highlight region between the two reference lines
            ax.axvspan(0.8, 0.9, color='darkblue', alpha=0.12, edgecolor='none', linewidth=0)
            ax.annotate(r'$U_c/U_e\in[0.8,0.9]$', xy=(0.6, -0.07), xycoords='data', rotation=90, fontsize=8, color='darkblue')

        custom_lines = [
            Line2D([0], [0], color='gray', linestyle='-', lw=1),
            Line2D([0], [0], color='gray', linestyle='--', lw=1),
        ]
        axs[0].legend(custom_lines, ['far', 'close'], loc='upper right', fontsize=8)

    axs[0].set_ylabel(r"$R_{pp}(\Delta t) / R_{pp}(0)$")
            
    plt.savefig(FIG_DIR / "2pt_correlation_speed_outer.png", dpi=600)

def plot_2pt_speed_inner():
    labels = ['0psig', '50psig', '100psig']
    Re_nom = [1_500, 4_500, 9_000]



    f_cutl, f_cuth = 100.0, 1_000.0  # Hz

    fig, axs = plt.subplots(1, 3, figsize=(8, 3), tight_layout=True,
                            sharex=True, sharey=True)

    with h5py.File(cfg.PH_PROCESSED_FILE, "r") as hf:
        g_fs = hf["wallp_production"]
        # fall back to global FS if attribute is missing
        fs = float(hf.attrs.get("fs_Hz", FS))
        for i, L in enumerate(labels):
            ax = axs[i]
            gL = g_fs[L]
            Ue = _get_ue(hf, gL, i)

            # scalarise attrs in case h5py gives small arrays
            u_tau = float(np.atleast_1d(gL.attrs["u_tau"])[0])
            nu    = float(np.atleast_1d(gL.attrs["nu"])[0])
            rho   = float(np.atleast_1d(gL.attrs["rho"])[0])
            Re_tau = float(np.atleast_1d(gL.attrs["Re_tau"])[0])
            u_tau_rel_unc = float(
                np.atleast_1d(gL.attrs.get("u_tau_rel_unc", 0.0))[0]
            )

            g_corr = gL["fs_noise_rejected_signals"]
            g_far   = g_corr["far"]
            far_space = g_far.attrs['spacing_m']
            g_close = g_corr["close"]
            close_space = g_close.attrs['spacing_m']

            # use PH2 for spectra / models as before
            ph1_far   = g_far["PH1_Pa"][:]
            ph2_far   = g_far["PH2_Pa"][:]
            ph1_close = g_close["PH1_Pa"][:]
            ph2_close = g_close["PH2_Pa"][:]

            nt = ph2_far.size
            mean_p2_far   = np.mean(ph1_far**2 + ph2_far**2) / 2.0
            mean_p2_close = np.mean(ph1_close**2 + ph2_close**2) / 2.0


            # log-spaced time lags between 0.01 and 1 s
            tmin, tmax = 0.001, 0.1 # seconds
            lags_sec = np.logspace(np.log10(tmin), np.log10(tmax), 64)
            lags_samp = np.unique(np.round(lags_sec * fs).astype(int))  # integer sample lags
            lags_sec = lags_samp / fs  # snap back to actual times

            R_norm_far   = np.empty(lags_samp.size, dtype=float)
            R_norm_close = np.empty(lags_samp.size, dtype=float)

            start = int(fs)  # discard first 1 s if you want

            for k, lag in tqdm(enumerate(lags_samp), total=lags_samp.size):
                # FAR
                p1 = ph1_far[start:nt-lag]
                p2 = ph2_far[start+lag:nt]
                R  = np.mean(p1 * p2)
                R_norm_far[k] = R / mean_p2_far

                # CLOSE (match trimming)
                p1 = ph1_close[start:nt-lag]
                p2 = ph2_close[start+lag:nt]
                R  = np.mean(p1 * p2)
                R_norm_close[k] = R / mean_p2_close

            # x-axis: convective distance (or just lags_sec if you want time)
            t_conv = lags_sec  # seconds
            speed_far_out = (far_space / t_conv)/u_tau
            speed_close_out = (close_space / t_conv)/u_tau

            ax.semilogx(speed_far_out, R_norm_far,   linestyle="-",  color=COLOURS[i])
            ax.semilogx(speed_close_out, R_norm_close, linestyle="--", color=COLOURS[i])
            re_labs = fr'$Re_\tau^{{nom}}={Re_nom[i]:,.0f}$; {labels[i]}'
        
            ax.set_title(f"{re_labs}")
            ax.set_xlabel(r"$U_c^+$")

        custom_lines = [
            Line2D([0], [0], color='gray', linestyle='-', lw=1),
            Line2D([0], [0], color='gray', linestyle='--', lw=1),
        ]
        axs[0].legend(custom_lines, ['far', 'close'], loc='upper right', fontsize=8)

    axs[0].set_ylabel(r"$R_{pp}(\Delta t) / R_{pp}(0)$")
            
    plt.savefig(FIG_DIR / "2pt_correlation_speed_inner.png", dpi=600)

if __name__ == "__main__":
    # plot_2pt_inner()
    # plot_2pt_outer()
    plot_2pt_speed_outer()
    # plot_2pt_speed_inner()
    
