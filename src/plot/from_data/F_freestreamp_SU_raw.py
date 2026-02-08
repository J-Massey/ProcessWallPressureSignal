from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, get_window
import scienceplots
from icecream import ic

from src.config_params import Config

cfg = Config()

plt.style.use(["science", "grid"])
plt.rcParams["font.size"] = 10.5
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")

# -------------------- constants --------------------
FS = cfg.FS
NPERSEG = 2**14          # keep one value for all runs
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

def plot_fs_raw():
    psigs = ['0psig', '50psig', '100psig']
    Re_noms = [1_500, 5_000, 9_000]
    fig, ax = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
    ax[0].set_title("NC--close run")
    ax[1].set_title("NC--far run")
    ax[0].set_xlabel(r"$T^+$")
    ax[1].set_xlabel(r"$T^+$")
    ax[0].set_ylabel(r"${f \phi_{pp}}_{\mathrm{raw}}^+$")

    with h5py.File(cfg.NKD_RAW_FILE, "r") as f_raw:
        for psig in psigs:
            nc_raw_close = f_raw[f'freestream_raw/{psig}/close/NC_Pa'][:]
            nc_raw_far = f_raw[f'freestream_raw/{psig}/far/NC_Pa'][:]
            ic(f_raw[f'freestream_raw/{psig}'].attrs.keys())
            rho = f_raw[f'freestream_raw/{psig}'].attrs['rho'][()]
            u_tau = f_raw[f'freestream_raw/{psig}'].attrs['u_tau'][()]
            nu = f_raw[f'freestream_raw/{psig}'].attrs['nu'][()]
    
            f, Pxx_close = compute_spec(nc_raw_close, fs=FS, nperseg=NPERSEG)
            f, Pxx_far = compute_spec(nc_raw_far, fs=FS, nperseg=NPERSEG)

            T_plus = (u_tau**2) / (nu * f)

            norm_factor = (rho**2) * (u_tau**4)
            ax[0].loglog(T_plus, f * Pxx_close / norm_factor, label=fr'$Re_\tau^{{\mathrm{{nom}}}}$={Re_noms[psigs.index(psig)]}', color=COLOURS[psigs.index(psig)])
            ax[1].loglog(T_plus, f * Pxx_far / norm_factor, label=psig, color=COLOURS[psigs.index(psig)])

    
    ax[0].legend()
    ax[1].legend()
    plt.savefig(FIG_DIR / "F_freestreamp_SU_raw.png", dpi=600)

if __name__ == "__main__":
    plot_fs_raw()
