from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, get_window
import scienceplots
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from icecream import ic

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
            ic(u_tau**2/(nu * 700), u_tau, nu)
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
            ph2_far   = g_far["PH2_Pa"][:]
            ph2_close = g_close["PH2_Pa"][:]

            # spectra
            f_far,   Pyy_far   = compute_spec(ph2_far,   fs=fs, nperseg=NPERSEG)
            f_close, Pyy_close = compute_spec(ph2_close, fs=fs, nperseg=NPERSEG)

            # T^+ based on far spectrum for the model curves
            T_plus_far = (u_tau**2) / (nu * f_far)

            # friction coefficient "cf_2" from u_tau and Ue (cf/2 = (u_tau/Ue)^2)
            cf_2 = (u_tau / Ue)**2

            # models
            g1_b, g2_b, rv_b = bl_model(T_plus_far, Re_tau, cf_2)
            g1_c, g2_c, rv_c = channel_model(T_plus_far, Re_tau,
                                             u_tau, u_cl=Ue)

            bl_fphipp_plus      = rv_b * (g1_b + g2_b)
            channel_fphipp_plus = rv_c * (g1_c + g2_c)

            ax.semilogx(T_plus_far, bl_fphipp_plus,
                        linestyle="--", color=COLOURS[i], lw=0.7)
            ax.semilogx(T_plus_far, channel_fphipp_plus,
                        linestyle="-.", color=COLOURS[i], lw=0.7)

            # ROI & u_tau-uncertainty fan based on PH2 close
            mask = (f_close > f_cutl) & (f_close < f_cuth)
            f_m = f_close[mask]
            P_m = Pyy_close[mask]

            u_nom = u_tau
            u_lo  = u_nom * (1.0 - u_tau_rel_unc)
            u_hi  = u_nom * (1.0 + u_tau_rel_unc)

            n = 16
            u_grid = np.linspace(u_lo, u_hi, n)

            mid  = 0.5 * (u_lo + u_hi)
            span = (u_hi - u_lo)

            def fade(u):
                w = 1.0 - np.abs(u - mid) / (0.5 * span)  # 1 at centre, 0 at edges
                return 0.15 + 0.75 * np.clip(w, 0.0, 1.0)

            base = "gray"
            # draw edges first, centre (nominal) last
            order = np.argsort(np.abs(u_grid - u_nom))[::-1]

            for j in order:
                u = u_grid[j]
                T = (u**2) / (nu * f_m)
                Y = (f_m * P_m) / (rho**2 * u**4)
                ax.semilogx(T, Y,
                            color=to_rgba(base, fade(u)),
                            linewidth=1.0)

            # nominal curve on top
            T_nom = (u_nom**2) / (nu * f_m)
            Y_nom = (f_m * P_m) / (rho**2 * u_nom**4)
            ax.semilogx(T_nom, Y_nom,
                        color=COLOURS[i], linewidth=1.0,
                        label=labels[i], zorder=10)

            ax.grid(True, which='major', linestyle='--',
                    linewidth=0.4, alpha=0.7)
            # ax.grid(True, which='minor', linestyle=':',
            #         linewidth=0.2, alpha=0.6)
            
            re_labs = fr'$Re_\tau^{{nom}}={Re_nom[i]:,.0f}$; {labels[i]}'
            ax.set_title(f"{re_labs}")

            ax.set_xlabel(r"$T^+$")
    # common axes / limits
    axs[0].set_ylabel(r"$({f \phi_{pp}}^+)_{\mathrm{corr.}}$")

    for ax in axs:
        ax.set_xlim(7, 7_000)
        ax.set_ylim(0, 6)

    # legend for Re_tau (or PH2 conditions)
    labels_handles = ['1,000 PH2', '5,000 PH2', '9,000 PH2']
    label_colours  = COLOURS
    label_styles   = ['-', '-', '-']
    custom_lines = [
        Line2D([0], [0],
               color=label_colours[i],
               linestyle=label_styles[i])
        for i in range(len(labels_handles))
    ]
    # leg1 = axs[-1].legend(custom_lines, labels_handles,
    #                       loc='upper right', fontsize=8)
    # axs[-1].add_artist(leg1)

    # legend for model types
    labels_handles2 = ['BL model', 'Channel model']
    label_colours2  = ['black', 'black']
    label_styles2   = ['--', '-.']
    custom_lines2 = [
        Line2D([0], [0],
               color=label_colours2[i],
               linestyle=label_styles2[i])
        for i in range(len(labels_handles2))
    ]
    axs[1].legend(custom_lines2, labels_handles2,
                  loc='upper center', fontsize=8)


    plt.savefig(FIG_DIR / "G_wallp_SU_production.png", dpi=600)

if __name__ == "__main__":
    plot_model_comparison_roi()
