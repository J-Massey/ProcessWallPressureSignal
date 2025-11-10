import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, get_window
import scienceplots

plt.style.use(["science", "grid"])
plt.rcParams["font.size"] = 10.5
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")

# -------------------- constants --------------------
FS = 50_000.0
NPERSEG = 2**14          # keep one value for all runs
WINDOW  = "hann"
PSI_TO_PA = 6_894.76

LABELS = ("0psig", "50psig", "100psig")
PSIGS  = (0.0, 50.0, 100.0)
COLOURS = ("#1e8ad8", "#ff7f0e", "#26bd26")  # hex equivalents of C0, C1, C2

CLEANED_BASE = "data/final_cleaned/"

# -------------------- helpers ----------------------
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

def correct_pressure_sensitivity(pa_signal: np.ndarray, psig: float,
                                 alpha_db_per_kPa: float = -0.01) -> np.ndarray:
    """
    Apply static-pressure sensitivity correction to a *pressure* time series (Pa),
    when the original Pa was computed using the nominal sensitivity S0.

    Sensor sensitivity vs. static pressure: Δ[dB] = alpha_db_per_kPa * p_kPa.
    Actual sensitivity S(p) = S0 * 10^(Δ/20). Original Pa_est = V / S0 = 10^(Δ/20) * Pa_true.
    => Pa_true = Pa_est * 10^(-Δ/20).

    Default alpha = -0.01 dB/kPa (less sensitive at higher p).
    """
    kPa = (psig * PSI_TO_PA) / 1000.0
    factor = 10.0 ** (-(alpha_db_per_kPa * kPa) / 20.0)  # note the minus
    return np.asarray(pa_signal, float) * factor

def plot_one_position(ax, position: str, fmin: float = 100.0, fmax: float = 1_000.0):
    """
    Plot raw normalized spectra for PH1 & PH2 at a given position ('close'/'far'),
    across 0/50/100 psig using the same Welch settings and pressure correction.
    """

    for idx, (lab, psig, col) in enumerate(zip(LABELS, PSIGS, COLOURS)):
        fn = os.path.join(CLEANED_BASE, f"{lab}_{position}_cleaned.h5")
        if not os.path.exists(fn):
            print(f"Missing: {fn} — skipping.")
            continue

        with h5py.File(fn, "r") as hf:
            ph1 = np.asarray(hf["ph1_clean"][:], float)
            ph2 = np.asarray(hf["ph2_clean"][:], float)
            # flow props for normalization
            rho = float(hf.attrs["rho"])
            u_tau = float(hf.attrs["u_tau"])
            nu = float(hf.attrs["nu"])

        # pressure sensitivity correction (−0.01 dB/kPa spec)
        ph1 = correct_pressure_sensitivity(ph1, psig, alpha_db_per_kPa=-0.01)
        ph2 = correct_pressure_sensitivity(ph2, psig, alpha_db_per_kPa=-0.01)

        # PSDs and normalization: (f * Pyy) / (rho^2 u_tau^4)
        f1, P11 = compute_spec(ph1)
        f2, P22 = compute_spec(ph2)

        # guard DC & band-limit for plotting
        m1 = (f1 >= fmin) & (f1 <= fmax)
        m2 = (f2 >= fmin) & (f2 <= fmax)

        def norm_pm(f, P): return (f * P) / (rho**2 * u_tau**4)

        ax[0].semilogx(f2, norm_pm(f2, P22),
                       color=col, lw=1.2, alpha=0.9, label=f"{lab} PH2")
        ax[0].semilogx(f1, norm_pm(f1, P11),
                       color=col, lw=1.0, alpha=0.9, ls="--", label=f"{lab} PH1")

        # T+ axis view
        # T+ = (u_tau^2 / nu) / f  (skip f=0 by construction)
        Tplus1 = (u_tau**2 / nu) / f1[m1]
        Tplus2 = (u_tau**2 / nu) / f2[m2]
        ax[1].semilogx(Tplus2, norm_pm(f2[m2], P22[m2]),
                       color=col, lw=1.2, alpha=0.9, label=f"{lab} PH2")
        ax[1].semilogx(Tplus1, norm_pm(f1[m1], P11[m1]),
                       color=col, lw=1.0, alpha=0.9, ls="--", label=f"{lab} PH1")

    # formatting
    for k in range(2):
        ax[k].grid(True, which="major", ls="--", lw=0.4, alpha=0.7)
        ax[k].grid(True, which="minor", ls=":",  lw=0.25, alpha=0.6)

    ax[0].set_xlim(10, 10_000)
    ax[0].set_ylim(0, 32)
    ax[0].set_xlabel(r"$f$ [Hz]")
    ax[0].set_ylabel(r"$({f \phi_{pp}}^+)_{\textrm{raw}}$")
    # shade if you like to mark out-of-band region on the left plot
    ax[0].fill_betweenx([0, 32], 1_200, 1_700, color="grey", alpha=0.15)
    ax[0].fill_betweenx([0, 32], 100, 1_000, color="green", alpha=0.15)

    ax[1].set_xlim(1, 1e4)
    ax[1].set_ylim(0, 8)
    ax[1].set_xlabel(r"$T^+$")
    ax[1].set_ylabel(r"$({f \phi_{pp}}^+)_{\textrm{raw}}$")

    # custom legend: solid for PH2, dashed for PH1
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=COLOURS[0], lw=1.2, ls="-"),
        Line2D([0], [0], color=COLOURS[0], lw=1.0, ls="--"),
        Line2D([0], [0], color=COLOURS[1], lw=1.2, ls="-"),
        Line2D([0], [0], color=COLOURS[1], lw=1.0, ls="--"),
        Line2D([0], [0], color=COLOURS[2], lw=1.2, ls="-"),
        Line2D([0], [0], color=COLOURS[2], lw=1.0, ls="--"),
    ]
    labels = ["1 000 PH2", "1 000 PH1", "5 000 PH2", "5 000 PH1", "9 000 PH2", "9 000 PH1"]
    ax[0].legend(handles, labels, loc="upper right", fontsize=8)

    # fig.suptitle(f"Raw spectra at {position}", fontsize=11)

            # if savepath:
            #     os.makedirs(os.path.dirname(savepath), exist_ok=True)
            #     # plt.close(fig)
            # else:
            #     plt.show()

def main():
    fig, ax = plt.subplots(2, 1, figsize=(7, 3), tight_layout=True, sharex=False)

    plot_one_position(ax, "far")
    plot_one_position(ax, "close")
    fig.savefig("figures/final/spectra_comparison.png", dpi=400)

if __name__ == "__main__":
    main()
