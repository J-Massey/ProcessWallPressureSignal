import itertools
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch, savgol_filter, find_peaks, peak_widths
from scipy.special import j1, struve

from icecream import ic

import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

sns.set_palette("colorblind")
plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")

def load_wallpressure(path):
    """Load and flatten frequency and pressure arrays from MAT file."""
    mat = loadmat(path)
    fs = mat["fs_pressure_fluc_Pa"].flatten()
    p_w = mat["wall_pressure_fluc_Pa"].flatten()
    return fs, p_w

def compute_psd(p_w, sample_rate, nperseg, noverlap):
    """Compute one-sided PSD via Welch's method."""
    psd =  welch(p_w,
                 fs=sample_rate,
                 window="hann",
                 nperseg=nperseg,
                 noverlap=noverlap)
    return psd
                

def propagate_error(f_raw, psd, nu0, rho0, u_tau0, err_frac):
    """Compute nominal and +/- error bands for f+ and phi+."""
    nu_min, nu_max = nu0 * (1 - err_frac), nu0 * (1 + err_frac)
    rho_min, rho_max = rho0 * (1 - err_frac), rho0 * (1 + err_frac)
    u_tau_min, u_tau_max = u_tau0 * (1 - err_frac), u_tau0 * (1 + err_frac)

    f_nom = f_raw * nu0 / u_tau0**2
    f_min = f_raw * nu_min / u_tau_max**2
    f_max = f_raw * nu_max / u_tau_min**2

    denom_min = (rho_max * u_tau_max**2)**2
    denom_max = (rho_min * u_tau_min**2)**2

    phi_nom = f_raw * psd / (rho0 * u_tau0**2)**2
    phi_min = f_raw * psd / denom_min
    phi_max = f_raw * psd / denom_max

    return {
        "f_nom": f_nom,
        "f_min": f_min,
        "f_max": f_max,
        "phi_nom": phi_nom,
        "phi_min": phi_min,
        "phi_max": phi_max,
    }

def duct_mode_freq(U, c, m, n, l, W, H, L):
    """
    Compute physical duct-mode frequency with quarter wave equation:
    upstream closed, downstream open.
    """
    delta2 = c**2 - U**2
    k_sq = (m * np.pi / W)**2 + (n * np.pi / H)**2
    kz_sq = ((2 * l + 1) * np.pi / L)**2
    return (1.0 / (2 * np.pi)) * np.sqrt(delta2 * k_sq + delta2**2 / (4 * c**2) * kz_sq)



def compute_duct_modes(U, c, mode_m, mode_n, mode_l,
                       W, H, L, nu0, u_tau0, err_frac):
    freqs = {"nom": [], "min": [], "max": []}
    nu_min, nu_max = nu0*(1-err_frac), nu0*(1+err_frac)
    u_tau_min, u_tau_max = u_tau0*(1-err_frac), u_tau0*(1+err_frac)

    for m in mode_m:
      for n in mode_n:
        for l in mode_l:
          # first get a “zero‐order” frequency
          f0 = duct_mode_freq(U, c, m, n, l, W, H, L)
          # compute ΔL at that f0
          # re‐compute with L_eff = L0 + dL
          f_phys = duct_mode_freq(U, c, m, n, l, W, H, L)

          # nondimensionalise
          freqs["nom"].append(f_phys * nu0 / u_tau0**2)
          freqs["min"].append(f_phys * nu_min / u_tau_max**2)
          freqs["max"].append(f_phys * nu_max / u_tau_min**2)

    return freqs


def plot_spectrum_and_modes(spec, modes, mode_l, outfile):
    """Plot spectrum with uncertainty band and duct-mode lines."""
    fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=600)

    # ax.fill_between(spec["f_nom"], spec["phi_min"], spec["phi_max"],
    #                 label="±3% spectrum", color="gray", alpha=0.5, edgecolor="none")
    # ax.fill_between(spec["f_max"], spec["phi_min"], spec["phi_max"],
    #                 color="gray", alpha=0.5, edgecolor="none")
    # ax.fill_between(spec["f_nom"], spec["phi_min"], spec["phi_max"],
    #                 label="±3% spectrum", color="gray", alpha=0.5, edgecolor="none")
    # ax.fill_betweenx(
    #     spec["phi_nom"],      # 1-D array of y-values
    #     spec["f_min"],        # left x-boundary at each y
    #     spec["f_max"],        # right x-boundary at each y
    #     color="gray",
    #     alpha=0.3,
    #     edgecolor="none"
    # )
    # inside plot_spectrum_and_modes, replace the placeholder with:
    if not hasattr(plot_spectrum_and_modes, "tit"):
        plot_spectrum_and_modes.tit = itertools.cycle([
            "Wall Pressure Spectrum",
            "Free-Stream Pressure Spectrum"
        ])
    ax.set_title(next(plot_spectrum_and_modes.tit))

    if not hasattr(plot_spectrum_and_modes, "col"):
        plot_spectrum_and_modes.col = itertools.cycle([
            "blue",
            "green"
        ])


    ax.plot(1/spec["f_nom"], spec["phi_nom"], next(plot_spectrum_and_modes.col), lw=0.5, alpha=0.8)

    ymax = spec["phi_max"].max()
    for idx, f0 in enumerate(modes["nom"]):
        if idx==0:
            lab = "Duct Mode Frequency"
        else:
            lab = None
        ax.axvline(1/f0, color="red", linestyle="--", lw=0.8, alpha=0.5, label=lab)
        ax.text(1/f0, 18, f"l={mode_l[idx]}", rotation=90,
                ha="right", va="center", fontsize=8, color="red")

    ax.set_xscale("log")
    ax.set_xlim(1/1e-1, 1/5e-4)
    ax.set_ylim(0, 25)
    ax.set_xlabel("$T^+$")
    ax.set_ylabel("$f \\Phi_{pp}^+$")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outfile)


def plot_pw_p_fs(fs_w, fs_fs, p_w, p_fs, outfile):
    """Plot wall pressure signal and PSD."""
    # Interpolate onto common frequency grid
    f_sample = np.logspace(np.log10(5e-4), np.log10(5e-1), 1024)
    p_w_interp = np.interp(f_sample, fs_w, p_w)
    p_fs_interp = np.interp(f_sample, fs_fs, p_fs)

    fig, ax = plt.subplots(figsize=(4, 2.5), dpi=600, sharex=True)
    ax.plot(1/fs_w, p_w, label="Wall pressure", lw=1)
    ax.plot(1/fs_fs, p_fs, label="F-S pressure", lw=1)
    ax.legend()

    ax.set_xlabel("$T^+$")
    ax.set_ylabel("$f \\Phi_{pp}^+$")  # Updated to use ax
    ax.set_xscale("log")

    ax.set_xlim(1/5e-1, 1/5e-4)
    ax.set_ylim(0, 20)
    plt.savefig(outfile)


def notch_filter(f_nom, phi_nom, f_min, f_max, mode_freqs,
                 min_height=None, prominence=0.001, rel_height=0.9):
    """
    Notch out the largest PSD peak around each duct-mode f0,
    using its error band defined by f_min,f_max curves.

    Parameters
    ----------
    f_nom : 1D array
    phi_nom : 1D array
    f_min, f_max : 1D arrays (same length as f_nom)
        Error-shifted lower/upper frequency at each f_nom.
    mode_freqs : sequence
        Nominal duct-mode freqs.
    min_height : float or None
        Scalar min peak height.
    prominence : float
    rel_height : float

    Returns
    -------
    phi_filtered : 1D array
    peaks_info : list of dict
        Each dict has keys mode_freq, peak_freq, f_left, f_right.
    """
    # 1) find all peaks once
    peaks, props = find_peaks(phi_nom,
                              height=min_height,
                              prominence=prominence)
    if peaks.size == 0:
        return phi_nom.copy(), []

    # 2) widths at rel_height
    widths, _, left_ips, right_ips = peak_widths(phi_nom, peaks, rel_height=rel_height)
    idx = np.arange(len(f_nom))
    phi_filt = phi_nom.copy()
    info = []

    for f0 in mode_freqs:
        # extract error‐band around f0
        band = (f_min <= f0) & (f_max >= f0)
        if not np.any(band):
            continue
        low, high = f_nom[band][0], f_nom[band][-1]

        # peaks in [low, high]
        in_band = peaks[(f_nom[peaks] >= low) & (f_nom[peaks] <= high)]
        if in_band.size == 0:
            continue

        # pick highest peak
        pk = in_band[np.argmax(phi_nom[in_band])]
        j  = np.where(peaks == pk)[0][0]

        fl = np.interp(left_ips[j],  idx, f_nom)
        fr = np.interp(right_ips[j], idx, f_nom)

        info.append({
            "mode_freq":  f0,
            "peak_freq":  f_nom[pk],
            "f_left":     fl,
            "f_right":    fr
        })

        # notch
        mask = (f_nom >= fl) & (f_nom <= fr)
        phi_filt[mask] = np.interp(
            f_nom[mask],
            [fl, fr],
            [phi_nom[int(np.floor(left_ips[j]))],
             phi_nom[int(np.ceil (right_ips[j]))]]
        )

    return phi_filt, info


def plot_filtered_spectrum(spec, spec_filtered, peak_info, outfile, line='b-'):
    """Plot spectrum with uncertainty band and duct-mode lines."""
    fig, ax = plt.subplots(1, 2, figsize=(5.6, 3.2), dpi=600, sharex=True, tight_layout=True)

    ax[0].plot(1/spec["f_nom"], spec["phi_nom"], 'k-', lw=0.5, alpha=0.5)
    ax[0].plot(1/spec["f_nom"], spec_filtered, 'r-', lw=0.5, alpha=0.5,
            label="Notched Duct Modes")
    ax[0].legend()

    f_sample = np.logspace(np.log10(5e-4), np.log10(5e-1), 1024)
    p_w_interp = np.interp(f_sample, spec["f_nom"], spec_filtered)
    
    ax[1].plot(1/f_sample, p_w_interp, 'r-', lw=0.7, alpha=0.5)
    sv_filter = savgol_filter(p_w_interp, 101, 1)
    ax[1].plot(1/f_sample, sv_filter, line, lw=0.7, alpha=0.8,
            label="Savitzky-Golay Filter")
    ax[1].legend()
    # ax[0].scatter(peak_info["peak_freq"], 15,)
    ax[0].set_xscale("log")
    ax[0].set_xlim(1/5e-1, 1/5e-4)
    ax[0].set_ylim(0, 25)
    ax[1].set_ylim(0, 7)
    ax[0].set_xlabel("$T^+$")
    ax[1].set_xlabel("$T^+$")
    ax[0].set_ylabel("$f \\Phi_{pp}^+$")
    if not hasattr(plot_filtered_spectrum, "suptit"):
        plot_filtered_spectrum.suptit = itertools.cycle([
            "Wall Pressure Spectrum with Notched Duct Modes",
            "Free-Stream Pressure Spectrum with Notched Duct Modes"
        ])

    fig.suptitle(next(plot_filtered_spectrum.suptit), y=0.95)
    plt.tight_layout()
    plt.savefig(outfile)

def plot_filtered_diff(spec, spec_filtered, spec_filtered_fs, outfile):
    """Plot spectrum with uncertainty band and duct-mode lines."""
    fig, ax = plt.subplots(figsize=(4, 2.5), dpi=600, tight_layout=True)

    f_sample = np.logspace(np.log10(5e-4), np.log10(5e-1), 1024)
    p_w_interp = np.interp(f_sample, spec["f_nom"], spec_filtered)
    sv_filter = savgol_filter(p_w_interp, 101, 1)
    ax.plot(1/f_sample, sv_filter, 'b-', lw=0.7, alpha=0.8,
            label="Wall Pressure")
    p_fs_interp = np.interp(f_sample, spec["f_nom"], spec_filtered_fs)
    sv_filter_fs = savgol_filter(p_fs_interp, 101, 1)
    ax.plot(1/f_sample, sv_filter_fs, 'g-', lw=0.7, alpha=0.8,
            label="Free-Stream Pressure")
    
    diff = sv_filter - sv_filter_fs
    # ax.plot(1/f_sample, diff, 'r-', lw=0.7, alpha=0.8,
    #         label="Difference")
    ax.legend()
    ax.set_xscale("log")
    ax.set_xlim(1/5e-1, 1/5e-4)
    ax.set_ylim(0, 4)
    ax.set_xlabel("$T^+$")
    ax.set_ylabel("$f \\Phi_{pp}^+$")
    plt.savefig(outfile)

if __name__ == "__main__":
    # load and process data
    fs, p_w = load_wallpressure("DuctModes/data/wallpressure_booman_Pa.mat")
    f_raw, psd = compute_psd(p_w, sample_rate=40000,
                             nperseg=len(p_w)//2000,
                             noverlap=(len(p_w)//2000)//2)

    # parameters and error propagation
    nu0, rho0, u_tau0 = 1.52e-5, 1.225, 0.58
    spec = propagate_error(f_raw, psd, nu0, rho0, u_tau0, err_frac=0.04)

    # duct-mode calculation
    W, H, L = 0.30, 0.152, 3.0
    a_eq = np.sqrt(W*H/np.pi)
    # low‐frequency end correction
    Delta_L0 = L*0.1
    L += Delta_L0
    K = 1
    U = 14.2
    c = np.sqrt(1.4 * 101325 / rho0)
    mode_m = [0]
    mode_n = [0]
    mode_l = [0, 1, 4, 5, 8, 11, 15]
    modes = compute_duct_modes(U, c, mode_m, mode_n, mode_l,
                           W, H, L, nu0, u_tau0, err_frac=0.03)

    plot_spectrum_and_modes(spec, modes, mode_l,
                            "DuctModes/figures/duct_and_wall_pressure_uncertainty.png")

    
    fs_fs, p_w_fs = load_wallpressure("DuctModes/data/booman_wallpressure_fspressure_650sec_40khz.mat")
    # ic(p_w_fs.shape)
    f_raw_fs, psd_fs = compute_psd(p_w_fs, sample_rate=40000,
                             nperseg=len(p_w_fs)//2000,
                             noverlap=(len(p_w_fs)//2000)//2)
    spec_fs = propagate_error(f_raw_fs, psd_fs, nu0, rho0, u_tau0, err_frac=0.03)
    modes_fs = compute_duct_modes(U, c, mode_m, mode_n, mode_l,
                               W, H, L, nu0, u_tau0, err_frac=0.03)
    plot_spectrum_and_modes(spec_fs, modes_fs, mode_l,
                            "DuctModes/figures/duct_and_fs_pressure_uncertainty.png")
    
    plot_pw_p_fs(spec['f_nom'], spec_fs['f_nom'], spec['phi_nom'], spec_fs['phi_nom'],
                  "DuctModes/figures/wall_pressure_and_fs_pressure.png")
    
    mode_m = range(0, 5)
    mode_n = range(0, 5)
    mode_l = range(0, 16)
    modes = compute_duct_modes(U, c, mode_m, mode_n, mode_l,
                               W, H, L, nu0, u_tau0, err_frac=0.03)
    # plot results
    
    phi_filtered, peaks_info = notch_filter(
        spec["f_nom"], spec["phi_nom"], spec['f_min'], spec['f_max'], modes["nom"],
    )
    
    plot_filtered_spectrum(spec, phi_filtered, peaks_info,
                            "DuctModes/figures/wall_pressure_duct_notch.png", line='b-')
    
    phi_filtered_fs, peaks_info = notch_filter(
        spec["f_nom"], spec_fs["phi_nom"], spec_fs['f_min'], spec_fs['f_max'], modes["nom"],
    )
    
    plot_filtered_spectrum(spec_fs, phi_filtered_fs, peaks_info,
                            "DuctModes/figures/fs_pressure_duct_notch.png", line='g-')
    
    plot_filtered_diff(spec, phi_filtered, phi_filtered_fs,
                            "DuctModes/figures/wall_pressure_fs_pressure_diff.png")