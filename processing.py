#!/usr/bin/env python3
"""
Wall‐ and Free‐Stream Pressure Processing
Hard-coded inputs and parameters for transparency.
"""

import itertools
import os
import numpy as np
import scipy.io as sio
from scipy.signal import welch, savgol_filter, find_peaks, peak_widths
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import pyplot as plt
import seaborn as sns
import scienceplots

plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")

# === Hard-coded file paths ===
WALL_PRESSURE_MAT      = "data/wallpressure_booman_Pa.mat"
FREESTREAM_PRESSURE_MAT = "data/booman_wallpressure_fspressure_650sec_40khz.mat"
OUTPUT_DIR             = "figures"

# === Physical & processing parameters ===
SAMPLE_RATE = 40000        # Hz
NU0         = 1.52e-5      # m^2/s
RHO0        = 1.225        # kg/m^3
U_TAU0      = 0.58         # m/s
ERR_FRAC    = 0.03         # ±3% uncertainty
W, H        = 0.30, 0.152  # duct width & height (m)
L0          = 3.0          # duct length (m)
DELTA_L0    = 0.1 * L0     # low-frequency end correction
U           = 14.2         # flow speed (m/s)
C           = np.sqrt(1.4 * 101325 / RHO0)  # speed of sound (m/s)
MODE_M      = [0]
MODE_N      = [0]
MODE_L      = [0, 1, 4, 5, 8, 11, 15]

# ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_wallpressure(path, var_name="wall_pressure_fluc_Pa"):
    """Load frequency and pressure from a .mat file."""
    data = sio.loadmat(path)
    fs = np.ravel(data.get("fs_pressure_fluc_Pa", data.get("fs", None)))
    p  = np.ravel(data.get(var_name))
    return fs, p

def compute_psd(signal, fs, nperseg=None, noverlap=None):
    """One-sided PSD via Welch."""
    if nperseg is None:
        nperseg = len(signal) // 2000
    if noverlap is None:
        noverlap = nperseg // 2
    return welch(signal, fs=fs, window="hann",
                 nperseg=nperseg, noverlap=noverlap)

def propagate_error(f_raw, psd, nu0, rho0, u_tau0, err_frac):
    """Compute nominal and ±error bounds for f+ and φ+."""
    nu_min, nu_max     = nu0*(1-err_frac), nu0*(1+err_frac)
    u_tau_min, u_tau_max = u_tau0*(1-err_frac), u_tau0*(1+err_frac)
    rho_min, rho_max   = rho0*(1-err_frac), rho0*(1+err_frac)

    f_nom = f_raw * nu0 / u_tau0**2
    f_min = f_raw * nu_min / u_tau_max**2
    f_max = f_raw * nu_max / u_tau_min**2

    denom_nom = (rho0*u_tau0**2)**2
    denom_min = (rho_max*u_tau_max**2)**2
    denom_max = (rho_min*u_tau_min**2)**2

    phi_nom = f_raw * psd / denom_nom
    phi_min = f_raw * psd / denom_min
    phi_max = f_raw * psd / denom_max

    return {"f_nom": f_nom, "f_min": f_min, "f_max": f_max,
            "phi_nom": phi_nom, "phi_min": phi_min, "phi_max": phi_max}

def duct_mode_freq(U, c, m, n, l, W, H, L):
    """Physical duct‐mode frequency (quarter-wave, closed‐open)."""
    delta2 = c**2 - U**2
    k_sq   = (m*np.pi/W)**2 + (n*np.pi/H)**2
    kz_sq  = ((2*l+1)*np.pi/L)**2
    return (1/(2*np.pi)) * np.sqrt(delta2*k_sq + delta2**2/(4*c**2)*kz_sq)

def compute_duct_modes(U, c, mode_m, mode_n, mode_l, W, H, L, nu0, u_tau0, err_frac):
    """Compute non-dimensional duct modes (nom, min, max)."""
    nu_min, nu_max     = nu0*(1-err_frac), nu0*(1+err_frac)
    u_tau_min, u_tau_max = u_tau0*(1-err_frac), u_tau0*(1+err_frac)
    freqs = {"nom": [], "min": [], "max": []}
    for m in mode_m:
        for n in mode_n:
            for l in mode_l:
                f_phys = duct_mode_freq(U,c,m,n,l,W,H,L)
                freqs["nom"].append(f_phys*nu0/u_tau0**2)
                freqs["min"].append(f_phys*nu_min/u_tau_max**2)
                freqs["max"].append(f_phys*nu_max/u_tau_min**2)
    return freqs

def notch_filter(f_nom, phi_nom, f_min, f_max, mode_freqs,
                 min_height=None, prominence=0.001, rel_height=0.9):
    """Notch out the largest PSD peak around each duct‐mode."""
    peaks, props = find_peaks(phi_nom, height=min_height, prominence=prominence)
    if not peaks.size:
        return phi_nom.copy(), []
    widths, _, left_ips, right_ips = peak_widths(phi_nom, peaks, rel_height=rel_height)
    idx = np.arange(len(f_nom))
    phi_filt = phi_nom.copy()
    info = []
    for f0 in mode_freqs:
        band = (f_min <= f0) & (f_max >= f0)
        if not np.any(band):
            continue
        low, high = f_nom[band][0], f_nom[band][-1]
        in_band = peaks[(f_nom[peaks]>=low)&(f_nom[peaks]<=high)]
        if not in_band.size:
            continue
        pk = in_band[np.argmax(phi_nom[in_band])]
        j = np.where(peaks==pk)[0][0]
        fl = np.interp(left_ips[j], idx, f_nom)
        fr = np.interp(right_ips[j], idx, f_nom)
        info.append({"mode_freq":f0, "peak_freq":f_nom[pk], "f_left":fl, "f_right":fr})
        mask = (f_nom>=fl)&(f_nom<=fr)
        phi_filt[mask] = np.interp(
            f_nom[mask],
            [fl, fr],
            [phi_nom[int(np.floor(left_ips[j]))],
             phi_nom[int(np.ceil (right_ips[j]))]]
        )
    return phi_filt, info

def plot_spectrum_and_modes(spec, modes, mode_l, outfile):
    """Plot φ+ vs T+ with uncertainty and duct‐mode lines."""
    fig, ax = plt.subplots(figsize=(5.6,3.2), dpi=600)
    if not hasattr(plot_spectrum_and_modes, "tit"):
        plot_spectrum_and_modes.tit = itertools.cycle([
            "Wall Pressure Spectrum",
            "Free‐Stream Pressure Spectrum"
        ])
    if not hasattr(plot_spectrum_and_modes, "col"):
        plot_spectrum_and_modes.col = itertools.cycle(["blue", "green"])
    title = next(plot_spectrum_and_modes.tit)
    color = next(plot_spectrum_and_modes.col)
    ax.plot(1/spec["f_nom"], spec["phi_nom"], color, lw=0.5, alpha=0.8)
    ax.fill_betweenx(spec["phi_nom"], spec["f_min"], spec["f_max"],
                     color="gray", alpha=0.3, edgecolor="none", label="±3%")
    for idx, f0 in enumerate(modes["nom"]):
        label = "Duct Mode" if idx==0 else None
        ax.axvline(1/f0, color="red", linestyle="--", lw=0.8, alpha=0.5, label=label)
        ax.text(1/f0, ax.get_ylim()[1]*0.9, f"l={mode_l[idx]}",
                rotation=90, ha="right", va="center", fontsize=8, color="red")
    ax.set_xscale("log")
    ax.set_xlim(1/1e-1, 1/5e-4)
    ax.set_ylim(0, 25)
    ax.set_title(title)
    ax.set_xlabel("$T^+$")
    ax.set_ylabel("$f\\Phi_{pp}^+$")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def plot_pw_p_fs(fs_w, fs_fs, p_w, p_fs, outfile):
    """Plot raw φ+ signals vs T+."""
    fig, ax = plt.subplots(figsize=(4,2.5), dpi=600)
    ax.plot(1/fs_w, p_w, label="Wall", lw=1)
    ax.plot(1/fs_fs, p_fs, label="Free‐Stream", lw=1)
    ax.set_xscale("log")
    ax.set_xlim(1/5e-1, 1/5e-4)
    ax.set_ylim(0, 20)
    ax.set_xlabel("$T^+$")
    ax.set_ylabel("$f\\Phi_{pp}^+$")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def plot_filtered_spectrum(spec, spec_filt, peaks_info, outfile, line):
    """Plot notched spectrum and Savitzky-Golay smoothing."""
    fig, ax = plt.subplots(1,2,figsize=(5.6,3.2),dpi=600,sharex=True)
    ax[0].plot(1/spec["f_nom"], spec["phi_nom"], 'k-', lw=0.5, alpha=0.5)
    ax[0].plot(1/spec["f_nom"], spec_filt, line, lw=0.7, alpha=0.8, label="Notched")
    ax[0].legend()
    f_samp = np.logspace(np.log10(5e-4), np.log10(5e-1), 1024)
    p_int = np.interp(f_samp, spec["f_nom"], spec_filt)
    sg = savgol_filter(p_int, 101, 1)
    ax[1].plot(1/f_samp, p_int, line, lw=0.7, alpha=0.5)
    ax[1].plot(1/f_samp, sg, 'r-', lw=0.7, alpha=0.8, label="Savitzky-Golay")
    ax[1].legend()
    for a in ax:
        a.set_xscale("log")
        a.set_xlim(1/5e-1, 1/5e-4)
    ax[0].set_ylim(0,25)
    ax[1].set_ylim(0,7)
    ax[0].set_xlabel("$T^+$");  ax[1].set_xlabel("$T^+$")
    ax[0].set_ylabel("$f\\Phi_{pp}^+$")
    if not hasattr(plot_filtered_spectrum, "suptit"):
        plot_filtered_spectrum.suptit = itertools.cycle([
            "Wall Spectrum w/ Notched Duct Modes",
            "Free‐Stream Spectrum w/ Notched Duct Modes"
        ])
    fig.suptitle(next(plot_filtered_spectrum.suptit), y=0.95)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def plot_filtered_diff(spec, spec_filt_w, spec_filt_fs, outfile):
    """Plot difference between wall and free‐stream SG-filtered spectra."""
    fig, ax = plt.subplots(figsize=(4,2.5),dpi=600)
    f_samp = np.logspace(np.log10(5e-4), np.log10(5e-1), 1024)
    sg_w  = savgol_filter(np.interp(f_samp, spec["f_nom"], spec_filt_w), 101,1)
    sg_fs = savgol_filter(np.interp(f_samp, spec["f_nom"], spec_filt_fs), 101,1)
    ax.plot(1/f_samp, sg_w, 'b-', label="Wall")
    ax.plot(1/f_samp, sg_fs,'g-', label="Free-Stream")
    ax.set_xscale("log")
    ax.set_xlim(1/5e-1,1/5e-4)
    ax.set_ylim(0,4)
    ax.set_xlabel("$T^+$")
    ax.set_ylabel("$f\\Phi_{pp}^+$")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def main():
    # load & PSD
    fs_w, p_w = load_wallpressure(WALL_PRESSURE_MAT)
    f_raw_w, psd_w = compute_psd(p_w, SAMPLE_RATE)
    spec_w = propagate_error(f_raw_w, psd_w, NU0, RHO0, U_TAU0, ERR_FRAC)

    # compute modes
    L = L0 + DELTA_L0
    modes = compute_duct_modes(U, C, MODE_M, MODE_N, MODE_L, W, H, L, NU0, U_TAU0, ERR_FRAC)
    plot_spectrum_and_modes(spec_w, modes, MODE_L,
                            os.path.join(OUTPUT_DIR, "wall_pressure_uncertainty.png"))

    # free‐stream
    fs_fs, p_fs = load_wallpressure(FREESTREAM_PRESSURE_MAT, var_name="wall_pressure_fluc_Pa")
    f_raw_fs, psd_fs = compute_psd(p_fs, SAMPLE_RATE)
    spec_fs = propagate_error(f_raw_fs, psd_fs, NU0, RHO0, U_TAU0, ERR_FRAC)
    modes_fs = compute_duct_modes(U, C, MODE_M, MODE_N, MODE_L, W, H, L, NU0, U_TAU0, ERR_FRAC)
    plot_spectrum_and_modes(spec_fs, modes_fs, MODE_L,
                            os.path.join(OUTPUT_DIR, "fs_pressure_uncertainty.png"))

    # raw signals overlay
    plot_pw_p_fs(spec_w["f_nom"], spec_fs["f_nom"],
                 spec_w["phi_nom"], spec_fs["phi_nom"],
                 os.path.join(OUTPUT_DIR, "wall_vs_fs.png"))

    # notch & plot filtered
    all_modes = compute_duct_modes(U, C, range(5), range(5), range(16), W, H, L, NU0, U_TAU0, ERR_FRAC)
    phi_filt_w, info_w   = notch_filter(spec_w["f_nom"], spec_w["phi_nom"],
                                        spec_w["f_min"], spec_w["f_max"], all_modes["nom"])
    phi_filt_fs, info_fs = notch_filter(spec_fs["f_nom"], spec_fs["phi_nom"],
                                        spec_fs["f_min"], spec_fs["f_max"], all_modes["nom"])
    plot_filtered_spectrum(spec_w, phi_filt_w, info_w,
                            os.path.join(OUTPUT_DIR, "wall_notched.png"), 'b-')
    plot_filtered_spectrum(spec_fs, phi_filt_fs, info_fs,
                            os.path.join(OUTPUT_DIR, "fs_notched.png"),  'g-')
    plot_filtered_diff(spec_w, phi_filt_w, phi_filt_fs,
                       os.path.join(OUTPUT_DIR, "difference.png"))

if __name__ == "__main__":
    main()
