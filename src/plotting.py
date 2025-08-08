import itertools

import numpy as np
import torch
from scipy.signal import savgol_filter

from matplotlib import pyplot as plt
import seaborn as sns
import scienceplots

plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")

def plot_spectrum_and_modes(spec, modes, mode_l, outfile):
    """Plot Phi+ vs T+ with uncertainty and duct-mode lines."""
    fig, ax = plt.subplots(figsize=(5.6,3.), dpi=600)
    if not hasattr(plot_spectrum_and_modes, "tit"):
        plot_spectrum_and_modes.tit = itertools.cycle([
            "Wall Pressure Spectrum",
            "Free-Stream Pressure Spectrum",
            "CSD"
        ])
    if not hasattr(plot_spectrum_and_modes, "col"):
        plot_spectrum_and_modes.col = itertools.cycle(["blue", "green", "orange"])
    title = next(plot_spectrum_and_modes.tit)
    color = next(plot_spectrum_and_modes.col)
    ax.plot(1/torch.as_tensor(spec["f_nom"]).cpu(), torch.as_tensor(spec["phi_nom"]).cpu(), color, lw=0.5, alpha=0.8)
    ax.fill_betweenx(torch.as_tensor(spec["phi_nom"]).cpu(), torch.as_tensor(spec["f_min"]).cpu(), torch.as_tensor(spec["f_max"]).cpu(),
                     color="gray", alpha=0.3, edgecolor="none", label="±3%")
    for idx, f0 in enumerate(modes["nom"]):
        label = "Duct Mode" if idx==0 else None
        ax.axvline(1/float(f0), color="red", linestyle="--", lw=0.8, alpha=0.5, label=label)
        ax.text(1/float(f0), ax.get_ylim()[1]*0.9, f"l={mode_l[idx]}",
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
    ax.plot(1/torch.as_tensor(fs_w).cpu(), torch.as_tensor(p_w).cpu(), label="Wall", lw=1)
    ax.plot(1/torch.as_tensor(fs_fs).cpu(), torch.as_tensor(p_fs).cpu(), label="Free-Stream", lw=1)
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
    fig, ax = plt.subplots(1,2,figsize=(5.6,3.),dpi=600,sharex=True)
    f_nom = torch.as_tensor(spec["f_nom"]).cpu()
    phi_nom = torch.as_tensor(spec["phi_nom"]).cpu()
    spec_filt = torch.as_tensor(spec_filt).cpu()
    ax[0].plot(1/f_nom, phi_nom, 'k-', lw=0.5, alpha=0.5)
    ax[0].plot(1/f_nom, spec_filt, line, lw=0.7, alpha=0.8, label="Notched")
    ax[0].legend()
    f_samp = np.logspace(np.log10(5e-4), np.log10(5e-1), 1024)
    p_int = np.interp(f_samp, f_nom.numpy(), spec_filt.numpy())
    sg = savgol_filter(p_int, 101, 1)
    ax[1].plot(1/f_samp, p_int, line, lw=0.7, alpha=0.5)
    ax[1].plot(1/f_samp, sg, 'r-', lw=0.7, alpha=0.8, label="Savitzky-Golay")
    ax[1].legend()
    for a in ax:
        a.set_xscale("log")
        a.set_xlim(1/5e-1, 1/5e-4)
    ax[0].set_ylim(-1, 1)
    ax[1].set_ylim(-1e-1, 1e-1)
    ax[0].set_xlabel("$T^+$");  ax[1].set_xlabel("$T^+$")
    ax[0].set_ylabel("$f\\Phi_{pp}^+$")
    if not hasattr(plot_filtered_spectrum, "suptit"):
        plot_filtered_spectrum.suptit = itertools.cycle([
            "Wall Spectrum w/ Notched Duct Modes",
            "Free-Stream Spectrum w/ Notched Duct Modes",
            "CSD Spectrum w/ Notched Duct Modes"
        ])
    fig.suptitle(next(plot_filtered_spectrum.suptit), y=0.95)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def plot_filtered_diff(spec, spec_filt_w, spec_filt_fs, outfile):
    """Plot difference between wall and free-stream SG-filtered spectra."""
    fig, ax = plt.subplots(figsize=(4,2.5),dpi=600)
    f_samp = np.logspace(np.log10(5e-4), np.log10(5e-1), 1024)
    sg_w  = savgol_filter(np.interp(f_samp, torch.as_tensor(spec["f_nom"]).cpu().numpy(), torch.as_tensor(spec_filt_w).cpu().numpy()), 101,1)
    sg_fs = savgol_filter(np.interp(f_samp, torch.as_tensor(spec["f_nom"]).cpu().numpy(), torch.as_tensor(spec_filt_fs).cpu().numpy()), 101,1)
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

def plot_transfer(f, H, outfile, decim=None):
    """
    Plot |H(f)| and arg H(f) versus frequency for sanity checks.

    Parameters
    ----------
    f : ndarray, shape (M,)
        Frequency bins [Hz].
    H : ndarray, shape (M,)
        Complex transfer function.
    decim : int or None
        If set, plot every decim-th point to reduce density.

    Returns
    -------
    fig, (ax_mag, ax_ph) : matplotlib objects
    """
    if decim:
        f = f[::decim]
        H = H[::decim]
    f = torch.as_tensor(f).cpu().numpy()
    H = torch.as_tensor(H).cpu().numpy()
    mag   = np.abs(H)
    phase = np.unwrap(np.angle(H))

    fig, ax_ph = plt.subplots(1, 1, sharex=True, figsize=(5.6, 2.6), dpi=600)
    ax_ph.semilogx(f, phase, lw=1)
    ax_ph.set_ylabel(r'$\arg H_{fw}(f)$')
    ax_ph.set_xlabel(r'$f\ \mathrm{[Hz]}$')
    fig.tight_layout()
    plt.savefig(outfile)
    plt.close()

def plot_phase_match_csd(f, P_w, P_fs, P_w_fs, P_w_fs_opt, outfile):
    fig, ax = plt.subplots(figsize=(5.6, 2.6), dpi=600)

    f = torch.as_tensor(f).cpu()
    ax.plot(f, torch.as_tensor(P_w_fs).real.cpu(), lw=0.5, alpha=0.8)
    ax.plot(f, torch.as_tensor(P_w).real.cpu(), lw=0.5, alpha=0.8)
    ax.plot(f, torch.as_tensor(P_fs).real.cpu(), lw=0.5, alpha=0.8)
    ax.plot(f, torch.as_tensor(P_w_fs_opt).real.cpu(), lw=0.5, alpha=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("$f$ [Hz]")
    ax.set_ylabel("$\\Phi$")
    ax.legend(["$P_{wf}$", "$P_{ww}$", "$P_{ff}$", "$P_{wf}^{opt}$"])
    plt.savefig(outfile)
    plt.close()

def plot_coherence(f, coh, f_match, coh_match, outfile):
    fig, ax = plt.subplots(figsize=(5.6, 2.6), dpi=600)
    ax.plot(torch.as_tensor(f).cpu(), torch.as_tensor(coh).cpu(), lw=0.5, alpha=0.8)
    ax.plot(torch.as_tensor(f_match).cpu(), torch.as_tensor(coh_match).cpu(), lw=0.5, alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$f$ [Hz]")
    ax.set_ylabel("$\\gamma^2$")
    ax.legend(["$\\gamma^2_{wf}$", "$\\gamma^{2, opt}_{wf}$"])
    plt.savefig(outfile)
    plt.close()

def plot_wiener_filter(f, P_w, P_fs, P_w_clean, outfile):
    fig, ax = plt.subplots(figsize=(5.6, 2.6), dpi=600)
    ax.plot(torch.as_tensor(f).cpu(), torch.as_tensor(P_w).cpu(), lw=0.5, alpha=0.8)
    ax.plot(torch.as_tensor(f).cpu(), torch.as_tensor(P_fs).cpu(), lw=0.5, alpha=0.8)
    ax.plot(torch.as_tensor(f).cpu(), torch.as_tensor(P_w_clean).cpu(), lw=0.5, alpha=0.8)
    ax.set_xscale("log")

    ax.set_xlabel("$f$ [Hz]")
    ax.set_ylabel("$\\Phi$")
    ax.legend(["$P_{ww}$", "$P_{ff}$", "$P_{ww}^{clean}$"])
    plt.savefig(outfile)
    plt.close()

def plot_reference_transfer_function(f_grid, H_mag, outfile):
    fig, ax = plt.subplots(figsize=(5.6, 2.6), dpi=600)
    ax.plot(torch.as_tensor(f_grid).cpu(), torch.as_tensor(H_mag).cpu(), lw=0.5, alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$f$ [Hz]")
    ax.set_ylabel("$|H_{\mathrm{ref}}(f)|$")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(outfile)
    plt.close()