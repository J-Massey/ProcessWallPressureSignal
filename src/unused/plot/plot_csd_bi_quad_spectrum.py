import h5py
import numpy as np
from scipy.signal import welch, csd, get_window
from icecream import ic
from pathlib import Path
import os

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import scienceplots
from matplotlib.colors import to_rgba, LogNorm
from matplotlib.ticker import LogLocator, LogFormatter
plt.style.use(["science", "grid"])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")

############################
# Constants & defaults
############################
FS = 50_000.0
NPERSEG = 2**12
WINDOW = "hann"
TRIM_CAL_SECS = 5  # seconds trimmed from the start of calibration runs (0 to disable)


DEFAULT_UNITS = {
    'channelData_300_plug': ('Pa', 'Pa'),  # PH, NKD
    'channelData_300_nose': ('Pa', 'Pa'),  # NKD, NC
    'channelData_300':      ('Pa', 'Pa'),  # NC,  PH
}

# --- constants (keep once, top of file) ---
R = 287.05        # J/kg/K
PSI_TO_PA = 6_894.76
P_ATM = 101_325.0
DELTA = 0.035  # m, bl-height of 'channel'
TDEG = [18, 20, 22]

TPLUS_CUT = 10  # picked so that we cut at half the inner peak

CHAIN_SENS_V_PER_PA = {
    'ph1': 50.9e-3,  # V/Pa
    'ph2': 51.7e-3,  # V/Pa
    'nc':  52.4e-3,  # V/Pa
}

COLOURS = ("#1e8ad8", "#ff7f0e", "#26bd26")  # hex equivalents of C0, C1, C2

CSD_BASE = "figures/CSDs/"
os.makedirs(CSD_BASE, exist_ok=True)

def quad_spectrum(x, fs=FS, nperseg=NPERSEG, window=WINDOW):
    win = get_window(window, nperseg)
    nfft = nperseg
    step = nperseg // 2
    segs = (len(x) - nperseg) // step + 1
    Q = 0.0
    for i in range(segs):
        s = x[i*step:i*step+nperseg] * win
        X = np.fft.rfft(s, nfft)
        Q += X * X * np.conj(X) * np.conj(X)  # 4th-order product
    Q /= segs
    f = np.fft.rfftfreq(nfft, 1/fs)
    return f, Q




def plot_csd_corrected_pressure():
    labels = ['0psig', '50psig', '100psig']
    with h5py.File("data/final_pressure/SU_2pt_pressure.h5", 'r') as hf_out:
        h_corrected = hf_out["corrected_data"]

        u_tau_uncertainty = [0.2, 0.1, 0.05]
        # --- main loop over datasets ---
        fig, axs = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True, sharey=True, sharex=True)
        for i, L in enumerate(labels):

            ph1_filt_far = h_corrected[f'{L}_far/ph1'][:]
            ph2_filt_far = h_corrected[f'{L}_far/ph2'][:]
            ph1_filt_close = h_corrected[f'{L}_close/ph1'][:]
            ph2_filt_close = h_corrected[f'{L}_close/ph2'][:]
            rho = h_corrected.attrs[f'rho_{L}']
            u_tau = h_corrected.attrs[f'u_tau_{L}']
            nu = h_corrected.attrs[f'nu_{L}']

            # compute CSDs
            f_clean, P12_close = csd(ph1_filt_close, ph2_filt_close, fs=FS, window=WINDOW, nperseg=NPERSEG)
            f_clean, P12_far = csd(ph1_filt_far, ph2_filt_far, fs=FS, window=WINDOW, nperseg=NPERSEG)
            roi_mask = (f_clean >= 50) & (f_clean <= 1000)
            pre_mult = 1/(rho**2 * u_tau**4)

            for ax, P12, pos in zip(axs, [P12_close, P12_far], ['close', 'far']):
                f_clean_roi = f_clean[roi_mask]
                P12_roi = P12[roi_mask]
                TPLUS = u_tau **2 / (nu * f_clean_roi)
                ax.loglog(TPLUS, f_clean_roi * np.abs(P12_roi)*pre_mult, label=f'{L} {pos}', color=COLOURS[i])
                ax.set_xlabel(r'$T^+$')
                ax.set_ylabel(r'${f\phi_{12}}^+$')
                ax.grid(True, which='both', ls='--', lw=0.5)
                ax.legend(fontsize=8, loc='upper right')

        plt.savefig(CSD_BASE + f'CSD.png', dpi=410)


def quad_cumulant_spectrum(x, y, fs=FS, nperseg=NPERSEG, noverlap=None, window=WINDOW):
    x = np.asarray(x) - np.mean(x)
    y = np.asarray(y) - np.mean(y)
    if noverlap is None: noverlap = nperseg // 2
    step = nperseg - noverlap
    w = get_window(window, nperseg, fftbins=True)
    U = (w**2).sum()

    nseg = (len(x)-nperseg)//step + 1
    if nseg <= 0: raise ValueError("Too few segments")

    Sxx = 0.0j; Syy = 0.0j; Sxy = 0.0j; C4 = 0.0j
    for i in range(nseg):
        sl = slice(i*step, i*step+nperseg)
        X = np.fft.rfft(x[sl]*w).astype(np.complex128, copy=False)
        Y = np.fft.rfft(y[sl]*w).astype(np.complex128, copy=False)
        Sxx += X*np.conj(X);  Syy += Y*np.conj(Y);  Sxy += X*np.conj(Y)
        C4  += X*X*np.conj(Y)*np.conj(Y)

    Sxx /= nseg; Syy /= nseg; Sxy /= nseg; C4 /= nseg

    # Welch density scaling
    scale2 = 1.0/(fs*U)
    Sxx *= scale2; Syy *= scale2; Sxy *= scale2
    C4  *= (scale2**2)

    # One-sided corrections (rfft)
    f  = np.fft.rfftfreq(nperseg, 1.0/fs)
    w2 = np.ones_like(f); w2[1:-1] *= 2.0
    w4 = np.ones_like(f); w4[1:-1] *= 4.0
    Sxx *= w2; Syy *= w2; Sxy *= w2; C4 *= w4

    # Cumulant
    C4c = C4 - Sxx*Syy - Sxy*Sxy
    return f, C4c



def plot_quad():
    labels = ['0psig', '50psig', '100psig']
    re_labs = [1000, 5000, 9000]
    with h5py.File("data/final_pressure/SU_2pt_pressure.h5", 'r') as hf_out:
        h_corrected = hf_out["corrected_data"]

        u_tau_uncertainty = [0.2, 0.1, 0.05]
        # --- main loop over datasets ---
        fig, axs = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True, sharey=True, sharex=True)
        for i, L in enumerate(labels):

            ph1_filt_far = h_corrected[f'{L}_far/ph1'][:]
            ph2_filt_far = h_corrected[f'{L}_far/ph2'][:]
            ph1_filt_close = h_corrected[f'{L}_close/ph1'][:]
            ph2_filt_close = h_corrected[f'{L}_close/ph2'][:]
            rho = h_corrected.attrs[f'rho_{L}']
            u_tau = h_corrected.attrs[f'u_tau_{L}']
            nu = h_corrected.attrs[f'nu_{L}']

            # compute CSDs
            f_clean, C4c_close = quad_cumulant_spectrum(ph1_filt_close, ph2_filt_close, fs=FS, nperseg=NPERSEG)
            f_clean, C4c_far   = quad_cumulant_spectrum(ph1_filt_far,   ph2_filt_far,   fs=FS, nperseg=NPERSEG)

            roi_mask = (f_clean >= 50) & (f_clean <= 1000)
            pre_mult = 1/(rho * u_tau**2)**4

            for ax, C4c, pos in zip(axs, [C4c_close, C4c_far], ['close', 'far']):
                f_clean_roi = f_clean[roi_mask]
                C4c_roi = C4c[roi_mask]
                TPLUS = u_tau **2 / (nu * f_clean_roi)
                ax.loglog(TPLUS, f_clean_roi * np.abs(C4c_roi)*pre_mult, label=fr'$Re_\tau \approx${re_labs[i]} {pos}', color=COLOURS[i])
                ax.set_xlabel(r'$T^+$')
                ax.set_ylabel(r'${f\phi_{12}^{(4)}}^+$')
                ax.grid(True, which='both', ls='--', lw=0.5)
                ax.legend(fontsize=8, loc='upper right')

        plt.savefig(CSD_BASE + f'quad.png', dpi=410)

def _stft_frames(x, fs, nperseg, noverlap, window):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    w = get_window(window, nperseg, fftbins=True)
    step = nperseg - noverlap
    if len(x) < nperseg:
        raise ValueError("Signal shorter than nperseg")
    nseg = 1 + (len(x) - nperseg) // step
    # collect frames
    X = np.empty((nseg, nperseg//2 + 1), dtype=np.complex128)
    for i in range(nseg):
        s = x[i*step : i*step + nperseg] * w
        X[i] = np.fft.rfft(s, n=nperseg)
    f = np.fft.rfftfreq(nperseg, 1.0/fs)
    return f, X  # (M,K)

def bispectrum_xyz(x, y, z, fs=FS, nperseg=NPERSEG, noverlap=None, window=WINDOW,
                   fmin=None, fmax=None):
    """
    Cross-bispectrum B_xyz(f1,f2) = < X(f1) Y(f2) Z*(f1+f2) >
    Returns: f1, f2, B (complex), b2 (real in [0,1]); all one-sided, ROI-limited.
    """
    if noverlap is None:
        noverlap = nperseg // 2

    f, X = _stft_frames(x, fs, nperseg, noverlap, window)
    _, Y = _stft_frames(y, fs, nperseg, noverlap, window)
    _, Z = _stft_frames(z, fs, nperseg, noverlap, window)

    # Apply ROI early to shrink K
    m = np.ones_like(f, dtype=bool)
    if fmin is not None: m &= (f >= fmin)
    if fmax is not None: m &= (f <= fmax)
    f = f[m]; X = X[:, m]; Y = Y[:, m]; Z = Z[:, m]

    M, K = X.shape
    I = np.arange(K)[:, None]
    J = np.arange(K)[None, :]
    S = I + J
    mask = (S < K)  # principal domain (f1>=0,f2>=0,f1+f2<=f_Nyq)

    # Accumulators
    B_acc   = np.zeros((K, K), dtype=np.complex128)
    den1acc = np.zeros((K, K), dtype=np.float64)
    den2acc = np.zeros((K, K), dtype=np.float64)

    # Vectorised over freqs; iterate over segments
    for mseg in range(M):
        XY = X[mseg][:, None] * Y[mseg][None, :]          # (K,K)
        Zsum = np.zeros((K, K), dtype=np.complex128)
        # safe masked gather: Z*(f1+f2) only where valid
        Zsum[mask] = np.conj(Z[mseg])[S[mask]]

        B_acc   += XY * Zsum
        den1acc += np.abs(XY)**2
        den2acc += np.abs(Zsum)**2

    B_raw = B_acc / M
    den1  = den1acc / M
    den2  = den2acc / M

    # bicoherence (dimensionless, robust to constant scaling)
    b2 = np.zeros_like(den1, dtype=np.float64)
    valid = mask & (den1 > 0) & (den2 > 0)
    b2[valid] = (np.abs(B_raw[valid])**2) / (den1[valid] * den2[valid])

    # One-sided density scaling (kept minimal; avoids extra constants)
    # You can add precise Welch/one-sided factors if you need absolute units.
    return f, f, B_raw * mask, b2 * mask



def plot_bispectrum_xyy():
    """
    Computes cross-bispectrum B_x y y and bicoherence for (ph1, ph2, ph2)
    for close/far at each pressure. Saves T1+,T2+ contour plots.
    """
    labels = ['0psig', '50psig', '100psig']
    roi = (50.0, 1000.0)
    os.makedirs(CSD_BASE, exist_ok=True)

    with h5py.File("data/final_pressure/SU_2pt_pressure.h5", 'r') as hf_out:
        h_corrected = hf_out["corrected_data"]

        for i, L in enumerate(labels):
            ph1_close = h_corrected[f'{L}_close/ph1'][:]
            ph2_close = h_corrected[f'{L}_close/ph2'][:]
            ph1_far   = h_corrected[f'{L}_far/ph1'][:]
            ph2_far   = h_corrected[f'{L}_far/ph2'][:]

            rho = h_corrected.attrs[f'rho_{L}']
            u_tau = h_corrected.attrs[f'u_tau_{L}']
            nu = h_corrected.attrs[f'nu_{L}']

            # Compute with ROI inside:
            f1, f2, Bc, b2c = bispectrum_xyz(ph1_close, ph2_close, ph2_close,
                                            fs=FS, nperseg=NPERSEG, window=WINDOW,
                                            fmin=roi[0], fmax=roi[1])
            _,  _, Bf, b2f  = bispectrum_xyz(ph1_far,   ph2_far,   ph2_far,
                                            fs=FS, nperseg=NPERSEG, window=WINDOW,
                                            fmin=roi[0], fmax=roi[1])

            T1p = (u_tau**2) / (nu * f1)
            T2p = (u_tau**2) / (nu * f2)

            # |B|^+ scaling (premultiplied)
            pre = 1.0 / ((rho * u_tau**2)**3)
            Fclose = (f1[:, None] * f2[None, :]) * np.abs(Bc) * pre
            Ffar   = (f1[:, None] * f2[None, :]) * np.abs(Bf) * pre

            # ---- Plot 1: premultiplied |B|^+ (close/far) ----
            fig1, axs1 = plt.subplots(1, 2, figsize=(8.5, 3.4),
                                    constrained_layout=True, sharex=True, sharey=True)
            pow1 = 10 - i/2; pow2 = 12 - i/2
            vmin_mag, vmax_mag = 1*10**pow1, 1*10**pow2
            levels_mag = np.logspace(np.log10(vmin_mag), np.log10(vmax_mag), 40)
            norm_mag = LogNorm(vmin=vmin_mag, vmax=vmax_mag)
            cmap_mag = sns.color_palette("magma", as_cmap=True)
            for ax, Z, title in zip(axs1, [Fclose, Ffar], [f'{L} close', f'{L} far']):
                cs = ax.contourf(
                                T1p, T2p, Z.T,
                                levels=levels_mag,
                                norm=norm_mag,
                                cmap=cmap_mag,
                                extend='both',
                                )
                ax.set_xscale('log'); ax.set_yscale('log')
                ax.set_xlabel(r'$T_1^+$'); ax.set_ylabel(r'$T_2^+$')
                ax.set_title(title, color=COLOURS[i])
                ax.grid(True, which='both', ls='--', lw=0.5, alpha=0.6)
                ax.set_xticks([30, 100, 300, 1000], labels=[30, 100, 300, 1000])
                ax.set_yticks([30, 100, 300, 1000], labels=[30, 100, 300, 1000])
                ax.set_xlim(20, 2100); ax.set_ylim(20, 2100)
            cbar = fig1.colorbar(cs, ax=axs1.ravel().tolist(),
                        label=r'$f_1 f_2\,|B|^+$ (log scale)')
            cbar.locator = LogLocator(base=10, subs=[1, 2, 5])
            cbar.formatter = LogFormatter(base=10, labelOnlyBase=False)
            cbar.update_ticks()
            fig1.savefig(os.path.join(CSD_BASE, f"bispec_mag_{L}.png"), dpi=410)
            plt.close(fig1)


            # ---- Plot 2: bicoherence b^2 (close/far) ----
            fig2, axs2 = plt.subplots(1, 2, figsize=(8.5, 3.4),
                                    constrained_layout=True, sharex=True, sharey=True)
            vmin_b, vmax_b = 1e-4, 1e-3
            levels_coh = np.logspace(np.log10(vmin_b), np.log10(vmax_b), 40)
            norm_coh = LogNorm(vmin=vmin_b, vmax=vmax_b)
            cmap_coh = sns.color_palette("viridis", as_cmap=True)
            for ax, Z, title in zip(axs2, [b2c, b2f], [f'{L} close', f'{L} far']):
                cs2 = ax.contourf(
                                T1p, T2p, Z.T,
                                levels=levels_coh,
                                norm=norm_coh,
                                cmap=cmap_coh,
                                extend='both',
                                )
                ax.set_xscale('log'); ax.set_yscale('log')
                ax.set_xlabel(r'$T_1^+$'); ax.set_ylabel(r'$T_2^+$')
                ax.set_title(title + r'  ($b^2$)', color=COLOURS[i])
                ax.grid(True, which='both', ls='--', lw=0.5, alpha=0.6)
                ax.set_xticks([30, 100, 300, 1000], labels=[30, 100, 300, 1000])
                ax.set_yticks([30, 100, 300, 1000], labels=[30, 100, 300, 1000])
                ax.set_xlim(20, 2100); ax.set_ylim(20, 2100)
            cbar2 = fig2.colorbar(cs2, ax=axs2.ravel().tolist(),
                        label=r'$b^2$ (log scale)')
            cbar2.locator = LogLocator(base=10, subs=[1, 2, 5])
            cbar2.formatter = LogFormatter(base=10, labelOnlyBase=False)
            cbar2.update_ticks()
            fig2.savefig(os.path.join(CSD_BASE, f"bispec_bicoherence_{L}.png"), dpi=410)
            plt.close(fig2)




if __name__ == "__main__":
    plot_csd_corrected_pressure()
    plot_quad()
    plot_bispectrum_xyy()
# ============================================================================ 