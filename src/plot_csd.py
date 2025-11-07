import h5py
import numpy as np
from scipy.signal import welch, csd, get_window
from icecream import ic
from pathlib import Path
import os

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import scienceplots
from matplotlib.colors import to_rgba
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


def quad_cumulant_spectrum(x, y, fs=1.0, nperseg=2048, noverlap=None, window='hann'):
    x = np.asarray(x) - np.mean(x)
    y = np.asarray(y) - np.mean(y)
    if noverlap is None: noverlap = nperseg//2
    step = nperseg - noverlap
    win = get_window(window, nperseg)
    U = (win**2).sum()

    nseg = (len(x)-nperseg)//step + 1
    Sxx = Syy = Sxy = C4 = 0.0
    for i in range(nseg):
        sl = slice(i*step, i*step+nperseg)
        X = np.fft.rfft(x[sl]*win)
        Y = np.fft.rfft(y[sl]*win)

        Sxx += (X*np.conj(X))
        Syy += (Y*np.conj(Y))
        Sxy += (X*np.conj(Y))
        C4  += (X*X*np.conj(Y)*np.conj(Y))   # raw 4th moment term

    # averages and bias removal (cumulant)
    Sxx /= (nseg*U); Syy /= (nseg*U); Sxy /= (nseg*U)
    C4  /= (nseg*U*U)
    C4c = C4 - Sxx*Syy - Sxy*Sxy

    f = np.fft.rfftfreq(nperseg, 1/fs)
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
            pre_mult = 1/(rho * u_tau**2)**2

            for ax, C4c, pos in zip(axs, [C4c_close, C4c_far], ['close', 'far']):
                f_clean_roi = f_clean[roi_mask]
                C4c_roi = C4c[roi_mask]
                TPLUS = u_tau **2 / (nu * f_clean_roi)
                ax.loglog(TPLUS, f_clean_roi * np.abs(C4c_roi)*pre_mult, label=fr'$Re_\tau \approx${re_labs[i]} {pos}', color=COLOURS[i])
                ax.set_xlabel(r'$T^+$')
                ax.set_ylabel(r'${f\phi_{12}}^+$')
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

def bispectrum_xyz(x, y, z, fs=FS, nperseg=NPERSEG, noverlap=None, window=WINDOW):
    """
    Cross-bispectrum B_xyz(f1,f2) = < X(f1) Y(f2) Z*(f1+f2) >
    Returns principal-domain (k1>=0, k2>=0, k1+k2<K) arrays:
      f1 (K,), f2 (K,), B (K,K complex), b2 (K,K real in [0,1])
    """
    if noverlap is None:
        noverlap = nperseg // 2

    f, X = _stft_frames(x, fs, nperseg, noverlap, window)
    _, Y = _stft_frames(y, fs, nperseg, noverlap, window)
    _, Z = _stft_frames(z, fs, nperseg, noverlap, window)
    M, K = X.shape

    B   = np.zeros((K, K), dtype=np.complex128)
    den1 = np.zeros((K, K), dtype=np.float64)
    den2 = np.zeros((K, K), dtype=np.float64)

    for k1 in range(K):
        # shapes: (M,1), (M, K-k1)
        X1  = X[:, k1][:, None]
        Y2  = Y[:, :K-k1]
        Z12 = Z[:, k1:K]
        prod = X1 * Y2 * np.conj(Z12)       # (M, K-k1)
        B[k1, :K-k1]    = prod.mean(axis=0)
        den1[k1, :K-k1] = np.mean(np.abs(X1 * Y2)**2, axis=0)
        den2[k1, :K-k1] = np.mean(np.abs(Z12)**2, axis=0)

    # principal-domain mask
    k1 = np.arange(K)[:, None]
    k2 = np.arange(K)[None, :]
    mask = (k1 + k2) < K

    eps = 1e-300
    b2 = np.zeros_like(den1)
    valid = mask & (den1 > eps) & (den2 > eps)
    b2[valid] = (np.abs(B[valid])**2) / (den1[valid] * den2[valid])

    # zero-out outside domain for clarity
    B[~mask]  = 0.0
    b2[~mask] = 0.0

    return f, f, B, b2

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

            # Compute bispectra
            f1, f2, Bc, b2c = bispectrum_xyz(ph1_close, ph2_close, ph2_close, fs=FS, nperseg=NPERSEG, window=WINDOW)
            _,  _, Bf, b2f  = bispectrum_xyz(ph1_far,   ph2_far,   ph2_far,   fs=FS, nperseg=NPERSEG, window=WINDOW)

            # ROI mask
            m1 = (f1 >= roi[0]) & (f1 <= roi[1])
            m2 = (f2 >= roi[0]) & (f2 <= roi[1])

            f1m, f2m = f1[m1], f2[m2]
            T1p = (u_tau**2) / (nu * f1m)
            T2p = (u_tau**2) / (nu * f2m)

            # Premultiplied |B| in wall units
            pre = 1.0 / (rho**3 * u_tau**6)
            Fclose = (f1m[:, None] * f2m[None, :]) * np.abs(Bc[np.ix_(m1, m2)] * pre)
            Ffar   = (f1m[:, None] * f2m[None, :]) * np.abs(Bf[np.ix_(m1, m2)] * pre)

            # Bicoherence (dimensionless)
            bclose = b2c[np.ix_(m1, m2)]
            bfar   = b2f[np.ix_(m1, m2)]

            # ---- Plot 1: premultiplied |B|^+ (close/far) ----
            fig1, axs1 = plt.subplots(1, 2, figsize=(8.5, 3.4), constrained_layout=True, sharex=True, sharey=True)
            for ax, Z, title in zip(axs1, [Fclose, Ffar], [f'{L} close', f'{L} far']):
                cs = ax.contourf(T1p[:, None], T2p[None, :], Z, levels=30)
                ax.set_xscale('log'); ax.set_yscale('log')
                ax.set_xlabel(r'$T_1^+$'); ax.set_ylabel(r'$T_2^+$')
                ax.set_title(title, color=COLOURS[i])
                ax.grid(True, which='both', ls='--', lw=0.5, alpha=0.6)
            cbar = fig1.colorbar(cs, ax=axs1.ravel().tolist(), label=r'$f_1 f_2\,|B|^+$')
            fig1.savefig(os.path.join(CSD_BASE, f"bispec_mag_{L}.png"), dpi=410)
            plt.close(fig1)

            # ---- Plot 2: bicoherence b^2 (close/far) ----
            fig2, axs2 = plt.subplots(1, 2, figsize=(8.5, 3.4), constrained_layout=True, sharex=True, sharey=True)
            levels = np.linspace(0.0, 1.0, 21)
            for ax, Z, title in zip(axs2, [bclose, bfar], [f'{L} close', f'{L} far']):
                cs2 = ax.contourf(T1p[:, None], T2p[None, :], Z, levels=levels)
                ax.set_xscale('log'); ax.set_yscale('log')
                ax.set_xlabel(r'$T_1^+$'); ax.set_ylabel(r'$T_2^+$')
                ax.set_title(title + r'  ($b^2$)', color=COLOURS[i])
                ax.grid(True, which='both', ls='--', lw=0.5, alpha=0.6)
            cbar2 = fig2.colorbar(cs2, ax=axs2.ravel().tolist(), label=r'$b^2$')
            fig2.savefig(os.path.join(CSD_BASE, f"bispec_bicoherence_{L}.png"), dpi=410)
            plt.close(fig2)

if __name__ == "__main__":
    # plot_csd_corrected_pressure()
    # plot_quad()
    plot_bispectrum_xyy()
# ============================================================================ 