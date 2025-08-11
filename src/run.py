"""
Wall- and Free-Stream Pressure Processing
This script demonstrates usage of :class:`WallPressureProcessor`.
"""

import os
import scipy.io as sio
import numpy as np
from scipy.signal import welch, coherence
from scipy.optimize import least_squares

from processor import WallPressureProcessor
from noise_rejection import *
from plotting import *

from matplotlib import pyplot as plt
import seaborn as sns
import scienceplots
plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")
sns.set_palette("colorblind")

# === Hard-coded file paths ===
WALL_PRESSURE_MAT = "data/wallpressure_booman_Pa.mat"
FREESTREAM_PRESSURE_MAT = "data/booman_wallpressure_fspressure_650sec_40khz.mat"

fn_train = [
    'Sensor1(naked)Sensor2(naked).mat',
    'Sensor1(naked)Sensor2(plug1).mat',
    'Sensor1(naked)Sensor3(naked).mat',
    'Sensor1(naked)Sensor3(plug2).mat',
    'Sensor1(naked)Sensor3(nosecone).mat',

]

TEST_MAT = f"data/calibration/{fn_train[0]}"
OUTPUT_DIR = "figures/calibration_08-11"

# === Physical & processing parameters ===
SAMPLE_RATE = 25000        # Hz
NU0 = 1.52e-5              # m^2/s
RHO0 = 1.225               # kg/m^3
U_TAU0 = 0.358              # m/s
ERR_FRAC = 0.03            # ±3% uncertainty
W, He = 0.30, 0.152         # duct width & height (m)
L0 = 3.0                   # duct length (m)
DELTA_L0 = 0.1 * L0        # low-frequency end correction
U = 14.2                   # flow speed (m/s)
C = np.sqrt(1.4 * 101325 / RHO0)  # speed of sound (m/s)
MODE_M = [0]
MODE_N = [0]
MODE_L = [0, 1, 4, 5, 8, 11, 15]

# ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

# ---------- 1) Transfer-function estimation (H1) ----------
def estimate_frf(x, y, fs, nperseg=4096, noverlap=2048, window='hann'):
    """
    x: reference mic (input); y: treated mic (output); fs: Hz.
    Returns: f [Hz], H1(f)=S_yx/S_xx (complex), coherence gamma^2(f).
    """
    x = np.asarray(x) - np.mean(x)
    y = np.asarray(y) - np.mean(y)
    f, Sxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)
    f, Syy = welch(y, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)
    f, Syx = csd(y, x, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)  # y * conj(x)
    H = Syx / Sxx
    coh2 = (np.abs(Syx)**2) / (Sxx * Syy)
    return f, H, coh2

# ---------- 2) Second-order resonator fit (for Fig. 19-style Bode) ----------
def G2nd(f, f0, zeta):
    r = f / f0
    return 1.0 / ((1.0 - r**2) + 1j*(2.0*zeta*r))  # complex FRF

def fit_second_order(f, H, frange=None, remove_delay=True):
    """
    Fits H(f) ≈ A*exp(j*phi0)*G2nd(f,f0,zeta)*exp(-j*2πfτ) (A>0).
    Returns dict with f0, zeta, A, phi0, tau, H_fit (including delay).
    """
    idx = np.ones_like(f, dtype=bool)
    if frange is not None:
        idx = (f >= frange[0]) & (f <= frange[1])
    ffit, Hfit = f[idx], H[idx]

    # Optional pure-delay estimate from phase slope
    tau = 0.0
    if remove_delay:
        ph = np.unwrap(np.angle(Hfit))
        # least-squares line: ph ≈ -2π f τ + φ0
        A = np.vstack([ffit, np.ones_like(ffit)]).T
        sol, _, _, _ = np.linalg.lstsq(A, -ph/(2*np.pi), rcond=None)
        tau, phi0_lin = sol[0], -2*np.pi*sol[1]  # phi0_lin unused later
        Hnod = Hfit * np.exp(1j*2*np.pi*ffit*tau)
    else:
        Hnod = Hfit

    # Initial guesses
    f0_guess = ffit[np.argmax(np.abs(Hnod))]
    p0 = [max(1.0, f0_guess), 0.2, np.abs(Hnod[0]), 0.0]  # f0, zeta, A, phi0

    def resid(p):
        f0, zt, Aamp, ph0 = p
        G = G2nd(ffit, f0, np.abs(zt))
        Hm = Aamp * np.exp(1j*ph0) * G
        return np.r_[ (Hnod.real - Hm.real), (Hnod.imag - Hm.imag) ]

    bounds = ([1e-3, 1e-4, 1e-6, -np.pi], [f[-1]*2, 5.0, 1e6, np.pi])
    res = least_squares(resid, p0, bounds=bounds, ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=20000)
    f0, zeta, Aamp, phi0 = res.x
    H_model = Aamp*np.exp(1j*phi0)*G2nd(f, f0, np.abs(zeta))
    if remove_delay:
        H_model = H_model * np.exp(-1j*2*np.pi*f*tau)

    return dict(f0=f0, zeta=float(np.abs(zeta)), A=Aamp, phi0=phi0, tau=tau, H_fit=H_model, success=res.success)

def plot_bode(f, H, Hm=None):
    mag = np.abs(H); ph = np.unwrap(np.angle(H))
    plt.figure(); 
    plt.subplot(2,1,1); plt.semilogx(f, mag); 
    if Hm is not None: plt.semilogx(f, np.abs(Hm), linestyle='--')
    plt.ylabel('|H|'); plt.grid(True, which='both'); 
    plt.subplot(2,1,2); plt.semilogx(f, ph)
    if Hm is not None: plt.semilogx(f, np.unwrap(np.angle(Hm)), linestyle='--')
    plt.xlabel('f [Hz]'); plt.ylabel('Phase(H) [rad]'); plt.grid(True, which='both'); plt.tight_layout()

# ---------- 3) Inverse filtering (for Fig. 20-style time traces) ----------
def inverse_filter(y, fs, f, H, fc_lp=None, reg=1e-4):
    """
    Regularised frequency-domain deconvolution: Ycorr = Y * H* / (|H|^2+λ).
    H is provided on 'f' (Welch grid); we interpolate to FFT bins.
    """
    N = len(y); Yr = np.fft.rfft(y)
    fr = np.fft.rfftfreq(N, d=1/fs)

    # interpolate magnitude/phase separately for stability
    mag = np.abs(H); phi = np.unwrap(np.angle(H))
    mag_i = np.interp(fr, f, mag, left=mag[0], right=mag[-1])
    phi_i = np.interp(fr, f, phi, left=phi[0], right=phi[-1])
    H_i = mag_i * np.exp(1j*phi_i)

    lam = reg * np.max(mag_i**2)
    H_inv = np.conj(H_i) / (mag_i**2 + lam)
    Ycorr = Yr * H_inv
    ycorr = np.fft.irfft(Ycorr, n=N)

    # optional low-pass (e.g. 3 kHz, as in the paper)
    if fc_lp is not None and fc_lp < fs/2:
        taps = sig.firwin(numtaps=513, cutoff=fc_lp/(fs/2))
        ycorr = sig.filtfilt(taps, [1.0], ycorr, padlen=3*len(taps))
    return ycorr

def plot_times(t, a, b, tspan=None, labels=('a','b')):
    if tspan is None: tspan = (t[0], t[0]+0.016)  # ~16 ms window like Fig. 20
    m = (t>=tspan[0]) & (t<=tspan[1])
    plt.figure()
    plt.plot(t[m], a[m], label=labels[0])
    plt.plot(t[m], b[m], label=labels[1])
    plt.xlabel(r'Time [s]'); plt.ylabel(r'Pressure [Pa]'); plt.legend(); plt.grid(True); plt.tight_layout()

def main(sanity=False):
    proc = WallPressureProcessor(
        sample_rate=SAMPLE_RATE,
        nu0=NU0,
        rho0=RHO0,
        u_tau0=U_TAU0,
        err_frac=ERR_FRAC,
        W=W,
        He=He,
        L0=L0,
        delta_L0=DELTA_L0,
        U=U,
        C=C,
        mode_m=MODE_M,
        mode_n=MODE_N,
        mode_l=MODE_L,
    )

    # proc.load_data(WALL_PRESSURE_MAT, FREESTREAM_PRESSURE_MAT)
    proc.load_test(TEST_MAT)
    ic(f"Loaded data: {proc.p_w.shape}, {proc.p_fs.shape}")
    ref, trt, fs = proc.p_fs.cpu().numpy(), proc.p_w.cpu().numpy(), SAMPLE_RATE

    f, H, coh2 = estimate_frf(ref, trt, fs, nperseg=4096, noverlap=2048)

    fit = fit_second_order(f, H, frange=(1, 5000), remove_delay=True)
    plot_bode(f, H, fit['H_fit'])

    trt_corr = inverse_filter(trt, fs, f, H, fc_lp=3000.0, reg=1e-4)

    t = np.arange(len(ref))/fs
    plot_times(t, trt, ref, labels=('treated (raw)','reference'))
    plot_times(t, trt_corr, ref, labels=('treated (corrected)','reference'))



def main2(sanity=False):
    proc = WallPressureProcessor(
        sample_rate=SAMPLE_RATE,
        nu0=NU0,
        rho0=RHO0,
        u_tau0=U_TAU0,
        err_frac=ERR_FRAC,
        W=W,
        He=H,
        L0=L0,
        delta_L0=DELTA_L0,
        U=U,
        C=C,
        mode_m=MODE_M,
        mode_n=MODE_N,
        mode_l=MODE_L,
    )

    # proc.load_data(WALL_PRESSURE_MAT, FREESTREAM_PRESSURE_MAT)
    proc.load_test(TEST_MAT)
    ic(f"Loaded data: {proc.p_w.shape}, {proc.p_fs.shape}")
    # proc.compute_duct_modes()
    # proc.notch_filter()
    plot_raw_signals(proc.p_w[100:800], proc.p_fs[100:800],
                     os.path.join(OUTPUT_DIR, "raw_signals.png"))

    if sanity:
        p_w_org = proc.p_w.cpu()
        nperseg = len(proc.p_w.cpu()) // 5000
        noverlap = nperseg // 2
        f, P_w = welch(proc.p_w.cpu(), fs=SAMPLE_RATE, nperseg=nperseg,
                    noverlap=noverlap, window="hann")
        f, P_fs = welch(proc.p_fs.cpu(), fs=SAMPLE_RATE, nperseg=nperseg,
                        noverlap=noverlap, window="hann")
        f, P_w_fs = csd(proc.p_w.cpu(), proc.p_fs.cpu(), fs=SAMPLE_RATE,
                        nperseg=nperseg, noverlap=noverlap, window="hann")
        f, P_w_fs_opt = csd(proc.p_w.cpu(), proc.p_fs.cpu(), fs=SAMPLE_RATE,
                            nperseg=nperseg, noverlap=noverlap, window="hann")

    # Find the transfer complex transfer function between the mics
    proc.phase_match(smoothing_len=1)
    ic(f"Phase-matched: {proc.p_w.shape}")

    if sanity:
        f, H_match, mag_diff = phase_match_transfer(p_w_org, proc.p_w.cpu(), SAMPLE_RATE, smoothing_len=1)
        plot_transfer(f, H_match, mag_diff, os.path.join(OUTPUT_DIR, "complex_transfer_function.png"))

        f, P_w_fs_opt = csd(p_w_org, proc.p_fs.cpu(), fs=SAMPLE_RATE,
                            nperseg=nperseg, noverlap=noverlap, window="hann")
        plot_phase_match_csd(f, P_w, P_fs, P_w_fs, P_w_fs_opt,
                            os.path.join(OUTPUT_DIR, "phase_match_csd.png"))
        
        std_corr = np.std(P_w_fs_opt)

        f, coh = coherence(p_w_org, proc.p_fs.cpu(), fs=SAMPLE_RATE,
                        nperseg=nperseg, noverlap=noverlap)
        f_match, coh_match = coherence(proc.p_w.cpu(), proc.p_fs.cpu(), fs=SAMPLE_RATE,
                                    nperseg=nperseg, noverlap=noverlap)
        # plot coherence
        plot_coherence(f, coh, f_match, coh_match,
                    os.path.join(OUTPUT_DIR, "coherence.png"))

        # Plot estimated transfer function

    # Wiener filter time series
    proc.reject_free_stream_noise()
    ic(f"Noise-rejected: {proc.p_w.shape}")
    if sanity:
        f, P_w_clean = welch(proc.p_w.cpu(), fs=SAMPLE_RATE, nperseg=nperseg,
                            noverlap=noverlap, window="hann")
        plot_wiener_filter(f, P_w, P_fs, P_w_clean,
                                    os.path.join(OUTPUT_DIR, "wiener_filtered_spectrum.png"))

    # Load reference spectrum for transfer function
    data  = sio.loadmat("data/premultiplied_spectra_Pw_ReT2000_Deshpande_JFM_2025.mat")
    T_plus_ref = data["Tplus"][0]
    f_ref_plus = 1/T_plus_ref  # convert T+ to f+
    f_Phi_ref_plus = data["premul_Pw_plus"][0]  # first column is the wall-pressure PSD
    u_tau_ref = 0.358
    rho_ref = 1.225
    nu_ref = 1.52e-5  # kinematic viscosity at reference conditions

    # Undo the normalisation to get the reference PSD in physical units
    denom_ref = (rho_ref * u_tau_ref**2)**2
    denom_ref = (RHO0 * U_TAU0**2) ** 2
    f_Pxx_ref = f_Phi_ref_plus * denom_ref  # premultiplied PSD in Pa^2/Hz^2
    f_ref = f_ref_plus / (nu_ref / u_tau_ref**2)  # dimensional frequency f [Hz]
    Pxx_ref = f_Pxx_ref / f_ref  # convert to physical units (Pa^2/Hz)

    f_grid, Phi_w_corrected, H_mag = proc.compute_transfer_function(f_ref, Pxx_ref)
    f_grid = f_grid.cpu()
    Phi_w_corrected = Phi_w_corrected.cpu()
    # Smooth
    Phi_w_corrected  = savgol_filter(Phi_w_corrected, 51, 1)  # window length 51, polynomial order 3
    H_mag = H_mag.cpu()


    plot_reference_transfer_function(
        f_grid, H_mag, os.path.join(OUTPUT_DIR, "reference_transfer_function.png")
    )
    
    denom = (RHO0 * U_TAU0**2) ** 2
    T_plus = 1/(f_grid * (nu_ref / U_TAU0**2))  # convert f to T+
    
    fig, ax = plt.subplots(figsize=(5.6, 2.5), dpi=600)
    ax.plot(T_plus_ref, f_Phi_ref_plus, lw=0.5, alpha=0.8)
    ax.plot(T_plus, f_grid * Phi_w_corrected/denom, lw=0.5, alpha=0.8)

    ax.set_xscale("log")
    ax.set_ylim(0, 4)
    # ax.set_yscale("log")
    ax.set_xlabel("$T^+$")
    ax.set_ylabel("$f\\Phi^+$")
    ax.legend(["Deshpande et al. (2025)", "$P_{ww}^{\\mathrm{corrected}}$"])
    ax.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(os.path.join(OUTPUT_DIR, "final_spectrum.png"))
    plt.close()


if __name__ == "__main__":
    # main(sanity=1)
    main(sanity=True)

