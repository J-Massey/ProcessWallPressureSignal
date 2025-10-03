import numpy as np
from scipy.signal import welch, csd, get_window, iirnotch, sosfiltfilt, find_peaks
import scipy.signal as signal
import scipy.io as sio
from matplotlib import pyplot as plt
from icecream import ic
import os

from plotting import (
    plot_spectrum,
    plot_raw_spectrum,
    plot_transfer_NKD,
    plot_transfer_PH,
    plot_transfer_NC,
    plot_corrected_trace_NKD,
    plot_corrected_trace_NC,
    plot_corrected_trace_PH,
    plot_time_series,
    plot_spectrum_pipeline,
)

############################
#Plan
#    \item Raw TS
#    \item Raw spectra
#    \item PH calibration
#    \item TF corrected TS
#    \item TF corrected spectra
#    \item Notched electrical facility noise
#    \item Notched FS and PH spectra
#    \item Coherence correction
#    \item Cleaned spectra
############################

############################
# Constants & defaults
############################
FS = 50_000.0
NPERSEG = 2**12
WINDOW = "hann"
CALIB_BASE_DIR = "data/calibration_30_09"  # base directory to save calibration .npy files
TRIM_CAL_SECS = 5  # seconds trimmed from the start of calibration runs (0 to disable)

R = 287.0         # J/kg/K
T = 298.0         # K (adjust if you have per-case temps)
P_ATM = 101_325.0 # Pa
PSI_TO_PA = 6_894.76

# Labels for operating conditions; first entry is assumed 'atm'
psi_labels = ['atm']
Re_tau = 5000
delta = 0.035
nu_utau = 1/(Re_tau / delta)
u_tau = 0.51
nu = 1/(u_tau/nu_utau)
# at T+=20, what is f [Hz]? T^+\equiv T u_\tau^2/\nu
ic(1/(20 * nu/(u_tau**2)))


# -------------------------------------------------------------------------
# >>> Units & column layout for loaded time-series (per MATLAB key) <<<
# Keys (column order):
#   - channelData_300_plug : col1=PH (pinhole), col2=NKD (naked)   [calibration sweep]
#   - channelData_300_nose : col1=NKD,            col2=NC          [calibration sweep]
#   - channelData_300      : col1=NC,             col2=PH          [real flow data]
DEFAULT_UNITS = {
    'channelData_300_plug': ('Pa', 'Pa'),  # PH, NKD
    'channelData_300_nose': ('Pa', 'Pa'),  # NKD, NC
    'channelData_300':      ('Pa', 'Pa'),  # NC,  PH
}

# If using volts, specify sensitivities (V/Pa) and preamp gains here:
SENSITIVITIES_V_PER_PA = {  # leave empty if not using 'V'
    # 'NKD': 0.05,
    # 'PH':  0.05,
    # 'NC':  0.05,
}
PREAMP_GAIN = {  # linear gain; leave 1.0 if unknown
    'NKD': 1.0,
    'PH':  1.0,
    'NC':  1.0,
}
# -------------------------------------------------------------------------
DATA_LAYOUT = {
    'channelData_300_plug': ('PH', 'NKD'),  # col1, col2
    'channelData_300_nose': ('NC', 'NKD'),
    'channelData_300':      ('NC',  'PH'),
}

def load_pair_pa(path: str, key: str):
    ch1, ch2 = DATA_LAYOUT[key]
    x, y = load_mat_to_pa(path, key=key, ch1_name=ch1, ch2_name=ch2)
    return x, y


def inner_scales(Re_taus, u_taus, nu_atm):
    """Return delta (from the atm case) and nu for each case via Re_tau relation."""
    Re_taus = np.asarray(Re_taus, dtype=float)
    u_taus = np.asarray(u_taus, dtype=float)
    delta = Re_taus[0] * nu_atm / u_taus[0]
    nus = delta * u_taus / Re_taus
    return float(delta), nus


def estimate_frf(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    window: str = WINDOW,
    detrend: str = "false",
    npsg: int = NPERSEG,
):
    """
    Estimate H1 FRF and magnitude-squared coherence using Welch/CSD.

    Returns
    -------
    f : array_like [Hz]
    H : array_like (complex) = S_yx / S_xx  (x → y)
    gamma2 : array_like in [0, 1]
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    nseg = int(min(npsg, x.size, y.size))
    if nseg < 8:
        raise ValueError(f"Signal too short for FRF: n={min(x.size, y.size)}")
    nov = int(min(npsg // 2, nseg // 2))
    w = get_window(window, nseg, fftbins=True)

    f, Sxx = welch(x, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)
    _, Syy = welch(y, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)
    # SciPy convention: csd(x, y) = E{ X * conj(Y) }
    _, Sxy = csd(x, y, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)  # x→y

    H = np.conj(Sxy) / Sxx               # H1 = Syx / Sxx = conj(Sxy)/Sxx
    gamma2 = (np.abs(Sxy) ** 2) / (Sxx * Syy)
    gamma2 = np.clip(gamma2.real, 0.0, 1.0)
    return f, H, gamma2


def _resolve_cal_dir(name: str) -> str:
    """
    Prefer new `data/calibration/<name>` if present; otherwise fall back to `figures/<name>`.
    This keeps backward compatibility with existing saved calibrations.
    """
    new_dir = os.path.join(CALIB_BASE_DIR, name)
    if os.path.isdir(new_dir):
        return new_dir
    old_dir = os.path.join("figures", name)
    return old_dir


def convert_to_pa(x: np.ndarray, units: str, *, channel_name: str = "unknown") -> np.ndarray:
    """
    Convert a pressure time series to Pa.
    Supported units: 'Pa', 'kPa', 'mbar', 'V'.
    For 'V', you must provide a sensitivity (V/Pa) and optional preamp gain via dicts above.
    """
    u = units.lower()
    if u == 'pa':
        return x.astype(float)
    elif u == 'kpa':
        return (x.astype(float) * 1e3)
    elif u == 'mbar':
        return (x.astype(float) * 100.0)  # 1 mbar = 100 Pa
    elif u in ('v', 'volt', 'volts'):
        if channel_name not in SENSITIVITIES_V_PER_PA or SENSITIVITIES_V_PER_PA[channel_name] is None:
            raise ValueError(
                f"Sensitivity (V/Pa) for channel '{channel_name}' not provided; cannot convert V→Pa."
            )
        sens = float(SENSITIVITIES_V_PER_PA[channel_name])  # V/Pa
        gain = float(PREAMP_GAIN.get(channel_name, 1.0))
        # Pa = V / (gain * (V/Pa))
        return x.astype(float) / (gain * sens)
    else:
        raise ValueError(f"Unsupported units '{units}' for channel '{channel_name}'")


def load_mat(path: str, key: str = "channelData"):
    """Load an Nx2 array from a MATLAB .mat file under `key` robustly (no unit conversion)."""
    mat = sio.loadmat(path, squeeze_me=True)
    if key not in mat:
        raise KeyError(f"Key '{key}' not found in {path}. Available: {list(mat.keys())}")
    data = np.asarray(mat[key])
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array under '{key}', got shape {data.shape} in {path}")
    # Handle either (N,2) or (2,N)
    if data.shape[1] == 2:
        x_r = data[:, 0].astype(float)
        y_r = data[:, 1].astype(float)
    elif data.shape[0] == 2:
        x_r = data[0, :].astype(float)
        y_r = data[1, :].astype(float)
    else:
        raise ValueError(f"Unsupported shape for '{key}': {data.shape} in {path}")
    return x_r, y_r


def load_mat_to_pa(path: str, key: str, ch1_name: str, ch2_name: str):
    """
    Load two channels and convert each to Pa using DEFAULT_UNITS[key].
    ch1_name/ch2_name are used only if units are 'V' (to look up sensitivity/gain).
    """
    x_r, y_r = load_mat(path, key=key)
    units_pair = DEFAULT_UNITS.get(key, ('Pa', 'Pa'))
    x_pa = convert_to_pa(x_r, units_pair[0], channel_name=ch1_name)
    y_pa = convert_to_pa(y_r, units_pair[1], channel_name=ch2_name)
    return x_pa, y_pa


def compute_spec(fs: float, x: np.ndarray, npsg : int = NPERSEG):
    """Welch PSD with sane defaults and shape guarding. Returns (f [Hz], Pxx [Pa^2/Hz])."""
    x = np.asarray(x, float)
    nseg = int(min(npsg, x.size))
    nov = nseg // 2
    w = get_window(WINDOW, nseg, fftbins=True)
    f, Pxx = welch(
        x,
        fs=fs,
        window=w,
        nperseg=nseg,
        noverlap=nov,
        detrend="constant",
        scaling="density",
        return_onesided=True,
    )
    return f, Pxx

def wiener_forward(x, fs, f, H, gamma2, nfft_pow=0, demean=True, zero_dc=True):
    """
    Forward FRF application: given x (PH) and H_{PH->NKD}, synthesize ŷ ≈ NKD.
    Uses coherence-weighted magnitude (sqrt(gamma2)) and safe OOB taper.
    """
    import numpy as np

    x = np.asarray(x, float)
    if demean:
        x = x - x.mean()

    N = x.size
    Nfft = int(2 ** np.ceil(np.log2(N))) if nfft_pow == 0 else 2**nfft_pow
    X = np.fft.rfft(x, n=Nfft)
    fr = np.fft.rfftfreq(Nfft, d=1.0/fs)

    mag = np.abs(H)
    phi = np.unwrap(np.angle(H))

    # Interp H and coherence to FFT grid
    mag_i = np.interp(fr, f, mag, left=0.0, right=0.0)
    phi_i = np.interp(fr, f, phi, left=0.0, right=0.0)
    H_i = mag_i * np.exp(1j * phi_i)

    # Coherence weighting (shrink toward zero where unreliable)
    g2_i = np.clip(np.interp(fr, f, gamma2, left=0.0, right=0.0), 0.0, 1.0)
    W = np.sqrt(g2_i)

    # Apply forward FRF with weighting
    Y = W * H_i * X

    # Optional zero DC
    if zero_dc and Y.size > 0:
        Y[0] = 0.0

    y_hat = np.fft.irfft(Y, n=Nfft)[:N]
    return y_hat



def wiener_inverse(
    y_r: np.ndarray,
    fs: float,
    f: np.ndarray,
    H: np.ndarray,
    gamma2: np.ndarray,
    demean: bool = True,
    zero_dc: bool = True,
):
    """
    Reconstruct source-domain signal from a measured signal using a
    coherence-weighted inverse filter: H_inv = gamma^2 * H* / |H|^2.

    Parameters
    ----------
    y_r : array_like
        Measured time series (maps from source via H).
    fs : float
        Sample rate [Hz].
    f, H, gamma2 : arrays
        FRF and coherence tabulated on frequency vector f (as from estimate_frf).
        H must correspond to x→y (so this operation aims to recover x from y).
    demean : bool
        Remove mean before FFT.
    zero_dc : bool
        Zero DC (and Nyquist if present) in the inverse filter.

    Returns
    -------
    x_hat : array_like
        Reconstructed source-domain time series.
    """
    y = np.asarray(y_r, float)
    if demean:
        y = y - y.mean()
    N = y.size
    Nfft = int(2 ** np.ceil(np.log2(N)))  # next power of 2

    # FFT of measurement
    Yr = np.fft.rfft(y, n=Nfft)
    fr = np.fft.rfftfreq(Nfft, d=1.0 / fs)

    # Interpolate |H| and unwrapped phase to FFT grid
    mag = np.abs(H)
    phi = np.unwrap(np.angle(H))
    # Safer OOB behaviour: taper magnitude to zero outside measured band
    mag_i = np.interp(fr, f, mag, left=0.0, right=0.0)
    phi_i = np.interp(fr, f, phi, left=phi[0], right=phi[-1])
    Hi = mag_i * np.exp(1j * phi_i)

    # Interpolate and clip coherence; set OOB to 0 as well
    g2_i = np.clip(np.interp(fr, f, gamma2, left=0.0, right=0.0), 0.0, 1.0)

    # Inverse filter
    eps = np.finfo(float).eps
    Hinv = g2_i * np.conj(Hi) / np.maximum(mag_i**2, eps)
    if zero_dc:
        Hinv[0] = 0.0
        if Nfft % 2 == 0:  # real Nyquist bin exists
            Hinv[-1] = 0.0

    x_hat = np.fft.irfft(Yr * Hinv, n=Nfft)[:N]
    return x_hat


def apply_frf(
    x: np.ndarray,
    fs: float,
    f: np.ndarray,
    H: np.ndarray,
    demean: bool = True,
    zero_dc: bool = True,
):
    """
    Apply a measured FRF H (x→y) to a time series x to synthesise y.
    This is the forward operation: Y = H · X in the frequency domain.
    """
    x = np.asarray(x, float)
    if demean:
        x = x - x.mean()

    N = x.size
    Nfft = int(2 ** np.ceil(np.log2(N)))
    X = np.fft.rfft(x, n=Nfft)
    fr = np.fft.rfftfreq(Nfft, d=1.0 / fs)

    mag = np.abs(H)
    phi = np.unwrap(np.angle(H))
    # Safer OOB behaviour: taper magnitude to zero outside measured band
    mag_i = np.interp(fr, f, mag, left=1.0, right=0.0)
    phi_i = np.interp(fr, f, phi, left=phi[0], right=phi[-1])
    Hi = mag_i * np.exp(1j * phi_i)

    if zero_dc:
        Hi[0] = 0.0
        if Nfft % 2 == 0:
            Hi[-1] = 0.0

    y = np.fft.irfft(X * Hi, n=Nfft)[:N]
    return y

def apply_frf_ph_corr(
    x: np.ndarray,
    fs: float,
    f: np.ndarray,
    H: np.ndarray,
    demean: bool = True,
    zero_dc: bool = True,
):
    """
    Apply a measured FRF H (x→y) to a time series x to synthesise y.
    This is the forward operation: Y = H · X in the frequency domain.
    """
    x = np.asarray(x, float)
    if demean:
        x = x - x.mean()

    N = x.size
    Nfft = int(2 ** np.ceil(np.log2(N)))
    X = np.fft.rfft(x, n=Nfft)
    fr = np.fft.rfftfreq(Nfft, d=1.0 / fs)

    mag = np.abs(H)
    phi = np.unwrap(np.angle(H))
    # Safer OOB behaviour: taper magnitude to zero outside measured band
    mag_i = np.interp(fr, f, mag, left=0.0, right=0.0)
    phi_i = np.interp(fr, f, phi, left=phi[0], right=phi[-1])
    Hi = mag_i * np.exp(1j * phi_i)

    if zero_dc:
        Hi[0] = 0.0
        if Nfft % 2 == 0:
            Hi[-1] = 0.0

    y = np.fft.irfft(X * Hi, n=Nfft)[:N]
    return y


def design_notches(fs, freqs, Q=30.0):
    """
    Make a cascade of IIR notch filters (as SOS).
    
    Parameters
    ----------
    fs : float
        Sampling rate [Hz].
    freqs : list of float
        Centre frequencies to notch [Hz].
    Q : float
        Quality factor (higher = narrower notch).
    """
    sos_list = []
    for f0 in freqs:
        w0 = f0 / (fs/2.0)   # normalised frequency (Nyquist=1)
        b, a = iirnotch(w0, Q)
        sos_list.append(np.hstack([b, a]))
    if not sos_list:
        return None
    return np.vstack(sos_list)

def apply_notches(x, sos):
    """Apply zero-phase notch filtering to signal x."""
    if sos is None:
        return x
    return sosfiltfilt(sos, x)



# --------- Inner scaling helpers (units & Jacobian are correct) ---------
def f_plus_from_f(f: np.ndarray, u_tau: float, nu: float) -> np.ndarray:
    """f⁺ = f * nu / u_tau²."""
    return f * (nu / (u_tau**2))


def phi_pp_plus_per_fplus(Pyy: np.ndarray, rho: float, u_tau: float, nu: float) -> np.ndarray:
    """
    Dimensionless PSD per unit f⁺:
    Φ_pp⁺(f⁺) = Φ_pp(f) / (ρ² u_τ² ν)
    (Jacobian df/df⁺ = u_τ²/ν has been applied.)
    """
    return Pyy / ((rho**2) * (u_tau**2) * nu)


def premultiplied_phi_pp_plus(f: np.ndarray, Pyy: np.ndarray, rho: float, u_tau: float, nu: float):
    """
    Return (1/f⁺, f⁺ Φ_pp⁺(f⁺)) with the zero bin safely excluded.
    Using the identity: f⁺ Φ_pp⁺ = f * Φ_pp / (ρ² u_τ⁴), but computed via f⁺·Φ_pp⁺ for clarity.
    """
    f = np.asarray(f, float)
    Pyy = np.asarray(Pyy, float)
    f_plus = f_plus_from_f(f, u_tau, nu)
    phi_plus = phi_pp_plus_per_fplus(Pyy, rho, u_tau, nu)   # per unit f⁺
    y = f_plus * phi_plus                                   # premultiplied
    mask = f_plus > 0
    return 1.0 / f_plus[mask], y[mask]


# --------- Robust forward equaliser for PH→NKD ---------
def _moving_average(y: np.ndarray, win: int = 7):
    if win <= 1:
        return y
    k = int(max(1, win))
    k = k + 1 - (k % 2)  # force odd
    w = np.ones(k, float) / k
    return np.convolve(y, w, mode='same')

def stabilise_forward_frf(f: np.ndarray,
                          H: np.ndarray,
                          gamma2: np.ndarray,
                          fs: float,
                          g2_thresh: float = 0.6,
                          smooth_bins: int = 9,
                          enforce_min_gain: bool = True,
                          monotone_hf_envelope: bool = True,
                          clip_gain_max: float = 50.0):
    """
    Produce a forward PH→NKD equaliser H_stab that:
      * keeps the measured phase (unwrapped),
      * smooths |H|,
      * enforces |H| >= 1 where coherence is reliable,
      * (optionally) makes the HF tail non-decreasing beyond the first unity crossing,
      * extends the last reliable gain into the low-coherence tail,
      * clips ridiculous gains to 'clip_gain_max'.
    """
    f = np.asarray(f, float)
    H = np.asarray(H, complex)
    gamma2 = np.asarray(gamma2, float)

    mag = np.abs(H).astype(float)
    phi = np.unwrap(np.angle(H))
    # smooth
    mag_s = _moving_average(mag, smooth_bins)

    # coherent region
    cmask = (gamma2 >= g2_thresh)
    if np.any(cmask) and enforce_min_gain:
        mag_s[cmask] = np.maximum(mag_s[cmask], 1.0)

    # find first index where |H|>=1 in coherent region
    idx_coh = np.where(cmask & (mag_s >= 1.0))[0]
    if monotone_hf_envelope and idx_coh.size > 0:
        start = int(idx_coh[0])
        # enforce non-decreasing envelope beyond 'start' (within coherent mask)
        # we also allow it to propagate a little into slightly lower g2 to avoid edge dips
        env = np.maximum.accumulate(mag_s[start:])
        mag_s[start:] = env

    # extend the last reliable coherent gain across the trailing low-coherence tail
    if np.any(cmask):
        last_coh = int(np.max(np.where(cmask)[0]))
        mag_s[last_coh+1:] = max(mag_s[last_coh], 1.0)

    # clip absurd gains
    mag_s = np.clip(mag_s, 0.0, clip_gain_max)

    H_stab = mag_s * np.exp(1j * phi)
    return H_stab


# --------- Coherent FS-noise cancellation ---------
def coherent_band_mask(f: np.ndarray, gamma2: np.ndarray, fs: float,
                       fmin: float = 50.0, fmax_frac: float = 0.4, g2_thresh: float = 0.6):
    """Return boolean mask for a sensible coherent band."""
    return (f >= fmin) & (f <= fmax_frac * fs) & (gamma2 >= g2_thresh)

def band_energy_from_psd(f: np.ndarray, Pxx: np.ndarray, mask: np.ndarray):
    """Integrate PSD over masked band to get band-limited variance [Pa^2]."""
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(Pxx[mask], f[mask]))

def coherent_cancel(ref: np.ndarray,
                    tgt: np.ndarray,
                    fs: float,
                    fmin: float = 50.0,
                    fmax_frac: float = 0.4,
                    g2_thresh: float = 0.6):
    """
    γ²-weighted coherent subtraction of the part of `tgt` that is linearly predictable from `ref`.
    Returns residual, predicted, and diagnostics.
    """
    ref0 = ref - np.mean(ref)
    tgt0 = tgt - np.mean(tgt)

    f_xy, H_xy, g2_xy = estimate_frf(ref0, tgt0, fs)
    H_eff = H_xy * np.clip(g2_xy, 0.0, 1.0)

    pred = apply_frf(ref0, fs, f_xy, H_eff)
    resid = tgt0 - pred

    # Diagnostics: band-limited energy ratio
    f_tgt, Pyy_tgt = compute_spec(fs, tgt0)
    f_res, Pyy_res = compute_spec(fs, resid)
    Pyy_tgt_i = np.interp(f_xy, f_tgt, Pyy_tgt, left=0.0, right=0.0)
    Pyy_res_i = np.interp(f_xy, f_res, Pyy_res, left=0.0, right=0.0)

    band = coherent_band_mask(f_xy, g2_xy, fs, fmin=fmin, fmax_frac=fmax_frac, g2_thresh=g2_thresh)
    E_tgt = band_energy_from_psd(f_xy, Pyy_tgt_i, band)
    E_res = band_energy_from_psd(f_xy, Pyy_res_i, band)
    ratio = (E_res / E_tgt) if E_tgt > 0 else np.nan
    ic({'FS_cancel_band_Eratio_res_over_tgt': ratio})

    return resid, pred, (f_xy, H_eff, g2_xy), (f_res, Pyy_res), (f_tgt, Pyy_tgt), ratio

def plot_white(ax):
    x = np.logspace(1, 4, 200)   # 10^1 to 10^4
    y = 1e-4 * (x / 1e1)         # slope +1 line, scaled to pass through (1e1, 1e-8)
    ax.loglog(x, y, '--', color='gray', label='White noise (slope +1)')


root = 'data/30092025'
fn = f'{root}/50psi/nkd-ph_nofacilitynoise.mat'
OUTPUT_DIR = "figures/sanity/50psi/PH-NKD"
CAL_DIR = os.path.join(CALIB_BASE_DIR, "PH-NKD")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CAL_DIR, exist_ok=True)

# Load data (to Pa): col1 = PH, col2 = NKD
nkd_nn, ph_nn = load_mat_to_pa(fn, 'channelData_nofacilitynoise', 'PH_300', 'NC_300')
# fig, ax = plt.subplots(1, 1, figsize=(5, 2.), sharex=True)
# ax.plot(np.arange(len(nkd)) / FS, nkd, label='NKD')
# ax.plot(np.arange(len(ph)) / FS, ph, label='PH')
# ax.set_xlabel("Time [s]")
# ax.set_ylabel("Voltage [V]")
# ax.legend()
# fig.tight_layout()
# fig.savefig(f"{OUTPUT_DIR}/calib_ts_signals_50psi_nonoise.pdf", dpi=400)
# Plot the spectra to make sure it's white
f, Pyy_nkd = compute_spec(FS, nkd_nn)
f, Pyy_ph = compute_spec(FS, ph_nn)
fig, ax = plt.subplots(1, 1, figsize=(5, 2.), sharex=True)
ax.loglog(f, f * Pyy_nkd, label='NKD')
ax.loglog(f, f * Pyy_ph, label='PH')
plot_white(ax)
ax.set_xlabel("$f$ [Hz]")
ax.set_ylabel(r"$f \phi_{pp}$")
ax.legend()
fig.savefig(f"{OUTPUT_DIR}/calib_spectra_50psi_nonoise.pdf", dpi=400)


# Trim initial transients
if TRIM_CAL_SECS > 0:
    start = int(TRIM_CAL_SECS * FS)
    ph_nn = ph_nn[start:]
    nkd_nn = nkd_nn[start:]

# FRF PH→NKD (x=PH, y=NKD)
f, H_nn, gamma2_nn = estimate_frf(ph_nn, nkd_nn, FS, npsg=2**9)
np.save(f"{CAL_DIR}/H_nn.npy", H_nn)
np.save(f"{CAL_DIR}/gamma2_nn.npy", gamma2_nn)
np.save(f"{CAL_DIR}/f_nn.npy", f)

plot_transfer_PH(f, H_nn, f"{OUTPUT_DIR}/H_50psi", "50psi")

# Sanity: reconstruct PH from NKD using the inverse (should resemble PH)
ph_hat_nn = wiener_forward(ph_nn, FS, f, H_nn, gamma2_nn)
t = np.arange(len(ph_hat_nn)) / FS
plot_corrected_trace_PH(t, nkd_nn, ph_nn, ph_hat_nn, f"{OUTPUT_DIR}/ph_recon_50psi_nn", "50psi")

# plot spectra of nkd, ph, and ph_hat
f, Pyy_nkd_nn = compute_spec(FS, nkd_nn)
f, Pyy_ph_nn = compute_spec(FS, ph_nn)
f, Pyy_ph_hat_nn = compute_spec(FS, ph_hat_nn)
fig, ax = plt.subplots(1, 1, figsize=(5, 2.), sharex=True)
ax.loglog(f, f * Pyy_nkd_nn, label='NKD')
ax.loglog(f, f * Pyy_ph_nn, label='PH')
ax.loglog(f, f * Pyy_ph_hat_nn, label='PH hat')
plot_white(ax)
ax.set_xlabel("$f$ [Hz]")
ax.set_ylabel(r"$f \phi_{pp}$")
ax.legend()
fig.savefig(f"{OUTPUT_DIR}/calib_spectra_50psi_nn_recon.pdf", dpi=400)

# Add high-pass filter at 0.1Hz and LP at 2kHz
sos = signal.butter(4, 0.1, btype='highpass', fs=FS, output='sos')
sos_lp = signal.butter(4, 2000.0, btype='lowpass', fs=FS, output='sos')
nkd_nn_filt = signal.sosfilt(sos, nkd_nn)
nkd_nn_filt = signal.sosfilt(sos_lp, nkd_nn_filt)
ph_nn_filt = signal.sosfilt(sos, ph_nn)
ph_nn_filt = signal.sosfilt(sos_lp, ph_nn_filt)
f, Pyy_nkd = compute_spec(FS, nkd_nn_filt)
f, Pyy_ph = compute_spec(FS, ph_nn_filt)
fig, ax = plt.subplots(1, 1, figsize=(5, 2.), sharex=True)
ax.loglog(f, f * Pyy_nkd, label='NKD')
ax.loglog(f, f * Pyy_ph, label='PH')
ax.set_xlabel("$f$ [Hz]")
ax.set_ylabel(r"$f \phi_{pp}$")
ax.legend()
fig.savefig(f"{OUTPUT_DIR}/calib_spectra_50psi_nn_filt.pdf", dpi=400)


root = 'data/30092025'
fn = f'{root}/50psi/nkd-ph_facilitynoise.mat'
OUTPUT_DIR = "figures/sanity/50psi/PH-NKD"
CAL_DIR = os.path.join(CALIB_BASE_DIR, "PH-NKD")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CAL_DIR, exist_ok=True)

# Load data (to Pa): col1 = PH, col2 = NKD
nkd_fn, ph_fn = load_mat_to_pa(fn, 'channelData_BKD', 'PH_300', 'NC_300')
# fig, ax = plt.subplots(1, 1, figsize=(5, 2.), sharex=True)
# ax.plot(np.arange(len(nkd)) / FS, nkd, label='NKD')
# ax.plot(np.arange(len(ph)) / FS, ph, label='PH')
# ax.set_xlabel("Time [s]")
# ax.set_ylabel("Voltage [V]")
# ax.legend()
# fig.tight_layout()
# fig.savefig(f"{OUTPUT_DIR}/calib_ts_signals_50psi_noise.pdf", dpi=400)
# Plot the spectra to make sure it's white

f, Pyy_nkd_fn = compute_spec(FS, nkd_fn)
f, Pyy_ph_fn = compute_spec(FS, ph_fn)
fig, ax = plt.subplots(1, 1, figsize=(5, 2.), sharex=True)
ax.loglog(f, f * Pyy_nkd_fn, label='NKD')
ax.loglog(f, f * Pyy_ph_fn, label='PH')
plot_white(ax)
ax.set_xlabel("$f$ [Hz]")
ax.set_ylabel(r"$f \phi_{pp}$")
ax.legend()
fig.savefig(f"{OUTPUT_DIR}/calib_spectra_50psi_noise.pdf", dpi=400)

# Trim initial transients
if TRIM_CAL_SECS > 0:
    start = int(TRIM_CAL_SECS * FS)
    ph_fn = ph_fn[start:]
    nkd_fn = nkd_fn[start:]

# FRF PH→NKD (x=PH, y=NKD)
f, H_fn, gamma2_fn = estimate_frf(ph_fn, nkd_fn, FS, npsg=2**9)
np.save(f"{CAL_DIR}/H_fn.npy", H_fn)
np.save(f"{CAL_DIR}/gamma2_fn.npy", gamma2_fn)
np.save(f"{CAL_DIR}/f_fn.npy", f)

plot_transfer_PH(f, H_fn, f"{OUTPUT_DIR}/H_50psi", "50psi")

# Sanity: reconstruct PH from NKD using the inverse (should resemble PH)
ph_hat_fn = wiener_forward(ph_fn, FS, f, H_fn, gamma2_fn)
t = np.arange(len(ph_hat_fn)) / FS
plot_corrected_trace_PH(t, nkd_fn, ph_fn, ph_hat_fn, f"{OUTPUT_DIR}/ph_recon_50psi_fn", "50psi")

# plot spectra of nkd, ph, and ph_hat
f, Pyy_nkd_fn = compute_spec(FS, nkd_fn)
f, Pyy_ph_fn = compute_spec(FS, ph_fn)
f, Pyy_ph_hat_fn = compute_spec(FS, ph_hat_fn)
fig, ax = plt.subplots(1, 1, figsize=(5, 2.), sharex=True)
ax.loglog(f, f * Pyy_nkd_fn, label='NKD')
ax.loglog(f, f * Pyy_ph_fn, label='PH')
ax.loglog(f, f * Pyy_ph_hat_fn, label='PH hat')
plot_white(ax)
ax.set_xlabel("$f$ [Hz]")
ax.set_ylabel(r"$f \phi_{pp}$")
ax.legend()
fig.savefig(f"{OUTPUT_DIR}/calib_spectra_50psi_fn_recon.pdf", dpi=400)

# Add high-pass filter at 0.1Hz and LP at 2kHz
sos = signal.butter(4, 0.1, btype='highpass', fs=FS, output='sos')
sos_lp = signal.butter(4, 2000.0, btype='lowpass', fs=FS, output='sos')
nkd_fn_filt = signal.sosfilt(sos, nkd_fn)
nkd_fn_filt = signal.sosfilt(sos_lp, nkd_fn_filt)
ph_fn_filt = signal.sosfilt(sos, ph_fn)
ph_fn_filt = signal.sosfilt(sos_lp, ph_fn_filt)
f, Pyy_nkd = compute_spec(FS, nkd_fn_filt)
f, Pyy_ph = compute_spec(FS, ph_fn_filt)
fig, ax = plt.subplots(1, 1, figsize=(5, 2.), sharex=True)
ax.loglog(f, f * Pyy_nkd, label='NKD')
ax.loglog(f, f * Pyy_ph, label='PH')
ax.set_xlabel("$f$ [Hz]")
ax.set_ylabel(r"$f \phi_{pp}$")
ax.legend()
fig.savefig(f"{OUTPUT_DIR}/calib_spectra_50psi_fn_filt.pdf", dpi=400)

root = 'data/30092025'
fn = f'{root}/50psi/nkd-ph_facilitynoise.mat'
OUTPUT_DIR = "figures/sanity/50psi/PH-NKD"
CAL_DIR = os.path.join(CALIB_BASE_DIR, "PH-NKD")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CAL_DIR, exist_ok=True)

# Load data (to Pa): col1 = PH, col2 = NKD
nkd_an, ph_an = load_mat_to_pa(fn, 'channelData_BKD_WN', 'PH_300', 'NC_300')
# fig, ax = plt.subplots(1, 1, figsize=(5, 2.), sharex=True)
# ax.plot(np.arange(len(nkd)) / FS, nkd, label='NKD')
# ax.plot(np.arange(len(ph)) / FS, ph, label='PH')
# ax.set_xlabel("Time [s]")
# ax.set_ylabel("Voltage [V]")
# ax.legend()
# fig.tight_layout()
# fig.savefig(f"{OUTPUT_DIR}/calib_ts_signals_50psi_noiseWN.pdf", dpi=400)
# Plot the spectra to make sure it's white
f, Pyy_nkd_an = compute_spec(FS, nkd_an)
f, Pyy_ph_an = compute_spec(FS, ph_an)
fig, ax = plt.subplots(1, 1, figsize=(5, 2.), sharex=True)
ax.loglog(f, f * Pyy_nkd_an, label='NKD')
ax.loglog(f, f * Pyy_ph_an, label='PH')
plot_white(ax)
ax.set_xlabel("$f$ [Hz]")
ax.set_ylabel(r"$f \phi_{pp}$")
ax.legend()
fig.savefig(f"{OUTPUT_DIR}/calib_spectra_50psi_noiseWN.pdf", dpi=400)

# Add high-pass filter at 0.1Hz
sos = signal.butter(4, 0.1, btype='highpass', fs=FS, output='sos')
nkd = signal.sosfilt(sos, nkd_an)
ph = signal.sosfilt(sos, ph_an)
f, Pyy_nkd = compute_spec(FS, nkd)
f, Pyy_ph = compute_spec(FS, ph)
fig, ax = plt.subplots(1, 1, figsize=(5, 2.), sharex=True)
ax.loglog(f, f * Pyy_nkd, label='NKD (HPF 0.1Hz)')
ax.loglog(f, f * Pyy_ph, label='PH (HPF 0.1Hz)')
ax.set_xlabel("$f$ [Hz]")
ax.set_ylabel(r"$f \phi_{pp}$")
ax.legend()
fig.savefig(f"{OUTPUT_DIR}/calib_spectra_50psi_hpf_noiseWN.pdf", dpi=400)

# Trim initial transients
if TRIM_CAL_SECS > 0:
    start = int(TRIM_CAL_SECS * FS)
    ph_an = ph_an[start:]
    nkd_an = nkd_an[start:]

# FRF PH→NKD (x=PH, y=NKD)
f, H_an, gamma2_an = estimate_frf(ph_an, nkd_an, FS, npsg=2**9)
np.save(f"{CAL_DIR}/H_an.npy", H_an)
np.save(f"{CAL_DIR}/gamma2_an.npy", gamma2_an)
np.save(f"{CAL_DIR}/f_an.npy", f)

plot_transfer_PH(f, H_an, f"{OUTPUT_DIR}/H_50psi_an", "50psi")

# Sanity: reconstruct PH from NKD using the inverse (should resemble PH)
ph_hat_an = wiener_forward(ph_an, FS, f, H_an, gamma2_an)
t = np.arange(len(ph_hat_an)) / FS
plot_corrected_trace_PH(t, nkd_an, ph_an, ph_hat_an, f"{OUTPUT_DIR}/ph_recon_50psi_an", "50psi")

# plot spectra of nkd, ph, and ph_hat
f, Pyy_nkd_an = compute_spec(FS, nkd_an)
f, Pyy_ph_an = compute_spec(FS, ph_an)
f, Pyy_ph_hat_an = compute_spec(FS, ph_hat_an)
fig, ax = plt.subplots(1, 1, figsize=(5, 2.), sharex=True)
ax.loglog(f, f * Pyy_nkd_an, label='NKD')
ax.loglog(f, f * Pyy_ph_an, label='PH')
ax.loglog(f, f * Pyy_ph_hat_an, label='PH hat')
plot_white(ax)
ax.set_xlabel("$f$ [Hz]")
ax.set_ylabel(r"$f \phi_{pp}$")
ax.legend()
fig.savefig(f"{OUTPUT_DIR}/calib_spectra_50psi_an_recon.pdf", dpi=400)

# Add high-pass filter at 0.1Hz and LP at 2kHz
sos = signal.butter(4, 0.1, btype='highpass', fs=FS, output='sos')
sos_lp = signal.butter(4, 2000.0, btype='lowpass', fs=FS, output='sos')
nkd_an_filt = signal.sosfilt(sos, nkd_an)
nkd_an_filt = signal.sosfilt(sos_lp, nkd_an_filt)
ph_an_filt = signal.sosfilt(sos, ph_an)
ph_an_filt = signal.sosfilt(sos_lp, ph_an_filt)
f, Pyy_nkd = compute_spec(FS, nkd_an_filt)
f, Pyy_ph = compute_spec(FS, ph_an_filt)
fig, ax = plt.subplots(1, 1, figsize=(5, 2.), sharex=True)
ax.loglog(f, f * Pyy_nkd, label='NKD')
ax.loglog(f, f * Pyy_ph, label='PH')
ax.set_xlabel("$f$ [Hz]")
ax.set_ylabel(r"$f \phi_{pp}$")
ax.legend()
fig.savefig(f"{OUTPUT_DIR}/calib_spectra_50psi_an_filt.pdf", dpi=400)

# Trim initial transients
if TRIM_CAL_SECS > 0:
    start = int(TRIM_CAL_SECS * FS)
    ph_an_filt = ph_an_filt[start:]
    nkd_an_filt = nkd_an_filt[start:]

# FRF PH→NKD (x=PH, y=NKD)
f, H_an_filt, gamma2_an_filt = estimate_frf(ph_an_filt, nkd_an_filt, FS, npsg=2**9)
np.save(f"{CAL_DIR}/H_an_filt.npy", H_an_filt)
np.save(f"{CAL_DIR}/gamma2_an_filt.npy", gamma2_an_filt)
np.save(f"{CAL_DIR}/f_an_filt.npy", f)

plot_transfer_PH(f, H_an_filt, f"{OUTPUT_DIR}/H_50psi_an_filt", "50psi")

# Sanity: reconstruct PH from NKD using the inverse (should resemble PH)
ph_hat_an_filt = wiener_forward(ph_an_filt, FS, f, H_an_filt, gamma2_an_filt)
t = np.arange(len(ph_hat_an_filt)) / FS
plot_corrected_trace_PH(t, nkd_an_filt, ph_an_filt, ph_hat_an_filt, f"{OUTPUT_DIR}/ph_recon_50psi_an_filt", "50psi")



# Does Pyy_nkd_an=Pyy_nkd_fn + Pyy_nkd_nn?
fig, ax = plt.subplots(1, 1, figsize=(5, 2.), sharex=True)
ax.loglog(f, Pyy_nkd_an, label='NKD (all)')
ax.loglog(f, Pyy_nkd_fn + Pyy_nkd_nn, label='NKD (fn+nn)', linestyle='--')
ax.set_xlabel("$f$ [Hz]")
ax.set_ylabel(r"$\phi_{pp}$ [Pa$^2$/Hz]")
ax.legend()
fig.savefig(f"{OUTPUT_DIR}/calib_spectra_50psi_nkd_fn_plus_nn.pdf", dpi=400)

# Does the transfer