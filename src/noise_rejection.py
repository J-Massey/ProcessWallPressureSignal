import numpy as np
import os
from scipy.signal import welch, csd, detrend, correlate, coherence, savgol_filter
from scipy.interpolate import interp1d
from icecream import ic


def align_signals(p_free, p_wall, fs, max_lag_s=None):
    """
    Align p_free to p_wall by cross-correlation and clip to common length.

    Inputs:
      p_free    : (N,) free-stream trace
      p_wall    : (M,) wall-pressure trace
      fs        : sampling rate [Hz]
      max_lag_s : optional max lag to search [s]

    Returns:
      p_free_al, p_wall_al, tau  : aligned arrays and time shift [s]
    """
    # cross-correlation
    R = correlate(p_wall, p_free, mode='full')
    lags = np.arange(-len(p_free)+1, len(p_wall))
    if max_lag_s is not None:
        max_lag = int(max_lag_s * fs)
        mask = (lags >= -max_lag) & (lags <= max_lag)
        lag = lags[mask][np.argmax(R[mask])]
    else:
        lag = lags[np.argmax(R)]
    tau = lag / fs

    # shift free-stream
    p_free_shifted = np.roll(p_free, -lag)

    # clip to overlap
    if lag > 0:
        start_f, start_w = lag, 0
    else:
        start_f, start_w = 0, -lag
    n = min(len(p_free_shifted)-start_f, len(p_wall)-start_w)

    p_free_al = p_free_shifted[start_f:start_f+n]
    p_wall_al = p_wall   [start_w:start_w+n]
    return p_free_al, p_wall_al, tau

def phase_match(sig1, sig2):
    """
    Phase-match two signals in the frequency domain.

    This function finds a frequency-dependent phase correction between sig1 (e.g., freestream pressure)
    and sig2 (e.g., wall pressure), and applies it to sig1 to align its phase with sig2's phase:contentReference[oaicite:0]{index=0}.
    It uses the cross power spectrum to estimate the phase difference at each frequency,
    which represents the phase lag between the two signals:contentReference[oaicite:1]{index=1}.

    Parameters:
    sig1 : array_like
        The first time-series signal (e.g., freestream pressure).
    sig2 : array_like
        The second time-series signal (reference signal, e.g., wall pressure).

    Returns:
    sig1_phase_matched : ndarray
        The phase-corrected version of sig1, aligned in phase with sig2.

    Notes:
    - The signals are assumed to be time-aligned (no major time lag).
    - The method computes the cross-spectrum and adjusts the phase of sig1 to match sig2
      at each frequency. This preserves sig1's magnitude spectrum and only alters its phase.
    - The output is a real signal, obtained via inverse FFT after phase correction.
    - This approach is efficient (uses FFT) and numerically stable for large signals, preserving the coherent phase relationships required for Wiener filtering:contentReference[oaicite:2]{index=2}.
    """
    sig1 = np.asarray(sig1)
    sig2 = np.asarray(sig2)
    N = len(sig1) if len(sig1) <= len(sig2) else len(sig2)  # ensure same length
    sig1 = sig1[:N]
    sig2 = sig2[:N]
    # Compute one-sided FFT (rfft) for efficiency with real signals
    F1 = np.fft.rfft(sig1)
    F2 = np.fft.rfft(sig2)
    # Compute phase difference for each frequency bin
    phase_diff = np.angle(F2) - np.angle(F1)
    # Unwrap the phase difference to avoid 2π discontinuities
    phase_diff = np.unwrap(phase_diff)
    # Construct complex phase correction (unit magnitude factors)
    phase_correction = np.exp(1j * phase_diff)
    # Apply phase correction to F1's spectrum
    F1_matched = F1 * phase_correction
    # Inverse FFT to get the phase-adjusted time-domain signal
    sig1_phase_matched = np.fft.irfft(F1_matched, n=N)
    return sig1_phase_matched

def phase_align(sig, ref, fs, max_lag_s=None):
    """
    Find a single time‐delay τ that phase-matches sig to ref,
    then apply that delay (fractionally) via a linear-phase filter.

    Parameters
    ----------
    sig : array_like, shape (N,)
        Signal to be shifted (e.g. free-stream pressure).
    ref : array_like, shape (M,)
        Reference signal (e.g. wall-pressure), already time-aligned.
    fs : float
        Sampling frequency [Hz].
    max_lag_s : float, optional
        Maximum lag (±max_lag_s) to search [s].  Default: full length.

    Returns
    -------
    sig_matched : ndarray, shape (L,)
        Phase-matched version of sig, clipped to overlap with ref.
    tau : float
        Estimated time delay (sig must be shifted by +τ to match ref).
    """
    # 1) cross-correlate to find best integer lag
    sig = np.asarray(sig)
    ref = np.asarray(ref)
    N, M = len(sig), len(ref)
    R = correlate(ref, sig, mode='full')
    lags = np.arange(-N+1, M)
    if max_lag_s is not None:
        max_lag = int(max_lag_s * fs)
        mask = (lags >= -max_lag) & (lags <= max_lag)
        lag = lags[mask][np.argmax(R[mask])]
    else:
        lag = lags[np.argmax(R)]
    tau = lag / fs

    # 2) fractional-delay via linear phase in freq-domain
    L = max(N, M)
    sig_pad = np.zeros(L, dtype=sig.dtype)
    sig_pad[:N] = sig
    F = np.fft.rfft(sig_pad)
    freqs = np.fft.rfftfreq(L, 1/fs)
    # linear phase shift e^(−j2πf τ)
    shift = np.exp(-1j * 2*np.pi * freqs * tau)
    F_shifted = F * shift
    sig_shifted = np.fft.irfft(F_shifted, n=L)

    # 3) clip to common overlap
    if lag >= 0:
        start_sig, start_ref = lag, 0
    else:
        start_sig, start_ref = 0, -lag
    n = min(L - start_sig, M - start_ref)
    sig_matched = sig_shifted[start_sig:start_sig+n]

    return sig_matched, tau

def match_via_transfer(p_free, p_wall, fs):
    """
    Use the full complex transfer function H(f)=S_wf/S_ff to align and scale
    p_free so it best matches p_wall in both amplitude and phase.

    Inputs
    ------
    p_free   : (N,) free‐stream trace
    p_wall   : (N,) wall‐pressure trace (time‐aligned)
    fs       : sampling rate [Hz]
    nperseg  : length of Welch segments
    noverlap : overlap between segments (defaults to nperseg//2)

    Returns
    -------
    p_free_matched : (N,) free‐stream after gain+phase correction
    H              : (M,) complex transfer function on Welch grid
    f              : (M,) frequency vector [Hz]
    """
    nperseg = len(p_wall) // 2000  # segment length for Welch's method
    noverlap = nperseg // 2  # overlap between segments

    # 1) estimate spectra & cross‐spectrum
    f,   S_ff = welch(p_free, fs, nperseg=nperseg, noverlap=noverlap)
    _,   S_ww = welch(p_wall, fs, nperseg=nperseg, noverlap=noverlap)
    _,   S_wf = csd(p_wall, p_free, fs, nperseg=nperseg, noverlap=noverlap)

    # 2) complex transfer function H(f)
    eps = 1e-16
    H = S_wf / (S_ff + eps)

    # 3) apply H in freq‐domain to p_free
    N = len(p_free)
    P_free = np.fft.rfft(p_free, n=N)
    # interpolate H onto full FFT grid if needed
    f_full = np.fft.rfftfreq(N, 1/fs)
    H_full = np.interp(f_full, f, H.real) + 1j*np.interp(f_full, f, H.imag)
    P_matched = P_free * H_full
    p_free_matched = np.fft.irfft(P_matched, n=N)

    return p_free_matched, H, f

def restore_coherence(free_stream_signal, wall_signal, fs):
    """
    Returns a filtered version of wall_signal with coherence-restored relative to free_stream_signal.
    """
    nperseg = len(wall_signal) // 4000  # segment length for Welch's method
    noverlap = nperseg // 2  # overlap between segments
    # 1. Estimate power spectral densities (PSDs) and cross-PSD using Welch’s method
    f, Pxx = welch(free_stream_signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, Pyy = welch(wall_signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, Pxy = csd(free_stream_signal, wall_signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # 2. Compute Wiener filter transfer function H(f) = Pxy / Pxx
    H = Pxy / (Pxx + 1e-12)   # small epsilon to avoid division by zero
    H = np.interp(np.fft.rfftfreq(len(free_stream_signal), 1/fs), f, H, left=1, right=1)  # interpolate to full FFT grid
    
    # 3. Apply H(f) to free_stream_signal via FFT to get the coherent part of wall_signal
    X = np.fft.rfft(free_stream_signal)         # FFT of input signal
    Y_coherent = np.fft.irfft(X * H, n=len(free_stream_signal))  # filter in freq domain
    
    return Y_coherent

def compute_coherence(p_free, p_wall, fs):
    """
    Compute coherence between free-stream and wall-pressure signals.

    Inputs:
      p_free    : (N,) free-stream trace
      p_wall    : (M,) wall-pressure trace
      fs        : sampling rate [Hz]
      max_lag_s : optional max lag to search [s]

    Returns:
      f         : frequency vector [Hz]
      coh       : magnitude-squared coherence
      Pyy       : original wall-pressure PSD
      Pyy_coh   : portion of Pyy coherent with p_free
    """
    nperseg = len(p_wall) // 2000  # segment length for Welch's method
    noverlap = nperseg // 2  # overlap between segments
    f, coh2 = coherence(p_wall, p_free, fs, nperseg=nperseg, noverlap=noverlap, window='hann')
    return f, coh2


def reject_freestream_noise(p_free, p_wall, fs):
    """
    Remove free-stream pressure noise from wall-pressure via spectral subtraction.

    Inputs:
      p_free  : 1D array of free-stream pressure fluctuations
      p_wall  : 1D array of wall-pressure fluctuations (duct modes already filtered)
      fs      : sampling frequency [Hz]
      nperseg : segment length for Welch's method
      noverlap: overlap between segments (defaults to nperseg//2 if None)

    Outputs:
      f        : frequency vector [Hz]
      Phi_orig : original wall-pressure PSD
      Phi_clean: wall-pressure PSD with free-stream noise rejected
    """
    nperseg = len(p_wall) // 2000  # segment length for Welch's method
    noverlap = nperseg // 2  # overlap between segments
    # 1) estimate spectra
    f, S_ww = welch(p_wall, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, S_ff = welch(p_free, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, S_wf = csd(p_wall, p_free, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # 2) optimal transfer function H(f) = S_wf / S_ff
    H_opt = S_wf / S_ff

    # 3) reconstruct noise in time domain
    #    compute FFT of free-stream, apply H, then IFFT
    P_free = np.fft.rfft(p_free, n=2*(len(H_opt)-1))
    # align lengths: len(P_free)==len(H_opt)
    noise_hat = np.fft.irfft(H_opt * P_free, n=len(p_free))

    # 4) subtract estimated noise
    p_clean = p_wall - noise_hat

    # 5) PSD of cleaned signal
    _, Phi_clean = welch(p_clean, fs=fs, nperseg=nperseg, noverlap=noverlap)

    return f, S_ww, Phi_clean

def reject_noise_by_coherence(p_free, p_wall, fs, thresh=3e-4):
    """
    Remove wall-PSD where free-stream coherence > thresh.

    Inputs:
      p_free   : (N,) free-stream trace
      p_wall   : (N,) wall-pressure trace (aligned & notched)
      fs       : sampling rate [Hz]
      thresh   : coherence^2 threshold
      nperseg  : Welch segment length
      noverlap : overlap (defaults to nperseg//2)

    Returns:
      f        : (M,) frequency vector
      Phi_orig : (M,) original wall-PSD
      Phi_clean: (M,) cleaned wall-PSD
    """
    nperseg = len(p_wall) // 2000  # segment length for Welch's method
    noverlap = nperseg // 2  # overlap between segments

    f, Phi_orig = welch(p_wall, fs=fs,
                        nperseg=nperseg, noverlap=noverlap)
    _, coh2       = coherence(p_wall, p_free, fs=fs,
                            nperseg=nperseg, noverlap=noverlap, window='hann')

    Phi_clean = Phi_orig.copy()
    Phi_clean[coh2.real > thresh] = np.nan
    return f, Phi_orig, Phi_clean



def fill_and_smooth_psd(f, Pclean,
                        interp_kind='linear',
                        smooth_window=301,
                        smooth_poly=2):
    """
    Fill NaNs in Pclean by log-log interpolation on f, then smooth.

    Inputs:
      f             : (M,) frequency vector
      Pclean        : (M,) PSD with NaNs where coherence>thresh
      interp_kind   : 'linear', 'cubic', etc.
      smooth_window : odd integer window length for Savitzky-Golay
      smooth_poly   : polynomial order for Savitzky-Golay

    Returns:
      Pfilled_sm : (M,) continuous, smoothed PSD
    """
    # mask valid points
    mask = ~np.isnan(Pclean)
    # log-log interp
    logf = np.log(f[mask])
    logP = np.log(Pclean[mask])
    interp = interp1d(logf, logP, kind=interp_kind,
                      fill_value='extrapolate')
    Pfilled = np.exp(interp(np.log(f)))
    # Savitzky–Golay smoothing
    if smooth_window >= 3 and smooth_window % 2 == 1:
        Pfilled_sm = savgol_filter(Pfilled,
                                   window_length=smooth_window,
                                   polyorder=smooth_poly,
                                   mode='interp')
    else:
        Pfilled_sm = Pfilled
    return Pfilled_sm


def wiener_psd_clean_thresh(p_free, p_wall, fs, thresh=1e-3):
    """
    Wiener‐filter PSD cleaning with coherence threshold.

    Inputs:
      p_free   : (N,) free-stream trace
      p_wall   : (N,) wall-pressure trace
      fs       : sampling rate [Hz]
      thresh   : coherence^2 threshold
      nperseg  : Welch segment length
      noverlap : overlap (defaults to nperseg//2)

    Returns:
      f, Sww_orig, Sww_clean
    """
    nperseg = len(p_wall) // 2000  # segment length for Welch's method
    noverlap = nperseg // 2  # overlap between segments

    f, Sww = welch(p_wall, fs, nperseg=nperseg, noverlap=noverlap)
    _, Sff = welch(p_free, fs, nperseg=nperseg, noverlap=noverlap)
    _, Swf = csd(p_wall, p_free, fs, nperseg=nperseg, noverlap=noverlap)
    _, coh2  = coherence(p_wall, p_free, fs, nperseg=nperseg, noverlap=noverlap)

    # thresholded transfer function
    eps = 1e-16
    H = Swf/(Sff+eps)
    H[coh2 < thresh] = 0

    # clean PSD
    Sww_clean = Sww - np.abs(H)**2 * Sff
    Sww_clean[Sww_clean < 0] = 0

    return f, Sww, Sww_clean