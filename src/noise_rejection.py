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