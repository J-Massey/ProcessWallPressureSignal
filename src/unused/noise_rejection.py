import numpy as np
import os
from scipy.signal import welch, csd, detrend, correlate, coherence, savgol_filter
from scipy.interpolate import interp1d
from icecream import ic
import torch
from tqdm import tqdm


def optimize_delay_by_csd_torch(p_wall, p_fs, fs,
                                max_lag_s=0.1, min_lag_s=0.001,
                                device=None):
    """
    Find the time‐delay τ that maximises ∫|S_wp(f;τ)|² df using PyTorch on CUDA.

    Parameters
    ----------
    p_free    : array_like or Tensor, shape (N,)
        Free‐stream signal.
    p_wall    : array_like or Tensor, shape (M,)
        Wall‐pressure signal (time‐aligned).
    fs        : float
        Sampling rate [Hz].
    max_lag_s : float
        Max lag to search ±[s] (default 0.01 s).
    nperseg   : int
        FFT segment length for Welch‐style CSD (default 1024).
    noverlap  : int or None
        Overlap between segments (default nperseg//2).
    device    : str or torch.device or None
        Compute device (e.g. 'cuda'). Defaults to CUDA if available.

    Returns
    -------
    p_free_opt : Tensor, shape (N,)
        p_free phase‐shifted by optimal lag.
    tau_opt    : float
        Optimal delay [s].
    score_opt  : float
        Maximum ∫|CSD|² df achieved.
    """
    nperseg = len(p_fs) // 2000  # segment length for Welch's method
    noverlap = nperseg // 2  # overlap between segments
    # --- setup ---
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.get_default_dtype()
    # convert inputs to tensors
    def to_tensor(x):
        if not torch.is_tensor(x):
            x = np.ascontiguousarray(x)
            x = torch.from_numpy(x)
        return x.to(device=device, dtype=dtype)
    x = to_tensor(p_wall)
    y = to_tensor(p_fs)

    hop    = nperseg - noverlap
    window = torch.hann_window(nperseg, device=device, dtype=dtype)
    U      = window.pow(2).sum()
    # frequency vector for integration
    f = torch.fft.rfftfreq(nperseg, 1/fs, device=device)

    # helper: compute CSD via STFT
    def csd_torch(a, b):
        A = torch.stft(a, n_fft=nperseg, hop_length=hop,
                       win_length=nperseg, window=window,
                       return_complex=True)
        B = torch.stft(b, n_fft=nperseg, hop_length=hop,
                       win_length=nperseg, window=window,
                       return_complex=True)
        S_ab = (A * B.conj() / (fs * U)).mean(dim=-1)
        return S_ab

    # search over integer lags
    max_lag = int(max_lag_s * fs)
    min_lag = int(min_lag_s * fs)
    best_score = float('-inf')
    best_lag   = 0

    for lag in tqdm(range(min_lag, max_lag+1)):
        # roll and clip to common length
        y_shift = torch.roll(x, -lag)
        L = min(y_shift.shape[0], y.shape[0])
        S_xy = csd_torch(y[:L], y_shift[:L])
        score = torch.trapz(torch.abs(S_xy)**2, f).item()
        if score > best_score:
            best_score = score
            best_lag   = lag

    # compute final outputs
    tau_opt    = best_lag / fs
    p_wall_opt = torch.roll(x, -best_lag)
    score_opt  = best_score

    return p_wall_opt.cpu(), tau_opt, score_opt

def phase_match(sig1, sig2, smoothing_len=10):
    """
    Phase-match two signals in the frequency domain.

    This function finds a frequency-dependent phase correction between sig1 (e.g., freestream pressure)
    and sig2 (e.g., wall pressure), and applies it to sig1 to align its phase with sig2.
    It uses the cross power spectrum to estimate the phase difference at each frequency,
    which represents the phase lag between the two signals.

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
    - This approach is efficient (uses FFT) and numerically stable for large signals,
      preserving the coherent phase relationships required for Wiener filtering.
    """
    sig1 = np.asarray(sig1)
    sig2 = np.asarray(sig2)
    N = len(sig1)
    # Compute one-sided FFT (rfft) for efficiency with real signals
    F1 = np.fft.rfft(sig1)
    F2 = np.fft.rfft(sig2)
    # Compute phase difference for each frequency bin
    phase_diff = np.angle(F2) - np.angle(F1)
    # Unwrap the phase difference to avoid 2π discontinuities
    phase_diff = np.unwrap(phase_diff)
    # smooth the unwrapped phase
    if smoothing_len > 1:
        kernel = np.ones(smoothing_len) / smoothing_len
        phase_diff = np.convolve(phase_diff, kernel, mode='same')
    # Construct complex phase correction (unit magnitude factors)
    phase_correction = np.exp(1j * phase_diff)
    # Apply phase correction to F1's spectrum
    F1_matched = F1 * phase_correction
    # Inverse FFT to get the phase-adjusted time-domain signal
    sig1_phase_matched = np.fft.irfft(F1_matched, n=N)
    return sig1_phase_matched


def phase_match_transfer(sig1, sig2, fs, smoothing_len=1):
    """
    Return the complex transfer function H(f) implemented by phase_match:
      H(f) = exp[i·(angle(F2) - angle(F1))_smoothed]

    Parameters
    ----------
    sig1 : array_like, shape (N,)
        Input signal (to be phase-matched).
    sig2 : array_like, shape (N,)
        Reference signal.
    fs : float
        Sampling rate [Hz].
    smoothing_len : int
        Moving-average window length for unwrapped phase.

    Returns
    -------
    f : ndarray, shape (N//2+1,)
        Frequency bins [Hz].
    H : ndarray, shape (N//2+1,)
        Complex transfer function.
    """
    sig1 = np.asarray(sig1)
    sig2 = np.asarray(sig2)
    N = min(sig1.size, sig2.size)
    # one-sided FFT
    F1 = np.fft.rfft(sig1[:N])
    F2 = np.fft.rfft(sig2[:N])

    # Δφ(f) = arg F2 – arg F1
    phase_diff = np.unwrap(np.angle(F2) - np.angle(F1))
    mag_diff = np.abs(F2) / np.abs(F1)

    # smooth Δφ
    if smoothing_len > 1:
        kernel = np.ones(smoothing_len) / smoothing_len
        phase_diff = np.convolve(phase_diff, kernel, mode='same')

    # all-pass: H(f) = exp[i·Δφ(f)]
    H = np.exp(1j * phase_diff)
    f = np.fft.rfftfreq(N, 1/fs)
    return f, H, mag_diff


def reject_free_stream_noise(p_free, p_wall, fs, eps=1e-8):
    """
    Wiener-filter rejection of free-stream noise from wall-pressure signal.

    We estimate
    \begin{equation}
    H(f)=\frac{S_{fw}(f)}{S_{ff}(f)},\quad
    \hat p_w(t)=\mathcal F^{-1}\{H(f)\,P_f(f)\},\quad
    p_{w,\mathrm{clean}}(t)=p_w(t)-\hat p_w(t).
    \end{equation}

    Parameters
    ----------
    p_free : array_like, shape (N,)
        Free-stream (noise) signal.
    p_wall : array_like, shape (N,)
        Wall-pressure signal.
    fs : float
        Sampling rate [Hz].
    nperseg : int or None
        FFT segment length for Welch (default = N).
    noverlap : int or None
        Overlap between segments (default = nperseg//2).
    eps : float
        Small regularisation to avoid division by zero.

    Returns
    -------
    p_clean : ndarray, shape (N,)
        Noise-rejected wall-pressure time series.
    f       : ndarray, shape (M,)
        Frequency bins [Hz].
    H       : ndarray, shape (M,)
        Complex Wiener filter transfer function.
    """
    N = len(p_free)
    nperseg = N//2000
    noverlap = nperseg // 2

    # estimate spectra via Welch
    f, P_ff = welch(p_free, fs=fs, nperseg=nperseg, noverlap=noverlap)
    f, P_fw = csd (p_free, p_wall, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Wiener filter
    H = P_fw / (P_ff + eps)

    # apply in frequency domain
    P_f = np.fft.rfft(p_free)
    F   = np.fft.rfftfreq(N, 1/fs)
    H_interp = np.interp(F, f, H)       # align filter to full-spectrum bins
    p_est    = np.fft.irfft(H_interp * P_f, n=N)

    # subtract predicted free-stream contribution
    p_clean = p_wall - p_est

    return p_clean, f, H


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

