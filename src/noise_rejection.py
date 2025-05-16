import numpy as np
import torch
import os
from scipy.signal import welch, csd, detrend, coherence
from icecream import ic

def compute_coherence_psd(p_fs, p_wall, fs):
    """
    Compute coherence and coherent PSD given raw time series.
    
    Parameters:
    -----------
    p_fs : ndarray
        Freestream pressure time series.
    p_wall : ndarray
        Wall-pressure time series.
    fs : float
        Sampling frequency (Hz).
    nperseg : int
        Length of each segment for Welch's method.
    noverlap : int
        Number of points to overlap between segments.
    
    Returns:
    --------
    f : ndarray
        Frequency array.
    coh : ndarray
        Magnitude-squared coherence between p_fs and p_wall.
    Pyy : ndarray
        Original wall-pressure PSD.
    Pyy_coh : ndarray
        Portion of Pyy coherent with p_fs.
    """
    # Estimate auto- and cross-spectra
    nperseg = len(p_fs)//2000
    noverlap = nperseg // 2
    f, Pxx = welch(p_fs, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, Pyy = welch(p_wall, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, Pxy = csd(p_fs, p_wall, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Compute coherence
    coh = np.abs(Pxy)**2 / (Pxx * Pyy)
    coh = np.clip(coh, 0.0, 1.0)

    # Coherent output PSD
    Pyy_coh = coh * Pyy

    return f, coh, Pyy, Pyy_coh

def torch_welch(x, fs, window):
    N = x.shape[-1]
    nperseg = N//2000
    noverlap = nperseg // 2
    hop = nperseg - noverlap
    # returns shape (..., n_freqs, n_frames)
    X = torch.stft(x,
                   n_fft     = nperseg,
                   hop_length= hop,
                   win_length= nperseg,
                   window    = window,
                   return_complex=True)
    U    = window.pow(2).sum()
    Sxx  = (X.abs().pow(2).mean(dim=-1) / (fs * U))  # (n_freqs,)
    freqs= torch.fft.rfftfreq(nperseg, 1/fs, device=x.device)
    return freqs, Sxx, X

def torch_csd(x, y, fs, window):
    _, _, spec_x = torch_welch(x, fs, window=window)
    _, _, spec_y = torch_welch(y, fs, window=window)
    U = window.pow(2).sum()
    csd = (spec_x * spec_y.conj() / (fs * U)).mean(dim=-1)
    return csd

def reject_noise_torch(p_free, p_wall, fs):
    device = "cuda"
    p_free = np.ascontiguousarray(p_free, dtype=np.float32)
    p_wall = np.ascontiguousarray(p_wall, dtype=np.float32)
    p_free = torch.as_tensor(p_free, device=device, dtype=torch.float32)
    p_wall = torch.as_tensor(p_wall, device=device, dtype=torch.float32)
    nperseg = p_free.shape[-1] // 2000
    noverlap = nperseg // 2
    window = torch.hann_window(nperseg, device=p_free.device, dtype=p_free.dtype)
    ic('windowed')
    f, Sww, _ = torch_welch(p_wall, fs, window=window)
    ic('psd1')
    _, Sff, _ = torch_welch(p_free, fs, window=window)
    ic('psd2')
    Swf      = torch_csd(p_wall, p_free, fs, window=window)
    ic('csd')
    coh2       = Swf.abs().pow(2) / (Sww * Sff + 1e-16)
    ic('coh2')
    Phi_clean = Sww * (1 - coh2)
    return f.cpu(), Sww.cpu(), Phi_clean.cpu()

# os.system("python src/run.py")