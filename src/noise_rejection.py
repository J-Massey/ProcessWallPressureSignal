import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, csd

def compute_coherence_psd(p_fs, p_wall, fs, nperseg=1024, noverlap=512):
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
    f, Pxx = welch(p_fs, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, Pyy = welch(p_wall, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, Pxy = csd(p_fs, p_wall, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Compute coherence
    coh = np.abs(Pxy)**2 / (Pxx * Pyy)
    coh = np.clip(coh, 0.0, 1.0)

    # Coherent output PSD
    Pyy_coh = coh * Pyy

    # Plot coherence
    plt.figure()
    plt.semilogx(f, coh, label='Coherence')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'$\gamma^2(f)$')
    plt.title('Coherence between Freestream and Wall-Pressure')
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()

    # Plot PSDs
    plt.figure()
    plt.semilogy(f, Pyy, label='Original Wall PSD')
    plt.semilogy(f, Pyy_coh, label='Coherent PSD')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.title('Wall PSD vs Coherent Output PSD')
    plt.grid(True)
    plt.legend()
    plt.show()

    return f, coh, Pyy, Pyy_coh


def wiener_filter_timeseries(p_fs, p_wall, fs, nperseg=1024, noverlap=512):
    """
    Apply frequency-domain Wiener filter to raw time series.

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
    y_clean : ndarray
        Wall-pressure time series with freestream-coherent part removed.
    y_coh : ndarray
        Predicted freestream-coherent component of the wall-pressure signal.
    """
    # Estimate spectra
    f, Pxx = welch(p_fs, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, Pyy = welch(p_wall, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, Pxy = csd(p_fs, p_wall, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Transfer function H(f) = Sxy / Sxx
    epsilon = 1e-12 * np.mean(Pxx)  # regularization to avoid division by zero
    H = Pxy / (Pxx + epsilon)

    # Full-length FFT of freestream signal
    N = len(p_fs)
    freqs_full = np.fft.rfftfreq(N, 1/fs)
    X_full = np.fft.rfft(p_fs)

    # Interpolate H onto full FFT frequency grid
    H_real = np.interp(freqs_full, f, np.real(H))
    H_imag = np.interp(freqs_full, f, np.imag(H))
    H_full = H_real + 1j * H_imag

    # Predicted coherent component in frequency domain
    Y_coh_full = H_full * X_full
    y_coh = np.fft.irfft(Y_coh_full, n=N)

    # Residual (cleaned) wall-pressure
    y_clean = p_wall - y_coh

    # Plot original vs cleaned PSD
    f2, Pyy2 = welch(p_wall, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, P_clean = welch(y_clean, fs=fs, nperseg=nperseg, noverlap=noverlap)

    plt.figure()
    plt.semilogy(f2, Pyy2, label='Original Wall PSD')
    plt.semilogy(f2, P_clean, label='Cleaned Wall PSD')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.title('Wiener Filter: Original vs Cleaned PSD')
    plt.grid(True)
    plt.legend()
    plt.show()

    return y_clean, y_coh