from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import savgol_filter, welch, csd

from i_o import load_stan_wallpressure
from processing import compute_duct_modes, notch_filter_timeseries, compute_psd

from icecream import ic


def torch_unwrap1d(x: torch.Tensor) -> torch.Tensor:
    # 1) Δφ along dim=0
    d = x.diff(dim=0)
    # 2) wrap to [−π, π)
    dd = ((d + math.pi) % (2*math.pi)) - math.pi
    dd = torch.where((dd == -math.pi) & (d > 0),
                     torch.tensor(math.pi, device=x.device),
                     dd)
    # 3) cumulative correction
    corr = (dd - d).cumsum(dim=0)
    # 4) pad so shapes match
    corr = F.pad(corr, (1, 0))
    return x + corr

def torch_welch(
    x: torch.Tensor,
    fs: float = 1.0,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = x.numel()
    L = nperseg or N
    O = noverlap or L//2
    S = L - O

    # pad so remainder fits exactly
    nseg = 1 + math.ceil((N-L)/S)
    pad  = (nseg-1)*S + L - N
    if pad>0:
        x = torch.cat([x, x.new_zeros(pad)], dim=0)

    # frame
    shape   = (L, nseg)
    strides = (x.stride(0), S*x.stride(0))
    frames  = x.as_strided(shape, strides)

    # detrend
    frames = frames - frames.mean(dim=0, keepdim=True)

    # window, FFT, PSD
    w  = torch.hann_window(L, dtype=x.dtype, device=x.device)
    X  = torch.fft.rfft(frames*w.unsqueeze(1), dim=0)
    U  = w.pow(2).sum()
    Pxx = (X.abs()**2).mean(dim=1) / (fs*U)

    # freqs
    nfreq = X.size(0)
    f     = torch.arange(nfreq, device=x.device)*fs/L

    return f, Pxx


def torch_csd(
    x: torch.Tensor,
    y: torch.Tensor,
    fs: float = 1.0,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (f, Pxy) via cross-spectral density in PyTorch,
    forcing a real-valued output by taking the real part.
    """
    N = x.numel()
    L = nperseg or N
    O = noverlap or (L // 2)
    S = L - O
    # frame the signals
    shape = (L, 1 + (N - L) // S)
    strides = (x.stride(0), S * x.stride(0))
    fx = x.as_strided(shape, strides)
    fy = y.as_strided(shape, strides)
    # window + FFT
    w = torch.hann_window(L, dtype=x.dtype, device=x.device)
    X = torch.fft.rfft(fx * w.unsqueeze(1), dim=0)
    Y = torch.fft.rfft(fy * w.unsqueeze(1), dim=0)
    U = w.pow(2).sum()
    # complex CSD
    Pxy_c = (X * Y.conj()).mean(dim=1) / (fs * U)
    # force real output (drop any residual imag)
    Pxy = Pxy_c.real
    # frequency vector
    f = torch.arange(Pxy.size(0), device=x.device) * fs / L
    return f, Pxy


def torch_interp(x: torch.Tensor,
                 xp: torch.Tensor,
                 fp: torch.Tensor,
                 left: float = 0.0,
                 right: float = 0.0) -> torch.Tensor:
    """
    Pure-torch equivalent of np.interp with left/right fill values.
    """
    # ensure same dtype/device
    N = xp.shape[0]
    # masks
    mask_l = x < xp[0]
    mask_r = x > xp[-1]
    # clamp for interp
    x_cl = x.clamp(xp[0], xp[-1])
    # find intervals
    idx = torch.searchsorted(xp, x_cl)
    idx1 = idx.clamp(1, N-1)
    idx0 = idx1 - 1

    x0, x1 = xp[idx0], xp[idx1]
    y0, y1 = fp[idx0], fp[idx1]
    slope = (y1 - y0) / (x1 - x0)
    y = y0 + slope * (x_cl - x0)

    # apply left/right
    y = torch.where(mask_l, left, y)
    y = torch.where(mask_r, right, y)
    return y


def torch_savgol_filter(
    x: torch.Tensor,
    window_length: int,
    polyorder: int,
    deriv: int = 0,
) -> torch.Tensor:
    """
    x: 1D torch.Tensor
    window_length: odd int > polyorder
    polyorder: int < window_length
    deriv: derivative order (0 for smoothing)
    """
    assert window_length % 2 == 1 and window_length > polyorder
    m = (window_length - 1)//2
    device, dtype = x.device, x.dtype

    # build Vandermonde (L×(p+1))
    pos = torch.arange(-m, m+1, dtype=torch.float64, device=device)
    A = torch.stack([pos**i for i in range(polyorder+1)], dim=1)
    ATAinv = torch.inverse(A.T @ A)
    C = ATAinv @ A.T                # (p+1)×L
    coeffs = (C[deriv] * math.factorial(deriv)).to(dtype)  # length L

    # reflect‐pad
    left  = x[:m].flip(0)
    right = x[-m:].flip(0)
    xp = torch.cat([left, x, right], dim=0)

    # sliding window and convolution
    windows = xp.unfold(0, window_length, 1)  # (N, L)
    return (windows * coeffs).sum(dim=1)


class WallPressureProcessor:
    """Processing pipeline for wall-pressure and free-stream signals."""

    def __init__(
        self,
        sample_rate: float,
        nu0: float,
        rho0: float,
        u_tau0: float,
        err_frac: float,
        W: float,
        H: float,
        L0: float,
        delta_L0: float,
        U: float,
        C: float,
        mode_m: Sequence[int],
        mode_n: Sequence[int],
        mode_l: Sequence[int],
    ) -> None:
        self.sample_rate = sample_rate
        self.nu0 = nu0
        self.rho0 = rho0
        self.u_tau0 = u_tau0
        self.err_frac = err_frac
        self.W = W
        self.H = H
        self.L0 = L0
        self.delta_L0 = delta_L0
        self.U = U
        self.C = C
        self.mode_m = mode_m
        self.mode_n = mode_n
        self.mode_l = mode_l

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.L = self.L0 + self.delta_L0

        self.duct_modes = None
        self.f_w = torch.tensor([], device=torch.device(device=self.device), dtype=torch.float64)
        self.p_w = torch.tensor([], device=torch.device(device=self.device), dtype=torch.float64)
        self.f_fs = torch.tensor([], device=torch.device(device=self.device), dtype=torch.float64)
        self.p_fs = torch.tensor([], device=torch.device(device=self.device), dtype=torch.float64)
        self.filtered = False

    # ------------------------------------------------------------------
    def compute_duct_modes(self) -> Dict[str, Any]:
        """Identify duct modes using current parameters."""
        self.duct_modes = compute_duct_modes(
            self.U, self.C,
            self.mode_m, self.mode_n, self.mode_l,
            self.W, self.H, self.L,
            self.nu0, self.u_tau0, self.err_frac,
        )
        return self.duct_modes

    # ------------------------------------------------------------------
    def load_data(
        self,
        wall_mat: str,
        fs_mat: str,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Load wall and freestream pressure signals from .mat files."""
        self.f_w, self.p_w = load_stan_wallpressure(wall_mat)
        self.f_fs, self.p_fs = load_stan_wallpressure(fs_mat)
        # Convert to torch tensors
        self.f_w = torch.tensor(self.f_w, device=self.device)
        self.p_w = torch.tensor(self.p_w, device=self.device)
        self.f_fs = torch.tensor(self.f_fs, device=self.device)
        self.p_fs = torch.tensor(self.p_fs, device=self.device)
        return (self.f_w, self.p_w), (self.f_fs, self.p_fs)

    # ------------------------------------------------------------------
    def notch_filter(self) -> Tuple[Any, Any]:
        """Apply duct-mode notch filtering to wall and free-stream signals."""
        if self.duct_modes is None:
            self.compute_duct_modes()

        modes_min = self.duct_modes["min"] * (self.u_tau0 ** 2 / self.nu0)
        modes_max = self.duct_modes["max"] * (self.u_tau0 ** 2 / self.nu0)
        modes_nom = self.duct_modes["nom"] * (self.u_tau0 ** 2 / self.nu0)

        results_w = notch_filter_timeseries(
            self.p_w, self.sample_rate, modes_min, modes_max, modes_nom
        )
        (
            self.p_w,
            self.f_wall_nom,
            self.Pxx_wall_nom,
            self.f_wall_filt,
            self.Pxx_wall_filt,
            self.info_wall,
        ) = results_w

        results_fs = notch_filter_timeseries(
            self.p_fs, self.sample_rate, modes_min, modes_max, modes_nom
        )
        (
            self.p_fs,
            self.f_fs_nom,
            self.Pxx_fs_nom,
            self.f_fs_filt,
            self.Pxx_fs_filt,
            self.info_fs,
        ) = results_fs

        self.filtered = True
        return results_w, results_fs
    
    def phase_match(self, smoothing_len: int = 1) -> None:
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
        sig1 = self.p_w
        sig2 = self.p_fs
        N = len(sig1)
        # Compute one-sided FFT (rfft) for efficiency with real signals
        F1 = torch.fft.rfft(sig1)
        F2 = torch.fft.rfft(sig2)
        # Compute phase difference for each frequency bin
        phase_diff = torch.angle(F2) - torch.angle(F1)
        # Unwrap the phase difference to avoid 2π discontinuities
        phase_diff = torch_unwrap1d(phase_diff)
        # smooth the unwrapped phase
        if smoothing_len > 1:
            kernel = torch.ones(smoothing_len, device=self.device) / smoothing_len
            pad = smoothing_len // 2
            phase_diff = torch.nn.functional.conv1d(
                phase_diff.view(1,1,-1), kernel.view(1,1,-1), padding=pad
            ).view(-1)
        # Construct complex phase correction (unit magnitude factors)
        phase_correction = torch.exp(1j * phase_diff)
        # Apply phase correction to F1's spectrum
        F1_matched = F1 * phase_correction
        # Inverse FFT to get the phase-adjusted time-domain signal
        self.p_w = torch.fft.irfft(F1_matched, n=N)
        self.phase_matched = True

    def reject_free_stream_noise(self, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Wiener-filter rejection of free-stream noise from wall-pressure signal.

        We estimate
        \\begin{equation}
        H(f)=\\frac{S_{fw}(f)}{S_{ff}(f)},\\quad
        \\hat p_w(t)=\\mathcal F^{-1}\\{H(f)\\,P_f(f)\\},\\quad
        p_{w,\\mathrm{clean}}(t)=p_w(t)-\\hat p_w(t).
        \\end{equation}

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
        p_w = self.p_w_filt if self.filtered else self.p_w
        p_fs = self.p_fs_filt if self.filtered else self.p_fs
        N = len(self.p_fs)
        nperseg = N//2000
        noverlap = nperseg // 2

        # estimate spectra via Welch
        f, P_ff = torch_welch(p_fs, fs=self.sample_rate, nperseg=nperseg, noverlap=noverlap)
        _, P_fw = torch_csd(p_fs, p_w, fs=self.sample_rate, nperseg=nperseg, noverlap=noverlap)

        # Wiener filter
        H = P_fw / (P_ff + eps)

        # apply in frequency domain
        P_f = torch.fft.rfft(self.p_fs)
        f_grid   = torch.fft.rfftfreq(N, 1/self.sample_rate).to(device=self.device)
        H_interp = torch_interp(f_grid, f, H)       # align filter to full-spectrum bins
        p_est    = torch.fft.irfft(H_interp * P_f, n=N)

        # subtract predicted free-stream contribution
        p_clean = self.p_w - p_est
        self.p_w = p_clean
        self.wiener_filtered = True
        return p_clean, f, H

    # --------------------------------------- Spectra ---------------------------------------
    def compute_wall_spectrum(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute PSD for the (optionally filtered) wall signal."""
        signal = self.p_w
        N = len(signal)
        nperseg = N//2000
        noverlap = nperseg // 2
        f_nom, phi_nom = torch_welch(
            signal, fs=self.sample_rate, nperseg=nperseg, noverlap=noverlap
        )
        return f_nom, phi_nom

    def compute_transfer_function(
        self,
        ref_freq: Sequence[float] | torch.Tensor,
        ref_pxx: Sequence[float] | torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # tensors on correct device
        f_ref = torch.as_tensor(ref_freq , device=self.device)
        P_ref = torch.as_tensor(ref_pxx  , device=self.device)
        f_nom, Phi_w = self.compute_wall_spectrum()

        # log grid
        f_min, f_max = 1e0, 3e3
        f_grid = torch.logspace(math.log10(f_min),
                                math.log10(f_max),
                                2048, device=self.device)

        # interpolate physical PSDs
        W = torch_interp(f_grid, f_nom,     Phi_w, left=0., right=0.)
        R = torch_interp(f_grid, f_ref,     P_ref, left=0., right=0.)

        # protect zeros
        eps = 1e-12
        W = W.clamp(min=eps)
        R = R.clamp(min=eps)

        H2 = R / W
        H2 = torch.nan_to_num(H2, nan=1., posinf=1., neginf=1.)

        # H  = torch.sqrt(H2)
        H  = torch.nan_to_num(H2,  nan=1., posinf=1., neginf=1.)
        # Smooth the transfer function
        # H = torch_savgol_filter(
        #     H, window_length=701, polyorder=1, deriv=0
        # )


        self.transfer_freq = f_grid
        self.transfer_mag  = H

        self.Phi_p_corr = H * W
        return f_grid, self.Phi_p_corr, H


    def apply_transfer_function(self) -> torch.Tensor:
        p = self.p_w.to(device=self.device,
                        dtype=self.transfer_mag.dtype)
        N = p.shape[0]

        freqs    = torch.fft.rfftfreq(N, 1/self.sample_rate,
                                    device=self.device,
                                    dtype=p.dtype)
        wall_fft = torch.fft.rfft(p, n=N)

        H_full = torch_interp(freqs,
                            self.transfer_freq.to(device=self.device, dtype=freqs.dtype),
                            self.transfer_mag .to(dtype=freqs.dtype),
                            left=1., right=1.)
        H_full = torch.nan_to_num(H_full, nan=1., posinf=1., neginf=1.)
        corrected = wall_fft * H_full

        self.p_w = torch.fft.irfft(corrected, n=N)
        return self.p_w

