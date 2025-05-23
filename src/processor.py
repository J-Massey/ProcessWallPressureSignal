import torch
import numpy as np
from scipy.signal import savgol_filter, welch, csd
from i_o import load_stan_wallpressure
from processing import compute_duct_modes, notch_filter_timeseries, compute_psd


class WallPressureProcessor:
    """Processing pipeline for wall-pressure and free-stream signals."""

    def __init__(self, sample_rate, nu0, rho0, u_tau0, err_frac,
                 W, H, L0, delta_L0, U, C,
                 mode_m, mode_n, mode_l):
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

        self.L = self.L0 + self.delta_L0

        self.duct_modes = None
        self.f_w = None
        self.p_w = None
        self.f_fs = None
        self.p_fs = None
        self.filtered = False
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    # ------------------------------------------------------------------
    def compute_duct_modes(self):
        """Identify duct modes using current parameters."""
        self.duct_modes = compute_duct_modes(
            self.U, self.C,
            self.mode_m, self.mode_n, self.mode_l,
            self.W, self.H, self.L,
            self.nu0, self.u_tau0, self.err_frac,
        )
        return self.duct_modes

    # ------------------------------------------------------------------
    def load_data(self, wall_mat, fs_mat):
        """Load wall and freestream pressure signals from .mat files."""
        self.f_w, self.p_w = load_stan_wallpressure(wall_mat)
        self.f_fs, self.p_fs = load_stan_wallpressure(fs_mat)
        return (self.f_w, self.p_w), (self.f_fs, self.p_fs)

    # ------------------------------------------------------------------
    def notch_filter(self):
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
            self.p_w_filt,
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
            self.p_fs_filt,
            self.f_fs_nom,
            self.Pxx_fs_nom,
            self.f_fs_filt,
            self.Pxx_fs_filt,
            self.info_fs,
        ) = results_fs

        self.filtered = True
        return results_w, results_fs
    
    def phase_match(self, smoothing_len=1):
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
        sig1 = self.p_w_filt if self.filtered else self.p_w
        sig2 = self.p_fs_filt if self.filtered else self.p_fs
        sig1 = torch.as_tensor(sig1)
        sig2 = torch.as_tensor(sig2)
        N = len(sig1)
        # Compute one-sided FFT (rfft) for efficiency with real signals
        F1 = torch.fft.rfft(sig1)
        F2 = torch.fft.rfft(sig2)
        # Compute phase difference for each frequency bin
        phase_diff = torch.angle(F2) - torch.angle(F1)
        # Unwrap the phase difference to avoid 2π discontinuities
        phase_diff = torch.from_numpy(
            np.unwrap(phase_diff.detach().cpu().numpy())
        )
        # smooth the unwrapped phase
        if smoothing_len > 1:
            kernel = torch.ones(smoothing_len) / smoothing_len
            pad = smoothing_len // 2
            phase_diff = torch.nn.functional.conv1d(
                phase_diff.view(1,1,-1), kernel.view(1,1,-1), padding=pad
            ).view(-1)
        # Construct complex phase correction (unit magnitude factors)
        phase_correction = torch.exp(1j * phase_diff)
        # Apply phase correction to F1's spectrum
        F1_matched = F1 * phase_correction
        # Inverse FFT to get the phase-adjusted time-domain signal
        self.p_w_matched = torch.fft.irfft(F1_matched, n=N)

    def reject_free_stream_noise(self, eps=1e-8):
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
        N = len(self.p_fs)
        nperseg = N//2000
        noverlap = nperseg // 2

        # estimate spectra via Welch
        f, P_ff = welch(self.p_fs.detach().cpu().numpy(), fs=self.sample_rate, nperseg=nperseg, noverlap=noverlap)
        _, P_fw = csd(self.p_fs.detach().cpu().numpy(), self.p_w_matched.detach().cpu().numpy(), fs=self.sample_rate, nperseg=nperseg, noverlap=noverlap)
        f = torch.tensor(f)
        P_ff = torch.tensor(P_ff)
        P_fw = torch.tensor(P_fw)

        # Wiener filter
        H = P_fw / (P_ff + eps)

        # apply in frequency domain
        P_f = torch.fft.rfft(self.p_fs)
        F   = torch.fft.rfftfreq(N, 1/self.sample_rate)
        H_interp = torch.tensor(np.interp(F.cpu().numpy(), f.numpy(), H.numpy()))
        p_est    = torch.fft.irfft(H_interp * P_f, n=N)

        # subtract predicted free-stream contribution
        p_clean = self.p_w_matched - p_est
        self.p_w_clean = p_clean
        return p_clean, f, H

    # ------------------------------------------------------------------
    def compute_wall_spectrum(self):
        """Compute PSD for the (optionally filtered) wall signal."""
        signal = self.p_w_clean if self.filtered else self.p_w
        f_nom, phi_nom = compute_psd(signal, fs=self.sample_rate)
        return f_nom, phi_nom

    # ------------------------------------------------------------------
    def compute_transfer_function(self, ref_freq, ref_pxx):
        """Compute magnitude transfer function from reference spectrum."""
        f_nom, phi_nom = self.compute_wall_spectrum()
        denom = (self.rho0 * self.u_tau0 ** 2) ** 2

        f_grid = np.logspace(
            np.log10(f_nom.min().item() + 1e-1), np.log10(f_nom.max().item()), 2048
        )
        phi_wall_grid = np.interp(f_grid, f_nom.numpy(), (phi_nom / denom).numpy(), left=0, right=0)
        phi_wall_grid = savgol_filter(phi_wall_grid, window_length=64, polyorder=1)
        ref_grid = np.interp(f_grid, ref_freq, (ref_pxx / denom), left=0, right=0)

        H_power_ratio = np.ones_like(f_grid)
        band_idx = np.logical_and(f_grid >= f_grid.min(), f_grid <= f_grid.max())
        H_power_ratio[band_idx] = ref_grid[band_idx] / phi_wall_grid[band_idx]
        H_mag = np.sqrt(H_power_ratio)
        H_mag = savgol_filter(H_mag, window_length=16, polyorder=1)

        self.transfer_freq = torch.tensor(f_grid)
        self.transfer_mag = torch.tensor(H_mag)
        return self.transfer_freq, self.transfer_mag

    # ------------------------------------------------------------------
    def apply_transfer_function(self):
        """Apply previously computed transfer function to wall signal."""
        if not hasattr(self, "transfer_mag"):
            raise RuntimeError("Transfer function not computed")
        N = len(self.p_w_clean)
        freqs = torch.fft.rfftfreq(N, 1 / self.sample_rate)
        wall_fft = torch.fft.rfft(self.p_w_clean)
        H_full = torch.tensor(
            np.interp(freqs.cpu().numpy(), self.transfer_freq.numpy(), self.transfer_mag.numpy(), left=1.0, right=1.0)
        )
        corrected_fft = wall_fft * H_full
        self.p_w_corrected = torch.fft.irfft(corrected_fft, n=N)
        return self.p_w_corrected

