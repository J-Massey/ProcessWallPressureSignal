import numpy as np
from scipy.signal import savgol_filter, welch
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
        self.fs_w = None
        self.p_w = None
        self.fs_fs = None
        self.p_fs = None
        self.filtered = False

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
        self.fs_w, self.p_w = load_stan_wallpressure(wall_mat)
        self.fs_fs, self.p_fs = load_stan_wallpressure(fs_mat)
        return (self.fs_w, self.p_w), (self.fs_fs, self.p_fs)

    # ------------------------------------------------------------------
    def notch_filter(self):
        """Apply duct-mode notch filtering to wall and free-stream signals."""
        if self.duct_modes is None:
            self.compute_duct_modes()

        modes_min = np.array(self.duct_modes["min"]) * (self.u_tau0 ** 2 / self.nu0)
        modes_max = np.array(self.duct_modes["max"]) * (self.u_tau0 ** 2 / self.nu0)
        modes_nom = np.array(self.duct_modes["nom"]) * (self.u_tau0 ** 2 / self.nu0)

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

    # ------------------------------------------------------------------
    def compute_wall_spectrum(self):
        """Compute PSD for the (optionally filtered) wall signal."""
        signal = self.p_w_filt if self.filtered else self.p_w
        f_nom, phi_nom = compute_psd(signal, fs=self.sample_rate)
        return f_nom, phi_nom

    # ------------------------------------------------------------------
    def compute_transfer_function(self, ref_freq, ref_pxx):
        """Compute magnitude transfer function from reference spectrum."""
        f_nom, phi_nom = self.compute_wall_spectrum()
        denom = (self.rho0 * self.u_tau0 ** 2) ** 2

        f_grid = np.logspace(
            np.log10(f_nom.min() + 1e-1), np.log10(f_nom.max()), 1024
        )
        phi_wall_grid = np.interp(f_grid, f_nom, phi_nom / denom, left=0, right=0)
        phi_wall_grid = savgol_filter(phi_wall_grid, window_length=64, polyorder=1)
        ref_grid = np.interp(f_grid, ref_freq, ref_pxx / denom, left=0, right=0)

        H_power_ratio = np.ones_like(f_grid)
        band_idx = np.logical_and(f_grid >= f_grid.min(), f_grid <= f_grid.max())
        H_power_ratio[band_idx] = ref_grid[band_idx] / phi_wall_grid[band_idx]
        H_mag = np.sqrt(H_power_ratio)
        H_mag = savgol_filter(H_mag, window_length=16, polyorder=1)

        self.transfer_freq = f_grid
        self.transfer_mag = H_mag
        return f_grid, H_mag

    # ------------------------------------------------------------------
    def apply_transfer_function(self):
        """Apply previously computed transfer function to wall signal."""
        if not hasattr(self, "transfer_mag"):
            raise RuntimeError("Transfer function not computed")
        N = len(self.p_w)
        freqs = np.fft.rfftfreq(N, 1 / self.sample_rate)
        wall_fft = np.fft.rfft(self.p_w)
        H_full = np.interp(freqs, self.transfer_freq, self.transfer_mag, left=1.0, right=1.0)
        corrected_fft = wall_fft * H_full
        self.p_w_corrected = np.fft.irfft(corrected_fft, n=N)
        return self.p_w_corrected

