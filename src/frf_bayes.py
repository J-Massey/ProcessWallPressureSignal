
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Dict
import numpy as np

def _build_tridiagonal(diag_w: np.ndarray, lambda_smooth: float, tau2: float):
    F = diag_w.shape[0]
    a = diag_w.astype(np.float64).copy()
    if not np.isinf(tau2):
        a = a + (1.0 / float(tau2))
    if lambda_smooth > 0.0:
        a[0] += lambda_smooth * 1.0
        if F > 1:
            a[1:-1] += lambda_smooth * 2.0
            a[-1] += lambda_smooth * 1.0
        b = -lambda_smooth * np.ones(F - 1, dtype=np.float64)
    else:
        b = np.zeros(F - 1, dtype=np.float64)
    a += 1e-18
    return a, b

def _thomas_solve_tridiagonal(a: np.ndarray, b: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    F = a.shape[0]
    ac = a.copy()
    bc = b.copy()
    rc = rhs.astype(np.complex128).copy()
    for i in range(1, F):
        w = bc[i-1] / ac[i-1]
        ac[i] = ac[i] - w * bc[i-1]
        rc[i] = rc[i] - w * rc[i-1]
    x = np.empty(F, dtype=np.complex128)
    x[-1] = rc[-1] / ac[-1]
    for i in range(F - 2, -1, -1):
        x[i] = (rc[i] - bc[i] * x[i + 1]) / ac[i]
    return x

def _tridiag_cholesky(a: np.ndarray, b: np.ndarray):
    F = a.shape[0]
    l = np.empty(F, dtype=np.float64)
    m = np.empty(F - 1, dtype=np.float64)
    l[0] = np.sqrt(max(a[0], 1e-300))
    for i in range(1, F):
        m[i-1] = b[i-1] / l[i-1]
        val = a[i] - m[i-1] * m[i-1]
        if val <= 0:
            val = 1e-18
        l[i] = np.sqrt(val)
    return l, m

def _chol_solve(L_diag: np.ndarray, L_sub: np.ndarray, rhs: np.ndarray, transpose: bool=False) -> np.ndarray:
    F = L_diag.shape[0]
    y = rhs.astype(np.complex128).copy()
    if not transpose:
        y[0] = y[0] / L_diag[0]
        for i in range(1, F):
            y[i] = (y[i] - L_sub[i-1] * y[i-1]) / L_diag[i]
    else:
        y[-1] = y[-1] / L_diag[-1]
        for i in range(F-2, -1, -1):
            y[i] = (y[i] - L_sub[i] * y[i+1]) / L_diag[i]
    return y

def _diag_from_cholesky_slow(Ld: np.ndarray, Ls: np.ndarray) -> np.ndarray:
    F = Ld.shape[0]
    diag = np.empty(F, dtype=np.float64)
    for i in range(F):
        y = np.zeros(F, dtype=np.float64)
        y[0] = (1.0 if i == 0 else 0.0) / Ld[0]
        for j in range(1, F):
            rhs = 1.0 if j == i else 0.0
            y[j] = (rhs - Ls[j-1] * y[j-1]) / Ld[j]
        diag[i] = np.dot(y, y)
    return diag

def coherence_to_variance(Hm: np.ndarray, gamma2: np.ndarray, nu: np.ndarray, gamma2_min: float = 0.02, var_floor_rel: float = 1e-12) -> np.ndarray:
    Hm = np.asarray(Hm, dtype=np.complex128)
    gamma2 = np.asarray(gamma2, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)
    if gamma2.shape != Hm.shape:
        gamma2 = np.broadcast_to(gamma2, Hm.shape)
    if nu.shape != Hm.shape:
        nu = np.broadcast_to(nu, Hm.shape)
    gamma2c = np.clip(gamma2, gamma2_min, 1.0)
    mag2 = np.maximum(np.abs(Hm) ** 2, var_floor_rel)
    sigma2 = ((1.0 - gamma2c) / gamma2c) * (mag2 / np.maximum(nu, 1.0))
    sigma2 = np.maximum(sigma2, var_floor_rel * np.median(mag2, axis=1, keepdims=True))
    return sigma2.astype(np.float64)

from dataclasses import dataclass

@dataclass
class FusedFRF:
    freq: np.ndarray
    mu: np.ndarray
    sigma2_complex: np.ndarray
    a_diag: np.ndarray
    a_off: np.ndarray
    weights_sum: np.ndarray
    lambda_smooth: float
    tau2: float

    def sample(self, K: int = 200, random_state: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        Ld, Ls = _tridiag_cholesky(self.a_diag, self.a_off)
        F = self.freq.shape[0]
        samples = np.empty((K, F), dtype=np.complex128)
        scale = np.sqrt(0.5)
        for k in range(K):
            z = rng.standard_normal(F)
            y = _chol_solve(Ld, Ls, z, transpose=False)
            real_part = _chol_solve(Ld, Ls, y, transpose=True) * scale
            z = rng.standard_normal(F)
            y = _chol_solve(Ld, Ls, z, transpose=False)
            imag_part = _chol_solve(Ld, Ls, y, transpose=True) * scale
            samples[k] = self.mu + real_part + 1j * imag_part
        return samples

    def credible_band(self, level: float = 0.95, K: int = 400, random_state: Optional[int] = None):
        samples = self.sample(K=K, random_state=random_state)
        mags = np.abs(samples)
        lower = np.quantile(mags, (1.0 - level) / 2.0, axis=0)
        upper = np.quantile(mags, 1.0 - (1.0 - level) / 2.0, axis=0)
        return lower, upper

    def regularized_inverse(self, mode: Literal["c_times_var", "cap_noise_gain"] = "c_times_var", c: float = 3.0, noise_gain_cap: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        mu = self.mu
        mu2 = np.abs(mu) ** 2
        if mode == "c_times_var":
            alpha = c * self.sigma2_complex
        elif mode == "cap_noise_gain":
            if noise_gain_cap is None:
                raise ValueError("noise_gain_cap must be provided for mode='cap_noise_gain'.")
            ng = np.asarray(noise_gain_cap, dtype=np.float64)
            if ng.ndim == 0:
                ng = np.full_like(mu2, fill_value=float(ng))
            target = np.sqrt(np.maximum(ng, 1e-300))
            alpha = (np.sqrt(mu2) / target) - mu2
            alpha = np.maximum(alpha, 0.0)
        else:
            raise ValueError("Unknown mode")
        denom = mu2 + alpha
        denom = np.where(denom <= 1e-300, 1e-300, denom)
        G = np.conjugate(mu) / denom
        gain = mu2 / (denom ** 2)
        return {"G": G, "alpha": alpha, "gain": gain}

def fuse_frf(freq: np.ndarray, Hm: np.ndarray, sigma2: Optional[np.ndarray] = None, gamma2: Optional[np.ndarray] = None, nu: Optional[np.ndarray] = None, gamma2_min: float = 0.02, var_floor_rel: float = 1e-12, lambda_smooth: float = 0.0, tau2: float = np.inf) -> FusedFRF:
    freq = np.asarray(freq, dtype=np.float64)
    Hm = np.asarray(Hm, dtype=np.complex128)
    M, F = Hm.shape
    if sigma2 is None:
        if gamma2 is None or nu is None:
            raise ValueError("Either provide sigma2 or (gamma2 and nu).")
        sigma2 = coherence_to_variance(Hm, gamma2, nu, gamma2_min=gamma2_min, var_floor_rel=var_floor_rel)
    else:
        sigma2 = np.asarray(sigma2, dtype=np.float64)
        if sigma2.shape != Hm.shape:
            sigma2 = np.broadcast_to(sigma2, Hm.shape).copy()
    w_runs = 1.0 / np.maximum(sigma2, 1e-300)
    w_sum = np.sum(w_runs, axis=0)
    b = np.sum(w_runs * Hm, axis=0)
    a, boff = _build_tridiagonal(w_sum, lambda_smooth=lambda_smooth, tau2=tau2)
    mu = _thomas_solve_tridiagonal(a, boff, b)
    # Stable diagonal using Cholesky-based method
    Ld, Ls = _tridiag_cholesky(a, boff)
    diag_Sigma = _diag_from_cholesky_slow(Ld, Ls)
    return FusedFRF(freq=freq, mu=mu, sigma2_complex=diag_Sigma, a_diag=a, a_off=boff, weights_sum=w_sum, lambda_smooth=lambda_smooth, tau2=tau2)


def inverse_credible_band(fused_obj: FusedFRF, alpha, level: float = 0.95, K: int = 400, random_state: int = 0):
    samples = fused_obj.sample(K=K, random_state=random_state)
    import numpy as np
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim == 0:
        alpha = np.full(samples.shape[1], float(alpha))
    mu2_samp = np.abs(samples) ** 2
    Gs = np.conjugate(samples) / (mu2_samp + alpha[None, :])
    mags = np.abs(Gs)
    lo = np.quantile(mags, (1.0 - level) / 2.0, axis=0)
    hi = np.quantile(mags, 1.0 - (1.0 - level) / 2.0, axis=0)
    return lo, hi

