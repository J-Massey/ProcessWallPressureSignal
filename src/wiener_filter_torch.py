import math
import torch
import torch.nn.functional as F

def wiener_cancel_background_torch(
    p0,
    pn,
    filter_order=2**6,
    device=None,
    dtype=torch.float32,
    solver="cg_fft",          # "cg_fft" (fast O(m log m) per iter) or "cholesky" (dense, for small m)
    regularization=0.0,       # small ridge on r_pn(0) to stabilize the Toeplitz system
    cg_tol=1e-6,
    cg_maxiter=128,
    preserve_mean=True,
    return_noise_estimate=False,
):
    """
    GPU-accelerated Wiener noise-cancelling filter (Appendix A, eqs. A5–A8; Gibeau & Ghaemi, 2021).

    Solves (A6) for FIR coefficients c using the autocorrelation of the noise microphone (pn)
    and the cross-correlation between p0 and pn, then estimates p̂_b = c * pn and returns
    p̂ = p0 − p̂_b (A8→A5). Correlations use unbiased normalization.

    Parameters
    ----------
    p0 : 1-D array-like
        Transfer-function/Helmholtz-corrected wall-pressure signal (paper’s p0).
    pn : 1-D array-like
        Simultaneous free-stream noise microphone signal (paper’s pn).
    filter_order : int
        FIR order m. The paper reports m ≈ 16,000 based on spectral convergence.
    device : str or torch.device, optional
        Defaults to "cuda" if available else "cpu".
    dtype : torch.dtype
        torch.float32 (fast) or torch.float64 (higher precision).
    solver : {"cg_fft", "cholesky"}
        "cg_fft": Conjugate Gradient with FFT-based Toeplitz mat-vec (recommended for large m).
        "cholesky": Dense build + Cholesky solve (OK for small m ≲ 4096).
    regularization : float
        Non-negative ridge added to r_pn(0) to improve conditioning.
    cg_tol : float
        Relative tolerance for CG solver.
    cg_maxiter : int
        Max CG iterations.
    preserve_mean : bool
        If True, re-adds mean(p0) after filtering.
    return_noise_estimate : bool
        If True, also return the estimated background noise p̂_b.

    Returns
    -------
    p_clean : torch.Tensor
        Cleaned wall-pressure estimate on the chosen device/dtype.
    pb_hat : torch.Tensor (optional)
        Estimated background noise (returned if return_noise_estimate=True).

    References
    ----------
    Appendix A, eqs. (A5)–(A8) and discussion on pp. 40–41 (filtering & order choice).
    """

    # ---------- helpers ----------
    def _as_t(x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x)

    def _next_pow2(n):
        return 1 << (int(n - 1).bit_length())

    def _full_conv_fft(x, y):
        # Linear convolution via FFT (same as numpy.convolve(x, y, 'full')).
        L = _next_pow2(x.numel() + y.numel() - 1)
        X = torch.fft.rfft(x, n=L)
        Y = torch.fft.rfft(y, n=L)
        z = torch.fft.irfft(X * Y, n=L)
        return z[: x.numel() + y.numel() - 1]

    def _correlate_full_like_numpy(a, v):
        # numpy.correlate(a, v, 'full') == convolve(a[::-1], v, 'full')
        return _full_conv_fft(torch.flip(a, dims=(0,)), v)

    def _prepare_circulant_fft(r_first_col):
        # Build circulant embedding C of size L for Toeplitz(r) with first column r_first_col
        m = r_first_col.numel()
        L = _next_pow2(2 * m - 1)
        c = torch.zeros(L, dtype=dtype, device=device)
        c[:m] = r_first_col
        # tail: r[1:], reversed, placed at the end
        if m > 1:
            c[L - (m - 1):] = torch.flip(r_first_col[1:], dims=(0,))
        Cf = torch.fft.rfft(c, n=L)
        return Cf, L

    def _toeplitz_mv_fft(Cf, L, v):
        # y = Toeplitz(r) @ v via circulant multiplication with FFT
        m = v.numel()
        V = torch.zeros(L, dtype=dtype, device=device)
        V[:m] = v
        y_full = torch.fft.irfft(Cf * torch.fft.rfft(V, n=L), n=L)
        return y_full[:m]

    def _cg_toeplitz_solve(r_first_col, b, ridge, tol, maxiter):
        # Solve (Toeplitz(r) + ridge I) x = b with CG and FFT mat-vec
        r0 = r_first_col.clone()
        r0[0] = r0[0] + ridge
        Cf, L = _prepare_circulant_fft(r0)
        x = torch.zeros_like(b)
        r_vec = b - _toeplitz_mv_fft(Cf, L, x)
        p_vec = r_vec.clone()
        rsold = torch.dot(r_vec, r_vec)
        # cg loop
        for _ in range(maxiter):
            Ap = _toeplitz_mv_fft(Cf, L, p_vec)
            denom = torch.dot(p_vec, Ap)
            alpha = rsold / (denom + torch.finfo(dtype).eps)
            x = x + alpha * p_vec
            r_vec = r_vec - alpha * Ap
            rsnew = torch.dot(r_vec, r_vec)
            if rsnew.sqrt() <= tol * rsold.sqrt():
                break
            p_vec = r_vec + (rsnew / (rsold + torch.finfo(dtype).eps)) * p_vec
            rsold = rsnew
        return x

    # ---------- to torch & device ----------
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    p0 = _as_t(p0).to(device=device, dtype=dtype).flatten()
    pn = _as_t(pn).to(device=device, dtype=dtype).flatten()

    # align and guard
    N = int(min(p0.numel(), pn.numel()))
    p0 = p0[:N]
    pn = pn[:N]

    m = int(min(filter_order, N))
    if m < 1:
        raise ValueError("filter_order must be >= 1 and <= len(signal).")

    # ---------- demean (used for correlation estimates) ----------
    mu_p0 = p0.mean()
    mu_pn = pn.mean()
    p0_zm = p0 - mu_p0
    pn_zm = pn - mu_pn

    # ---------- unbiased auto/cross correlations for lags k = 0..m-1 ----------
    # r_pn(k)   = sum_{n} pn[n] * pn[n+k]     (unbiased: divide by N - k)
    # r_p0pn(k) = sum_{n} pn[n] * p0[n+k]
    rpn_full   = _correlate_full_like_numpy(pn_zm, pn_zm)
    rp0pn_full = _correlate_full_like_numpy(pn_zm, p0_zm)
    start = N - 1
    rpn   = rpn_full[start : start + m].clone()
    rp0pn = rp0pn_full[start : start + m].clone()

    denom = (N - torch.arange(m, device=device, dtype=dtype))
    rpn   = rpn / denom
    rp0pn = rp0pn / denom

    # ridge for stability (A6 system is SPD; ridge improves conditioning if needed)
    if regularization > 0.0:
        rpn[0] = rpn[0] + regularization

    # ---------- solve Toeplitz system R_pn c = r_{p0,pn} for c (A6, A7) ----------
    if solver == "cholesky":
        # build dense Toeplitz (OK for small m)
        idx = torch.abs(torch.arange(m, device=device).unsqueeze(0) - torch.arange(m, device=device).unsqueeze(1))
        R = rpn[idx]
        c = torch.cholesky_solve(rp0pn.unsqueeze(1), torch.linalg.cholesky(R)).squeeze(1)
    else:
        # CG with FFT-based Toeplitz mat-vec (recommended for large m/GPU)
        c = _cg_toeplitz_solve(rpn, rp0pn, ridge=0.0, tol=cg_tol, maxiter=cg_maxiter)

    # ---------- estimate noise p̂_b = c * pn  (causal FIR; A8) ----------
    # causal "full" convolution truncated to N samples:
    # y[n] = sum_{k=0}^{m-1} c[k] * pn_zm[n-k], so use conv1d with flipped kernel and left padding.
    x = pn_zm.view(1, 1, N)
    w = torch.flip(c, dims=(0,)).view(1, 1, m)  # conv1d is correlation; flip to realize convolution
    x_pad = F.pad(x, (m - 1, 0))
    pb_hat = F.conv1d(x_pad, w).view(-1)  # length N

    # ---------- cleaned wall pressure p̂ = p0 - p̂_b (A5) ----------
    p_clean = p0_zm - pb_hat
    if preserve_mean:
        p_clean = p_clean + mu_p0

    return (p_clean, pb_hat) if return_noise_estimate else p_clean
