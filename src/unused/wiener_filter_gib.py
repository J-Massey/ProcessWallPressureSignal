import numpy as np

def wiener_cancel_background(
    p0,
    pn,
    filter_order=4096,
    regularization=0.0,
    preserve_mean=False,
    prefer_scipy=True,
    return_noise_estimate=False,
):
    """
    Wiener noise-cancelling filter for wall-pressure measurements (Appendix A, eqs. A5–A8).

    Parameters
    ----------
    p0 : (N,) array_like
        Helmholtz/transfer-function-corrected wall-pressure time series (contains true wall pressure + background noise).
        In the paper this is the pinhole microphone signal AFTER Helmholtz correction (their p0(t)).
    pn : (N,) array_like
        Simultaneous free-stream (background noise) microphone time series (their pn(t)).
    filter_order : int, optional
        FIR Wiener filter length m (how many lags to use). Larger values provide more
        modelling capacity but cost more memory/compute. The paper reports using m≈16000.
    regularization : float, optional
        Small non-negative value added to the zero-lag autocorrelation r_pn(0) to stabilize
        the Toeplitz system if needed (Tikhonov-style ridge).
    preserve_mean : bool, optional
        If True, re-adds the mean of p0 after filtering (both signals are demeaned
        internally to compute correlations).
    prefer_scipy : bool, optional
        If True, tries to use SciPy's efficient Toeplitz solver when available.
    return_noise_estimate : bool, optional
        If True, also returns the estimated background noise time series (c * pn).

    Returns
    -------
    p_clean : (N,) ndarray
        Cleaned wall-pressure estimate p̂(t) = p0(t) - p̂_b(t).
    pb_hat : (N,) ndarray, optional
        Estimated background noise p̂_b(t) (returned if return_noise_estimate is True).

    Notes
    -----
    Implements:
      • Wiener–Hopf equations R_pn c = r_{p0,pn}  (Appendix A, eq. A6 / A7)
      • Noise estimate p̂_b = c * pn              (Appendix A, eq. A8)
      • Clean signal  p̂   = p0 - p̂_b           (Appendix A, eq. A5)

    Correlations are computed for nonnegative lags k=0..m-1 and use an unbiased
    normalization (divide by N-k). Filtering is causal; the first m-1 samples can
    be treated as a transient if you wish.
    """
    # Convert inputs to contiguous float64 arrays and align lengths
    p0 = np.asarray(p0, dtype=np.float64).ravel()
    pn = np.asarray(pn, dtype=np.float64).ravel()
    N = min(p0.size, pn.size)
    p0 = p0[:N]
    pn = pn[:N]

    # Guardrails
    if filter_order < 1:
        raise ValueError("filter_order must be >= 1.")
    m = int(min(filter_order, N))
    if m < filter_order:
        # If requested order exceeds available samples, cap it
        filter_order = m

    # Demean for correlation estimation
    mu_p0 = p0.mean()
    mu_pn = pn.mean()
    p0_zm = p0 - mu_p0
    pn_zm = pn - mu_pn

    # --------- Estimate autocorrelation r_pn[k] and cross-correlation r_{p0,pn}[k], k=0..m-1 ---------
    # Use numpy.correlate (unbiased normalization). For cross-corr we need sum pn[i] * p0[i+k] (k>=0),
    # which corresponds to np.correlate(pn, p0, 'full')[N-1 + k].
    # Autocorr (nonnegative lags): np.correlate(pn, pn, 'full')[N-1 + k]
    rpn_full = np.correlate(pn_zm, pn_zm, mode='full')
    rpn = rpn_full[N-1:N-1+m].copy()

    rp0pn_full = np.correlate(pn_zm, p0_zm, mode='full')
    rp0pn = rp0pn_full[N-1:N-1+m].copy()

    # Unbiased normalization: divide by (N - k)
    denom = (N - np.arange(m))
    rpn /= denom
    rp0pn /= denom

    # Optional small ridge to help conditioning
    if regularization > 0:
        rpn[0] += float(regularization)

    # --------- Solve Toeplitz system R_pn c = r_{p0,pn} for FIR coefficients c ---------
    c = None
    used_scipy = False
    if prefer_scipy:
        try:
            # SciPy provides an O(m^2) solver with Toeplitz structure (no full matrix build)
            from scipy.linalg import solve_toeplitz  # type: ignore
            c = solve_toeplitz((rpn, rpn), rp0pn)
            used_scipy = True
        except Exception:
            used_scipy = False

    if c is None:
        # Fallback: build the Toeplitz matrix explicitly (O(m^2) memory) and solve with NumPy.
        # For large m this can be memory-heavy; reduce filter_order if needed.
        # Toeplitz[i,j] = rpn[|i-j|]
        idx = np.abs(np.subtract.outer(np.arange(m), np.arange(m)))
        R = rpn[idx]
        c = np.linalg.solve(R, rp0pn)

    # --------- Estimate noise and clean the wall-pressure signal ---------
    # Causal FIR filtering: y[n] = sum_{k=0}^{m-1} c[k]*pn_zm[n-k]
    # Implement via convolution; take the first N samples to align causally.
    pb_hat = np.convolve(pn_zm, c, mode='full')[:N]

    # Cleaned signal: p̂ = p0_zm - p̂_b
    p_clean = p0_zm - pb_hat

    # Restore original mean if requested (typical for fluctuation signals mean≈0)
    if preserve_mean:
        p_clean = p_clean + mu_p0

    if return_noise_estimate:
        # Return noise estimate in the same mean convention as p_clean
        pb_out = pb_hat + (mu_pn * 0.0)  # pb_hat was built from zero-mean pn; keep zero-mean
        return p_clean, pb_out
    return p_clean
