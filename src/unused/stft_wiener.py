import torch
import torch.nn.functional as F

def wiener_cancel_background_stft_torch(
    p0,
    pn,
    fs,
    *,
    n_fft=2**14,
    win_length=None,
    hop_length=None,
    window="hann",
    center=True,
    causal=False,
    smooth_frames=128,
    ewma_alpha=None,
    regularization=1e-6,
    coherence_threshold=0.0,
    over_subtraction=1.0,
    coherence_guard=True,
    guard_floor_db=0.0,
    preserve_mean=True,
    device=None,
    dtype=torch.float32,
    return_noise_estimate=False,
    # ---------- NEW knobs for stronger low‑f cleanup ----------
    freq_smooth_bins=7,           # small odd integer (e.g., 5–9) smooths spectra across frequency
    lf_shelf=None,                # (f0, f1, gain) => extra subtraction below f1; gain ≥ 1 (e.g., (0.0, 25.0, 1.4))
    lf_shelf_coh_thresh=0.1,      # apply the shelf only where gamma^2 ≥ this threshold
    snap_to_floor_beta=0.0        # 0..1: pull residual power toward (1-γ^2)Syy by fraction β (never below floor)
):
    """
    Block/time‑varying Wiener canceller with added low‑frequency control.

    Implements frequency‑domain A5–A8 from the paper (estimate H = S_x y / S_xx, predict PN path,
    subtract, iSTFT). New options improve low‑frequency removal without violating coherence limits.
    See Appendix A and Fig. 21 for the low‑f noise discussion; Section 4.1 discusses true low‑f (VLSP)
    content you may wish to preserve.  :contentReference[oaicite:2]{index=2}
    """
    # ---------- device & tensors ----------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _as_t(x):
        return x if isinstance(x, torch.Tensor) else torch.tensor(x)

    p0 = _as_t(p0).to(device=device, dtype=dtype).flatten()
    pn = _as_t(pn).to(device=device, dtype=dtype).flatten()
    N = int(min(p0.numel(), pn.numel()))
    p0 = p0[:N]; pn = pn[:N]

    # Demean before spectral work
    mu_p0 = p0.mean(); mu_pn = pn.mean()
    p0_zm = p0 - mu_p0
    pn_zm = pn - mu_pn

    # ---------- STFT ----------
    if win_length is None: win_length = n_fft
    if hop_length is None: hop_length = win_length // 4
    if isinstance(window, str):
        if window.lower() == "hann":
            win = torch.hann_window(win_length, periodic=True, device=device, dtype=dtype)
        else:
            raise ValueError("Only 'hann' string is supported; pass a custom tensor for others.")
    else:
        win = window.to(device=device, dtype=dtype)

    P0 = torch.stft(p0_zm, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    window=win, center=center, return_complex=True)
    PN = torch.stft(pn_zm, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    window=win, center=center, return_complex=True)
    Fbins, Tframes = P0.shape
    eps = torch.finfo(dtype).eps

    # ---------- instantaneous spectra ----------
    S_xx_inst = (PN.conj() * PN).real
    S_yy_inst = (P0.conj() * P0).real
    S_xy_inst = PN.conj() * P0

    # ---------- temporal smoothing ----------
    if causal:
        a = 2.0/(float(smooth_frames)+1.0) if ewma_alpha is None else float(ewma_alpha)
        b = 1.0 - a
        S_xx = torch.empty_like(S_xx_inst); S_yy = torch.empty_like(S_yy_inst); S_xy = torch.empty_like(S_xy_inst)
        S_xx[:,0] = S_xx_inst[:,0]; S_yy[:,0] = S_yy_inst[:,0]; S_xy[:,0] = S_xy_inst[:,0]
        for t in range(1, Tframes):
            S_xx[:,t] = b*S_xx[:,t-1] + a*S_xx_inst[:,t]
            S_yy[:,t] = b*S_yy[:,t-1] + a*S_yy_inst[:,t]
            S_xy_real = b*S_xy[:,t-1].real + a*S_xy_inst[:,t].real
            S_xy_imag = b*S_xy[:,t-1].imag + a*S_xy_inst[:,t].imag
            S_xy[:,t] = torch.complex(S_xy_real, S_xy_imag)
    else:
        Kt = max(1, int(smooth_frames))
        if Kt == 1:
            S_xx, S_yy, S_xy = S_xx_inst, S_yy_inst, S_xy_inst
        else:
            ker_t = torch.ones((Fbins,1,Kt), device=device, dtype=dtype)/float(Kt)
            pad = Kt//2
            def movavg_time(x_real):  # [F,T] -> [F,T]
                y = F.conv1d(x_real.unsqueeze(0), ker_t, padding=pad, groups=Fbins).squeeze(0)
                if y.shape[-1] > x_real.shape[-1]:
                    y = y[..., : x_real.shape[-1]]
                return y
            S_xx = movavg_time(S_xx_inst); S_yy = movavg_time(S_yy_inst)
            S_xy = torch.complex(movavg_time(S_xy_inst.real), movavg_time(S_xy_inst.imag))

    # ---------- NEW: small frequency smoothing (stabilizes low‑f bins) ----------
    Kf = int(freq_smooth_bins)
    if Kf > 1:
        ker_f = torch.ones((1,1,Kf,1), device=device, dtype=dtype)/float(Kf)
        pad_f = (Kf//2, 0)
        def movavg_freq(x_real):  # [F,T] -> [F,T]
            x = x_real.unsqueeze(0).unsqueeze(0)        # [1,1,F,T]
            y = F.conv2d(x, ker_f, padding=pad_f)       # [1,1,F,T]
            y = y.squeeze(0).squeeze(0)
            if y.shape[-2] > x_real.shape[-2]:
                y = y[: x_real.shape[-2], :]
            if y.shape[-1] > x_real.shape[-1]:
                y = y[:, : x_real.shape[-1]]
            return y
        S_xx = movavg_freq(S_xx); S_yy = movavg_freq(S_yy)
        S_xy = torch.complex(movavg_freq(S_xy.real), movavg_freq(S_xy.imag))

    # ---------- Wiener gain per bin ----------
    denom = S_xx + regularization
    H = over_subtraction * (S_xy / (denom + eps))
    gamma2 = (S_xy.abs()**2) / (S_xx*S_yy + eps)  # magnitude-squared coherence

    # Optional coherence threshold (mask)
    if coherence_threshold > 0.0:
        mask = (gamma2 >= float(coherence_threshold)).to(H.dtype)
        H = H * mask

    # ---------- NEW: coherence‑gated low‑f over‑subtraction shelf ----------
    if lf_shelf is not None:
        f0, f1, gain = float(lf_shelf[0]), float(lf_shelf[1]), float(lf_shelf[2])
        freqs = torch.linspace(0.0, fs/2.0, Fbins, device=device, dtype=dtype)
        shelf = torch.ones_like(freqs)
        if f1 > f0:
            lo = freqs <= f0
            mid = (freqs > f0) & (freqs < f1)
            shelf[lo] = gain
            shelf[mid] = 1.0 + (gain - 1.0) * ( (f1 - freqs[mid]) / (f1 - f0) )
        else:
            shelf[:] = gain
        # apply shelf only where coherence is at least lf_shelf_coh_thresh
        coh_w = ( (gamma2 - float(lf_shelf_coh_thresh)) / max(1e-12, 1.0 - float(lf_shelf_coh_thresh)) ).clamp(min=0.0, max=1.0)
        shelf_eff = 1.0 + (shelf[:,None] - 1.0) * coh_w
        H = H * shelf_eff

    # ---------- subtract & guards ----------
    PB_hat = H * PN
    P_clean = P0 - PB_hat

    # NEW: pull residual toward the coherence floor by fraction beta (never below the floor)
    if snap_to_floor_beta > 0.0:
        beta = float(snap_to_floor_beta)
        floor_lin = (10.0 ** (guard_floor_db / 10.0)) if guard_floor_db != 0.0 else 1.0
        S_res_floor = torch.clamp(1.0 - gamma2, min=0.0) * S_yy * floor_lin
        abs2 = (P_clean.conj()*P_clean).real
        target = torch.where(abs2 > S_res_floor, (1.0 - beta)*abs2 + beta*S_res_floor, abs2)
        scale = torch.sqrt((target + eps)/(abs2 + eps))
        P_clean = P_clean * scale
        PB_hat = P0 - P_clean  # keep consistency

    # Coherence guard (ensures we don't dip below the theoretical floor)
    if coherence_guard:
        floor_lin = (10.0 ** (guard_floor_db / 10.0)) if guard_floor_db != 0.0 else 1.0
        S_res_floor = torch.clamp(1.0 - gamma2, min=0.0) * S_yy * floor_lin
        abs2 = (P_clean.conj() * P_clean).real
        need = abs2 < S_res_floor
        if need.any():
            scale = torch.sqrt((S_res_floor + eps)/(abs2 + eps))
            P_clean = torch.where(need, P_clean * scale, P_clean)
            PB_hat = P0 - P_clean

    # ---------- iSTFT ----------
    p_clean = torch.istft(P_clean, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                          window=win, center=center, length=N)
    if preserve_mean:
        p_clean = p_clean + mu_p0

    if return_noise_estimate:
        pb_hat = torch.istft(PB_hat, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                             window=win, center=center, length=N)
        return p_clean, pb_hat
    return p_clean
