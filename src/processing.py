import torch



def compute_psd(signal, fs, nperseg=None, noverlap=None):
    """One-sided PSD via Welch computed with PyTorch."""
    signal = torch.as_tensor(signal)
    if nperseg is None:
        nperseg = len(signal) // 2000
    if nperseg < 1:
        nperseg = len(signal)
    if noverlap is None:
        noverlap = nperseg // 2

    step = nperseg - noverlap
    window = torch.hann_window(nperseg, dtype=signal.dtype, device=signal.device)
    U = window.pow(2).sum()
    num = 0
    pxx = None
    for start in range(0, len(signal) - nperseg + 1, step):
        seg = signal[start:start + nperseg] * window
        seg_fft = torch.fft.rfft(seg)
        spec = seg_fft.abs() ** 2 / (fs * U)
        pxx = spec if pxx is None else pxx + spec
        num += 1
    pxx = pxx / max(1, num)
    freqs = torch.fft.rfftfreq(nperseg, 1 / fs)
    return freqs, pxx

def compute_csd(signal1, signal2, fs, nperseg=None, noverlap=None):
    """Cross-spectral density computed with PyTorch."""
    signal1 = torch.as_tensor(signal1)
    signal2 = torch.as_tensor(signal2)
    if nperseg is None:
        nperseg = len(signal1) // 2000
    if nperseg < 1:
        nperseg = len(signal1)
    if noverlap is None:
        noverlap = nperseg // 2

    step = nperseg - noverlap
    window = torch.hann_window(nperseg, dtype=signal1.dtype, device=signal1.device)
    U = window.pow(2).sum()
    num = 0
    csd_accum = None
    for start in range(0, len(signal1) - nperseg + 1, step):
        seg1 = signal1[start:start + nperseg] * window
        seg2 = signal2[start:start + nperseg] * window
        fft1 = torch.fft.rfft(seg1)
        fft2 = torch.fft.rfft(seg2)
        spec = fft1.conj() * fft2 / (fs * U)
        csd_accum = spec if csd_accum is None else csd_accum + spec
        num += 1
    csd_avg = csd_accum / max(1, num)
    freqs = torch.fft.rfftfreq(nperseg, 1 / fs)
    return freqs, csd_avg

def propagate_frequency_error(f_duct, nu0, u_tau0, err_frac):
    """Compute nominal and ±error bounds for f_duct"""
    nu_min, nu_max     = nu0*(1-err_frac), nu0*(1+err_frac)
    u_tau_min, u_tau_max = u_tau0*(1-err_frac), u_tau0*(1+err_frac)

    f_nom = f_duct * nu0 / u_tau0**2
    f_min = f_duct * nu_min / u_tau_max**2
    f_max = f_duct * nu_max / u_tau_min**2
    if not torch.is_tensor(f_nom):
        f_nom = torch.tensor(f_nom)
    if not torch.is_tensor(f_min):
        f_min = torch.tensor(f_min)
    if not torch.is_tensor(f_max):
        f_max = torch.tensor(f_max)

    return f_nom, f_min, f_max

def propagate_PSD_error(f_raw, psd, nu0, rho0, u_tau0, err_frac):
    """Compute nominal and ±error bounds for f+ and Phi+."""
    nu_min, nu_max     = nu0*(1-err_frac), nu0*(1+err_frac)
    u_tau_min, u_tau_max = u_tau0*(1-err_frac), u_tau0*(1+err_frac)
    rho_min, rho_max   = rho0*(1-err_frac), rho0*(1+err_frac)

    denom_nom = (rho0*u_tau0**2)**2
    denom_min = (rho_max*u_tau_max**2)**2
    denom_max = (rho_min*u_tau_min**2)**2

    phi_nom = f_raw * psd / denom_nom
    phi_min = f_raw * psd / denom_min
    phi_max = f_raw * psd / denom_max
    if not torch.is_tensor(phi_nom):
        phi_nom = torch.tensor(phi_nom)
    if not torch.is_tensor(phi_min):
        phi_min = torch.tensor(phi_min)
    if not torch.is_tensor(phi_max):
        phi_max = torch.tensor(phi_max)

    return phi_nom, phi_min, phi_max

def duct_mode_freq(U, c, m, n, l, W, H, L):
    """Physical duct-mode frequency (quarter-wave, closed-open)."""
    delta2 = c**2 - U**2
    k_sq   = (m*torch.pi/W)**2 + (n*torch.pi/H)**2
    kz_sq  = ((2*l+1)*torch.pi/L)**2
    return (1/(2*torch.pi)) * torch.sqrt(delta2*k_sq + delta2**2/(4*c**2)*kz_sq)

def compute_duct_modes(U, c, mode_m, mode_n, mode_l, W, H, L, nu0, u_tau0, err_frac):
    """Compute non-dimensional duct modes (nom, min, max)."""
    nu_min, nu_max     = nu0*(1-err_frac), nu0*(1+err_frac)
    u_tau_min, u_tau_max = u_tau0*(1-err_frac), u_tau0*(1+err_frac)
    freqs = {"nom": [], "min": [], "max": []}
    for m in mode_m:
        for n in mode_n:
            for l in mode_l:
                f_phys = duct_mode_freq(U,c,m,n,l,W,H,L)
                freqs["nom"].append(float(f_phys*nu0/u_tau0**2))
                freqs["min"].append(float(f_phys*nu_min/u_tau_max**2))
                freqs["max"].append(float(f_phys*nu_max/u_tau_min**2))
    for k in freqs:
        freqs[k] = torch.tensor(freqs[k])
    return freqs

def _find_peaks(x):
    """Return indices of local maxima in 1-D tensor."""
    cond = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
    return torch.nonzero(cond).view(-1) + 1


def notch_filter_fourier(f_nom, phi_nom, f_min, f_max, mode_freqs,
                 min_height=None, prominence=0.001, rel_height=0.9):
    """Notch out the largest PSD peak around each duct-mode using PyTorch."""
    f_nom = torch.as_tensor(f_nom)
    phi_nom = torch.as_tensor(phi_nom)
    f_min = torch.as_tensor(f_min)
    f_max = torch.as_tensor(f_max)
    mode_freqs = torch.as_tensor(mode_freqs)

    peaks = _find_peaks(phi_nom)
    if peaks.numel() == 0:
        return phi_nom.clone(), []

    phi_filt = phi_nom.clone()
    info = []
    for f0, fmin, fmax in zip(mode_freqs, f_min, f_max):
        band = (f_nom >= fmin) & (f_nom <= fmax)
        band_peaks = peaks[band[peaks]]
        if band_peaks.numel() == 0:
            continue
        pk = band_peaks[torch.argmax(phi_nom[band_peaks])]
        peak_val = phi_nom[pk]
        left = pk
        while left > 0 and phi_nom[left] > peak_val * rel_height:
            left -= 1
        right = pk
        while right < len(phi_nom) - 1 and phi_nom[right] > peak_val * rel_height:
            right += 1
        fl = f_nom[left]
        fr = f_nom[right]
        info.append({
            "mode_freq": float(f0),
            "peak_freq": float(f_nom[pk]),
            "f_left": float(fl),
            "f_right": float(fr),
        })
        phi_filt[left:right + 1] = torch.linspace(
            phi_nom[left], phi_nom[right], right - left + 1, device=phi_nom.device
        )
    return phi_filt, info

def notch_filter_timeseries(p, fs,
                            f_duct_min, f_duct_max, f_duct_nom,
                            min_height=None, prominence=0.002, rel_height=0.5):
    """Notch out the largest PSD peak around each duct-mode frequency."""
    p = torch.as_tensor(p)
    N = len(p)
    nperseg = max(256, N // 2000)
    noverlap = nperseg // 2

    f_nom, Pxx_nom = compute_psd(p, fs, nperseg=nperseg, noverlap=noverlap)
    phi_filt, info = notch_filter_fourier(
        f_nom, Pxx_nom, f_duct_min, f_duct_max, f_duct_nom,
        min_height=min_height, prominence=prominence, rel_height=rel_height
    )

    freqs = torch.fft.rfftfreq(N, 1 / fs)
    X = torch.fft.rfft(p)
    for fmin, fmax in zip(f_duct_min, f_duct_max):
        band = (freqs >= fmin) & (freqs <= fmax)
        X[band] = 0
    x_filt = torch.fft.irfft(X, n=N)

    f_filt, Pxx_filt = compute_psd(x_filt, fs, nperseg=nperseg, noverlap=noverlap)
    return x_filt, f_nom, Pxx_nom, f_filt, Pxx_filt, info
