import torch
import scipy.io as sio
from scipy.signal import welch, find_peaks, peak_widths, csd, iirnotch, filtfilt
from icecream import ic


def compute_psd(signal, fs, nperseg=None, noverlap=None):
    """One-sided PSD via Welch."""
    if nperseg is None:
        nperseg = len(signal) // 2000
    if noverlap is None:
        noverlap = nperseg // 2
    if torch.is_tensor(signal):
        signal_np = signal.detach().cpu().numpy()
    else:
        signal_np = signal
    f, pxx = welch(signal_np, fs=fs, window="hann",
                   nperseg=nperseg, noverlap=noverlap)
    return torch.tensor(f), torch.tensor(pxx)

def compute_csd(signal1, signal2, fs, nperseg=None, noverlap=None):
    """One-sided PSD via Welch."""
    if nperseg is None:
        nperseg = len(signal1) // 2000
    if noverlap is None:
        noverlap = nperseg // 2
    if torch.is_tensor(signal1):
        sig1_np = signal1.detach().cpu().numpy()
    else:
        sig1_np = signal1
    if torch.is_tensor(signal2):
        sig2_np = signal2.detach().cpu().numpy()
    else:
        sig2_np = signal2
    f, Pxy = csd(sig1_np, sig2_np, fs=fs, window="hann",
                 nperseg=nperseg, noverlap=noverlap)
    return torch.tensor(f), torch.tensor(Pxy)

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

def notch_filter_fourier(f_nom, phi_nom, f_min, f_max, mode_freqs,
                 min_height=None, prominence=0.001, rel_height=0.9):
    """Notch out the largest PSD peak around each duct-mode."""
    if torch.is_tensor(phi_nom):
        phi_nom_np = phi_nom.detach().cpu().numpy()
        f_nom_np = f_nom.detach().cpu().numpy()
        f_min_np = f_min.detach().cpu().numpy()
        f_max_np = f_max.detach().cpu().numpy()
    else:
        phi_nom_np = phi_nom
        f_nom_np = f_nom
        f_min_np = f_min
        f_max_np = f_max
    peaks, props = find_peaks(phi_nom_np, height=min_height, prominence=prominence)
    if not peaks.size:
        return phi_nom.clone() if torch.is_tensor(phi_nom) else phi_nom.copy(), []
    widths, _, left_ips, right_ips = peak_widths(phi_nom_np, peaks, rel_height=rel_height)
    idx = np.arange(len(f_nom_np))
    phi_filt_np = phi_nom_np.copy()
    info = []
    for f0, fmin, fmax in zip(f_nom_np, f_min_np, f_max_np):
        band = (f_nom_np >= fmin) & (f_nom_np <= fmax)
        if not np.any(band):
            continue
        low, high = f_nom_np[band][0], f_nom_np[band][-1]
        in_band = peaks[(f_nom_np[peaks]>=low)&(f_nom_np[peaks]<=high)]
        if not in_band.size:
            continue
        pk = in_band[np.argmax(phi_nom_np[in_band])]
        j = np.where(peaks==pk)[0][0]
        fl = np.interp(left_ips[j], idx, f_nom_np)
        fr = np.interp(right_ips[j], idx, f_nom_np)
        info.append({"mode_freq":float(f0), "peak_freq":float(f_nom_np[pk]), "f_left":float(fl), "f_right":float(fr)})
        mask = (f_nom_np>=fl)&(f_nom_np<=fr)
        phi_filt_np[mask] = np.interp(
            f_nom_np[mask],
            [fl, fr],
            [phi_nom_np[int(np.floor(left_ips[j]))],
             phi_nom_np[int(np.ceil (right_ips[j]))]]
        )
    if torch.is_tensor(phi_nom):
        phi_filt = torch.tensor(phi_filt_np)
    else:
        phi_filt = phi_filt_np
    return phi_filt, info

def notch_filter_timeseries(p, fs,
                            f_duct_min, f_duct_max, f_duct_nom,
                            min_height=None, prominence=0.002, rel_height=0.5):
    """
    Notch out the largest PSD peak around each duct-mode frequency.
    Returns:
      x_filt     : filtered time series
      f_nom,Pxx_nom : original PSD
      f_filt,Pxx_filt : PSD after notch
      info       : list of dicts with notch details
    """
    N = len(p)
    nperseg = max(256, N // 2000)
    noverlap = nperseg // 2

    # original PSD
    if torch.is_tensor(p):
        p_np = p.detach().cpu().numpy()
    else:
        p_np = p
    f_nom_np, Pxx_nom_np = welch(p_np, fs=fs, nperseg=nperseg, noverlap=noverlap)
    f_nom = torch.tensor(f_nom_np)
    Pxx_nom = torch.tensor(Pxx_nom_np)

    # find peaks
    peaks, _ = find_peaks(Pxx_nom_np, height=min_height, prominence=prominence)
    if peaks.size == 0:
        x_copy = p.clone() if torch.is_tensor(p) else p.copy()
        return x_copy, (f_nom, Pxx_nom), (None, None), []

    x_filt = p.clone() if torch.is_tensor(p) else p.copy()
    x_filt_np = x_filt.detach().cpu().numpy() if torch.is_tensor(x_filt) else x_filt
    info = []

    f_duct_nom_np = f_duct_nom.detach().cpu().numpy() if torch.is_tensor(f_duct_nom) else f_duct_nom
    f_duct_min_np = f_duct_min.detach().cpu().numpy() if torch.is_tensor(f_duct_min) else f_duct_min
    f_duct_max_np = f_duct_max.detach().cpu().numpy() if torch.is_tensor(f_duct_max) else f_duct_max
    for f0, fmin, fmax in zip(f_duct_nom_np, f_duct_min_np, f_duct_max_np):
        mask = (f_nom_np >= fmin) & (f_nom_np <= fmax)
        in_band = peaks[mask[peaks]]
        if in_band.size == 0:
            continue

        # pick largest peak
        pk = in_band[np.argmax(Pxx_nom_np[in_band])]
        widths, _, left_ips, right_ips = peak_widths(Pxx_nom_np, [pk], rel_height=rel_height)
        fl = np.interp(left_ips[0], np.arange(len(f_nom_np)), f_nom_np)
        fr = np.interp(right_ips[0], np.arange(len(f_nom_np)), f_nom_np)
        bw = fr - fl
        Q = f_nom_np[pk] / bw if bw > 0 else np.inf

        # notch filter design
        w0 = f_nom_np[pk] / (fs/2)
        b, a = iirnotch(w0, Q)
        x_filt_np = filtfilt(b, a, x_filt_np)

        info.append({
            "mode_freq": float(f0),
            "peak_freq": float(f_nom_np[pk]),
            "f_left": float(fl),
            "f_right": float(fr),
            "Q": float(Q)
        })

    # PSD of filtered signal
    f_filt_np, Pxx_filt_np = welch(x_filt_np, fs=fs, nperseg=nperseg, noverlap=noverlap)
    f_filt = torch.tensor(f_filt_np)
    Pxx_filt = torch.tensor(Pxx_filt_np)
    x_filt = torch.tensor(x_filt_np) if torch.is_tensor(p) else x_filt_np
    return x_filt, f_nom, Pxx_nom, f_filt, Pxx_filt, info
