import numpy as np
import scipy.io as sio
from scipy.signal import welch, find_peaks, peak_widths


def compute_psd(signal, fs, nperseg=None, noverlap=None):
    """One-sided PSD via Welch."""
    if nperseg is None:
        nperseg = len(signal) // 2000
    if noverlap is None:
        noverlap = nperseg // 2
    return welch(signal, fs=fs, window="hann",
                 nperseg=nperseg, noverlap=noverlap)

def propagate_error(f_raw, psd, nu0, rho0, u_tau0, err_frac):
    """Compute nominal and ±error bounds for f+ and Phi+."""
    nu_min, nu_max     = nu0*(1-err_frac), nu0*(1+err_frac)
    u_tau_min, u_tau_max = u_tau0*(1-err_frac), u_tau0*(1+err_frac)
    rho_min, rho_max   = rho0*(1-err_frac), rho0*(1+err_frac)

    f_nom = f_raw * nu0 / u_tau0**2
    f_min = f_raw * nu_min / u_tau_max**2
    f_max = f_raw * nu_max / u_tau_min**2

    denom_nom = (rho0*u_tau0**2)**2
    denom_min = (rho_max*u_tau_max**2)**2
    denom_max = (rho_min*u_tau_min**2)**2

    phi_nom = f_raw * psd / denom_nom
    phi_min = f_raw * psd / denom_min
    phi_max = f_raw * psd / denom_max

    return {"f_nom": f_nom, "f_min": f_min, "f_max": f_max,
            "phi_nom": phi_nom, "phi_min": phi_min, "phi_max": phi_max}

def duct_mode_freq(U, c, m, n, l, W, H, L):
    """Physical duct-mode frequency (quarter-wave, closed-open)."""
    delta2 = c**2 - U**2
    k_sq   = (m*np.pi/W)**2 + (n*np.pi/H)**2
    kz_sq  = ((2*l+1)*np.pi/L)**2
    return (1/(2*np.pi)) * np.sqrt(delta2*k_sq + delta2**2/(4*c**2)*kz_sq)

def compute_duct_modes(U, c, mode_m, mode_n, mode_l, W, H, L, nu0, u_tau0, err_frac):
    """Compute non-dimensional duct modes (nom, min, max)."""
    nu_min, nu_max     = nu0*(1-err_frac), nu0*(1+err_frac)
    u_tau_min, u_tau_max = u_tau0*(1-err_frac), u_tau0*(1+err_frac)
    freqs = {"nom": [], "min": [], "max": []}
    for m in mode_m:
        for n in mode_n:
            for l in mode_l:
                f_phys = duct_mode_freq(U,c,m,n,l,W,H,L)
                freqs["nom"].append(f_phys*nu0/u_tau0**2)
                freqs["min"].append(f_phys*nu_min/u_tau_max**2)
                freqs["max"].append(f_phys*nu_max/u_tau_min**2)
    return freqs

def notch_filter(f_nom, phi_nom, f_min, f_max, mode_freqs,
                 min_height=None, prominence=0.001, rel_height=0.9):
    """Notch out the largest PSD peak around each duct-mode."""
    peaks, props = find_peaks(phi_nom, height=min_height, prominence=prominence)
    if not peaks.size:
        return phi_nom.copy(), []
    widths, _, left_ips, right_ips = peak_widths(phi_nom, peaks, rel_height=rel_height)
    idx = np.arange(len(f_nom))
    phi_filt = phi_nom.copy()
    info = []
    for f0 in mode_freqs:
        band = (f_min <= f0) & (f_max >= f0)
        if not np.any(band):
            continue
        low, high = f_nom[band][0], f_nom[band][-1]
        in_band = peaks[(f_nom[peaks]>=low)&(f_nom[peaks]<=high)]
        if not in_band.size:
            continue
        pk = in_band[np.argmax(phi_nom[in_band])]
        j = np.where(peaks==pk)[0][0]
        fl = np.interp(left_ips[j], idx, f_nom)
        fr = np.interp(right_ips[j], idx, f_nom)
        info.append({"mode_freq":f0, "peak_freq":f_nom[pk], "f_left":fl, "f_right":fr})
        mask = (f_nom>=fl)&(f_nom<=fr)
        phi_filt[mask] = np.interp(
            f_nom[mask],
            [fl, fr],
            [phi_nom[int(np.floor(left_ips[j]))],
             phi_nom[int(np.ceil (right_ips[j]))]]
        )
    return phi_filt, info