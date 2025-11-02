# tf_compute.py
from __future__ import annotations

import os
from typing import Optional, Callable, Dict, Tuple

import numpy as np
import h5py
from scipy.signal import welch, csd, get_window

# ---------------------------------------------------------------------
# Constants (exported so your other scripts can import them)
# ---------------------------------------------------------------------
FS: float = 50_000.0
NPERSEG: int = 2**10          # matches your current pipeline
WINDOW: str = "hann"

# Colors (kept for plotting modules that import these)
PH1_COLOR = "#c76713"  # orange
PH2_COLOR = "#9fda16"  # green-ish
NC_COLOR  = "#2ca02c"  # mpl green

# I/O bases used by your current workflow
CLEANED_BASE = "data/final_cleaned/"
TARGET_BASE  = "data/final_target/"

# ---------------------------------------------------------------------
# Welch helpers
# ---------------------------------------------------------------------
def estimate_frf(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    window: str = WINDOW,
    npsg: int = NPERSEG,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    H1 FRF and coherence via Welch/CSD.
        H1 = conj(Sxy)/Sxx    (x -> y)
    Returns f [Hz], H (complex), gamma2 [0..1]
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    nseg = int(min(npsg, x.size, y.size))
    if nseg < 8:
        raise ValueError(f"Signal too short for FRF: n={min(x.size, y.size)}")
    nov = int(min(npsg // 2, nseg // 2))
    w = get_window(window, nseg, fftbins=True)

    f, Sxx = welch(x, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)
    _, Syy = welch(y, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)
    # SciPy: csd(x,y) = E{ X * conj(Y) }
    _, Sxy = csd(x, y, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)

    H = np.conj(Sxy) / Sxx
    gamma2 = (np.abs(Sxy) ** 2) / (Sxx * Syy)
    gamma2 = np.clip(gamma2.real, 0.0, 1.0)
    return f, H, gamma2


def compute_spec(fs: float, x: np.ndarray, npsg: int = NPERSEG) -> Tuple[np.ndarray, np.ndarray]:
    """
    Welch PSD with sane defaults and shape guarding.
    Returns (f [Hz], Pxx [units^2/Hz]).
    """
    x = np.asarray(x, float)
    nseg = int(min(npsg, x.size))
    nov = nseg // 2
    w = get_window(WINDOW, nseg, fftbins=True)
    f, Pxx = welch(
        x,
        fs=fs,
        window=w,
        nperseg=nseg,
        noverlap=nov,
        detrend="constant",
        scaling="density",
        return_onesided=True,
    )
    return f, Pxx

# ---------------------------------------------------------------------
# Target builder (exactly the ratio you described): sqrt(data/model)
# ---------------------------------------------------------------------
# Uses your boundary-layer model g(T+) from `models.bl_model`
from models import bl_model  # your local module that provides the BL spectral shape

def save_scaling_target():
    """
    Build and save the per-series target ratio:
        scaling_ratio(f) = sqrt( data / model )  (amplitude)
    for each cleaned dataset in CLEANED_BASE, then write:
        TARGET_BASE/target_{label}.h5
      with datasets: 'frequencies', 'scaling_ratio'
      and attrs: rho, u_tau, nu, psig
    """
    fns = [
        '0psig_close_cleaned.h5',
        '0psig_far_cleaned.h5',
        '50psig_close_cleaned.h5',
        '50psig_far_cleaned.h5',
        '100psig_close_cleaned.h5',
        '100psig_far_cleaned.h5',
    ]
    labels = [
        '0psig_close',
        '0psig_far',
        '50psig_close',
        '50psig_far',
        '100psig_close',
        '100psig_far',
    ]
    pgs = [0, 0, 50, 50, 100, 100]

    for idx, fn in enumerate(fns):
        with h5py.File(os.path.join(CLEANED_BASE, fn), 'r') as hf:
            ph1 = hf['ph1_clean'][:]
            ph2 = hf['ph2_clean'][:]
            u_tau = float(hf.attrs['u_tau'])
            nu    = float(hf.attrs['nu'])
            rho   = float(hf.attrs['rho'])
            Re_tau = float(hf.attrs['Re_tau'])
            cf_2   = float(hf.attrs['cf_2'])

        # Spectra
        f1, P11 = compute_spec(FS, ph1)
        f2, P22 = compute_spec(FS, ph2)
        # Normalize: f * Pyy / (rho^2 u_tau^4)
        D1 = (f1 * P11) / (rho**2 * u_tau**4)
        D2 = (f2 * P22) / (rho**2 * u_tau**4)

        # Model BL spectrum on same f-grid using your g(T+)
        Tplus1 = (u_tau**2) / (nu * f1)
        g1, g2, rv = bl_model(Tplus1, Re_tau, cf_2)
        M = rv * (g1 + g2)  # model on f1 grid

        # Band-limit to 100–1000 Hz (Helmholtz safely out of band)
        band = (f1 < 1000.0) & (f1 >= 100.0)
        f_out = f1[band]
        M_out = M[band]
        D1_out = D1[band]
        # interp channel 2 onto f1 grid then clip
        D2i = np.interp(f1, f2, D2)[band]

        # Target AMPLITUDE ratio = sqrt(data/model), average over PH1/PH2
        r1 = np.sqrt(np.maximum(D1_out, 0.0) / np.maximum(M_out, 1e-30))
        r2 = np.sqrt(np.maximum(D2i,    0.0) / np.maximum(M_out, 1e-30))
        scaling_ratio = 0.5 * (r1 + r2)

        # Save target for this series
        outp = os.path.join(TARGET_BASE, f"target_{labels[idx]}.h5")
        with h5py.File(outp, 'w') as hf:
            hf.create_dataset('frequencies',    data=f_out)
            hf.create_dataset('scaling_ratio',  data=scaling_ratio)
            hf.attrs['rho']  = rho
            hf.attrs['u_tau'] = u_tau
            hf.attrs['nu']   = nu
            hf.attrs['psig'] = pgs[idx]

# ---------------------------------------------------------------------
# FRF application with optional (rho,f)-dependent magnitude scaling
# ---------------------------------------------------------------------
def apply_frf(
    x: np.ndarray,
    fs: float,
    f: np.ndarray,
    H: np.ndarray,
    demean: bool = True,
    zero_dc: bool = True,
    R: float = 1.0,
    *,
    rho: Optional[float] = None,
    scale_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    scale_params: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Apply a measured FRF H (x -> y) to a time series x to synthesize y, with optional
    rho–frequency magnitude scaling S(f, rho) that you fitted (power-law).

    Effective magnitude used:
        |H|(f) * S(f, rho) * R
    where S is either:
        - scale_fn(f, rho)    # callable returning linear amplitude multiplier, or
        - computed from scale_params={'c0_db','a','b','rho_ref','f_ref'}.

    Out-of-band behavior: magnitude -> 1 outside measured band;
    phase is linearly extrapolated from endpoints.
    """
    x = np.asarray(x, float)
    fH = np.asarray(f, float).copy()
    H  = np.asarray(H)

    if demean:
        x = x - x.mean()

    if fH.ndim != 1 or H.shape[-1] != fH.shape[0]:
        raise ValueError("f and H must have matching lengths (H along last axis).")

    N = x.size
    Nfft = int(2 ** np.ceil(np.log2(N)))
    X = np.fft.rfft(x, n=Nfft)
    fr = np.fft.rfftfreq(Nfft, d=1.0 / fs)

    # Measured FRF: base magnitude & phase
    fH[0] = max(fH[0], 1.0)  # avoid log(0)
    mag = np.abs(H).astype(float)
    phi = np.unwrap(np.angle(H))

    # Optional rho–f magnitude scaling (power-law)
    if scale_fn is not None or scale_params is not None:
        if rho is None:
            raise ValueError("rho must be provided when applying rho–f scaling.")
        if scale_fn is not None:
            S = np.asarray(scale_fn(fH, float(rho)), float)
        else:
            c0_db  = float(scale_params["c0_db"])
            a      = float(scale_params["a"])
            b      = float(scale_params["b"])
            rho_ref = float(scale_params["rho_ref"])
            f_ref   = float(scale_params["f_ref"])
            S_db = (c0_db
                    + a * (20.0*np.log10(float(rho) / rho_ref))
                    + b * (20.0*np.log10(fH / f_ref)))
            S = 10.0 ** (S_db / 20.0)
        S = np.clip(S, 0.0, np.inf)
        mag *= S

    mag *= float(R)
    mag[0] = 0.0  # enforce DC behavior on the measured grid

    # Interpolate onto FFT grid
    mag_i = np.interp(fr, fH, mag, left=1.0, right=1.0)
    phi_i = np.interp(fr, fH, phi, left=phi[0], right=phi[-1])
    Hi = mag_i * np.exp(1j * phi_i)

    if zero_dc:
        Hi[0] = 0.0
        if Nfft % 2 == 0:
            Hi[-1] = 0.0

    y = np.fft.irfft(X * Hi, n=Nfft)[:N]
    return y


# ---------------------------------------------------------------------
# Script entry for (re)building targets if you want it as a one-off
# ---------------------------------------------------------------------
if __name__ == "__main__":
    save_scaling_target()
