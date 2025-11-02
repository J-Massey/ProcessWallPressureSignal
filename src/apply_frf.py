from typing import Optional, Callable, Dict
import numpy as np

def apply_frf(
    x: np.ndarray,
    fs: float,
    f: np.ndarray,
    H: Optional[np.ndarray] = None,        # measured complex FRF (optional when using LEM)
    demean: bool = True,
    zero_dc: bool = True,
    R: float = 1.0,
    *,
    # --- legacy power-law scaling (kept for backward compatibility) ---
    rho: Optional[float] = None,
    scale_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    scale_params: Optional[Dict[str, float]] = None,
    # --- new: LEM-target application ---
    lem_params: Optional[Dict[str, float]] = None,  # {'g_db','fD_Hz','QD'}
    invert_target: bool = True,          # target= scaling_ratio → required |H| = 1/target
    phase_strategy: str = "auto",        # 'auto'|'measured'|'lem'|'minphase'|'zero'
):
    """
    Apply an FRF (x→y) to a time series x to synthesize y.

    Two ways to define the FRF magnitude:
      1) Original path: measured |H| (optionally scaled by power-law S(f, rho))
      2) New path: diaphragm-only LEM magnitude from lem_params
                   (typically inverted to get the required |H| in your pipeline)

    Phase options:
      - 'measured' : use ∠H from measured calibration (recommended when available)
      - 'lem'      : use LEM phase (diaphragm-only, DC zero over 2nd order)
      - 'minphase' : minimum-phase phase reconstructed from the target magnitude
      - 'zero'     : zero phase (symmetric, non-causal in time)
      - 'auto'     : 'measured' if H is provided, else 'lem'
    """
    def _lem_complex(fHz: np.ndarray, fD_Hz: float, QD: float) -> np.ndarray:
        w = 2.0 * np.pi * fHz
        wD = 2.0 * np.pi * fD_Hz
        den = (1.0 - (w / wD)**2) + 1j * (w / (QD * wD))
        return (1j * w) / den  # diaphragm-only: DC zero × 2nd-order poles

    def _minphase_phase_from_mag(fr: np.ndarray, mag: np.ndarray) -> np.ndarray:
        """
        Minimum-phase reconstruction for uniformly spaced fr (0..fs/2).
        Uses real-cepstrum liftering. Returns ∠H_minphase(fr).
        """
        # ensure finite and positive
        mag = np.maximum(mag, 1e-16)
        log_mag = np.log(mag)

        # cepstrum on the rfft grid that matches 'fr'
        Nfft = (len(fr) - 1) * 2
        c = np.fft.irfft(log_mag, n=Nfft)

        # minimum-phase liftering: double positive quefrencies, keep DC & Nyquist
        c_min = c.copy()
        c_min[1:Nfft//2] *= 2.0
        # reconstruct complex spectrum and take phase
        H_min = np.fft.rfft(np.exp(c_min), n=Nfft)
        return np.unwrap(np.angle(H_min))

    # --- input guards
    x = np.asarray(x, float)
    fH = np.asarray(f, float).copy()
    if H is not None:
        H = np.asarray(H)
        if fH.ndim != 1 or H.shape[-1] != fH.shape[0]:
            raise ValueError("f and H must have matching lengths (H along last axis).")

    if lem_params is not None and (scale_fn is not None or scale_params is not None):
        raise ValueError("Pass either lem_params or scale_{fn,params}, not both.")

    if demean:
        x = x - x.mean()

    # FFT grid
    N = x.size
    Nfft = int(2 ** np.ceil(np.log2(N)))
    X = np.fft.rfft(x, n=Nfft)
    fr = np.fft.rfftfreq(Nfft, d=1.0 / fs)

    # ------------------------------------------------------------------
    # 1) Build magnitude on the measured grid fH (later interpolated to fr)
    # ------------------------------------------------------------------
    if lem_params is not None:
        # LEM magnitude on fH
        if not all(k in lem_params for k in ("g_db","fD_Hz","QD")):
            raise ValueError("lem_params must have keys: 'g_db','fD_Hz','QD'.")
        g_db = float(lem_params["g_db"])
        fD   = float(lem_params["fD_Hz"])
        QD   = float(lem_params["QD"])

        Hlem = _lem_complex(fH, fD_Hz=fD, QD=QD)          # complex, unitless shape
        mag  = np.abs(Hlem) * (10.0**(g_db/20.0))          # amplitude target on fH

        # In most pipelines you want required |H| = 1 / scaling_ratio
        if invert_target:
            mag = 1.0 / np.maximum(mag, 1e-16)

        # Phase selection
        if phase_strategy == "auto":
            use = "measured" if H is not None else "lem"
        else:
            use = phase_strategy.lower()

        if use == "measured":
            if H is None:
                raise ValueError("phase_strategy='measured' requires H (measured FRF).")
            phi = np.unwrap(np.angle(H))
        elif use == "lem":
            # LEM phase for diaphragm-only section
            phi = np.unwrap(np.angle(Hlem))
        elif use == "minphase":
            # compute on FFT grid, so interpolate mag to fr first
            mag_i = np.interp(fr, fH, mag, left=mag[0], right=mag[-1])
            phi_i = _minphase_phase_from_mag(fr, mag_i)
        elif use == "zero":
            phi = np.zeros_like(fH)
        else:
            raise ValueError("phase_strategy must be 'auto','measured','lem','minphase','zero'.")

    else:
        # --- original path: measured H (+ optional rho–f scaling) ---
        if H is None:
            raise ValueError("Either pass (H) or (lem_params).")
        fH[0] = max(fH[0], 1.0)  # avoid DC ratio/log issues
        mag = np.abs(H).astype(float)
        phi = np.unwrap(np.angle(H))

        # optional rho–f power-law scaling
        if scale_fn is not None or scale_params is not None:
            if rho is None:
                raise ValueError("rho must be provided when applying rho–f scaling.")
            if scale_fn is not None:
                S = np.asarray(scale_fn(fH, float(rho)), float)
            else:
                c0_db = float(scale_params["c0_db"])
                a = float(scale_params["a"])
                b = float(scale_params["b"])
                rho_ref = float(scale_params["rho_ref"])
                f_ref = float(scale_params["f_ref"])
                S_db = c0_db \
                     + a * (20.0 * np.log10(float(rho) / rho_ref)) \
                     + b * (20.0 * np.log10(fH / f_ref))
                S = 10.0 ** (S_db / 20.0)
            S = np.clip(S, 0.0, np.inf)
            mag *= S

    # constant gain
    mag *= float(R)

    # DC handling on design grid
    if H is not None or lem_params is not None:
        if fH.size > 0:
            mag[0] = 0.0

    # ------------------------------------------------------------------
    # 2) Interpolate to FFT grid and form complex FRF
    # ------------------------------------------------------------------
    if lem_params is not None and phase_strategy.lower() == "minphase":
        # Already produced mag_i and phi_i on fr above
        pass
    else:
        # interpolate magnitude and phase from fH to fr
        mag_i = np.interp(fr, fH, mag, left=1.0, right=1.0)
        phi_i = np.interp(fr, fH, phi, left=(phi[0] if np.size(mag)>0 else 0.0),
                                   right=(phi[-1] if np.size(mag)>0 else 0.0))
    Hi = mag_i * np.exp(1j * phi_i)

    if zero_dc:
        Hi[0] = 0.0
        if Nfft % 2 == 0:
            Hi[-1] = 0.0

    y = np.fft.irfft(X * Hi, n=Nfft)[:N]
    return y
