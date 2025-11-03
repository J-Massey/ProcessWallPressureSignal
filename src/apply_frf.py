from typing import Optional, Callable, Dict
import numpy as np

# def apply_frf(
#     x: np.ndarray,
#     fs: float,
#     f: np.ndarray,
#     H: np.ndarray,
#     demean: bool = True,
#     zero_dc: bool = True,
#     R: float = 1.0,
#     *,
#     rho: Optional[float] = None,
#     scale_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
#     scale_params: Optional[Dict[str, float]] = None,
# ):
#     """
#     Apply a measured FRF H (x→y) to a time series x to synthesise y, with optional
#     rho–frequency magnitude scaling S(f, rho) that you fitted earlier.

#     If provided, the magnitude used becomes:
#         |H|(f) * S(f, rho) * R
#     where S is either:
#         - scale_fn(f, rho)  (callable returning a linear multiplier), or
#         - computed from scale_params = {'c0_db','a','b','rho_ref','f_ref'}.

#     Out-of-band behavior: magnitude is set to unity outside the measured band,
#     phase is linearly extrapolated from endpoints.
#     """
#     x = np.asarray(x, float)
#     fH = np.asarray(f, float).copy()
#     H = np.asarray(H)

#     if demean:
#         x = x - x.mean()

#     if fH.ndim != 1 or H.shape[-1] != fH.shape[0]:
#         raise ValueError("f and H must have matching lengths (H along last axis).")

#     N = x.size
#     Nfft = int(2 ** np.ceil(np.log2(N)))
#     X = np.fft.rfft(x, n=Nfft)
#     fr = np.fft.rfftfreq(Nfft, d=1.0 / fs)

#     # Base magnitude/phase from calibration FRF
#     fH[0] = max(fH[0], 1.0)  # avoid log/ratio issues at DC
#     mag = np.abs(H).astype(float)
#     phi = np.unwrap(np.angle(H))

#     # Optional rho–frequency scaling
#     if scale_fn is not None or scale_params is not None:
#         if rho is None:
#             raise ValueError("rho must be provided when applying rho–f scaling.")
#         if scale_fn is not None:
#             S = np.asarray(scale_fn(fH, float(rho)), float)
#         else:
#             # scale_params: {'c0_db','a','b','rho_ref','f_ref'}
#             c0_db = float(scale_params["c0_db"])
#             a = float(scale_params["a"])
#             b = float(scale_params["b"])
#             rho_ref = float(scale_params["rho_ref"])
#             f_ref = float(scale_params["f_ref"])
#             S_db = c0_db \
#                  + a * (20.0 * np.log10(float(rho) / rho_ref)) \
#                  + b * (20.0 * np.log10(fH / f_ref))
#             S = 10.0 ** (S_db / 20.0)
#         # keep magnitudes finite & non-negative
#         S = np.clip(S, 0.0, np.inf)
#         mag *= S

#     # Optional extra constant factor (kept from your signature)
#     mag *= float(R)

#     # Enforce DC handling on the measured grid
#     mag[0] = 0.0

#     # Interpolate H onto the FFT grid; unity magnitude outside band
#     mag_i = np.interp(fr, fH, mag, left=1.0, right=1.0)
#     phi_i = np.interp(fr, fH, phi, left=phi[0], right=phi[-1])
#     Hi = mag_i * np.exp(1j * phi_i)

#     if zero_dc:
#         Hi[0] = 0.0
#         if Nfft % 2 == 0:
#             Hi[-1] = 0.0

#     y = np.fft.irfft(X * Hi, n=Nfft)[:N]
#     return y


def apply_frf(
    x: np.ndarray,
    fs: float,
    f: np.ndarray,
    H: np.ndarray,
    demean: bool = True,
    zero_dc: bool = True,
):
    """
    Apply a measured FRF H (x→y) to a time series x to synthesise y.
    This is the forward operation: Y = H · X in the frequency domain.
    """
    x = np.asarray(x, float)
    if demean:
        x = x - x.mean()

    N = x.size
    Nfft = int(2 ** np.ceil(np.log2(N)))
    X = np.fft.rfft(x, n=Nfft)
    fr = np.fft.rfftfreq(Nfft, d=1.0 / fs)

    mag = np.abs(H)
    phi = np.unwrap(np.angle(H))
    # Safer OOB behaviour: taper magnitude to zero outside measured band
    mag_i = np.interp(fr, f, mag, left=1.0, right=1.0)
    phi_i = np.interp(fr, f, phi, left=phi[0], right=phi[-1])
    Hi = mag_i * np.exp(1j * phi_i)

    if zero_dc:
        Hi[0] = 0.0
        if Nfft % 2 == 0:
            Hi[-1] = 0.0

    y = np.fft.irfft(X * Hi, n=Nfft)[:N]
    return y