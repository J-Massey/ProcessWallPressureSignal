# tf_compute.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple
import inspect

import numpy as np
import h5py
from scipy.io import loadmat
from scipy.signal import welch, csd, get_window, iirnotch, sosfiltfilt
from scipy.interpolate import UnivariateSpline

from icecream import ic
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import scienceplots
plt.style.use(["science", "grid"])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")

# =============================================================================
# Constants & styling (exported so tf_plot.py can import them)
# =============================================================================
FS: float = 50_000.0
NPERSEG: int = 2**12
WINDOW: str = "hann"

# Colors (exported for plotting)
PH1_COLOR = "#c76713"  # orange
PH2_COLOR = "#9fda16"  # green-ish
NC_COLOR = "#2ca02c"   # matplotlib default green (kept for reference)

# --- constants (keep once, top of file) ---
R = 287.05        # J/kg/K
PSI_TO_PA = 6_894.76
P_ATM = 101_325.0
DELTA = 0.035  # m, bl-height of 'channel'
TDEG = [18, 20, 22]

# =============================================================================
# Units & optional conversions (kept for compatibility with other workflows)
# =============================================================================


SENSITIVITIES_V_PER_PA: dict[str, float] = {
    'nc': 50e-3,
    'PH1': 50e-3,
    'PH2': 50e-3,
    'NC': 50e-3,
}
PREAMP_GAIN: dict[str, float] = {"nc": 1.0, "PH1": 1.0, "PH2": 1.0, "NC": 1.0}
TONAL_BASE = "data/2025-10-28/tonal/"

def bl_model(Tplus, Re_tau: float, cf_2: float) -> np.ndarray:
    A1 = 2.2
    sig1 = 3.9
    mean_Tplus = 20
    A2 = 1.4 * (np.log10(Re_tau) - 2.2)
    sig2 = 1.2
    mean_To = 0.82
    r1 = 0.5
    r2 = 7
    rv = np.exp(r1 * Tplus)/(np.exp(r1*r2) + np.exp(r1 * Tplus)) # correct
    rv = np.nan_to_num(rv, nan=1)  # replace NaNs with 0
    mean_To_plus = mean_To * Re_tau * np.sqrt(cf_2)
    g1 = A1 * np.exp(-sig1 * (np.log10(Tplus) - np.log10(mean_Tplus))**2)
    g2 = A2 * np.exp(-sig2 * (np.log10(Tplus) - np.log10(mean_To_plus))**2)
    return g1, g2, rv

def channel_model(Tplus, Re_tau: float, u_tau: float, u_cl) -> np.ndarray:
    A1 = 2.1*(1 - 100/Re_tau)
    sig1 = 4.4
    mean_Tplus = 12
    A2 = 0.9 * (np.log10(Re_tau) - 2.2)
    sig2 = 1.0
    mean_To = 0.6
    r1 = 0.5
    r2 = 3
    rv = np.exp(r1 * Tplus)/(np.exp(r1*r2) + np.exp(r1 * Tplus)) # correct
    rv = np.nan_to_num(rv, nan=1)  # replace NaNs with 0
    mean_To_plus = mean_To * Re_tau * u_tau/u_cl
    g1 = A1 * np.exp(-sig1 * (np.log10(Tplus) - np.log10(mean_Tplus))**2)
    g2 = A2 * np.exp(-sig2 * (np.log10(Tplus) - np.log10(mean_To_plus))**2)
    return g1, g2, rv


def convert_to_pa(x: np.ndarray, units: str, *, channel_name: str = "unknown") -> np.ndarray:
    """Convert a pressure time series to Pa."""
    u = units.lower()
    x = np.asarray(x, float)
    if u == "pa":
        return x
    if u == "kpa":
        return x * 1e3
    if u == "mbar":
        return x * 100.0
    if u in ("v", "volt", "volts"):
        if channel_name not in SENSITIVITIES_V_PER_PA or SENSITIVITIES_V_PER_PA[channel_name] is None:
            raise ValueError(
                f"Sensitivity (V/Pa) for channel '{channel_name}' not provided; cannot convert V→Pa."
            )
        sens = float(SENSITIVITIES_V_PER_PA[channel_name])  # V/Pa
        gain = float(PREAMP_GAIN.get(channel_name, 1.0))
        return x / (gain * sens)
    raise ValueError(f"Unsupported units '{units}' for channel '{channel_name}'")


def air_props_from_gauge(psi_gauge: float, T_K: float):
    """
    Return rho [kg/m^3], mu [Pa·s], nu [m^2/s] from gauge pressure [psi] and temperature [K].
    Sutherland's law for mu; nu = mu/rho.
    """
    p_abs = P_ATM + psi_gauge * PSI_TO_PA
    # Sutherland's
    mu0, T0, S = 1.716e-5, 273.15, 110.4
    mu = mu0 * (T_K/T0)**1.5 * (T0 + S)/(T_K + S)
    rho = p_abs / (R * T_K)
    nu = mu / rho
    return rho, mu, nu


def estimate_frf(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    window: str = WINDOW,
    npsg: int = NPERSEG,
):
    """
    Estimate H1 FRF and magnitude-squared coherence using Welch/CSD.

    Returns
    -------
    f : array_like [Hz]
    H : array_like (complex) = S_yx / S_xx  (x → y)
    gamma2 : array_like in [0, 1]
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
    # SciPy convention: csd(x, y) = E{ X * conj(Y) }
    _, Sxy = csd(x, y, fs=fs, window=w, nperseg=nseg, noverlap=nov, detrend=False)  # x→y

    H = np.conj(Sxy) / Sxx               # H1 = Syx / Sxx = conj(Sxy)/Sxx
    gamma2 = (np.abs(Sxy) ** 2) / (Sxx * Syy)
    gamma2 = np.clip(gamma2.real, 0.0, 1.0)
    return f, H, gamma2


def save_calibs(pressures):
    for i, pressure in enumerate(pressures):
        frequencies = np.arange(100, 3100, 100)
        dat = loadmat(TONAL_BASE + f"calib_{pressure}.mat")
        ph1, ph2, nc, _ = dat["channelData_WN"].T
        nc_pa = convert_to_pa(nc, "V", channel_name="NC")
        ph1_pa = convert_to_pa(ph1, "V", channel_name="PH1")
        f1, H1, _ = estimate_frf(ph1_pa, nc_pa, fs=FS)
        # save frf
        np.save(TONAL_BASE + f"wn_frequencies_{pressure}.npy", f1)
        np.save(TONAL_BASE + f"wn_H1_{pressure}.npy", H1)
        with h5py.File(TONAL_BASE + f"lumped_scaling_{pressure}.h5", 'w') as hf:
            hf.create_dataset('frequencies', data=f1)
            hf.create_dataset('H1', data=H1)
            hf.attrs['psig'] = pressure


def compute_spec(fs: float, x: np.ndarray, npsg : int = NPERSEG):
    """Welch PSD with sane defaults and shape guarding. Returns (f [Hz], Pxx [Pa^2/Hz])."""
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


def save_lumped_scaling_target():
    fn_atm = '0psig_cleaned.h5'
    fn_50psig = '50psig_cleaned.h5'
    fn_100psig = '100psig_cleaned.h5'
    labels = ['0psig', '50psig', '100psig']
    colours = ['C0', 'C1', 'C2']

    pgs = [0, 50, 100]
    rho0, *_  = air_props_from_gauge(pgs[0], TDEG[0]+273)
    R = 1, 2, 4

    fig, ax = plt.subplots(1, 1, figsize=(7, 3), tight_layout=True)
    for idxfn, fn in enumerate([fn_atm, fn_50psig, fn_100psig]):
        with h5py.File(f'data/{fn}', 'r') as hf:
            ph1_clean = hf['ph1_clean'][:]
            ph2_clean = hf['ph2_clean'][:]
            u_tau = hf.attrs['u_tau']
            nu = hf.attrs['nu']
            rho = hf.attrs['rho']
            f_cut = hf.attrs['f_cut']
            Re_tau = hf.attrs['Re_tau']
            cf_2 = hf.attrs['cf_2']  # default if missing
        f_clean, Pyy_ph1_clean = compute_spec(FS, ph1_clean)
        f_clean, Pyy_ph2_clean = compute_spec(FS, ph2_clean)
        T_plus = 1/f_clean * (u_tau**2)/nu

        # R = rho/rho0
        # ic(R)

        g1_c, g2_c, rv_c = channel_model(T_plus, Re_tau, u_tau, 14)
        g1_b, g2_b, rv_b = bl_model(T_plus, Re_tau, cf_2)
        channel_fphipp_plus = rv_c*(g1_c+g2_c)
        bl_fphipp_plus = rv_b*(g1_b+g2_b)

        f_clean_clipped = f_clean[f_clean < 1_000]
        bl_fphipp_plus = bl_fphipp_plus[f_clean < 1_000]

        f_clean_tf, Pyy_ph1_clean_tf = compute_spec(FS, ph1_clean)
        f_clean_tf, Pyy_ph2_clean_tf = compute_spec(FS, ph2_clean)

        T_plus_tf = 1/f_clean_tf * (u_tau**2)/nu

        data_fphipp_plus1_tf = (f_clean_tf * Pyy_ph1_clean_tf)/(rho**2 * u_tau**4)
        data_fphipp_plus2_tf = (f_clean_tf * Pyy_ph2_clean_tf)/(rho**2 * u_tau**4)

        # clip at the helmholtz resonance
        f_clean = f_clean_tf[f_clean_tf < 1_000]
        data_fphipp_plus1_tf_m = data_fphipp_plus1_tf[f_clean_tf < 1_000]
        data_fphipp_plus2_tf_m = data_fphipp_plus2_tf[f_clean_tf < 1_000]

        model_data_ratio1 = data_fphipp_plus1_tf_m / bl_fphipp_plus
        model_data_ratio2 = data_fphipp_plus2_tf_m / bl_fphipp_plus

        model_ratio_avg = (model_data_ratio1 + model_data_ratio2) / 2
        with h5py.File(TONAL_BASE + f"lumped_scaling_{labels[idxfn]}.h5", 'w') as hf:
            hf.create_dataset('frequencies', data=f_clean)
            hf.create_dataset('scaling_ratio', data=model_ratio_avg)
            hf.attrs['rho'] = rho
            hf.attrs['u_tau'] = u_tau
            hf.attrs['nu'] = nu
            hf.attrs['psig'] = pgs[idxfn]

def fit_speaker_scaling_from_files(
    labels: Tuple[str, ...] = ("0psig", "50psig", "100psig"),
    *,
    f_ref: float = 1000.0,
    rho_ref: Optional[float] = None,
    fmin: Optional[float] = 100.0,
    fmax: Optional[float] = 1000.0,
    invert_target: bool = True,
):
    """
    Fit a power-law multiplier S(f, rho) so that the *calibration* FRF magnitude |H_cal|
    matches your saved *lumped scaling target* magnitude.

    Files expected (per label L):
      Target (your save_lumped_scaling_target):
        TONAL_BASE + f"lumped_scaling_{L}.h5" with datasets:
           - 'frequencies' (Hz)
           - 'scaling_ratio'  (dimensionless)
           - attrs: 'rho' [kg/m^3] (used here)
        NOTE: if your "target" is model_data_ratio, set invert_target=True (default),
              because required |H| ≈ 1 / scaling_ratio.
      Calibration (your save_calibs):
        TONAL_BASE + f"wn_frequencies_{L}.npy"
        TONAL_BASE + f"wn_H1_{L}.npy"  (complex FRF; only |H| is used)

    Model in dB:
        20log10 S = c0 + a * 20log10(rho/rho_ref) + b * 20log10(f/f_ref)

    Returns
    -------
    params : (c0_db, a, b)
    scale  : function (f[Hz], rho[kg/m^3]) -> linear multiplier S(f,rho)
             (Multiply |H_cal| by this to match the target.)
    diag   : dict with
             - 'rho_ref', 'f_ref'
             - 'rmse_db_global'
             - 'rmse_db_per_label' {label: RMSE dB}
             - 'counts_per_label'  {label: N points used}
    """
    def _load_target(L: str):
        path = TONAL_BASE + f"lumped_scaling_{L}.h5"
        with h5py.File(path, "r") as hf:
            f = np.asarray(hf["frequencies"][:], float)
            s = np.asarray(hf["scaling_ratio"][:], float)  # POWER ratio (data/model)
            rho = float(hf.attrs["rho"]) if "rho" in hf.attrs else np.nan
        # Convert to required AMPLITUDE magnitude for |H_cal|*S:
        # start from data/model (power), invert to model/data, then take sqrt.
        if invert_target:
            s = 1.0 / np.maximum(s, 1e-16)   # POWER: model/data
        s = np.sqrt(s)                        # AMPLITUDE: required |H|
        return f, s, rho

    def _load_cal(L: str):
        f = np.load(TONAL_BASE + f"wn_frequencies_{L}.npy").astype(float)
        H = np.load(TONAL_BASE + f"wn_H1_{L}.npy")
        return f, np.abs(H)

    def _align_band(fa, ya, fb, yb, lo, hi):
        lo = lo if lo is not None else max(fa.min(), fb.min())
        hi = hi if hi is not None else min(fa.max(), fb.max())
        lo = max(lo, fa.min(), fb.min())
        hi = min(hi, fa.max(), fb.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError("No valid overlap between target and calibration frequency bands.")
        ma = (fa >= lo) & (fa <= hi)
        mb = (fb >= lo) & (fb <= hi)
        # choose denser grid to interpolate onto
        use_a = ma.sum() >= mb.sum()
        f_common = (fa[ma] if use_a else fb[mb]).astype(float)
        f_common = np.unique(f_common)
        ya_i = np.interp(f_common, fa[ma], ya[ma])
        yb_i = np.interp(f_common, fb[mb], yb[mb])
        good = (f_common > 0) & np.isfinite(ya_i) & np.isfinite(yb_i) & (ya_i > 0) & (yb_i > 0)
        return f_common[good], ya_i[good], yb_i[good]

    # 1) Load all datasets
    f_tgt, tgt_mag, rho_list, f_cal, cal_mag = {}, {}, {}, {}, {}
    for L in labels:
        ft, st, rho = _load_target(L)
        fc, Hc = _load_cal(L)
        f_tgt[L], tgt_mag[L], rho_list[L] = ft, st, rho
        f_cal[L], cal_mag[L] = fc, Hc

    # 2) Reference density/frequency
    rho_vals = np.array([rho_list[L] for L in labels], float)
    if np.any(~np.isfinite(rho_vals)):
        raise ValueError("Missing density 'rho' attribute in one or more target .h5 files.")
    rho_ref_val = float(np.mean(rho_vals)) if rho_ref is None else float(rho_ref)
    f_ref_val = float(f_ref)

    # 3) Build linear system: y = c0 + a*X1 + b*X2, with y = dB(target) - dB(cal)
    X_blocks, y_blocks, counts = [], [], {}
    for L in labels:
        fC, HC = f_cal[L], cal_mag[L]
        fT, ST = f_tgt[L], tgt_mag[L]
        fCmn, STi, HCi = _align_band(fT, ST, fC, HC, fmin, fmax)
        if fCmn.size < 2:
            continue
        y = 20.0 * np.log10(STi) - 20.0 * np.log10(HCi)
        X1 = 20.0 * np.log10(float(rho_list[L]) / rho_ref_val) * np.ones_like(fCmn)
        X2 = 20.0 * np.log10(fCmn / f_ref_val)
        X = np.column_stack([np.ones_like(fCmn), X1, X2])
        X_blocks.append(X)
        y_blocks.append(y)
        counts[L] = int(fCmn.size)

    if not X_blocks:
        raise ValueError("No usable points after alignment/band-limiting.")
    X_all = np.vstack(X_blocks)
    y_all = np.concatenate(y_blocks)

    # 4) Least-squares fit for [c0_db, a, b]
    beta, *_ = np.linalg.lstsq(X_all, y_all, rcond=None)
    c0_db, a, b = (float(beta[0]), float(beta[1]), float(beta[2]))

    # 5) Diagnostics
    rmse_per, resid_all = {}, []
    for L in labels:
        fC, HC = f_cal[L], cal_mag[L]
        fT, ST = f_tgt[L], tgt_mag[L]
        fCmn, STi, HCi = _align_band(fT, ST, fC, HC, fmin, fmax)
        if fCmn.size < 2:
            continue
        y = 20.0 * np.log10(STi) - 20.0 * np.log10(HCi)
        yhat = (c0_db
                + a * (20.0 * np.log10(float(rho_list[L]) / rho_ref_val))
                + b * (20.0 * np.log10(fCmn / f_ref_val)))
        r = y - yhat
        rmse_per[L] = float(np.sqrt(np.mean(r**2)))
        resid_all.append(r)
    rmse_global = float(np.sqrt(np.mean(np.concatenate(resid_all)**2)))

    # 6) Scaling function (linear gain)
    def scale(f: np.ndarray, rho: float) -> np.ndarray:
        f = np.asarray(f, float)
        S_db = (c0_db
                + a * (20.0 * np.log10(float(rho) / rho_ref_val))
                + b * (20.0 * np.log10(f / f_ref_val)))
        return 10.0 ** (S_db / 20.0)

    diag = dict(
        rho_ref=rho_ref_val,
        f_ref=f_ref_val,
        rmse_db_global=rmse_global,
        rmse_db_per_label=rmse_per,
        counts_per_label=counts,
        params_db=dict(c0_db=c0_db, a=a, b=b),
        invert_target=invert_target,
        band=dict(fmin=fmin, fmax=fmax),
    )
    return (c0_db, a, b), scale, diag


from typing import Optional, Callable, Dict

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
):
    """
    Apply a measured FRF H (x→y) to a time series x to synthesise y, with optional
    rho–frequency magnitude scaling S(f, rho) that you fitted earlier.

    If provided, the magnitude used becomes:
        |H|(f) * S(f, rho) * R
    where S is either:
        - scale_fn(f, rho)  (callable returning a linear multiplier), or
        - computed from scale_params = {'c0_db','a','b','rho_ref','f_ref'}.

    Out-of-band behavior: magnitude is set to unity outside the measured band,
    phase is linearly extrapolated from endpoints.
    """
    x = np.asarray(x, float)
    fH = np.asarray(f, float).copy()
    H = np.asarray(H)

    if demean:
        x = x - x.mean()

    if fH.ndim != 1 or H.shape[-1] != fH.shape[0]:
        raise ValueError("f and H must have matching lengths (H along last axis).")

    N = x.size
    Nfft = int(2 ** np.ceil(np.log2(N)))
    X = np.fft.rfft(x, n=Nfft)
    fr = np.fft.rfftfreq(Nfft, d=1.0 / fs)

    # Base magnitude/phase from calibration FRF
    fH[0] = max(fH[0], 1.0)  # avoid log/ratio issues at DC
    mag = np.abs(H).astype(float)
    phi = np.unwrap(np.angle(H))

    # Optional rho–frequency scaling
    if scale_fn is not None or scale_params is not None:
        if rho is None:
            raise ValueError("rho must be provided when applying rho–f scaling.")
        if scale_fn is not None:
            S = np.asarray(scale_fn(fH, float(rho)), float)
        else:
            # scale_params: {'c0_db','a','b','rho_ref','f_ref'}
            c0_db = float(scale_params["c0_db"])
            a = float(scale_params["a"])
            b = float(scale_params["b"])
            rho_ref = float(scale_params["rho_ref"])
            f_ref = float(scale_params["f_ref"])
            S_db = c0_db \
                 + a * (20.0 * np.log10(float(rho) / rho_ref)) \
                 + b * (20.0 * np.log10(fH / f_ref))
            S = 10.0 ** (S_db / 20.0)
        # keep magnitudes finite & non-negative
        S = np.clip(S, 0.0, np.inf)
        mag *= S

    # Optional extra constant factor (kept from your signature)
    mag *= float(R)

    # Enforce DC handling on the measured grid
    mag[0] = 0.0

    # Interpolate H onto the FFT grid; unity magnitude outside band
    mag_i = np.interp(fr, fH, mag, left=1.0, right=1.0)
    phi_i = np.interp(fr, fH, phi, left=phi[0], right=phi[-1])
    Hi = mag_i * np.exp(1j * phi_i)

    if zero_dc:
        Hi[0] = 0.0
        if Nfft % 2 == 0:
            Hi[-1] = 0.0

    y = np.fft.irfft(X * Hi, n=Nfft)[:N]
    return y


def plot_tf_model_comparison():
    fn_atm = '0psig_cleaned.h5'
    fn_50psig = '50psig_cleaned.h5'
    fn_100psig = '100psig_cleaned.h5'
    labels = ['0psig', '50psig', '100psig']
    colours = ['C0', 'C1', 'C2']

    pgs = [0, 50, 100]
    rho0, *_  = air_props_from_gauge(pgs[0], TDEG[0]+273)
    fig, ax = plt.subplots(1, 1, figsize=(7, 3), tight_layout=True)
    for idxfn, fn in enumerate([fn_atm, fn_50psig, fn_100psig]):
        with h5py.File(f'data/{fn}', 'r') as hf:
            ph1_clean = hf['ph1_clean'][:]
            ph2_clean = hf['ph2_clean'][:]
            u_tau = hf.attrs['u_tau']
            nu = hf.attrs['nu']
            rho = hf.attrs['rho']
            f_cut = hf.attrs['f_cut']
            Re_tau = hf.attrs['Re_tau']
            cf_2 = hf.attrs['cf_2']  # default if missing
        f_clean, Pyy_ph1_clean = compute_spec(FS, ph1_clean)
        f_clean, Pyy_ph2_clean = compute_spec(FS, ph2_clean)
        T_plus = 1/f_clean * (u_tau**2)/nu

        R = rho/rho0

        g1_c, g2_c, rv_c = channel_model(T_plus, Re_tau, u_tau, 14)
        g1_b, g2_b, rv_b = bl_model(T_plus, Re_tau, cf_2)
        channel_fphipp_plus = rv_c*(g1_c+g2_c)
        bl_fphipp_plus = rv_b*(g1_b+g2_b)

        # ax.semilogx(f_clean, channel_fphipp_plus, label=f'Model {labels[idxfn]}', linestyle='--', color=colours[idxfn], lw=0.7)
        # ax.semilogx(f_clean, bl_fphipp_plus, label=f'BL Model {labels[idxfn]}', linestyle='-.', color=colours[idxfn], lw=0.7)

        # Get the ratios from above
        f_new = np.load(TONAL_BASE + f"wn_frequencies_{labels[idxfn]}.npy")
        H_new = np.load(TONAL_BASE + f"wn_H1_{labels[idxfn]}.npy")

        ph1_clean_tf = apply_frf(ph1_clean, FS, f_new, H_new, rho=rho)
        ph2_clean_tf = apply_frf(ph2_clean, FS, f_new, H_new, rho=rho)
        f_clean_tf, Pyy_ph1_clean_tf = compute_spec(FS, ph1_clean_tf)
        f_clean_tf, Pyy_ph2_clean_tf = compute_spec(FS, ph2_clean_tf)

        T_plus_tf = 1/f_clean_tf * (u_tau**2)/nu

        data_fphipp_plus1_tf = (f_clean_tf * Pyy_ph1_clean_tf)/(rho**2 * u_tau**4)
        data_fphipp_plus2_tf = (f_clean_tf * Pyy_ph2_clean_tf)/(rho**2 * u_tau**4)

        # clip at the helmholtz resonance
        f_clean = f_clean_tf[f_clean_tf < 1_000]
        data_fphipp_plus1_tf_m = data_fphipp_plus1_tf[f_clean_tf < 1_000]
        data_fphipp_plus2_tf_m = data_fphipp_plus2_tf[f_clean_tf < 1_000]

        ax.semilogx(f_clean, data_fphipp_plus1_tf_m, linestyle='-', color=colours[idxfn], alpha=0.7, lw=0.7)
        ax.semilogx(f_clean, data_fphipp_plus2_tf_m, linestyle='-', color=colours[idxfn], alpha=0.7, lw=0.7)

        # Now plot upper and lower bounds of u_tau error in light grey
        u_tau_bounds = (u_tau * (1 - 0.1), u_tau * (1 + 0.1))
        data_fphipp_plus1_tf_u = ((f_clean_tf * Pyy_ph1_clean_tf)/(rho**2 * u_tau_bounds[0]**4))[f_clean_tf < 1_000]
        data_fphipp_plus2_tf_u = ((f_clean_tf * Pyy_ph2_clean_tf)/(rho**2 * u_tau_bounds[0]**4))[f_clean_tf < 1_000]
        data_fphipp_plus1_tf_l = ((f_clean_tf * Pyy_ph1_clean_tf)/(rho**2 * u_tau_bounds[-1]**4))[f_clean_tf < 1_000]
        data_fphipp_plus2_tf_l = ((f_clean_tf * Pyy_ph2_clean_tf)/(rho**2 * u_tau_bounds[-1]**4))[f_clean_tf < 1_000]

        ax.fill_between(f_clean, data_fphipp_plus1_tf_u, data_fphipp_plus1_tf_l, color=colours[idxfn], alpha=0.3, edgecolor='none')
        ax.fill_between(f_clean, data_fphipp_plus2_tf_u, data_fphipp_plus2_tf_l, color=colours[idxfn], alpha=0.3, edgecolor='none')

    ax.set_xlabel(r"$T^+$")
    ax.set_xlabel(r"$f$ [Hz]")
    ax.set_ylabel(r"${f \phi_{pp}}^+$")
    ax.set_xlim(50, 1e4)
    ax.set_ylim(0, 10)
    ax.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

    labels_handles = ['0 psig Data',
                      '50 psig Data',
                      '100 psig Data',]
    label_colours = ['C0', 'C1', 'C2']
    label_styles = ['solid',  'solid', 'solid']
    custom_lines = [Line2D([0], [0], color=label_colours[i], linestyle=label_styles[i]) for i in range(len(labels_handles))]
    ax.legend(custom_lines, labels_handles, loc='upper right', fontsize=8)

    fig.savefig('figures/tonal_ratios/spectra_comparison_tf_freq.png', dpi=410)


def plot_target_calib_modeled(
    labels: tuple[str, ...] = ("0psig", "50psig", "100psig"),
    *,
    fmin: float = 100.0,
    fmax: float = 1000.0,
    f_ref: float = 1000.0,
    rho_ref: float | None = None,
    invert_target: bool = True,   # your saved "scaling_ratio" is model/data; required |H| = 1/scaling_ratio
    to_db: bool = False,          # set True to plot in dB
    colours: list[str] | None = None,
    savepath: str | None = None,
):
    """
    Plot target (required |H|), measured calibration |H_cal|, and modeled |H_cal| * S(f,\rho) together.

    Data sources per label L:
      - Target (from save_lumped_scaling_target):
           TONAL_BASE + f"lumped_scaling_{L}.h5"
           datasets: 'frequencies', 'scaling_ratio', attrs: 'rho'
        If invert_target=True, we plot required |H| = 1/scaling_ratio.

      - Calibration (from save_calibs):
           TONAL_BASE + f"wn_frequencies_{L}.npy"
           TONAL_BASE + f"wn_H1_{L}.npy"   (complex; we use |.|)

    The modeled curve uses the fitted scale S(f,\rho) from fit_speaker_scaling_from_files(...).

    NOTE: If you also created an H5 named 'lumped_scaling_{L}.h5' in save_calibs, that will
          collide with the target filename. This function *ignores* that and only loads
          calibration from the 'wn_*.npy' files to avoid confusion.
    """
    # --- Fit rho–f scaling from your files (uses same target/cal scheme) ---
    (c0_db, a, b), scale, diag = fit_speaker_scaling_from_files(
        labels=labels,
        fmin=fmin, fmax=fmax,
        f_ref=f_ref, rho_ref=rho_ref,
        invert_target=invert_target
    )

    # --- helpers ---
    def as_db(x):
        x = np.asarray(x, float)
        return 20.0 * np.log10(np.maximum(x, 1e-16))

    if colours is None:
        colours = ['C0', 'C1', 'C2', 'C3', 'C4']  # auto-extend if more labels

    fig, ax = plt.subplots(1, 1, figsize=(7, 3), tight_layout=True)

    rmse_txt = []  # collect per-label RMSE (dB) for title/legend text

    for i, L in enumerate(labels):
        color = colours[i % len(colours)]

        # --- load target ---
        with h5py.File(TONAL_BASE + f"lumped_scaling_{L}.h5", "r") as hf:
            f_tgt = np.asarray(hf["frequencies"][:], float)
            s_ratio = np.asarray(hf["scaling_ratio"][:], float)
            rho = float(hf.attrs["rho"]) if "rho" in hf.attrs else np.nan
        if invert_target:
            tgt_mag = 1.0 / np.maximum(s_ratio, 1e-16)  # required |H|
        else:
            tgt_mag = s_ratio

        # band-limit target
        mt = (f_tgt >= fmin) & (f_tgt <= fmax)
        f_tgt, tgt_mag = f_tgt[mt], tgt_mag[mt]

        # --- load measured calibration ---
        f_cal = np.load(TONAL_BASE + f"wn_frequencies_{L}.npy").astype(float)
        H_cal = np.load(TONAL_BASE + f"wn_H1_{L}.npy")
        cal_mag = np.abs(H_cal)

        mc = (f_cal >= fmin) & (f_cal <= fmax)
        f_cal, cal_mag = f_cal[mc], cal_mag[mc]

        # --- modeled calibration: |H_cal| * S(f, rho) ---
        S = scale(f_cal, rho)
        modeled_mag = cal_mag * S

        # --- plot ---
        if to_db:
            ax.semilogx(f_tgt, as_db(tgt_mag), color=color, lw=1.4, label=f"{L} target")
            ax.semilogx(f_cal, as_db(cal_mag), color=color, lw=1.0, ls="--", alpha=0.9, label=f"{L} cal (meas)")
            ax.semilogx(f_cal, as_db(modeled_mag), color=color, lw=1.0, ls="-.", alpha=0.9, label=f"{L} calxS")
        else:
            ax.semilogx(f_tgt, tgt_mag, color=color, lw=1.4, label=f"{L} target")
            ax.semilogx(f_cal, cal_mag, color=color, lw=1.0, ls="--", alpha=0.9, label=f"{L} cal (meas)")
            ax.semilogx(f_cal, modeled_mag, color=color, lw=1.0, ls="-.", alpha=0.9, label=f"{L} calxS")

        # --- RMSE (in dB) between modeled and target on target grid ---
        # interpolate modeled onto target grid
        modeled_on_tgt = np.interp(f_tgt, f_cal, modeled_mag)
        e_db = as_db(tgt_mag) - as_db(modeled_on_tgt)
        rmse = float(np.sqrt(np.mean(e_db**2)))
        rmse_txt.append(f"{L}: {rmse:.2f} dB")

    ax.set_xlabel(r"$f$ [Hz]")
    ax.set_ylabel("Magnitude (dB)" if to_db else "Magnitude")
    ax.set_xlim(fmin, fmax)
    ax.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)
    ax.legend(loc="best", fontsize=8, ncol=1)

    # title_params = f"fit: c0={c0_db:.2f} dB, a={a:.3f}, b={b:.3f}  |  " \
    #                f"ρ_ref={diag['rho_ref']:.3g} kg/m3, f_ref={diag['f_ref']:.0f} Hz"
    # ax.set_title(title_params + "\n" + "RMSE (modeled vs target): " + ", ".join(rmse_txt), fontsize=9)

    if savepath:
        fig.savefig(savepath, dpi=350)
    return fig, ax


def plot_tf_model_comparison():
    """
    Apply the (rho, f)-scaled calibration FRF to measured time series and plot
    pre-multiplied, normalized spectra:  f * Pyy / (rho^2 * u_tau^4).
    """
    # --- fit rho–f scaling once from your saved target + calibration ---
    labels = ['0psig', '50psig', '100psig']
    (c0_db, a, b), scale, diag = fit_speaker_scaling_from_files(
        labels=tuple(labels),
        fmin=100.0, fmax=1000.0,   # fitting band
        f_ref=1000.0,
        invert_target=True         # your "scaling_ratio" -> required |H| = 1/scaling_ratio
    )

    # files per label
    fn_map = {
        '0psig': '0psig_cleaned.h5',
        '50psig': '50psig_cleaned.h5',
        '100psig': '100psig_cleaned.h5',
    }
    colours = ['C0', 'C1', 'C2']

    # (rho0 only used if you want to compare densities directly elsewhere)
    pgs = [0, 50, 100]
    rho0, *_  = air_props_from_gauge(pgs[0], TDEG[0]+273)

    fig, ax = plt.subplots(1, 1, figsize=(7, 3), tight_layout=True)

    # --- main loop over datasets ---
    for i, L in enumerate(labels):
        fn = fn_map[L]
        color = colours[i]

        # Load cleaned signals and attributes
        with h5py.File(f'data/{fn}', 'r') as hf:
            ph1_clean = hf['ph1_clean'][:]
            ph2_clean = hf['ph2_clean'][:]
            u_tau = float(hf.attrs['u_tau'])
            nu = float(hf.attrs['nu'])
            rho = float(hf.attrs['rho'])
            # f_cut = hf.attrs.get('f_cut', np.nan)  # unused here
            # Re_tau = hf.attrs.get('Re_tau', np.nan)
            # cf_2   = hf.attrs.get('cf_2',   np.nan)

        # Load measured calibration FRF (frequency + complex H)
        f_cal = np.load(TONAL_BASE + f"wn_frequencies_{L}.npy").astype(float)
        H_cal = np.load(TONAL_BASE + f"wn_H1_{L}.npy")

        # --- apply FRF with fitted rho–f magnitude scaling ---
        # (uses your updated apply_frf that accepts scale_fn and rho)
        ph1_filt = apply_frf(ph1_clean, FS, f_cal, H_cal, rho=rho, scale_fn=scale)
        ph2_filt = apply_frf(ph2_clean, FS, f_cal, H_cal, rho=rho, scale_fn=scale)

        # --- PSDs ---
        f1, Pyy1 = compute_spec(FS, ph1_filt)
        f2, Pyy2 = compute_spec(FS, ph2_filt)
        # unify grids (Welch outputs match with identical settings)
        if not np.allclose(f1, f2):
            # If extremely picky, interpolate one onto the other; here we pick f1
            Pyy2 = np.interp(f1, f2, Pyy2)
        f_sp = f1

        # --- pre-multiplied, normalized spectra ---
        Y1_pm = (f_sp * Pyy1) / (rho**2 * u_tau**4)
        Y2_pm = (f_sp * Pyy2) / (rho**2 * u_tau**4)

        # clip to display band (avoid Helmholtz etc.)
        band = (f_sp >= 50.0) & (f_sp < 1000.0)
        f_plot = f_sp[band]
        y1_plot = Y1_pm[band]
        y2_plot = Y2_pm[band]

        # plot PH1 & PH2 for this label
        ax.semilogx(f_plot, y1_plot, linestyle='-', color=colours[i], alpha=0.8, lw=0.8)
        ax.semilogx(f_plot, y2_plot, linestyle='-', color=colours[i], alpha=0.8, lw=0.8)

        # ±10% u_tau uncertainty bands (repeat for each channel)
        u_low, u_high = u_tau*(1 - 0.1), u_tau*(1 + 0.1)
        y1_upper = ((f_sp * Pyy1) / (rho**2 * u_low**4))[band]
        y1_lower = ((f_sp * Pyy1) / (rho**2 * u_high**4))[band]
        y2_upper = ((f_sp * Pyy2) / (rho**2 * u_low**4))[band]
        y2_lower = ((f_sp * Pyy2) / (rho**2 * u_high**4))[band]
        ax.fill_between(f_plot, y1_upper, y1_lower, color=colours[i], alpha=0.25, edgecolor='none')
        ax.fill_between(f_plot, y2_upper, y2_lower, color=colours[i], alpha=0.25, edgecolor='none')

    # --- axes, legend, save ---
    ax.set_xlabel(r"$f$ [Hz]")
    ax.set_ylabel(r"${f \,\phi_{pp}}^+$")
    ax.set_xlim(50, 1_000)
    ax.set_ylim(0, 10)
    ax.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.6)

    labels_handles = ['0 psig Data', '50 psig Data', '100 psig Data']
    label_colours = ['C0', 'C1', 'C2']
    label_styles = ['solid',  'solid', 'solid']
    custom_lines = [Line2D([0], [0], color=label_colours[i], linestyle=label_styles[i]) for i in range(len(labels_handles))]
    ax.legend(custom_lines, labels_handles, loc='upper right', fontsize=8)

    # title_params = f"scaled FRF: c0={c0_db:.2f} dB, a={a:.3f}, b={b:.3f} | " \
    #                f"ρ_ref={diag['rho_ref']:.3g} kg/m³, f_ref={diag['f_ref']:.0f} Hz"
    # ax.set_title(title_params, fontsize=9)

    fig.savefig('figures/tonal_ratios/spectra_comparison_tf_freq_scaled.png', dpi=410)





if __name__ == "__main__":
    # scale_0psig(['0psig', '50psig', '100psig'])
    plot_tf_model_comparison()
    # save_lumped_scaling_target()
    # fit_speaker_scaling_from_files()
    # plot_target_calib_modeled(
    #     labels=('0psig', '50psig', '100psig'),
    #     fmin=100.0,
    #     fmax=1000.0,
    #     to_db=True,
    #     savepath='figures/tonal_ratios/target_calib_modeled_comparison_db.png',
    # )