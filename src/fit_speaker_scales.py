
from typing import Callable, Dict, Iterable, Optional, Tuple


import h5py
import numpy as np



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
    TONAL_BASE = "data/2025-10-28/tonal/"
    def _load_target(L: str):
        path = TONAL_BASE + f"lumped_scaling_{L}.h5"
        with h5py.File(path, "r") as hf:
            f = np.asarray(hf["frequencies"][:], float)
            s = np.asarray(hf["scaling_ratio"][:], float)  # POWER ratio (data/model)
            rho = float(hf.attrs["rho"]) if "rho" in hf.attrs else np.nan
        # Convert to required AMPLITUDE magnitude for |H_cal|*S:
        # start from data/model (power), invert to model/data.
        if invert_target:
            s = 1.0 / np.maximum(s, 1e-16)   # POWER: model/data
        return f, s, rho

    def _load_cal(L: str):
        with h5py.File(TONAL_BASE + f"calibs_{L}.h5", 'r') as hf:
            f1 = np.asarray(hf["frequencies"][:], float)
            H1 = np.asarray(hf["H1"][:], float)
        return f1, np.abs(H1)

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

    X_all = np.vstack(X_blocks)
    y_all = np.concatenate(y_blocks)

    # 4) Least-squares fit for [c0_db, a, b]
    beta, *_ = np.linalg.lstsq(X_all, y_all, rcond=None)
    c0_db, a, b = (float(beta[0]), float(beta[1]), float(beta[2]))

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
        counts_per_label=counts,
        params_db=dict(c0_db=c0_db, a=a, b=b),
        invert_target=invert_target,
        band=dict(fmin=fmin, fmax=fmax),
    )
    return (c0_db, a, b), scale, diag

def fit_speaker_scaling_with_stokes(
    labels: tuple[str, ...] = ("0psig", "50psig", "100psig"),
    *,
    a_orifice: float,          # [m] orifice radius a
    L_orifice: float,          # [m] physical length L
    St_pivot: float = 10.0,    # ~10 per Fig. 4-2/4-3 (transition in profile / R_aN rise)
    p_shape: float = 2.0,      # sharpness of slope transition
    f_ref: float = 1000.0,
    rho_ref: float | None = None,
    fmin: float = 100.0,
    fmax: float = 1000.0,
    invert_target: bool = True,
):
    """
    Fit a Stokes-aware broken-power-law magnitude scaling:
        20log10 S = c0_db
                    + a_rho * 20log10(rho/rho_ref)
                    + b_lo  * 20log10(f_eff/f_ref)
                    + c_st  * log10(1 + (f_eff/f_b(nu))**p_shape)
    where f_b(nu) = St_pivot * nu / (2*pi*a^2) and f_eff = f * sqrt(L/(L + 8a/(3pi))).

    Returns:
      params dict, scale_factory(nu)->callable scale(f, rho), diag
    """
    TONAL_BASE = "data/2025-10-28/tonal/"

    # --- helpers reused from your other fitter ---
    def _load_target(L: str):
        path = TONAL_BASE + f"lumped_scaling_{L}.h5"
        with h5py.File(path, "r") as hf:
            f = np.asarray(hf["frequencies"][:], float)
            s = np.asarray(hf["scaling_ratio"][:], float)  # POWER ratio (data/model)
            rho = float(hf.attrs["rho"])
            nu  = float(hf.attrs["nu"])
        if invert_target:
            s = 1.0 / np.maximum(s, 1e-16)
        s = np.sqrt(s)   # AMPLITUDE
        return f, s, rho, nu

    def _load_cal(L: str):
        f = np.load(TONAL_BASE + f"wn_frequencies_{L}.npy").astype(float)
        H = np.load(TONAL_BASE + f"wn_H1_{L}.npy")
        return f, np.abs(H)

    def _align_band(fa, ya, fb, yb, lo, hi):
        lo = max(lo, fa.min(), fb.min()); hi = min(hi, fa.max(), fb.max())
        ma = (fa >= lo) & (fa <= hi); mb = (fb >= lo) & (fb <= hi)
        f_common = (fa[ma] if ma.sum() >= mb.sum() else fb[mb]).astype(float)
        f_common = np.unique(f_common)
        ya_i = np.interp(f_common, fa[ma], ya[ma])
        yb_i = np.interp(f_common, fb[mb], yb[mb])
        good = (f_common > 0) & np.isfinite(ya_i) & np.isfinite(yb_i) & (ya_i > 0) & (yb_i > 0)
        return f_common[good], ya_i[good], yb_i[good]

    # --- load ---
    f_tgt, tgt_mag, rho_L, nu_L, f_cal, cal_mag = {}, {}, {}, {}, {}, {}
    for L in labels:
        ft, st, rho, nu = _load_target(L); f_tgt[L], tgt_mag[L], rho_L[L], nu_L[L] = ft, st, rho, nu
        fc, Hc = _load_cal(L); f_cal[L], cal_mag[L] = fc, Hc

    rho_vals = np.array([rho_L[L] for L in labels], float)
    rho_ref_val = float(np.mean(rho_vals)) if rho_ref is None else float(rho_ref)
    f_ref_val = float(f_ref)

    # --- geometry + end correction ---
    a = float(a_orifice)
    L = float(L_orifice)
    delta_e = 8.0 * a / (3.0 * np.pi)        # end correction (radiation added mass)
    f_warp = np.sqrt(L / (L + delta_e))      # frequency warp factor
    def f_eff(f): return f * f_warp

    # --- build design matrix across all labels ---
    X_blocks, y_blocks, counts = [], [], {}
    for L in labels:
        fC, HC = f_cal[L], cal_mag[L]
        fT, ST = f_tgt[L], tgt_mag[L]
        fCmn, STi, HCi = _align_band(fT, ST, fC, HC, fmin, fmax)
        if fCmn.size < 2: continue

        # response to fit (in dB): target minus measured cal
        y = 20.0*np.log10(STi) - 20.0*np.log10(HCi)

        # regressors
        Xrho = 20.0*np.log10(rho_L[L] / rho_ref_val) * np.ones_like(fCmn)
        fe   = f_eff(fCmn)
        Xf   = 20.0*np.log10(fe / f_ref_val)
        fb   = (St_pivot * nu_L[L]) / (2.0*np.pi*a*a)   # Stokes pivot frequency for this nu
        Xst  = np.log10(1.0 + (fe / fb)**p_shape)

        X = np.column_stack([np.ones_like(fCmn), Xrho, Xf, Xst])
        X_blocks.append(X); y_blocks.append(y); counts[L] = int(fCmn.size)

    if not X_blocks:
        raise ValueError("No usable points after alignment/band-limiting.")
    X_all = np.vstack(X_blocks); y_all = np.concatenate(y_blocks)

    # --- fit [c0_db, a_rho, b_lo, c_st] ---
    beta, *_ = np.linalg.lstsq(X_all, y_all, rcond=None)
    c0_db, a_rho, b_lo, c_st = map(float, beta)

    # --- diagnostics ---
    rmse_per = {}; resid_all = []
    for L in labels:
        fC, HC = f_cal[L], cal_mag[L]
        fT, ST = f_tgt[L], tgt_mag[L]
        fCmn, STi, HCi = _align_band(fT, ST, fC, HC, fmin, fmax)
        if fCmn.size < 2: continue
        fe = f_eff(fCmn)
        fb = (St_pivot * nu_L[L]) / (2.0*np.pi*a*a)
        y  = 20.0*np.log10(STi) - 20.0*np.log10(HCi)
        yhat = (c0_db
                + a_rho*20.0*np.log10(rho_L[L]/rho_ref_val)
                + b_lo *20.0*np.log10(fe/f_ref_val)
                + c_st *np.log10(1.0 + (fe/fb)**p_shape))
        r = y - yhat
        rmse_per[L] = float(np.sqrt(np.mean(r*r))); resid_all.append(r)
    rmse_global = float(np.sqrt(np.mean(np.concatenate(resid_all)**2)))

    params = dict(c0_db=c0_db, a_rho=a_rho, b_lo=b_lo, c_st=c_st,
                  p_shape=float(p_shape), St_pivot=float(St_pivot),
                  a=a, L=L, delta_e=delta_e, rho_ref=rho_ref_val, f_ref=f_ref_val)

    def scale_factory(nu: float):
        """Bind viscosity so scale(f, rho) matches apply_frf's expected signature."""
        fb = (St_pivot * float(nu)) / (2.0*np.pi*a*a)
        def scale(f: np.ndarray, rho: float) -> np.ndarray:
            f = np.asarray(f, float)
            fe = f_eff(f)
            S_db = (c0_db
                    + a_rho*20.0*np.log10(float(rho)/rho_ref_val)
                    + b_lo *20.0*np.log10(fe/f_ref_val)
                    + c_st *np.log10(1.0 + (fe/fb)**p_shape))
            return 10.0**(S_db/20.0)
        return scale

    diag = dict(rmse_db_global=rmse_global,
                rmse_db_per_label=rmse_per,
                counts_per_label=counts,
                params_db=params,
                band=dict(fmin=fmin, fmax=fmax),
                note="Stokes-aware broken power law with end correction.")

    return params, scale_factory, diag
