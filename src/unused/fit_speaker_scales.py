
from typing import Callable, Dict, Iterable, Optional, Tuple


import h5py
import numpy as np



def fit_speaker_scaling_from_files(
    labels: Tuple[str, ...] = ("0psig", "50psig", "100psig"),
    *,
    f_ref: float = 700.0,
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
    CALIB_BASE = "data/final_calibration/"
    TARGET_BASE = "data/final_target/"
    CLEANED_BASE = "data/final_cleaned/"
    def _load_target(L: str):
        path = TARGET_BASE + f"target_{L}_close.h5"
        with h5py.File(path, "r") as hf:
            fc = np.asarray(hf["frequencies"][:], float)
            sc = np.asarray(hf["scaling_ratio"][:], float)  # POWER ratio (data/model)
            rhoc = float(hf.attrs["rho"]) if "rho" in hf.attrs else np.nan
        path = TARGET_BASE + f"target_{L}_far.h5"
        with h5py.File(path, "r") as hf:
            ff = np.asarray(hf["frequencies"][:], float)
            sf = np.asarray(hf["scaling_ratio"][:], float)  # POWER ratio (data/model)
            rhof = float(hf.attrs["rho"]) if "rho" in hf.attrs else np.nan
        
        # Combine close & far targets by geometric mean of required |H|:
        fc = np.asarray(fc, float); ff = np.asarray(ff, float)
        sc = np.asarray(sc, float); sf = np.asarray(sf, float)
        rhoc = float(rhoc); rhof = float(rhof)
        f = np.unique( np.concatenate([fc, ff]) )
        s = (sc + sf) / 2.0
        rho = (rhoc + rhof) / 2.0
        # Convert to required AMPLITUDE magnitude for |H_cal|*S:
        # start from data/model (power), invert to model/data.
        if invert_target:
            s = 1.0 / np.maximum(s, 1e-16)   # POWER: model/data
        return f, s, rho

    def _load_cal(L: str):
        with h5py.File(CALIB_BASE + f"calibs_{L}.h5", 'r') as hf:
            f1 = np.asarray(hf["frequencies"][:], float)
            H1 = np.asarray(hf["H_fused"][:], complex)
        return f1, np.abs(H1)

    def _align_band(fa, ya, fb, yb, lo, hi):
        lo = lo if lo is not None else max(fa.min(), fb.min())
        hi = hi if hi is not None else min(fa.max(), fb.max())
        lo = max(lo, fa.min(), fb.min())
        hi = min(hi, fa.max(), fb.max())
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
