
from typing import Callable, Dict, Iterable, Optional, Tuple

import h5py
import numpy as np


def fit_speaker_scaling_from_files_with_positions(
    labels: Tuple[str, ...] = ("0psig","50psig","100psig"),
    positions: Tuple[str, ...] = ("close","far"),
    *,
    f_ref: float = 700.0,
    rho_ref: Optional[float] = None,
    fmin: float = 100.0,
    fmax: float = 1000.0,
    invert_target: bool = True,
    TARGET_BASE: str = "data/final_target/",
    CLEANED_BASE: str = "data/final_cleaned/",
    CALIB_BASE: str = "data/final_calibrations/",
    calibs_fmt: str = "{label}_{pos}.h5",   # <-- matches your '0psig_close.h5'
    dataset_name: str = "H_fused",          # complex FRF in the calib files
    add_label_offsets: bool = False         # rarely needed
):
    """
    Fits the same power-law S(f,ρ) but uses all (label,position) calibrations at once.
    Adds one per-position dB offset (δ_pos) so 'close' and 'far' can differ in level.

    Model: 20log10 S = c0 + a*20log10(ρ/ρref) + b*20log10(f/fref)
           y_{L,pos}(f) = [20log10 target] - [20log10 |H_cal|]
                        = c0 + a*Xρ + b*Xf + δ_pos [+ δ_label]
    """
    import numpy as np, h5py

    # --- loaders
    def _load_target(L: str):
        path = f"{TARGET_BASE}target_{L}.h5"
        with h5py.File(path, "r") as hf:
            f = np.asarray(hf["frequencies"][:], float)
            s = np.asarray(hf["scaling_ratio"][:], float)  # already AMPLITUDE (you took sqrt)
            rho = float(hf.attrs["rho"])
        if invert_target:
            # your saved ratio = sqrt(data/model) → required |H| = 1/ratio
            s = 1.0 / np.maximum(s, 1e-16)
        return f, s, rho

    def _load_cal(L: str, pos: str):
        path = f"{CALIB_BASE}{calibs_fmt.format(label=L, pos=pos)}"
        with h5py.File(path, "r") as hf:
            f = np.asarray(hf["frequencies"][:], float)
            H = np.asarray(hf[dataset_name][:])          # complex
        return f, np.abs(H)

    def _align_band(fa, ya, fb, yb, lo, hi):
        lo = max(lo, fa.min(), fb.min())
        hi = min(hi, fa.max(), fb.max())
        ma = (fa >= lo) & (fa <= hi)
        mb = (fb >= lo) & (fb <= hi)
        f_common = (fa[ma] if ma.sum() >= mb.sum() else fb[mb]).astype(float)
        f_common = np.unique(f_common)
        ya_i = np.interp(f_common, fa[ma], ya[ma])
        yb_i = np.interp(f_common, fb[mb], yb[mb])
        good = (f_common > 0) & np.isfinite(ya_i) & np.isfinite(yb_i) & (ya_i > 0) & (yb_i > 0)
        return f_common[good], ya_i[good], yb_i[good]

    # --- cache targets (per label) and calibrations (per label, pos)
    f_tgt, tgt_mag, rho_L = {}, {}, {}
    for L in labels:
        ft, st, rho = _load_target(L)
        f_tgt[L], tgt_mag[L], rho_L[L] = ft, st, rho

    # references
    rho_vals = np.array([rho_L[L] for L in labels], float)
    rho_ref_val = float(np.mean(rho_vals)) if rho_ref is None else float(rho_ref)
    f_ref_val = float(f_ref)

    # --- build regression: columns = [1, Xρ, Xf, POS_OFFSETS(>baseline), LABEL_OFFSETS(>baseline?)]
    pos_to_col = {positions[0]: None}  # baseline pos has no column
    for p in positions[1:]:
        pos_to_col[p] = None  # will assign after we know counts
    label_to_col = {}  # optional

    X_rows, y_rows = [], []
    counts = {}

    # Pre-assign column indices
    base_cols = 3  # [1, Xρ, Xf]
    col = base_cols
    for p in positions[1:]:
        pos_to_col[p] = col; col += 1
    if add_label_offsets and len(labels) > 1:
        for L in labels[1:]:
            label_to_col[L] = col; col += 1

    for L in labels:
        fT, ST = f_tgt[L], tgt_mag[L]
        for p in positions:
            fC, HC = _load_cal(L, p)
            fCmn, STi, HCi = _align_band(fT, ST, fC, HC, fmin, fmax)
            if fCmn.size < 2:
                continue
            # response (target - cal) in dB
            y = 20.0*np.log10(STi) - 20.0*np.log10(HCi)
            # base regressors
            Xρ = 20.0*np.log10(float(rho_L[L]) / rho_ref_val) * np.ones_like(fCmn)
            Xf = 20.0*np.log10(fCmn / f_ref_val)
            X = np.column_stack([np.ones_like(fCmn), Xρ, Xf])

            # position offset column (0 for baseline position)
            if p in pos_to_col and pos_to_col[p] is not None:
                pos_col = np.zeros((fCmn.size, len(positions)-1))
                pos_col[:, pos_to_col[p]-base_cols] = 1.0
                X = np.hstack([X, pos_col])
            else:
                X = np.hstack([X, np.zeros((fCmn.size, len(positions)-1))])

            # optional label offsets
            if add_label_offsets and len(labels) > 1:
                lab_col = np.zeros((fCmn.size, len(labels)-1))
                if L in label_to_col:
                    lab_col[:, label_to_col[L]-base_cols-(len(positions)-1)] = 1.0
                X = np.hstack([X, lab_col])

            X_rows.append(X); y_rows.append(y)
            counts[(L,p)] = int(fCmn.size)

    X_all = np.vstack(X_rows)
    y_all = np.concatenate(y_rows)

    # solve
    beta, *_ = np.linalg.lstsq(X_all, y_all, rcond=None)
    c0_db, a, b = map(float, beta[:3])

    # unpack position offsets
    pos_offsets_db = {positions[0]: 0.0}
    idx = 3
    for p in positions[1:]:
        pos_offsets_db[p] = float(beta[idx]); idx += 1

    label_offsets_db = None
    if add_label_offsets and len(labels) > 1:
        label_offsets_db = {labels[0]: 0.0}
        for L in labels[1:]:
            label_offsets_db[L] = float(beta[idx]); idx += 1

    def scale(f: np.ndarray, rho: float) -> np.ndarray:
        f = np.asarray(f, float)
        S_db = c0_db + a*20*np.log10(float(rho)/rho_ref_val) + b*20*np.log10(f/f_ref_val)
        return 10.0**(S_db/20.0)

    diag = dict(
        params_db=dict(c0_db=c0_db, a=a, b=b),
        rho_ref=rho_ref_val, f_ref=f_ref_val,
        counts_per_series=counts,
        pos_offsets_db=pos_offsets_db,
        label_offsets_db=label_offsets_db,
        band=dict(fmin=fmin, fmax=fmax),
        invert_target=invert_target,
        calibs_fmt=calibs_fmt,
    )
    return (c0_db, a, b), scale, diag
