# fit_powerlaw_with_positions.py
from __future__ import annotations
import os
from typing import Dict, Optional, Tuple

import h5py
import numpy as np

# ----------------------------------------------------------------------
# Fit S(f, ρ) across all pressures and both microphone positions
# Calib files are per *label* (0psig / 50psig / 100psig), target files
# are per (label,position) i.e. 0psig_close, 0psig_far, ...
# ----------------------------------------------------------------------
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
    CALIB_BASE: str = "data/final_calibration/",   # <-- singular, per your saver
    calib_fmt: str = "calibs_{label}.h5",          # <-- per your saver
    dataset_name: str = "H_fused",                 # <-- use the fused FRF
    add_pos_offsets: bool = True,
    add_label_offsets: bool = False
):
    """
    Model (in dB):
        20log10 S = c0 + a*20log10(ρ/ρ_ref) + b*20log10(f/f_ref)
                    + δ_pos[close|far] (+ δ_label[label] optional)

    Target files per (label,position):
      TARGET_BASE/target_{label}_{pos}.h5
        datasets: 'frequencies', 'scaling_ratio' (amplitude = sqrt(data/model))
        attrs: 'rho', 'u_tau', 'nu', 'psig', ...

    Calibration files per label (shared across positions):
      CALIB_BASE/calibs_{label}.h5
        datasets: 'frequencies', dataset_name='H_fused' (complex FRF)
    """
    def _load_target(label: str, pos: str):
        path = os.path.join(TARGET_BASE, f"target_{label}_{pos}.h5")
        with h5py.File(path, "r") as hf:
            f  = np.asarray(hf["frequencies"][:], float)
            sA = np.asarray(hf["scaling_ratio"][:], float)  # sqrt(data/model)
            rho = float(hf.attrs["rho"])
        if invert_target:
            sA = 1.0 / np.maximum(sA, 1e-16)  # required |H|
        return f, sA, rho

    def _load_cal(label: str):
        path = os.path.join(CALIB_BASE, calib_fmt.format(label=label))
        with h5py.File(path, "r") as hf:
            f = np.asarray(hf["frequencies"][:], float)
            H = np.asarray(hf[dataset_name][:])  # complex
        return f, np.abs(H)

    def _align_band(fa, ya, fb, yb, lo, hi):
        lo = max(lo, float(min(fa.min(), fb.min())))
        hi = min(hi, float(max(fa.max(), fb.max())))
        ma = (fa >= lo) & (fa <= hi)
        mb = (fb >= lo) & (fb <= hi)
        f_common = (fa[ma] if ma.sum() >= mb.sum() else fb[mb]).astype(float)
        f_common = np.unique(f_common)
        ya_i = np.interp(f_common, fa[ma], ya[ma])
        yb_i = np.interp(f_common, fb[mb], yb[mb])
        good = (f_common > 0) & np.isfinite(ya_i) & np.isfinite(yb_i) & (ya_i > 0) & (yb_i > 0)
        return f_common[good], ya_i[good], yb_i[good]

    # gather data rows
    X_rows, y_rows, counts = [], [], {}
    rho_per_label: Dict[str, float] = {}

    # column bookkeeping
    base_cols = 3  # [1, Xρ, Xf]
    col = base_cols
    pos_cols = {positions[0]: None}
    if add_pos_offsets:
        for p in positions[1:]:
            pos_cols[p] = col; col += 1
    lab_cols = {}
    if add_label_offsets and len(labels) > 1:
        for L in labels[1:]:
            lab_cols[L] = col; col += 1

    for L in labels:
        fC, HC = _load_cal(L)  # per-label calibration
        for P in positions:
            fT, ST, rho = _load_target(L, P)
            rho_per_label[L] = rho
            f, STi, HCi = _align_band(fT, ST, fC, HC, fmin, fmax)
            if f.size < 2:
                continue
            y = 20.0*np.log10(STi) - 20.0*np.log10(HCi)   # target - |H_cal| (dB)
            Xrho = 20.0*np.log10(rho) * np.ones_like(f)   # normalize later
            Xf   = 20.0*np.log10(f)                       # normalize later
            X = np.column_stack([np.ones_like(f), Xrho, Xf])

            # position offsets
            if add_pos_offsets:
                pos_block = np.zeros((f.size, len(positions)-1))
                if P in pos_cols and pos_cols[P] is not None:
                    pos_block[:, pos_cols[P]-base_cols] = 1.0
                X = np.hstack([X, pos_block])

            # optional per-label offsets
            if add_label_offsets and len(labels) > 1:
                lab_block = np.zeros((f.size, len(labels)-1))
                if L in lab_cols:
                    lab_block[:, lab_cols[L] - base_cols - (len(positions)-1 if add_pos_offsets else 0)] = 1.0
                X = np.hstack([X, lab_block])

            X_rows.append(X); y_rows.append(y)
            counts[(L, P)] = int(f.size)

    if not X_rows:
        raise RuntimeError("No usable points between fmin and fmax.")

    X_all = np.vstack(X_rows)
    y_all = np.concatenate(y_rows)

    # choose references
    rhos = np.array([rho_per_label[L] for L in labels], float)
    rho_ref_val = float(np.mean(rhos)) if rho_ref is None else float(rho_ref)
    f_ref_val   = float(f_ref)

    # solve in raw coords (20log10 rho, 20log10 f); then rebase intercept
    beta, *_ = np.linalg.lstsq(X_all, y_all, rcond=None)
    c0_raw, a, b = map(float, beta[:3])
    c0_db = c0_raw + a*20.0*np.log10(1.0/rho_ref_val) + b*20.0*np.log10(1.0/f_ref_val)

    # unpack offsets
    idx = 3
    pos_offsets_db = {positions[0]: 0.0}
    if add_pos_offsets:
        for p in positions[1:]:
            pos_offsets_db[p] = float(beta[idx]); idx += 1
    label_offsets_db = None
    if add_label_offsets and len(labels) > 1:
        label_offsets_db = {labels[0]: 0.0}
        for L in labels[1:]:
            label_offsets_db[L] = float(beta[idx]); idx += 1

    def scale(f: np.ndarray, rho: float, pos: Optional[str] = None) -> np.ndarray:
        f = np.asarray(f, float)
        S_db = c0_db + a*20.0*np.log10(float(rho)/rho_ref_val) + b*20.0*np.log10(f/f_ref_val)
        if pos is not None and add_pos_offsets:
            S_db += float(pos_offsets_db.get(pos, 0.0))
        return 10.0**(S_db/20.0)

    diag = dict(
        params_db=dict(c0_db=c0_db, a=a, b=b),
        rho_ref=rho_ref_val, f_ref=f_ref_val,
        counts_per_series=counts,
        pos_offsets_db=pos_offsets_db if add_pos_offsets else None,
        label_offsets_db=label_offsets_db,
        band=dict(fmin=fmin, fmax=fmax),
        invert_target=invert_target,
        calib_fmt=calib_fmt, dataset_name=dataset_name,
        TARGET_BASE=TARGET_BASE, CALIB_BASE=CALIB_BASE,
    )
    return (c0_db, a, b), scale, diag
