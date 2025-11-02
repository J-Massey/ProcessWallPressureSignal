# === powerlaw_fit_and_plots.py ===============================================
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt

# Reuse your Welch & constants (keeps spectra comparable to your pipeline)
from tf_compute import FS, compute_spec  # FS=50k, Welch config, etc. :contentReference[oaicite:2]{index=2}
# Your existing time-domain applicator (measured phase + amplitude scaling):
from tf_compute import estimate_frf  # if you need it elsewhere
from tf_compute import WINDOW, NPERSEG  # optional
# If apply_frf lives elsewhere, import it; otherwise adjust import path.
from tf_compute import apply_frf  # uses measured H; accepts scale_fn, rho


# -----------------------------------------------------------------------------
# 1) Fit power-law S(f, rho) using targets & calibrations for BOTH positions
# -----------------------------------------------------------------------------
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
    CALIB_BASE: str = "data/final_calibration/",
    calibs_fmt: str = "calibs_{label}.h5",   # e.g. 0psig_close.h5
    dataset_name: str = "H_fused",          # complex FRF in the calib files
    add_label_offsets: bool = False,        # rarely needed
    add_pos_offsets: bool = True            # keep the per-position δ_pos (handy)
):
    """
    Model in dB (global across pressures/positions):
        20log10 S = c0 + a * 20log10(rho/rho_ref) + b * 20log10(f/f_ref)
                    + δ_pos[close|far] (+ δ_label[label] optional)

    Target files (per label, per position):
        TARGET_BASE/target_{label}_{pos}.h5
          - datasets: 'frequencies' (Hz), 'scaling_ratio' (amplitude = sqrt(data/model))
          - attrs:    'rho' [kg/m^3], 'u_tau', 'nu', 'psig', ...

    Calibration files (per label, per position):
        CALIB_BASE/{label}_{pos}.h5
          - datasets: 'frequencies' (Hz), dataset_name='H_fused' (complex FRF)

    Returns
    -------
    (c0_db, a, b), scale_fn, diag
    where scale_fn can be called as: scale_fn(f, rho, pos=None)  # pos applies δ_pos
    """
    # --- loaders --------------------------------------------------------------
    def _load_target(label: str, pos: str):
        path = os.path.join(TARGET_BASE, f"target_{label}_{pos}.h5")
        with h5py.File(path, "r") as hf:
            f  = np.asarray(hf["frequencies"][:], float)
            sA = np.asarray(hf["scaling_ratio"][:], float)  # amplitude (sqrt(data/model))
            rho = float(hf.attrs["rho"])
        if invert_target:
            # Required magnitude the actuator must supply:
            sA = 1.0 / np.maximum(sA, 1e-16)
        return f, sA, rho

    def _load_cal(label: str, pos: str):
        path = os.path.join(CALIB_BASE, calibs_fmt.format(label=label, pos=pos))
        with h5py.File(path, "r") as hf:
            f = np.asarray(hf["frequencies"][:], float)
            H = np.asarray(hf[dataset_name][:])  # complex FRF
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

    # --- build regression -----------------------------------------------------
    # collect rho per *label*; positions may share the same rho (at same psig)
    rho_per_label: Dict[str, float] = {}

    base_cols = 3  # [1, Xρ, Xf]
    pos_cols: Dict[str, Optional[int]] = {positions[0]: None}
    lab_cols: Dict[str, Optional[int]] = {}

    col = base_cols
    if add_pos_offsets:
        for p in positions[1:]:
            pos_cols[p] = col; col += 1
    if add_label_offsets and len(labels) > 1:
        for L in labels[1:]:
            lab_cols[L] = col; col += 1

    X_rows, y_rows = [], []
    counts = {}

    for L in labels:
        for p in positions:
            # load target & calibration for *this* label/position
            fT, ST, rho = _load_target(L, p)
            fC, HC = _load_cal(L, p)
            rho_per_label[L] = rho  # any position will do; same psig

            fCmn, STi, HCi = _align_band(fT, ST, fC, HC, fmin, fmax)
            if fCmn.size < 2:
                continue

            # response (in dB): target (required |H|) - measured |H_cal|
            y = 20.0*np.log10(STi) - 20.0*np.log10(HCi)

            # base regressors
            Xrho = 20.0*np.log10(rho / 1.0)  # we'll normalize by rho_ref later
            Xf   = 20.0*np.log10(fCmn / 1.0)
            X = np.column_stack([np.ones_like(fCmn), Xrho*np.ones_like(fCmn), Xf])

            # position offsets (baseline = first in 'positions')
            if add_pos_offsets:
                pos_block = np.zeros((fCmn.size, len(positions)-1))
                if p in pos_cols and pos_cols[p] is not None:
                    pos_block[:, pos_cols[p] - base_cols] = 1.0
                X = np.hstack([X, pos_block])

            # optional label offsets (baseline = first in 'labels')
            if add_label_offsets and len(labels) > 1:
                lab_block = np.zeros((fCmn.size, len(labels)-1))
                if L in lab_cols:
                    lab_block[:, lab_cols[L] - base_cols - (len(positions)-1 if add_pos_offsets else 0)] = 1.0
                X = np.hstack([X, lab_block])

            X_rows.append(X); y_rows.append(y)
            counts[(L, p)] = int(fCmn.size)

    if not X_rows:
        raise RuntimeError("No usable points in the specified band.")

    X_all = np.vstack(X_rows)
    y_all = np.concatenate(y_rows)

    # choose rho_ref if needed
    rhos = np.array([rho_per_label[L] for L in labels], float)
    rho_ref_val = float(np.mean(rhos)) if rho_ref is None else float(rho_ref)
    # normalize X: subtract the rho_ref, f_ref by adding a post-fit shift
    # We built X with Xrho = 20log10(rho) and Xf = 20log10(f)
    # Solve in those raw coordinates; then when evaluating we subtract 20log10(rho_ref) and f_ref.

    # least-squares solve
    beta, *_ = np.linalg.lstsq(X_all, y_all, rcond=None)
    c0_db_raw, a_raw, b_raw = map(float, beta[:3])

    # adjust intercept so the model is exactly:
    #   20log10 S = c0_db + a*20log10(rho/rho_ref) + b*20log10(f/f_ref) + offsets
    c0_db = c0_db_raw \
            + a_raw*20.0*np.log10(1.0/rho_ref_val) \
            + b_raw*20.0*np.log10(1.0/f_ref)
    a = a_raw
    b = b_raw

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

    # scale function with optional per-position offset
    def scale(f: np.ndarray, rho: float, pos: Optional[str] = None) -> np.ndarray:
        f = np.asarray(f, float)
        S_db = c0_db \
             + a * 20.0*np.log10(float(rho)/rho_ref_val) \
             + b * 20.0*np.log10(f/f_ref)
        if pos is not None and add_pos_offsets:
            S_db = S_db + float(pos_offsets_db.get(pos, 0.0))
        return 10.0**(S_db/20.0)

    diag = dict(
        params_db=dict(c0_db=c0_db, a=a, b=b),
        rho_ref=rho_ref_val, f_ref=f_ref,
        counts_per_series=counts,
        pos_offsets_db=pos_offsets_db if add_pos_offsets else None,
        label_offsets_db=label_offsets_db,
        band=dict(fmin=fmin, fmax=fmax),
        invert_target=invert_target,
        calibs_fmt=calibs_fmt,
        dataset_name=dataset_name,
    )
    return (c0_db, a, b), scale, diag


# -----------------------------------------------------------------------------
# 2) Plot: scaled calibration |H| vs. target for one label_pos (100–1000 Hz)
# -----------------------------------------------------------------------------
def plot_scaled_tf_vs_target(
    label_pos: str,                   # e.g. '0psig_close'
    *,
    target_base: str = "data/final_target/",
    calib_base:  str = "data/final_calibration",
    dataset_name: str = "H_fused",
    fmin: float = 100.0,
    fmax: float = 1000.0,
    to_db: bool = True,
    invert: bool = True,              # compare as "required |H|" = 1/target
    scale_fn=None,                    # from the fitter above
    rho_for_label: Optional[float] = None,  # if None, read from target file
    savepath: Optional[str] = None,
):
    def _as_db(x): return 20.0*np.log10(np.maximum(np.asarray(x,float), 1e-16))

    # split "0psig_close" → ("0psig","close")
    if "_" not in label_pos:
        raise ValueError("label_pos must look like '0psig_close', '50psig_far', etc.")
    L, P = label_pos.split("_", 1)

    # --- load target (same grid we’ll use for plotting)
    tgt_path = os.path.join(target_base, f"target_{L}_{P}.h5")
    with h5py.File(tgt_path, "r") as hf:
        f_tgt = np.asarray(hf["frequencies"][:], float)
        S_tgt = np.asarray(hf["scaling_ratio"][:], float)  # amplitude sqrt(data/model)
        rho_t = float(hf.attrs.get("rho", np.nan))

    rho = rho_t if rho_for_label is None else float(rho_for_label)
    band = (f_tgt >= fmin) & (f_tgt <= fmax)
    fB = f_tgt[band]; S_tgtB = S_tgt[band]

    # --- load calibration |H|
    with h5py.File(os.path.join(calib_base, f"{L}_{P}.h5"), "r") as hf:
        f_cal = np.asarray(hf["frequencies"][:], float)
        H_cal = np.asarray(hf[dataset_name][:])
    # interp |H| onto target grid
    Hc_mag = np.abs(np.interp(fB, f_cal, np.abs(H_cal)))

    # --- apply power-law scale on the SAME grid
    if scale_fn is None:
        raise ValueError("Provide scale_fn from the fitter (call fit_* first).")
    S = scale_fn(fB, rho, pos=P)
    S_model = Hc_mag * S

    # orientation
    if invert:
        tgt_plot = 1.0/np.maximum(S_tgtB, 1e-16)
        mdl_plot = 1.0/np.maximum(S_model, 1e-16)
        mode = "required |H| (inverted)"
    else:
        tgt_plot = S_tgtB
        mdl_plot = S_model
        mode = "as saved (sqrt(data/model))"

    # RMSE (dB)
    rmse_db = float(np.sqrt(np.mean((_as_db(tgt_plot)-_as_db(mdl_plot))**2)))

    # --- plot
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.4), tight_layout=True)
    if to_db:
        ax.semilogx(fB, _as_db(tgt_plot), lw=1.6, label="target")
        ax.semilogx(fB, _as_db(mdl_plot), lw=1.2, ls="--", label="|H_cal| × S")
        ax.set_ylabel("Magnitude [dB]")
    else:
        ax.semilogx(fB, tgt_plot, lw=1.6, label="target")
        ax.semilogx(fB, mdl_plot, lw=1.2, ls="--", label="|H_cal| × S")
        ax.set_ylabel("Magnitude [linear]")
    ax.set_xlim(fmin, fmax)
    ax.set_xlabel("f [Hz]")
    ax.grid(True, which="both", ls=":", alpha=0.7)
    ax.set_title(f"{label_pos} | RMSE={rmse_db:.2f} dB | {mode}")
    ax.legend(loc="best", fontsize=8)

    if savepath:
        fig.savefig(savepath, dpi=350)
    return fig, ax, rmse_db


# -----------------------------------------------------------------------------
# 3) Plot: RAW vs FRF‑scaled (time‑domain) spectra on f*Pyy/(rho^2 u_tau^4)
# -----------------------------------------------------------------------------
def plot_raw_vs_scaled_corrected(
    label_pos: str,                   # e.g. '0psig_close'
    *,
    cleaned_base: str = "data/final_cleaned/",
    calib_base:   str = "data/final_calibration/",
    dataset_name: str = "H_fused",
    fmin: float = 100.0,
    fmax: float = 1000.0,
    scale_fn=None,                    # from the fitter above
    savepath: Optional[str] = None,
):
    import numpy as np

    if "_" not in label_pos:
        raise ValueError("label_pos must look like '0psig_close', '50psig_far', etc.")
    L, P = label_pos.split("_", 1)

    # --- cleaned time series + meta
    with h5py.File(os.path.join(cleaned_base, f"{label_pos}_cleaned.h5"), "r") as hf:
        ph1 = np.asarray(hf["ph1_clean"][:], float)
        ph2 = np.asarray(hf["ph2_clean"][:], float)
        rho = float(hf.attrs["rho"])
        u_tau = float(hf.attrs["u_tau"])

    # --- measured calibration (phase source)
    with h5py.File(os.path.join(calib_base, f"{L}_{P}.h5"), "r") as hf:
        f_cal = np.asarray(hf["frequencies"][:], float)
        H_cal = np.asarray(hf[dataset_name][:])

    # --- closure to include position offset in S
    if scale_fn is None:
        raise ValueError("Provide scale_fn from the fitter (call fit_* first).")
    def scale_pos(f: np.ndarray, rho_val: float) -> np.ndarray:
        return scale_fn(f, rho_val, pos=P)

    # --- apply FRF (measured magnitude × S, measured phase)
    y1 = apply_frf(ph1, fs=FS, f=f_cal, H=H_cal,
                   rho=rho, scale_fn=scale_pos, demean=True, zero_dc=True)
    y2 = apply_frf(ph2, fs=FS, f=f_cal, H=H_cal,
                   rho=rho, scale_fn=scale_pos, demean=True, zero_dc=True)

    # --- PSDs and normalization
    def pm_norm(f, P): return (f * P) / (rho**2 * u_tau**4)
    f1r, P1r = compute_spec(FS, ph1); f2r, P2r = compute_spec(FS, ph2)
    f1c, P1c = compute_spec(FS, y1);  f2c, P2c = compute_spec(FS, y2)

    # unify grids (Welch identical, but keep robust)
    if not np.allclose(f1r, f2r): P2r = np.interp(f1r, f2r, P2r)
    if not np.allclose(f1c, f2c): P2c = np.interp(f1c, f2c, P2c)

    f_raw = f1r;  f_cor = f1c
    Y1_raw, Y2_raw = pm_norm(f_raw, P1r), pm_norm(f_raw, P2r)
    Y1_cor, Y2_cor = pm_norm(f_cor, P1c), pm_norm(f_cor, P2c)
    mraw = (f_raw >= fmin) & (f_raw <= fmax)
    mcor = (f_cor >= fmin) & (f_cor <= fmax)

    # --- plot
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.4), tight_layout=True)
    ax.semilogx(f_raw[mraw], Y1_raw[mraw], "C0--", lw=0.9, alpha=0.9, label="PH1 raw")
    ax.semilogx(f_raw[mraw], Y2_raw[mraw], "C1--", lw=0.9, alpha=0.9, label="PH2 raw")
    ax.semilogx(f_cor[mcor], Y1_cor[mcor], "C0-",  lw=1.2, alpha=0.95, label="PH1 scaled")
    ax.semilogx(f_cor[mcor], Y2_cor[mcor], "C1-",  lw=1.2, alpha=0.95, label="PH2 scaled")

    ax.set_xlim(fmin, fmax)
    ax.set_xlabel(r"$f$ [Hz]")
    ax.set_ylabel(r"$f\,\phi_{pp}^+$")
    ax.grid(True, which="major", ls="--", lw=0.4, alpha=0.7)
    ax.grid(True, which="minor", ls=":",  lw=0.25, alpha=0.6)
    ax.legend(ncol=2, fontsize=8, loc="best")
    ax.set_title(f"{label_pos} | power-law scaled FRF")

    if savepath:
        fig.savefig(savepath, dpi=350)
    return fig, ax
# === end file =================================================================


if __name__ == "__main__":
    # 1) Fit once across all labels & positions
    (c0_db, a, b), scale, diag = fit_speaker_scaling_from_files_with_positions(
        labels=("0psig","50psig","100psig"),
        positions=("close","far"),
        f_ref=700.0, fmin=100.0, fmax=1000.0,
        invert_target=True,
        TARGET_BASE="data/final_target/",
        CALIB_BASE="data/final_calibration/",
        dataset_name="H_fused",
        add_pos_offsets=True,
    )
    print("Fit params (dB):", diag["params_db"])
    print("Pos offsets (dB):", diag["pos_offsets_db"])

    # 2) Quick overlays & spectra for a couple of series
    for lbl in ("0psig_close","0psig_far","50psig_close","100psig_far"):
        plot_scaled_tf_vs_target(lbl, scale_fn=scale,
                                 savepath=f"figures/scaled_tf_vs_target_{lbl}.png")
        plot_raw_vs_scaled_corrected(lbl, scale_fn=scale,
                                     savepath=f"figures/raw_vs_scaled_{lbl}.png")
