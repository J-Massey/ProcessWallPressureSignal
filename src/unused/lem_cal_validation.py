# lem_cal_validation.py
from __future__ import annotations
import os, re
from typing import Dict, Iterable, Tuple, Optional

import h5py
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------------------
CALIB_BASE = "data/final_calibration/"     # expects calibs_{label}.h5 with H_fused
FIG_DIR    = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

FMIN = 100.0
FMAX = 1000.0
F_REF = 700.0        # your usual choice

# simple ideal-gas density (good enough for validation)
R_AIR = 287.05       # J/(kg·K)
P_ATM = 101_325.0    # Pa
PSI_TO_PA = 6_894.76

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
import re

def _parse_psig(v, default=None):
    """Return psig as float from number, '50psig', '50', b'50psig', etc."""
    if v is None:
        return default
    if isinstance(v, (int, float, np.floating)):
        return float(v)
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", "ignore")
    if isinstance(v, str):
        m = re.search(r"[-+]?\d*\.?\d+(?=\s*psig|\s*$)", v.strip(), re.I)
        if m:
            return float(m.group(0))
    return default

def air_density_from_psig(psig: float, TdegC: float = 20.0) -> float:
    """Ideal-gas density [kg/m^3] from gauge pressure (psig) and ambient T."""
    p_abs = P_ATM + float(psig) * PSI_TO_PA
    T_K   = 273.15 + float(TdegC)
    return p_abs / (R_AIR * T_K)

def _load_calibration(label: str,
                      dataset_name: str = "H_fused",
                      default_TdegC: float = 20.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load measured calibration FRF for a given label (e.g. '0psig', '50psig', ...).
    Returns: (f [Hz], |H|(f), rho [kg/m^3])
    """
    path = os.path.join(CALIB_BASE, f"calibs_{label}.h5")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Calibration file not found: {path}")

    with h5py.File(os.path.join(CALIB_BASE, f"calibs_{label}.h5"), "r") as hf:
        f = np.asarray(hf["frequencies"][:], float)
        H = np.asarray(hf["H_fused"][:])
        psig_attr = hf.attrs.get("psig", None)
    psig = _parse_psig(psig_attr, default=_parse_psig(label))
    if psig is None:
        raise ValueError(f"Could not determine psig for label '{label}' "
                        f"(attr={psig_attr!r}). Fix the file or the callsite.")

    rho  = air_density_from_psig(psig, default_TdegC)

    return f, np.abs(H).astype(float), rho

def _align_to_common(fa, ya, fb, yb, lo, hi):
    """Intersect bands and interpolate onto the denser grid within [lo,hi]."""
    lo = max(lo, float(np.nanmin(fa)), float(np.nanmin(fb)))
    hi = min(hi, float(np.nanmax(fa)), float(np.nanmax(fb)))
    ma = (fa >= lo) & (fa <= hi)
    mb = (fb >= lo) & (fb <= hi)
    if ma.sum() < 2 or mb.sum() < 2 or hi <= lo:
        raise ValueError("Insufficient overlap for alignment.")
    use_a = ma.sum() >= mb.sum()
    f_common = (fa[ma] if use_a else fb[mb]).astype(float)
    f_common = np.unique(f_common)
    ya_i = np.interp(f_common, fa[ma], ya[ma])
    yb_i = np.interp(f_common, fb[mb], yb[mb])
    good = (f_common > 0) & np.isfinite(ya_i) & np.isfinite(yb_i) & (ya_i > 0) & (yb_i > 0)
    return f_common[good], ya_i[good], yb_i[good]

def _as_db(x) -> np.ndarray:
    x = np.maximum(np.asarray(x, float), 1e-16)
    return 20.0 * np.log10(x)

# --------------------------------------------------------------------------------------
# Core validation: estimate a, b from *only* measured FRFs (H_fused) via pairwise ratios
# --------------------------------------------------------------------------------------
def validate_powerlaw_with_calibrations(
    labels: Tuple[str, ...] = ("0psig", "50psig", "100psig"),
    *,
    fmin: float = FMIN,
    fmax: float = FMAX,
    f_ref: float = F_REF,
    dataset_name: str = "H_fused",
    TdegC_for_rho: float = 20.0,
    make_plots: bool = True,
) -> Dict[str, object]:
    """
    Estimate (a, b) from measured calibrations only by regressing pairwise dB differences:

        ΔY_L(f) = [20log10|H_L| - 20log10|H_base|]
                = a*Δρ_dB + b*20log10(f/f_ref) + δ_L + ε

    where base = labels[0] and δ_L is a per-pair constant offset.
    Returns diagnostics and optionally emits two figures to ./figures/.
    """
    # 1) Load all calibrations
    f_dict, mag_dict, rho = {}, {}, {}
    for L in labels:
        fL, ML, rL = _load_calibration(L, dataset_name=dataset_name, default_TdegC=TdegC_for_rho)
        f_dict[L], mag_dict[L], rho[L] = fL, ML, rL

    base = labels[0]
    pair_blocks = []
    pair_y      = []
    pair_info   = []   # [(L, f_common, dy_meas)]
    # 2) Build regression for each pair (L vs base)
    for idx, L in enumerate(labels[1:], start=0):
        f_common, M_L, M_B = _align_to_common(f_dict[L], mag_dict[L], f_dict[base], mag_dict[base], fmin, fmax)
        dy = _as_db(M_L) - _as_db(M_B)                    # measured ΔY(f) in dB
        Xrho = 20.0 * np.log10(rho[L] / rho[base]) * np.ones_like(dy)
        Xf   = 20.0 * np.log10(f_common / f_ref)
        # Build columns: [Xrho, Xf, I_pair] where I_pair is one-hot for this pair
        I = np.zeros((dy.size, len(labels)-1))
        I[:, idx] = 1.0
        X = np.column_stack([Xrho, Xf, I])
        pair_blocks.append(X)
        pair_y.append(dy)
        pair_info.append((L, f_common, dy))

    X_all = np.vstack(pair_blocks)              # shape: (Npts, 2 + (n_pairs))
    y_all = np.concatenate(pair_y)              # all measured ΔY
    # 3) Solve least squares for [a, b, δ_1...δ_n]
    beta, *_ = np.linalg.lstsq(X_all, y_all, rcond=None)
    a_hat, b_hat = float(beta[0]), float(beta[1])
    deltas = beta[2:]                            # per-pair constants (diagnostic)

    # 4) Predictions, residuals, per-pair RMSE
    rmse_per_pair = {}
    curves = []  # for plotting
    start = 0
    for j, (L, f_common, dy_meas) in enumerate(pair_info):
        n = f_common.size
        Xrho = 20.0 * np.log10(rho[L] / rho[base]) * np.ones_like(dy_meas)
        Xf   = 20.0 * np.log10(f_common / f_ref)
        dy_pred = a_hat * Xrho + b_hat * Xf + float(deltas[j])
        res = dy_meas - dy_pred
        rmse_per_pair[L] = float(np.sqrt(np.mean(res**2)))
        curves.append((L, f_common, dy_meas, dy_pred))
        start += n

    rmse_global = float(np.sqrt(np.mean((y_all - X_all @ beta)**2)))

    # 5) Optional plots
    if make_plots:
        # (A) Pairwise dB differences: measured vs model
        fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.3), tight_layout=True)
        for (L, f_common, dy_meas, dy_pred) in curves:
            ax.semilogx(f_common, dy_meas, lw=1.2, label=f"{L} – {base}: meas")
            ax.semilogx(f_common, dy_pred, lw=1.0, ls="--", label=f"{L} – {base}: model")
        ax.set_xlim(fmin, fmax)
        ax.set_xlabel("f [Hz]")
        ax.set_ylabel(r"$\Delta$ magnitude [dB]")
        ax.grid(True, which="both", ls=":", alpha=0.7)
        ax.legend(ncol=2, fontsize=8, loc="best")
        ax.set_title(f"Cal-only fit → a={a_hat:+.3f},  b={b_hat:+.3f}   (global RMSE={rmse_global:.2f} dB)")
        fig.savefig(os.path.join(FIG_DIR, "cal_pairs_meas_vs_model.png"), dpi=300)

        # (B) Cross-prediction: predict |H_L| from |H_base| via a,b (pure S; NO pair delta)
        fig2, ax2 = plt.subplots(1, 1, figsize=(7.0, 3.3), tight_layout=True)
        base_f, base_M = f_dict[base], mag_dict[base]
        ax2.semilogx(base_f, _as_db(base_M), color="k", lw=1.2, label=f"{base}: measured")
        for L in labels[1:]:
            f_common, M_L, M_B = _align_to_common(f_dict[L], mag_dict[L], base_f, base_M, fmin, fmax)
            Xrho = 20.0 * np.log10(rho[L] / rho[base]) * np.ones_like(f_common)
            Xf   = 20.0 * np.log10(f_common / f_ref)
            # pure power-law scale factor (no constant delta)
            scale_db = a_hat * Xrho + b_hat * Xf
            M_pred_db = _as_db(np.interp(f_common, base_f, base_M)) + scale_db
            ax2.semilogx(f_common, _as_db(M_L), lw=1.1, label=f"{L}: measured")
            ax2.semilogx(f_common, M_pred_db, lw=1.0, ls="--", label=f"{L}: predicted from {base}")
        ax2.set_xlim(fmin, fmax)
        ax2.set_xlabel("f [Hz]")
        ax2.set_ylabel(r"|H| [dB]")
        ax2.grid(True, which="both", ls=":", alpha=0.7)
        ax2.legend(ncol=2, fontsize=8, loc="best")
        ax2.set_title("Cross-prediction using cal-only a,b (pure S ratio, no constant offset)")
        fig2.savefig(os.path.join(FIG_DIR, "cal_cross_prediction.png"), dpi=300)

    # 6) Return diagnostics
    return dict(
        a=a_hat, b=b_hat,
        deltas_per_pair={labels[i+1]: float(d) for i, d in enumerate(deltas)},
        rmse_db_global=rmse_global,
        rmse_db_per_pair=rmse_per_pair,
        base_label=base, f_ref=f_ref,
        f_band=dict(fmin=fmin, fmax=fmax)
    )

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    labels = ("0psig", "50psig", "100psig")
    diag = validate_powerlaw_with_calibrations(labels=labels,
                                               fmin=FMIN, fmax=FMAX,
                                               f_ref=F_REF,
                                               dataset_name="H_fused",
                                               TdegC_for_rho=20.0,
                                               make_plots=True)
    print("\n=== Cal-only validation (H_fused) ===")
    print(f"a (density exponent): {diag['a']:+.4f}")
    print(f"b (frequency exponent): {diag['b']:+.4f}")
    print("Pair constant offsets δ_L [dB]:")
    for L, d in diag["deltas_per_pair"].items():
        print(f"  {L:>7s}: {d:+.3f} dB")
    print(f"Global RMSE (dB): {diag['rmse_db_global']:.3f}")
    for L, r in diag["rmse_db_per_pair"].items():
        print(f"  RMSE {L:>7s}: {r:.3f} dB")
    print(f"Saved plots → {os.path.join(FIG_DIR,'cal_pairs_meas_vs_model.png')} and cal_cross_prediction.png")
