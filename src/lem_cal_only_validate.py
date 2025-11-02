# lem_cal_only_validate.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, Optional

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ----------------------- constants (air, units) -----------------------
R = 287.05          # J/(kg·K)
PSI_TO_PA = 6894.76 # Pa/psi
P_ATM = 101_325.0   # Pa

# ----------------------- I/O helpers ---------------------------------
def _parse_psig(x) -> float:
    """Accept 0, '0', '0psig', 50.0, '50psig', etc."""
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    s = str(x).strip().lower()
    if s.endswith("psig"):
        s = s[:-4]
    return float(s)

def air_density_from_psig(psig: float, TdegC: float = 20.0) -> float:
    """Ideal-gas density at T and (patm + psig). Good enough for scaling."""
    P = P_ATM + psig * PSI_TO_PA
    T = 273.15 + float(TdegC)
    return P / (R * T)

def load_calibration(label: str,
                     calib_base: str = "data/final_calibration/",
                     dataset_name: str = "H_fused",
                     default_TdegC: float = 20.0
                     ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (f, |H|, rho) for given label like '0psig'."""
    fn = os.path.join(calib_base, f"calibs_{label}.h5")
    with h5py.File(fn, "r") as hf:
        f = np.asarray(hf["frequencies"][:], float)
        H = np.asarray(hf[dataset_name][:])
        psig_attr = hf.attrs.get("psig", label)
    psig = _parse_psig(psig_attr)
    rho = air_density_from_psig(psig, default_TdegC)
    return f, np.abs(H).astype(float), rho

def align_band(fa, ya, fb, yb, fmin: float, fmax: float):
    """Common grid over [fmin,fmax] using the denser of the two grids."""
    lo = max(fmin, max(fa.min(), fb.min()))
    hi = min(fmax, min(fa.max(), fb.max()))
    ma = (fa >= lo) & (fa <= hi)
    mb = (fb >= lo) & (fb <= hi)
    if ma.sum() < 2 or mb.sum() < 2:
        return np.array([]), np.array([]), np.array([])
    f_use = fa[ma] if ma.sum() >= mb.sum() else fb[mb]
    f_use = np.unique(f_use.astype(float))
    ya_i = np.interp(f_use, fa[ma], ya[ma])
    yb_i = np.interp(f_use, fb[mb], yb[mb])
    good = np.isfinite(ya_i) & np.isfinite(yb_i) & (f_use > 0) & (ya_i > 0) & (yb_i > 0)
    return f_use[good], ya_i[good], yb_i[good]

# ----------------------- core: fit a (density exponent) ---------------
@dataclass
class CalData:
    label: str
    f: np.ndarray
    mag: np.ndarray
    rho: float

def fit_density_exponent_only(
    labels: Iterable[str],
    *,
    calib_base: str = "data/final_calibration/",
    dataset_name: str = "H_fused",
    fmin: float = 100.0,
    fmax: float = 1000.0,
    TdegC_for_rho: float = 20.0,
    ref_label: Optional[str] = None
):
    """
    Estimate 'a' from pairwise differences across pressures at the *same frequency*.
    Model in dB for pair L vs ref: y(f) = a * ΔXρ_db + Δδ_L (constant)
    -> identify a and one Δδ per non-ref label via linear LS. No target ratio used.
    """
    labels = list(labels)
    data: Dict[str, CalData] = {}
    for L in labels:
        f, H, rho = load_calibration(L, calib_base, dataset_name, default_TdegC=TdegC_for_rho)
        data[L] = CalData(L, f, H, rho)

    if ref_label is None:
        ref_label = labels[0]
    ref = data[ref_label]

    rows = []
    y_all = []
    non_ref_labels = [L for L in labels if L != ref_label]
    label_ids = {L: i for i, L in enumerate(non_ref_labels)}  # no column for ref
    for L in labels:
        if L == ref_label:
            continue
        fB, HL, HR = align_band(data[L].f, data[L].mag, ref.f, ref.mag, fmin, fmax)
        if fB.size < 2:
            continue
        # response in dB
        y = 20*np.log10(HL) - 20*np.log10(HR)
        # regressors: global 'a' (constant across f for this pair) and one Δδ per non-ref label
        dXrho_db = 20.0*np.log10(data[L].rho / ref.rho)
        Xa = np.ones_like(y) * dXrho_db          # the 'a' column (scaled)
        Xdelta = np.zeros((y.size, len(label_ids)))
        Xdelta[:, label_ids[L]] = 1.0            # one constant per non-ref label
        X = np.column_stack([Xa, Xdelta])
        rows.append(X); y_all.append(y)

    Xmat = np.vstack(rows)
    yvec = np.concatenate(y_all)
    # least squares: [a, Δδ_1, Δδ_2, ...]
    beta, *_ = np.linalg.lstsq(Xmat, yvec, rcond=None)
    a_hat = float(beta[0])
    deltas = {ref_label: 0.0}
    for L, idx in label_ids.items():
        deltas[L] = float(beta[1+idx])

    # residual stats
    resid = yvec - Xmat.dot(beta)
    rmse_db = float(np.sqrt(np.mean(resid**2)))

    return a_hat, deltas, rmse_db, data

# ----------------------- optional: apparent b -------------------------
def fit_apparent_b(
    data: Dict[str, CalData],
    a_hat: float,
    deltas: Dict[str, float],
    *,
    fmin: float = 100.0,
    fmax: float = 1000.0,
    f_ref: float = 700.0,
    ref_label: Optional[str] = None
):
    """
    After removing 'a' and the per-label constants, regress what's left against 20log10(f/f_ref).
    NOTE: b_app depends on the unknown baseline shape Γ(f); treat it as a *modeling choice*.
    """
    labels = list(data.keys())
    if ref_label is None:
        ref_label = labels[0]

    Xf_all, y_all = [], []
    for L in labels:
        f = data[L].f
        band = (f >= fmin) & (f <= fmax)
        fB = f[band]
        if fB.size < 2:
            continue
        M = 20*np.log10(data[L].mag[band])
        # subtract density term and label constant:
        Xrho_db = 20.0*np.log10(data[L].rho / data[ref_label].rho)
        M_res = M - a_hat*Xrho_db - deltas[L]
        Xf = 20.0*np.log10(fB / float(f_ref))
        Xf_all.append(Xf); y_all.append(M_res)

    Xf_all = np.concatenate(Xf_all)
    y_all  = np.concatenate(y_all)
    b_app = float(np.dot(Xf_all, y_all) / np.dot(Xf_all, Xf_all))  # 1D LS
    resid = y_all - b_app*Xf_all
    rmse_db = float(np.sqrt(np.mean(resid**2)))
    return b_app, rmse_db

# ----------------------- optional: LEM shape check -------------------
def lem_mag(f: np.ndarray, fD_Hz: float, QD: float) -> np.ndarray:
    """
    Simple diaphragm-dominated magnitude shape (no Helmholtz term here):
    |H| ∝ |(jω) / [1 - (ω/ωD)^2 + j ω/(Q_D ωD)]|.
    This captures the jω high-pass and single mechanical resonance shape.
    See Gallas (2002), discussion of the numerator ∝ jω and resonant structure. """
    w  = 2*np.pi*np.asarray(f, float)
    wD = 2*np.pi*float(fD_Hz)
    den = (1.0 - (w/wD)**2) + 1j*(w/(QD*wD))
    H = (1j*w)/den
    return np.abs(H)

def fit_lem_shape_to_calibrations(
    data: Dict[str, CalData],
    a_hat: float,
    deltas: Dict[str, float],
    *,
    fmin: float = 100.0,
    fmax: float = 1000.0,
    fD0: float = 500.0,
    QD0: float = 1.0
):
    """
    Fit a *common* LEM magnitude shape (fD, QD) using only measured |H|.
    For each label L we allow one constant gain g_L (solved in closed form).
    We do **not** fit any target; this is purely a shape check on the calibrations.
    """
    labels = list(data.keys())
    # Build master grid (union) for stability
    f_union = np.unique(np.concatenate([d.f[(d.f>=fmin)&(d.f<=fmax)] for d in data.values()]))
    if f_union.size < 8:
        raise ValueError("Too few points in band to fit LEM shape.")

    # Precompute magnitudes on the union grid and density terms
    Ms, Xrhos = {}, {}
    for L in labels:
        M = 20*np.log10(np.interp(f_union, data[L].f, data[L].mag))
        Ms[L] = M
        Xrhos[L] = 20.0*np.log10(data[L].rho / data[labels[0]].rho)

    def pack_params(p):
        fD = float(p[0]); QD = float(p[1])
        return fD, QD

    def residuals(p):
        fD, QD = pack_params(p)
        Hlem = 20*np.log10(np.maximum(lem_mag(f_union, fD, QD), 1e-16))
        # For each label: best constant g_L in dB is the mean of [M - a*Xrho - Hlem]
        res = []
        for L in labels:
            M = Ms[L]
            base = M - a_hat*Xrhos[L] - Hlem
            gL = float(base.mean())  # closed form
            res.append(base - gL)
        return np.concatenate(res)

    p0 = np.array([fD0, QD0], float)
    lb = np.array([100.0,  0.2], float)
    ub = np.array([2000.0, 5.0], float)
    sol = least_squares(residuals, p0, bounds=(lb, ub), method="trf")
    fD_hat, QD_hat = pack_params(sol.x)
    res = residuals(sol.x)
    rmse_db = float(np.sqrt(np.mean(res**2)))
    return fD_hat, QD_hat, f_union, rmse_db

# ----------------------- plotting ------------------------------------
def plot_collapse(data: Dict[str, CalData],
                  a_hat: float, deltas: Dict[str, float],
                  b_app: Optional[float] = None,
                  *,
                  fmin: float = 100.0, fmax: float = 1000.0,
                  f_ref: float = 700.0,
                  title: str = "Cal-only collapse",
                  savepath: Optional[str] = None):
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.6), tight_layout=True)
    for L, d in data.items():
        m = (d.f >= fmin) & (d.f <= fmax)
        fB = d.f[m]
        M = 20*np.log10(d.mag[m])
        Xrho = 20.0*np.log10(d.rho / list(data.values())[0].rho)
        M_corr = M - a_hat*Xrho - deltas[L]
        if b_app is not None:
            M_corr = M_corr - b_app*20.0*np.log10(fB/f_ref)
        ax.semilogx(fB, M_corr, lw=1.1, label=L)
    ax.grid(True, which="both", ls=":", alpha=0.7)
    ax.set_xlim(fmin, fmax)
    ax.set_xlabel("f [Hz]"); ax.set_ylabel("collapsed magnitude [dB]")
    mode = " (−a − b_app)" if b_app is not None else " (−a only)"
    ax.set_title(title + mode)
    ax.legend(ncol=3, fontsize=8, loc="best")
    if savepath:
        fig.savefig(savepath, dpi=300)
    return fig, ax

# ----------------------- main (example usage) ------------------------
if __name__ == "__main__":
    labels = ("0psig", "50psig", "100psig")
    FMIN, FMAX = 100.0, 1000.0
    f_ref = 700.0

    print("=== Cal-only validation (H_fused; no target) ===")
    a_hat, deltas, rmse_pair_db, data = fit_density_exponent_only(
        labels, fmin=FMIN, fmax=FMAX,
        calib_base="data/final_calibration/", dataset_name="H_fused",
        TdegC_for_rho=20.0, ref_label="0psig"
    )
    print(f"a (density exponent): {a_hat:+.4f}")
    print("Per-label constants Δδ_L [dB]:")
    for L in labels:
        print(f" {L:>7s}: {deltas[L]:+6.3f} dB")
    print(f"Global pairwise-difference RMSE: {rmse_pair_db:.3f} dB")

    # Optional: apparent b (see note)
    b_app, rmse_after_b = fit_apparent_b(
        data, a_hat, deltas,
        fmin=FMIN, fmax=FMAX, f_ref=f_ref, ref_label="0psig"
    )
    print(f"b_app (frequency exponent, not strictly identifiable): {b_app:+.4f}")
    print(f"Residual RMSE after removing a and b_app: {rmse_after_b:.3f} dB")

    # Collapse plots
    os.makedirs("figures", exist_ok=True)
    plot_collapse(data, a_hat, deltas, None,
                  fmin=FMIN, fmax=FMAX, f_ref=f_ref,
                  title="Cal-only collapse", savepath="figures/cal_only_collapse_no_b.png")
    plot_collapse(data, a_hat, deltas, b_app,
                  fmin=FMIN, fmax=FMAX, f_ref=f_ref,
                  title="Cal-only collapse", savepath="figures/cal_only_collapse_with_b.png")

    # Optional: Fit a common LEM magnitude shape (diaphragm)
    try:
        fD_hat, QD_hat, fgrid, rmse_shape = fit_lem_shape_to_calibrations(
            data, a_hat, deltas, fmin=FMIN, fmax=FMAX, fD0=500.0, QD0=1.0
        )
        print(f"LEM-shape fit (cal-only): fD={fD_hat:.1f} Hz, QD={QD_hat:.2f}, RMSE={rmse_shape:.3f} dB")
        # Plot calibrations divided by the fitted LEM shape and density term
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.6), tight_layout=True)
        Hlem_db = 20*np.log10(np.maximum(lem_mag(fgrid, fD_hat, QD_hat), 1e-16))
        for L in labels:
            M = 20*np.log10(np.interp(fgrid, data[L].f, data[L].mag))
            Xrho = 20.0*np.log10(data[L].rho / data[labels[0]].rho)
            base = M - a_hat*Xrho - Hlem_db
            gL = base.mean()
            ax.semilogx(fgrid, base - gL, lw=1.1, label=L)
        ax.grid(True, which="both", ls=":", alpha=0.7)
        ax.set_xlim(FMIN, FMAX)
        ax.set_xlabel("f [Hz]"); ax.set_ylabel("magnitude – (a, shape, g_L) [dB]")
        ax.set_title("Collapse after removing density and fitted LEM shape")
        ax.legend(ncol=3, fontsize=8, loc="best")
        fig.savefig("figures/cal_only_collapse_with_LEMshape.png", dpi=300)
    except Exception as e:
        print("LEM-shape fit skipped:", e)
