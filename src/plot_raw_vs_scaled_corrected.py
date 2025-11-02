# plot_raw_vs_scaled_corrected.py
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from tf_compute import FS, compute_spec, apply_frf  # your stable helpers

def plot_raw_vs_scaled_corrected(
    label_pos: str,                   # e.g. '0psig_close'
    *,
    cleaned_base: str = "data/final_cleaned/",
    calib_base:   str = "data/final_calibration/",
    dataset_name: str = "H_fused",
    fmin: float = 100.0,
    fmax: float = 1000.0,
    scale_fn=None,                    # from the fitter above
    savepath: str | None = None,
):
    if "_" not in label_pos:
        raise ValueError("label_pos must look like '0psig_close', '50psig_far', etc.")
    L, P = label_pos.split("_", 1)

    # cleaned time series
    with h5py.File(os.path.join(cleaned_base, f"{label_pos}_cleaned.h5"), "r") as hf:
        ph1 = np.asarray(hf["ph1_clean"][:], float)
        ph2 = np.asarray(hf["ph2_clean"][:], float)
        rho = float(hf.attrs["rho"])
        u_tau = float(hf.attrs["u_tau"])

    # measured FRF (phase source) — per label
    with h5py.File(os.path.join(calib_base, f"calibs_{L}.h5"), "r") as hf:
        f_cal = np.asarray(hf["frequencies"][:], float)
        H_cal = np.asarray(hf[dataset_name][:])

    if scale_fn is None:
        raise ValueError("scale_fn is required (use the fitter's returned function).")
    def scale_pos(f, rho_val):  # include per-position offset
        return scale_fn(f, rho_val, pos=P)

    # apply measured |H|*S with measured phase
    y1 = apply_frf(ph1, fs=FS, f=f_cal, H=H_cal, rho=rho, scale_fn=scale_pos, demean=True, zero_dc=True)
    y2 = apply_frf(ph2, fs=FS, f=f_cal, H=H_cal, rho=rho, scale_fn=scale_pos, demean=True, zero_dc=True)

    # PSDs + normalization
    def pm_norm(f, P): return (f*P)/(rho**2 * u_tau**4)
    f1r, P1r = compute_spec(FS, ph1); f2r, P2r = compute_spec(FS, ph2)
    f1c, P1c = compute_spec(FS, y1);  f2c, P2c = compute_spec(FS, y2)
    if not np.allclose(f1r, f2r): P2r = np.interp(f1r, f2r, P2r)
    if not np.allclose(f1c, f2c): P2c = np.interp(f1c, f2c, P2c)
    f_raw = f1r; f_cor = f1c
    Y1_raw, Y2_raw = pm_norm(f_raw, P1r), pm_norm(f_raw, P2r)
    Y1_cor, Y2_cor = pm_norm(f_cor, P1c), pm_norm(f_cor, P2c)
    mraw = (f_raw >= fmin) & (f_raw <= fmax)
    mcor = (f_cor >= fmin) & (f_cor <= fmax)

    # plot
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
    ax.set_title(f"{label_pos} | measured-phase | power-law magnitude scale")
    if savepath:
        fig.savefig(savepath, dpi=350)
    return fig, ax
