# plot_scaled_tf_vs_target.py
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def plot_scaled_tf_vs_target(
    label_pos: str,                   # e.g. '0psig_close'
    *,
    target_base: str = "data/final_target/",
    calib_base:  str = "data/final_calibration/",
    dataset_name: str = "H_fused",
    fmin: float = 100.0,
    fmax: float = 1000.0,
    to_db: bool = True,
    invert: bool = True,              # compare as "required |H|" = 1/target
    scale_fn=None,                    # from the fitter above
    rho_for_label: float | None = None,
    savepath: str | None = None,
):
    def _as_db(x): return 20.0*np.log10(np.maximum(np.asarray(x,float), 1e-16))

    if "_" not in label_pos:
        raise ValueError("label_pos must look like '0psig_close', '50psig_far', etc.")
    L, P = label_pos.split("_", 1)

    # target (amplitude ratio sqrt(data/model))
    tgt_path = os.path.join(target_base, f"target_{label_pos}.h5")
    with h5py.File(tgt_path, "r") as hf:
        f_tgt = np.asarray(hf["frequencies"][:], float)
        S_tgt = np.asarray(hf["scaling_ratio"][:], float)
        rho_t = float(hf.attrs.get("rho", np.nan))

    rho = rho_for_label if rho_for_label is not None else rho_t

    mB = (f_tgt >= fmin) & (f_tgt <= fmax)
    fB = f_tgt[mB]; S_tgtB = S_tgt[mB]

    # |H_cal| (per-label calibration)
    with h5py.File(os.path.join(calib_base, f"calibs_{L}.h5"), "r") as hf:
        f_cal = np.asarray(hf["frequencies"][:], float)
        H_cal = np.asarray(hf[dataset_name][:])
    Hc_mag = np.interp(fB, f_cal, np.abs(H_cal))

    # apply S(f,ρ) from fit (include position offset)
    if scale_fn is None:
        raise ValueError("scale_fn is required (use the fitter's returned function).")
    S = scale_fn(fB, rho, pos=P)
    S_model = Hc_mag * S

    # orientation for comparison
    if invert:
        tgt = 1.0/np.maximum(S_tgtB, 1e-16)   # required |H|
        mdl = 1.0/np.maximum(S_model, 1e-16)
        mode = "required |H| (inverted)"
    else:
        tgt = S_tgtB
        mdl = S_model
        mode = "as saved (sqrt(data/model))"

    rmse_db = float(np.sqrt(np.mean((_as_db(tgt)-_as_db(mdl))**2)))

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.4), tight_layout=True)
    if to_db:
        ax.semilogx(fB, _as_db(tgt), lw=1.6, label="target")
        ax.semilogx(fB, _as_db(mdl), lw=1.2, ls="--", label="|H_cal|×S")
        ax.set_ylabel("Magnitude [dB]")
    else:
        ax.semilogx(fB, tgt, lw=1.6, label="target")
        ax.semilogx(fB, mdl, lw=1.2, ls="--", label="|H_cal|×S")
        ax.set_ylabel("Magnitude [linear]")
    ax.set_xlim(fmin, fmax)
    ax.set_xlabel("f [Hz]")
    ax.grid(True, which="both", ls=":", alpha=0.7)
    ax.set_title(f"{label_pos} | RMSE={rmse_db:.2f} dB | {mode}")
    ax.legend(loc="best", fontsize=8)
    if savepath:
        fig.savefig(savepath, dpi=350)
    return fig, ax, rmse_db
