# phys_tf_calib.py - pressure-dependent physical FRFs (PH to NC) from semi-anechoic runs
from __future__ import annotations
from pathlib import Path
from typing import Sequence
from icecream import ic

import h5py
from scipy.io import loadmat

from src.core.phys_helpers import volts_to_pa  # expects volts to Pa conversion by channel
from src.core.pressure_sensitivity import correct_pressure_sensitivity
from src.core.tf_definition import estimate_frf, combine_anechoic_calibrations

from src.config_params import Config

cfg = Config()  # load the config parameters (file paths, constants, etc.) from a central location to ensure consistency

# ----------------- Main API: save per-pressure physical FRFs ---------- 
def save_PH_calibs(
    *,
    gmin: float = 0.4,
    smooth_oct: float = 1/6,
    points_per_oct: int = 32,
    eps: float = 1e-12
) -> None:
    """
    Build PH to NC H1 FRF for each pressure from dual-position (..._1, ..._2) semi-anechoic runs:
      - Convert both channels to Pa (volts_to_pa) and compensate mic sensitivity vs psig.
      - Estimate H1 with x=PH, y=NC, using welch/csd: H = conj(Sxy)/Sxx (SciPy's definition).
      - Fuse PH1 and PH2 FRFs on a **common frequency grid**, coherence-weighted, optionally smoothed.
      - Save **f_fused**, **H_fused (complex)**, optional raw H1/H2 and fused gamma2, and numeric psig.
    """
    base = Path(cfg.RAW_CAL_BASE) / "PH"
    out_dir = Path(cfg.TF_BASE) / "PH"
    out_dir.mkdir(parents=True, exist_ok=True)
    pressures = [int(p) for p in cfg.PSIGS]
    if len(cfg.F_CUTS) != len(pressures):
        raise ValueError("f_cuts length must match number of pressures")

    for p_si, fcut in zip(pressures, cfg.F_CUTS):
        psig = float(p_si)
        # ---- run 1: PH1 to NC
        m1 = loadmat(base / f"calib_{p_si}psig_1.mat")
        ph1_v, _, nc_v, *_ = m1["channelData_WN"].T
        ph1_pa = volts_to_pa(ph1_v, "PH1")
        nc1_pa = volts_to_pa(nc_v,  "NC")
        # compensate sensor sensitivity vs psig (amplitude gain)
        ph1_pa = correct_pressure_sensitivity(ph1_pa, psig)
        nc1_pa =  correct_pressure_sensitivity(nc1_pa,  psig)
        f1, H1, g2_1 = estimate_frf(
            ph1_pa,
            nc1_pa,
            fs=cfg.FS,
            window=cfg.WINDOW,
            nperseg=cfg.NPERSEG,
        )  # x=PH1, y=NC, H:=H_{PH_to_NC}

        # ---- run 2: PH2 to NC
        m2 = loadmat(base / f"calib_{p_si}psig_2.mat")
        _, ph2_v, nc_v2, *_ = m2["channelData_WN"].T
        ph2_pa = volts_to_pa(ph2_v, "PH2")
        nc2_pa = volts_to_pa(nc_v2,  "NC")
        ph2_pa = correct_pressure_sensitivity(ph2_pa, psig)
        nc2_pa =  correct_pressure_sensitivity(nc2_pa,  psig)
        f2, H2, g2_2 = estimate_frf(
            ph2_pa,
            nc2_pa,
            fs=cfg.FS,
            window=cfg.WINDOW,
            nperseg=cfg.NPERSEG,
        )  # x=PH2 to y=NC

        # ---- fuse to physical anchor on a **common grid** and optionally smooth
        f_fused, H_fused, g2_fused = combine_anechoic_calibrations(
            f1, H1, g2_1, f2, H2, g2_2,
            gmin=gmin, smooth_oct=smooth_oct, points_per_oct=points_per_oct, eps=eps
        )

        # ---- persist (note: save the fused frequency vector)
        out = out_dir / f"calibs_{p_si}.h5"   # or f"calibs_{p_si}psig.h5"
        with h5py.File(out, "w") as hf:
            hf.create_dataset("frequencies", data=f_fused)   # use fused grid
            hf.create_dataset("H1", data=H1)                 # optional raw
            hf.create_dataset("H2", data=H2)
            hf.create_dataset("H_fused", data=H_fused)       # complex, PH to NC
            hf.create_dataset("gamma2_fused", data=g2_fused)
            hf.attrs["psig"] = psig
            hf.attrs["orientation"] = "H = NC/PH (H1 = conj(Sxy)/Sxx with x=PH, y=NC)"
            hf.attrs["fs_Hz"] = cfg.FS
            hf.attrs["fcut_Hz"] = fcut

        print(f"[ok] {p_si:>3} psig to {out}")


def save_NC_calibs(
    *,
    fs: float | None = None,
    f_cuts: Sequence[float] | None = None,
) -> None:
    """
    Build PH to NC H1 FRF for each pressure from dual-position (..._1, ..._2) semi-anechoic runs:
      - Convert both channels to Pa (volts_to_pa) and compensate mic sensitivity vs psig.
      - Estimate H1 with x=PH, y=NC, using welch/csd: H = conj(Sxy)/Sxx (SciPy's definition).
      - Fuse PH1 and PH2 FRFs on a **common frequency grid**, coherence-weighted, optionally smoothed.
      - Save **f_fused**, **H_fused (complex)**, optional raw H1/H2 and fused gamma2, and numeric psig.
    """
    base = Path(cfg.RAW_CAL_BASE) / "NC"
    out_dir = Path(cfg.TF_BASE) / "NC"
    out_dir.mkdir(parents=True, exist_ok=True)

    pressures = [int(p) for p in cfg.PSIGS]
    fs = cfg.FS if fs is None else fs
    f_cuts = cfg.F_CUTS if f_cuts is None else f_cuts
    if len(f_cuts) != len(pressures):
        raise ValueError("f_cuts length must match number of pressures")

    for p_si, fcut in zip(pressures, f_cuts):
        psig = float(p_si)
        # ---- run 1: PH1 to NC
        m1 = loadmat(base / f"{p_si}psig/nkd-ns_nofacilitynoise.mat")
        ic(m1.keys())
        if p_si == 100:
            nkd, nc = m1["channelData_nofacitynoise"].T
        else:
            nkd, nc = m1["channelData"].T

        f1, H1, g2_1 = estimate_frf(
            nc,
            nkd,
            fs=fs,
            window=cfg.WINDOW,
            nperseg=cfg.NPERSEG,
        )
        # ---- persist (note: save the fused frequency vector)
        out = out_dir / f"calibs_{p_si}.h5"   # or f"calibs_{p_si}psig.h5"
        with h5py.File(out, "w") as hf:
            hf.create_dataset("frequencies", data=f1)   # use fused grid
            hf.create_dataset("H_fused", data=H1)       # complex, PH to NC
            hf.create_dataset("gamma2_fused", data=g2_1)
            hf.attrs["psig"] = psig
            hf.attrs["orientation"] = "H = NC/PH (H1 = conj(Sxy)/Sxx with x=nc, y=nkd)"
            hf.attrs["fs_Hz"] = fs
            hf.attrs["fcut_Hz"] = fcut
        print(f"[ok] {p_si:>3} psig to {out}")

# --------------- example CLI ---------------------------------------------------
if __name__ == "__main__":
    save_PH_calibs()
    save_NC_calibs()
