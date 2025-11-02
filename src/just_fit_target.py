#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit a diaphragm-only LEM envelope to the target TF magnitude (100–1000 Hz)
and save parameters per dataset.

Model (Helmholtz assumed out-of-band):
    |H_LEM(jω)| ∝ | (jω) / (1 - (ω/ωD)^2 + j ω/(Q_D ωD)) |

We fit three parameters on each target (amplitude) dataset:
    - g_db : overall gain in dB
    - fD   : diaphragm natural frequency [Hz]
    - QD   : diaphragm quality factor [-]

INPUT FILES (one per dataset):
    TARGET_BASE/target_{label}.h5
      - datasets: 'frequencies' [Hz], 'scaling_ratio' [amplitude]
      - attrs   : 'rho','u_tau','nu','psig'

OUTPUT:
    TARGET_BASE/lem_params.h5 with one group per label:
      attrs: g_db, fD_Hz, QD, rmse_db, npts, fmin, fmax, rho, psig, timestamp

Labels covered (edit as needed):
    0psig_close, 0psig_far, 50psig_close, 50psig_far, 100psig_close, 100psig_far
"""

from __future__ import annotations
import h5py
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
from scipy.optimize import least_squares
from datetime import datetime, timezone
import os

# ---------------------------------------------------------------------
# Paths / labels
# ---------------------------------------------------------------------
CLEANED_BASE = "data/final_cleaned/"
TARGET_BASE  = "data/final_target/"

LABELS = (
    "0psig_close",
    "0psig_far",
    "50psig_close",
    "50psig_far",
    "100psig_close",
    "100psig_far",
)

# Fit band (away from Helmholtz)
FMIN = 100.0
FMAX = 1000.0


# ---------------------------------------------------------------------
# LEM diaphragm-only magnitude (DC zero × 2nd-order diaphragm)
# ---------------------------------------------------------------------
def _lem_mag_diaph(f: np.ndarray, fD: float, QD: float) -> np.ndarray:
    """
    |H_LEM| ∝ |(jω) / (1 - (ω/ωD)^2 + j ω/(Q_D ωD))|
    """
    f = np.asarray(f, float)
    w  = 2.0 * np.pi * f
    wD = 2.0 * np.pi * float(fD)
    den = (1.0 - (w / wD)**2) + 1j * (w / (QD * wD))
    H = (1j * w) / den
    return np.abs(H)


# ---------------------------------------------------------------------
# Fit one target (amplitude) curve in [FMIN, FMAX]
# ---------------------------------------------------------------------
def fit_lem_target(f_tgt: np.ndarray, S_tgt: np.ndarray,
                   fmin: float = FMIN, fmax: float = FMAX, invert_in_fit: bool = True) -> Tuple[Dict[str, float], float, int]:
    """
    Returns: (params_dict, rmse_db, n_used)
      params_dict = {'g_db':..., 'fD':..., 'QD':...}
    """
    f_tgt = np.asarray(f_tgt, float)
    S_tgt = np.asarray(S_tgt, float)

    m = np.isfinite(f_tgt) & np.isfinite(S_tgt) & (S_tgt > 0)
    f_tgt, S_tgt = f_tgt[m], S_tgt[m]

    band = (f_tgt >= float(fmin)) & (f_tgt <= float(fmax))
    fB, SB = f_tgt[band], S_tgt[band]
    if invert_in_fit:
        SB = 1.0 / np.maximum(SB, 1e-16)   # <--- use required magnitude
    y_db = 20.0 * np.log10(np.maximum(SB, 1e-16))

    def unpack(theta):
        g_db = float(theta[0])
        fD   = float(np.exp(theta[1]))
        QD   = float(np.exp(theta[2]))
        return g_db, fD, QD

    def resid(theta):
        g_db, fD, QD = unpack(theta)
        H = _lem_mag_diaph(fB, fD=fD, QD=QD)
        y_hat = g_db + 20.0 * np.log10(np.maximum(H, 1e-16))
        return (y_db - y_hat)

    # Initial guess (robust, guaranteed feasible)
    H0 = _lem_mag_diaph(fB, 500.0, 10.0)
    g0_raw = float(np.median(y_db - 20.0 * np.log10(np.maximum(H0, 1e-16))))

    # Bounds: relax gain a bit to avoid infeasible seeds on odd targets
    lb = np.array([-80.0, np.log(150.0),  np.log(1.0)],    float)
    ub = np.array([+80.0, np.log(3000.0), np.log(300.0)],  float)

    # Seed safely inside bounds
    eps = 1e-6
    g0 = float(np.clip(g0_raw, lb[0] + eps, ub[0] - eps))
    x0 = np.array([g0, np.log(500.0), np.log(10.0)], float)
    x0 = np.minimum(np.maximum(x0, lb + eps), ub - eps)

    sol = least_squares(resid, x0, bounds=(lb, ub), method="trf")

    g_db, fD, QD = unpack(sol.x)

    # RMSE in dB
    Hf = _lem_mag_diaph(fB, fD=fD, QD=QD)
    y_hat = g_db + 20.0 * np.log10(np.maximum(Hf, 1e-16))
    rmse_db = float(np.sqrt(np.mean((y_db - y_hat)**2)))

    return dict(g_db=g_db, fD=fD, QD=QD), rmse_db, int(fB.size)


# ---------------------------------------------------------------------
# Main: loop over labels, fit, and save
# ---------------------------------------------------------------------
def main():
    os.makedirs(TARGET_BASE, exist_ok=True)
    out_path = os.path.join(TARGET_BASE, "lem_params.h5")
    ts = datetime.now(timezone.utc).isoformat()

    with h5py.File(out_path, "w") as hf_out:
        hf_out.attrs["note"] = "Diaphragm-only LEM fit on target amplitude in 100–1000 Hz"
        hf_out.attrs["timestamp_utc"] = ts
        hf_out.attrs["model"] = "|H| ∝ |(jω)/(1-(ω/ωD)^2 + j ω/(Q_D ωD))|"
        hf_out.attrs["band_fmin_Hz"] = FMIN
        hf_out.attrs["band_fmax_Hz"] = FMAX

        for label in LABELS:
            tgt_path = os.path.join(TARGET_BASE, f"target_{label}.h5")
            if not os.path.exists(tgt_path):
                print(f"[WARN] Missing target file: {tgt_path} — skipping.")
                continue

            with h5py.File(tgt_path, "r") as hf_t:
                f_tgt = np.asarray(hf_t["frequencies"][:], float)
                S_tgt = np.asarray(hf_t["scaling_ratio"][:], float)  # already AMPLITUDE
                rho   = float(hf_t.attrs.get("rho", np.nan))
                psig  = float(hf_t.attrs.get("psig", np.nan))

            params, rmse_db, npts = fit_lem_target(f_tgt, S_tgt, fmin=FMIN, fmax=FMAX)

            grp = hf_out.create_group(label)
            grp.attrs["g_db"]   = float(params["g_db"])
            grp.attrs["fD_Hz"]  = float(params["fD"])
            grp.attrs["QD"]     = float(params["QD"])
            grp.attrs["rmse_db"] = rmse_db
            grp.attrs["npts_used"] = npts
            grp.attrs["rho"]    = rho
            grp.attrs["psig"]   = psig
            grp.attrs["timestamp_utc"] = ts

            print(f"{label:>14s}  g={params['g_db']:+6.2f} dB, "
                  f"fD={params['fD']:6.1f} Hz, QD={params['QD']:5.1f}  | RMSE={rmse_db:4.2f} dB")

    print(f"\nSaved LEM parameters to: {out_path}")


if __name__ == "__main__":
    main()
