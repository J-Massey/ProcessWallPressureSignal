#!/usr/bin/env python
"""
duct_mode_table.py
------------------
Generate a PNG image of a LaTeX table listing quarter-wave acoustic duct modes
for a rectangular (square) duct with mean flow.
"""

from __future__ import annotations
import numpy as np

import tempfile
import subprocess
from pathlib import Path

from icecream import ic

# --------------------------- core functions -------------------------------- #
def speed_of_sound(p: float, rho: float) -> float:
    """
    Ideal‑gas speed of sound   c = sqrt( γ * p / ρ )

    Parameters
    ----------
    p   : static pressure [Pa]
    rho : density        [kg m⁻³]

    Returns
    -------
    c   : speed of sound [m s⁻¹]
    """
    return np.sqrt(GAMMA_AIR * p / rho)


def duct_mode_freq(U, c, m, n, l, W, H, L):
    """Physical duct-mode frequency (quarter-wave, closed-open)."""
    delta2 = c**2 - U**2
    k_sq   = (m*np.pi/W)**2 + (n*np.pi/H)**2
    kz_sq  = ((2*l+1)*np.pi/L)**2
    return (1/(2*np.pi)) * np.sqrt(delta2*k_sq + delta2**2/(4*c**2)*kz_sq)


# ------------------------- LaTeX generation -------------------------------- #
TEMPLATE = r"""
\documentclass[varwidth,border=6pt]{standalone}
\usepackage{booktabs,siunitx}
\sisetup{
  detect-all,
  round-mode = places,
  round-precision = 1,
  table-format = 4.0
}
\begin{document}
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{@{}cccc@{}}
\toprule
$\ell$ & $m$ & $n$ & $f$ / \si{\hertz} \\
\midrule
%s
\bottomrule
\end{tabular}\\[6pt]
\small Pressure = %g\,Pa,\quad Mean speed = %g\,m\,s$^{-1}$
\end{document}
"""

def table_to_png(rows: list[tuple[int,int,int,float]], p: float, U: float, out_png: Path):
    """
    Fill the LaTeX template, run pdflatex, and convert page to PNG (via Ghostscript).

    Requires: a working LaTeX system (pdflatex) and Ghostscript (`gs`).
    """
    # rows → LaTeX table lines
    body = "\n".join(f"{l} & {m} & {n} & {f:.1f} \\\\" for l, m, n, f in rows)
    tex  = TEMPLATE % (body, p, U)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        texfile = tmpdir / "table.tex"
        texfile.write_text(tex)

        # compile LaTeX
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", texfile.name],
            cwd=tmpdir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # convert first page of resulting PDF to PNG at 300 dpi
        subprocess.run(
            [
                "gs",
                "-q",
                "-dNOPAUSE",
                "-dBATCH",
                "-r300",
                "-sDEVICE=pngalpha",
                f"-sOutputFile={out_png}",
                str(texfile.with_suffix(".pdf")),
            ],
            cwd=tmpdir,
            check=True,
        )


if __name__ == "__main__":
    # ---- 1. fixed user inputs --------------------------------------------- #
    GAMMA_AIR = 1.4
    R_AIR     = 287.05       # J kg⁻¹ K⁻¹
    W, H = 0.30, 0.152          # duct width & height (m)
    L0   = 3.0                  # duct length (m)
    DELTA_L0 = 0.1 * L0         # low‑frequency end correction
    L    = L0 + DELTA_L0        # effective duct length

    Us         = [9, 14]                                  # mean speeds (m/s)
    pressures  = [101_325, 344_738, 413_685, 482_633]     # static p (Pa abs)
    T_K        = 293.15                                   # temperature (K)

    M_MAX, N_MAX, L_MAX = 3, 3, 10                        # scan limits

    # ---- 2. sweep every (U, p) pair -------------------------------------- #
    for U in Us:
        for p in pressures:
            rho = p / (R_AIR * T_K)       # ideal‑gas density [kg m⁻³]
            c   = speed_of_sound(p, rho)  # speed of sound   [m s⁻¹]

            rows = []                     # <-- this will hold (ℓ,m,n,f)
            for m in range(M_MAX + 1):
                for n in range(N_MAX + 1):
                    for l in range(L_MAX + 1):
                        f_phys = duct_mode_freq(U, c, m, n, l, W, H, L)
                        if 10.0 <= f_phys <= 1000.0:       # keep only 10‑1000 Hz
                            rows.append((l, m, n, f_phys))

            # ---- 3. write PNG -------------------------------------------- #
            out_png = Path.cwd() / f"figures/duct_tables/duct_modes_p{p}_U{U}.png"
            table_to_png(rows, p, U, out_png)
            print(f"✓  {out_png.name} written")
