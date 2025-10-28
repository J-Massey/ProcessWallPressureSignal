# tf_plot.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import sys

# Try to keep your original style; fall back gracefully if LaTeX/scienceplots missing
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "grid"])
except Exception:  # pragma: no cover
    plt.style.use("default")
try:
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")
except Exception:  # pragma: no cover
    plt.rc("text", usetex=False)
plt.rcParams["font.size"] = "10.5"

# Import shared constants/paths/colors from compute module (filename starts with digits, so load by path)
_compute_path = Path(__file__).with_name("2025_10_28_tf_compute.py")
_spec = importlib.util.spec_from_file_location("tf_compute", _compute_path)
if _spec is None or _spec.loader is None:  # pragma: no cover
    raise ImportError(f"Could not load module from {_compute_path}")
_tf_compute = importlib.util.module_from_spec(_spec)
sys.modules["tf_compute"] = _tf_compute
_spec.loader.exec_module(_tf_compute)

# Re-export the needed names locally for convenience
CalibCase = _tf_compute.CalibCase
OUTPUT_DIR = _tf_compute.OUTPUT_DIR
CAL_DIR_FAR = _tf_compute.CAL_DIR_FAR
CAL_DIR_CLOSE = _tf_compute.CAL_DIR_CLOSE
CAL_DIR_COMB = _tf_compute.CAL_DIR_COMB
PH1_COLOR = _tf_compute.PH1_COLOR
PH2_COLOR = _tf_compute.PH2_COLOR

# =============================================================================
# Plot registry
# =============================================================================
PLOT_REGISTRY: Dict[str, Callable[["PlotContext", CalibCase], None]] = {}


def register_plot(name: str, help: str):
    """Decorator to register a plot function."""
    def _decorator(fn: Callable[["PlotContext", CalibCase], None]):
        fn._help = help  # type: ignore[attr-defined]
        PLOT_REGISTRY[name] = fn
        return fn
    return _decorator


@dataclass
class PlotContext:
    """Holds IO and helpers so plot functions stay tiny and easy to add."""
    outdir: Path = OUTPUT_DIR

    # -------- IO helpers (match the filenames saved by tf_compute) --------
    def load_triplet(self, base_dir: Path, tag: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        f = np.load(base_dir / f"f{tag}.npy")
        H = np.load(base_dir / f"H{tag}.npy")
        g = np.load(base_dir / f"gamma{tag}.npy")
        return f, H, g

    def load_comb(self, stem: str) -> np.ndarray:
        # For combined outputs saved as "<stem>.npy"
        return np.load(CAL_DIR_COMB / f"{stem}.npy")

    def savefig(self, filename: Path) -> None:
        self.outdir.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, dpi=600)
        plt.close()


# =============================================================================
# Concrete plots (drop-in extendable)
# =============================================================================
@register_plot("overlay", help="Overlay H(f) for PH1/PH2 across far/close + in-situ")
def plot_overlay(ctx: PlotContext, case: CalibCase) -> None:
    runs: List[Tuple[str, np.ndarray, np.ndarray]] = []

    # far/close anechoic
    for label, base_dir in (("far", CAL_DIR_FAR), ("close", CAL_DIR_CLOSE)):
        for ph in ("1", "2"):
            tag = f"{ph}_700_{case.psi_g}"
            f, H, _ = ctx.load_triplet(base_dir, tag)
            runs.append((f"PH{ph} {label}", f, H))

    # in-situ (no-flow)
    for label, base_dir in (("far", CAL_DIR_FAR), ("close", CAL_DIR_CLOSE)):
        for ph in ("1", "2"):
            tag = f"{ph}_700_{case.psi_label}_is"
            f, H, _ = ctx.load_triplet(base_dir, tag)
            runs.append((f"PH{ph} in-situ {label}", f, H))

    fig, (ax_mag, ax_ph) = plt.subplots(2, 1, sharex=True, figsize=(9, 3))
    ax_mag.set_title(r"$H_{\mathrm{PH-NC}}$ " + case.title_suffix + ", with suggested cutoffs")
    for label, f, H in runs:
        color = PH1_COLOR if "PH1" in label else PH2_COLOR if "PH2" in label else "k"
        ls = "--" if "close" in label else (":" if "in-situ" in label and "PH1" in label else "-.")
        mag = np.abs(H); phase = np.unwrap(np.angle(H))
        ax_mag.loglog(f, mag, lw=1, color=color, ls=ls, label=label)
        ax_ph.semilogx(f, phase, lw=1, color=color, ls=ls)

    ax_mag.legend(fontsize=8, ncol=2)
    ax_mag.set_ylabel(r"$|H_{\mathrm{PH-NC}}(f)|$")
    ax_mag.axvline(case.f_cut, color="red", linestyle="--", lw=1)
    ax_mag.text(case.f_cut, 10, fr"$T^+ \approx {case.Tplus_at_fcut:.1f}$", color="red",
                va="center", ha="right", rotation=90)

    ax_ph.set_ylabel(r"$\angle H_{\mathrm{PH-NC}}(f)\,[\mathrm{rad}]$")
    ax_ph.set_xlabel(r"$f\ \mathrm{[Hz]}$")
    ax_ph.axvline(case.f_cut, color="red", linestyle="--", lw=1)

    fig.tight_layout()
    ctx.savefig(OUTPUT_DIR / f"700_{case.psi_label}_H_2cal.png")


@register_plot("anechoic_fused", help="Fused anechoic H(f) for PH1 and PH2")
def plot_anechoic_fused(ctx: PlotContext, case: CalibCase) -> None:
    f1 = ctx.load_comb(f"700_{case.psi_label}_fused_anechoic_f1")
    H1 = ctx.load_comb(f"700_{case.psi_label}_fused_anechoic_H1")
    f2 = ctx.load_comb(f"700_{case.psi_label}_fused_anechoic_f2")
    H2 = ctx.load_comb(f"700_{case.psi_label}_fused_anechoic_H2")

    fig, (ax_mag, ax_ph) = plt.subplots(2, 1, sharex=True, figsize=(9, 3))
    ax_mag.set_title(r"Fused $H_{\mathrm{PH-NC}}$ " + case.title_suffix + ", with suggested cutoffs")

    for f, H, label, color in [(f1, H1, "PH1 lab fused", PH1_COLOR), (f2, H2, "PH2 lab fused", PH2_COLOR)]:
        mag = np.abs(H); phase = np.unwrap(np.angle(H))
        ax_mag.loglog(f, mag, lw=1, color=color, label=label)
        ax_ph.semilogx(f, phase, lw=1, color=color)

    ax_mag.set_ylabel(r"$|H_{\mathrm{PH-NC}}(f)|$")
    ax_mag.axvline(case.f_cut, color="red", linestyle="--", lw=1)
    ax_mag.text(case.f_cut, 10, fr"$T^+ \approx {case.Tplus_at_fcut:.1f}$", color="red",
                va="center", ha="right", rotation=90)

    ax_ph.set_ylabel(r"$\angle H_{\mathrm{PH-NC}}(f)\,[\mathrm{rad}]$")
    ax_ph.set_xlabel(r"$f\ \mathrm{[Hz]}$")
    ax_ph.axvline(case.f_cut, color="red", linestyle="--", lw=1)
    ax_mag.legend(fontsize=8)

    fig.tight_layout()
    ctx.savefig(OUTPUT_DIR / f"700_{case.psi_label}_H_anechoic_fused.png")


@register_plot("final_fused", help="Final H(f) after incorporating in-situ (PH1 and PH2)")
def plot_final_fused(ctx: PlotContext, case: CalibCase) -> None:
    f1 = ctx.load_comb(f"700_{case.psi_g}_fused_insitu_f1")
    H1 = ctx.load_comb(f"700_{case.psi_g}_fused_insitu_H1")
    f2 = ctx.load_comb(f"700_{case.psi_g}_fused_insitu_f2")
    H2 = ctx.load_comb(f"700_{case.psi_g}_fused_insitu_H2")

    fig, (ax_mag, ax_ph) = plt.subplots(2, 1, sharex=True, figsize=(9, 3))
    ax_mag.set_title(r"Final $H_{\mathrm{PH-NC}}$ " + case.title_suffix + ", with suggested cutoffs")

    for f, H, color in [(f1, H1, PH1_COLOR), (f2, H2, PH2_COLOR)]:
        mag = np.abs(H); phase = np.unwrap(np.angle(H))
        ax_mag.loglog(f, mag, lw=1, color=color)
        ax_ph.semilogx(f, phase, lw=1, color=color)

    ax_mag.set_ylabel(r"$|H_{\mathrm{PH-NC}}(f)|$")
    ax_mag.set_ylim(1e-3, 100)  # from your original figure
    ax_mag.axvline(case.f_cut, color="red", linestyle="--", lw=1)
    ax_mag.text(case.f_cut, 10, fr"$T^+ \approx {case.Tplus_at_fcut:.1f}$", color="red",
                va="center", ha="right", rotation=90)

    ax_ph.set_ylabel(r"$\angle H_{\mathrm{PH-NC}}(f)\,[\mathrm{rad}]$")
    ax_ph.set_xlabel(r"$f\ \mathrm{[Hz]}$")
    ax_ph.axvline(case.f_cut, color="red", linestyle="--", lw=1)

    fig.tight_layout()
    ctx.savefig(OUTPUT_DIR / f"700_{case.psi_label}_H_fuse_situ.png")


@register_plot("gamma_fused", help="γ² used in anechoic fusion (PH1/PH2)")
def plot_gamma_fused(ctx: PlotContext, case: CalibCase) -> None:
    f1 = ctx.load_comb(f"700_{case.psi_label}_fused_anechoic_f1")
    g1 = ctx.load_comb(f"700_{case.psi_label}_fused_anechoic_gamma1")
    f2 = ctx.load_comb(f"700_{case.psi_label}_fused_anechoic_f2")
    g2 = ctx.load_comb(f"700_{case.psi_label}_fused_anechoic_gamma2")

    fig, ax = plt.subplots(1, 1, figsize=(9, 2))
    ax.set_title(r"Coherence $\gamma^2$ used in calibration fusion " + case.title_suffix)
    ax.semilogx(f1, g1, lw=1, color=PH1_COLOR, label="PH1 lab fused")
    ax.semilogx(f2, g2, lw=1, color=PH2_COLOR, label="PH2 lab fused")
    ax.set_ylabel(r"$\gamma^2_{\mathrm{PH-NC}}(f)$")
    ax.set_xlabel(r"$f\ \mathrm{[Hz]}$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    ctx.savefig(OUTPUT_DIR / f"700_{case.psi_label}_gamma_fuse.png")


# =============================================================================
# Driver
# =============================================================================
DEFAULT_PLOTS = ["overlay", "anechoic_fused", "final_fused", "gamma_fused"]


def _cases_list(sel: str) -> List[str]:
    if sel.strip().lower() == "all":
        return ["0psi", "50psi", "100psi"]
    return [s.strip() for s in sel.split(",") if s.strip()]


def _case_obj(lbl: str) -> CalibCase:
    table = {
        "0psi":   CalibCase("0psi",   u_tau=0.58, nu_utau=27e-6,  f_cut=2_100.0),
        "50psi":  CalibCase("50psi",  u_tau=0.47, nu_utau=7.5e-6, f_cut=4_700.0),
        "100psi": CalibCase("100psi", u_tau=0.52, nu_utau=3.7e-6, f_cut=14_100.0),
    }
    return table[lbl]


def run_plots(*, plots: Iterable[str], cases: Iterable[str], outdir: Path = OUTPUT_DIR) -> None:
    ctx = PlotContext(outdir=outdir)
    for c in cases:
        case = _case_obj(c)
        for p in plots:
            if p not in PLOT_REGISTRY:
                raise KeyError(f"Unknown plot '{p}'. Known: {list(PLOT_REGISTRY)}")
            print(f"[tf_plot] case={c} plot={p}")
            PLOT_REGISTRY[p](ctx, case)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot-only frontend for TF calibration.")
    parser.add_argument("--plots", type=str, default=",".join(DEFAULT_PLOTS),
                        help=f"Comma-separated plot names or 'all'. Use --list-plots to see options.")
    parser.add_argument("--cases", type=str, default="0psi,50psi,100psi",
                        help="Comma-separated list (e.g. 0psi,100psi) or 'all'.")
    parser.add_argument("--outdir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for figures.")
    parser.add_argument("--list-plots", action="store_true", help="List available plot functions and exit.")
    args = parser.parse_args()

    if args.list_plots:
        print("Available plots:")
        for name, fn in PLOT_REGISTRY.items():
            desc = getattr(fn, "_help", "")
            print(f"  - {name:15s} {desc}")
        raise SystemExit(0)

    plots = list(PLOT_REGISTRY) if args.plots.strip().lower() == "all" else \
            [s.strip() for s in args.plots.split(",") if s.strip()]
    cases = _cases_list(args.cases)
    run_plots(plots=plots, cases=cases, outdir=Path(args.outdir))
