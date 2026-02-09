"""
Run the full processing pipeline and then all plots.
"""

from __future__ import annotations

from src.save.run_all import run_all as run_processing
from src.checks.plot.run_all import run_all as run_plots


def run_pipeline() -> None:
    run_processing()
    run_plots()


if __name__ == "__main__":
    run_pipeline()
