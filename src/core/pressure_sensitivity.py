from __future__ import annotations

import numpy as np

from src.config_params import Config

cfg = Config()


def correct_pressure_sensitivity(
    p_pa: np.ndarray,
    psig: float,
    alpha_db_per_kpa: float = 0.01,
) -> np.ndarray:
    """
    Compensate pressure-sensor sensitivity drift vs. gauge pressure.
    If sensitivity drops by ~alpha dB/kPa, multiply the pressure signal by:
      gain = 10^( + alpha_dB_per_kPa * p_kPa / 20 )
    """
    p_pa = np.asarray(p_pa, float)
    p_kpa = (float(psig) * cfg.PSI_TO_PA) / 1000.0
    gain = 10.0 ** (p_kpa * (alpha_db_per_kpa / 20.0))
    return p_pa * gain
