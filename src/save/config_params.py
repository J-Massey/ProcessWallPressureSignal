"""
Config parameters for the processing pipeline.
Keep all hard-coded paths and constants here so the rest of the pipeline stays consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    # --- Sampling / spectral defaults ---
    FS: float = 50_000.0
    NPERSEG: int = 2**12
    WINDOW: str = "hann"
    TRIM_CAL_SECS: int = 5

    # --- Physical constants ---
    R: float = 287.05        # J/kg/K
    PSI_TO_PA: float = 6_894.76
    P_ATM: float = 101_325.0
    DELTA: float = 0.035  # m, bl-height of 'channel'
    TDEG: tuple[float, float, float] = (18.0, 20.0, 22.0)
    TPLUS_CUT: float = 10.0  # picked so that we cut at half the inner peak

    # --- Data paths ---
    RAW_DIR: str = "data/final_pressure/wallp_raw"
    CALIB_DIR: str = field(init=False)
    TARGET_DIR: str = field(init=False)
    CLEANED_DIR: str = field(init=False)

    CAL_BASE: str = "data/final_calibration"
    TARGET_BASE: str = "data/final_target"
    CLEANED_BASE: str = "data/final_cleaned"
    RAW_BASE: str = "data/20251031"
    TONAL_BASE: str = "data/2025-10-28/tonal"
    SEMI_ANECHOIC_BASE: str = "data/20250930"
    FINAL_PRESSURE_DIR: str = "data/final_pressure"
    FINAL_CLEANED_DIR: str = "data/final_cleaned"
    PH_RAW_FILE: str = "data/final_pressure/G_wallp_SU_raw.hdf5"
    PH_PROCESSED_FILE: str = "data/final_pressure/G_wallp_SU_production.hdf5"
    NKD_RAW_FILE: str = "data/final_pressure/F_freestreamp_SU_raw.hdf5"
    NKD_PROCESSED_FILE: str = "data/final_pressure/F_freestreamp_SU_production.hdf5"

    # --- Experiment/run metadata defaults ---
    LABELS: tuple[str, str, str] = ("0psig", "50psig", "100psig")
    PSIGS: tuple[float, float, float] = (0.0, 50.0, 100.0)
    U_TAU: tuple[float, float, float] = (0.537, 0.522, 0.506)
    U_TAU_REL_UNC: tuple[float, float, float] = (0.2, 0.1, 0.05)
    U_E: float = 14.0
    ANALOG_LP_FILTER: tuple[int, int, int] = (2100, 4700, 14100)

    # --- Sensor constants ---
    SENSITIVITIES_V_PER_PA: dict[str, float] = field(
        default_factory=lambda: {
            "PH1": 50.9e-3,
            "PH2": 51.7e-3,
            "NC": 52.4e-3,
            "nkd": 50.9e-3,
        }
    )
    PREAMP_GAIN: dict[str, float] = field(
        default_factory=lambda: {"nc": 1.0, "PH1": 1.0, "PH2": 1.0, "NC": 1.0}
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "CALIB_DIR", f"{self.RAW_DIR}/calib")
        object.__setattr__(self, "TARGET_DIR", f"{self.RAW_DIR}/target")
        object.__setattr__(self, "CLEANED_DIR", f"{self.RAW_DIR}/cleaned")
