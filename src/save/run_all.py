"""
Run the full processing pipeline from raw .mat files.

Order:
1) calibs
2a) fs_raw
2b) pw_raw
3a) fs_proc
3b) pw_proc
"""

from __future__ import annotations

from src.config_params import Config
from src.save import calibs, fs_raw, pw_raw, fs_proc, pw_proc


def run_all(
    *,
    run_nc_calibs: bool | None = None,
    include_nc_calib_raw: bool | None = None,
    spacings: tuple[str, ...] | None = None,
) -> None:
    cfg = Config()
    run_nc_calibs = cfg.RUN_NC_CALIBS if run_nc_calibs is None else run_nc_calibs
    include_nc_calib_raw = (
        cfg.INCLUDE_NC_CALIB_RAW if include_nc_calib_raw is None else include_nc_calib_raw
    )
    spacings = cfg.SPACINGS if spacings is None else spacings

    print("[1] calibs: PH" + (" + NC" if run_nc_calibs else ""))
    calibs.save_PH_calibs()
    if run_nc_calibs:
        calibs.save_NC_calibs()

    print("[2a] fs_raw")
    fs_raw.save_raw_fs_pressure(
        spacings=spacings,
        include_nc_calib=include_nc_calib_raw,
    )

    print("[2b] pw_raw")
    pw_raw.save_raw_ph_pressure(spacings=spacings)

    print("[3a] fs_proc")
    fs_proc.save_prod_fs_pressure(spacings=spacings)

    print("[3b] pw_proc")
    pw_proc.save_corrected_pressure(spacings=spacings)



if __name__ == "__main__":
    run_all()
