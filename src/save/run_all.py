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

from src.save import calibs, fs_raw, pw_raw, fs_proc, pw_proc


def run_all() -> None:
    print("[1] calibs: PH + NC")
    calibs.save_PH_calibs()
    calibs.save_NC_calibs()

    print("[2] pw_raw")
    pw_raw.save_raw_ph_pressure()

    print("[3] fs_raw")
    fs_raw.save_raw_fs_pressure()

    print("[4a] fs_proc")
    fs_proc.save_prod_fs_pressure()

    print("[4b] pw_proc")
    pw_proc.save_corrected_pressure()



if __name__ == "__main__":
    run_all()
