PressureProcess
===============

This module performs pressure processing for two microphones. It ingests raw
mat files, computes calibration transfer functions, saves raw HDF5 products,
then applies those transfer functions to produce processed HDF5 outputs. It
also includes plotting sanity checks.

Install
-------
From the repository root:
```
python -m pip install -e .
```
or:
```
python -m pip install .
```

Usage overview
--------------
1) Define the required file structure and raw variable names.
2) Set user parameters in `src/config_params.py`.
3) Run the pipeline with `python -m src.run_pipeline`.

Required file structure
-----------------------
All paths are rooted at `ROOT_DIR` in `src/config_params.py`.

Raw wall pressure mat files
```
data/phase1/raw_wallp/
  close/
    0psig.mat
    50psig.mat
    100psig.mat
  far/
    0psig.mat
    50psig.mat
    100psig.mat
```

Raw calibration mat files for PH to NC
```
data/phase1/raw_calib/PH/
  calib_0psig_1.mat
  calib_0psig_2.mat
  calib_50psig_1.mat
  calib_50psig_2.mat
  calib_100psig_1.mat
  calib_100psig_2.mat
```

Raw calibration mat files for NC to NKD
```
data/phase1/raw_calib/NC/
  0psig/
    nkd-ns_nofacilitynoise.mat
  50psig/
    nkd-ns_nofacilitynoise.mat
  100psig/
    nkd-ns_nofacilitynoise.mat
```

Raw variable names expected in mat files
----------------------------------------
For wall pressure and freestream raw files, the loader expects:
```
channelData
```
with columns ordered as:
```
PH1, PH2, NC
```

For PH to NC calibration files, the loader expects:
```
channelData_WN
```
with columns ordered as:
```
PH1, PH2, NC, ...
```

For NC to NKD calibration files, the loader expects:
```
channelData
```
and for 100psig only:
```
channelData_nofacitynoise
```

Configure user parameters
-------------------------
Edit `src/config_params.py` to set:
 - ROOT_DIR
 - RAW_CAL_BASE, RAW_BASE
 - TF_BASE
 - PH_RAW_FILE, PH_PROCESSED_FILE
 - NKD_RAW_FILE, NKD_PROCESSED_FILE
 - LABELS, PSIGS, U_TAU, U_TAU_REL_UNC, U_E, ANALOG_LP_FILTER, F_CUTS
 - SENSITIVITIES_V_PER_PA
 - SPACINGS
 - RUN_NC_CALIBS
 - INCLUDE_NC_CALIB_RAW

Sarah iso option
----------------
If you are using the sarah iso layout where only one spacing is measured and
the NC calibration is not rerun, set:
 - SPACINGS to ("close",) or ("far",)
 - RUN_NC_CALIBS to False
 - INCLUDE_NC_CALIB_RAW to False

Run the pipeline
----------------
```
python -m src.run_pipeline
```

What run_pipeline does
----------------------
`src/run_pipeline.py` calls two runner scripts in order.

1) Processing runner: `src/save/run_all.py`
   - `calibs.save_PH_calibs()` and `calibs.save_NC_calibs()`
   - `fs_raw.save_raw_fs_pressure()`
   - `pw_raw.save_raw_ph_pressure()`
   - `fs_proc.save_prod_fs_pressure()`
   - `pw_proc.save_corrected_pressure()`

2) Plot runner: `src/checks/plot/run_all.py`
   - `F_freestreamp_SU_raw.plot_fs_raw()`
   - `F_freestreamp_SU_production.plot_fs_raw()`
   - `G_wallp_SU_raw.plot_raw()`
   - `G_wallp_SU_production.plot_model_comparison_roi()`
   - `SU_two_point.plot_2pt_inner()`
   - `SU_two_point.plot_2pt_outer()`
   - `SU_two_point.plot_2pt_speed_outer()`
   - `SU_two_point.plot_2pt_speed_inner()`
