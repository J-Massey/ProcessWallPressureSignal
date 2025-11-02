from fit_powerlaw_with_positions import fit_speaker_scaling_from_files_with_positions
from plot_scaled_tf_vs_target import plot_scaled_tf_vs_target
from plot_raw_vs_scaled_corrected import plot_raw_vs_scaled_corrected

# 1) Fit once
(params, scale, diag) = fit_speaker_scaling_from_files_with_positions(
    labels=("0psig","50psig","100psig"),
    positions=("close","far"),
    f_ref=700.0, fmin=100.0, fmax=1000.0,
    invert_target=False,
    TARGET_BASE="data/final_target/",
    CALIB_BASE="data/final_calibration/",
    dataset_name="H_fused",
)

# 2) Quick checks
for s in ("0psig_close","0psig_far","50psig_close","50psig_far","100psig_close","100psig_far"):
    plot_scaled_tf_vs_target(s, scale_fn=scale, savepath=f"figures/scaled_tf_vs_target_{s}.png")
    plot_raw_vs_scaled_corrected(s, scale_fn=scale, savepath=f"figures/raw_vs_scaled_{s}.png")
