# tf_corrected_spectra_roi.py  —  raw vs. calibration‑corrected spectra with pressure‑invariant source

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import h5py
from scipy.io import loadmat
from scipy.signal import welch, csd, get_window, butter, sosfiltfilt
from matplotlib import pyplot as plt

# ---------------- basic constants / Welch
FS: float = 50_000.0
NPERSEG: int = 2**12
WINDOW: str = "hann"
PSI_TO_PA: float = 6_894.76
PA_TO_KPA: float = 1e-3
PSI_TO_KPA: float = PSI_TO_PA * PA_TO_KPA

# ---------------- chain sensitivities (V/Pa)
CHAIN_SENS_V_PER_PA: Dict[str, float] = {
    "nc":  52.4e-3,
    "ph1": 50.9e-3,
    "ph2": 51.7e-3,
}

# ---------------- I/O locations
CALIB_BASE = Path("data/final_calibration")
CLEANED_BASE = Path("data/final_cleaned")
FIG_DIR = Path("figures/final"); FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- plotting band
FMIN, FMAX = 100.0, 1000.0

# ---------------- anti‑alias / cleanup bandpass cutoffs
F_CUT_BY_LABEL = {"0psig": 2_100.0, "50psig": 4_700.0, "100psig": 14_100.0}

# ---------------- source‑invariance you FIT elsewhere (plug your numbers here)
SRC_Z_MODEL = "rho_c"   # "rho_c" (rho*c) or "rho"
SRC_B_DENSITY = -0.12   # <- put your fitted b here (from your cal‑only validation)

# =====================================================================
# small helpers
# =====================================================================
def _psig_to_float(label: str) -> float:
    # "50psig" -> 50.0 ; "0" -> 0.0
    s = str(label).strip().lower()
    return float(s.replace("psig", ""))

def pressure_sens_gain(psig: float, slope_db_per_kpa: float = -0.01) -> float:
    """Sensor sensitivity change (amplitude) with gauge pressure."""
    kpa = float(psig) * PSI_TO_KPA
    return 10.0 ** (slope_db_per_kpa * kpa / 20.0)

def bandpass_filter(x: np.ndarray, fs: float, f_low: float, f_high: float, order: int = 4) -> np.ndarray:
    x = np.asarray(x, float)
    f_high = min(f_high, 0.45 * fs)
    sos = butter(order, [max(0.1, f_low), f_high], btype="band", fs=fs, output="sos")
    y = sosfiltfilt(sos, x)
    return np.nan_to_num(y, copy=False)

def volts_to_pa_corrected(x_volts: np.ndarray, *, channel: str, psig: float, f_high: float) -> np.ndarray:
    """Volts -> Pa and apply the -0.01 dB/kPa chain sensitivity for this pressure."""
    sens_v_per_pa = CHAIN_SENS_V_PER_PA[channel.lower()]
    x_bp = bandpass_filter(x_volts, FS, f_low=0.1, f_high=f_high)
    x_pa = x_bp / sens_v_per_pa
    return x_pa * pressure_sens_gain(psig)

def compute_spec(fs: float, x: np.ndarray, nperseg: int = NPERSEG) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    nseg = min(nperseg, x.size)
    w = get_window(WINDOW, nseg, fftbins=True)
    f, Pxx = welch(x, fs=fs, window=w, nperseg=nseg, noverlap=nseg // 2,
                   detrend="constant", scaling="density", return_onesided=True)
    return f, Pxx

def estimate_frf(x: np.ndarray, y: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    H1 FRF: H = conj(Sxy) / Sxx, gamma2 = |Sxy|^2 / (Sxx*Syy).
    NOTE: this cancels any input amplitude at each f, so it is unaffected by source level.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    nseg = min(NPERSEG, x.size, y.size)
    if nseg < 8: raise ValueError(f"Signals too short for FRF (n={min(x.size, y.size)}).")
    w = get_window(WINDOW, nseg, fftbins=True)
    f, Sxx = welch(x, fs=fs, window=w, nperseg=nseg, noverlap=nseg // 2, detrend=False)
    _, Syy = welch(y, fs=fs, window=w, nperseg=nseg, noverlap=nseg // 2, detrend=False)
    _, Sxy = csd(x, y, fs=fs, window=w, nperseg=nseg, noverlap=nseg // 2, detrend=False)
    H = np.conj(Sxy) / Sxx
    gamma2 = np.clip((np.abs(Sxy) ** 2) / (Sxx * Syy), 0.0, 1.0).real
    return f, H, gamma2

def apply_frf(x: np.ndarray, fs: float, fH: np.ndarray, H: np.ndarray,
              *, demean: bool = True, zero_dc: bool = True) -> np.ndarray:
    """Apply a measured FRF (magnitude+phase) to a time series x."""
    x = np.asarray(x, float)
    if demean: x = x - x.mean()
    N = x.size
    Nfft = int(2 ** np.ceil(np.log2(max(8, N))))
    X = np.fft.rfft(x, n=Nfft)
    fr = np.fft.rfftfreq(Nfft, d=1.0 / fs)
    mag = np.abs(H).astype(float); phi = np.unwrap(np.angle(H))
    mag_i = np.interp(fr, fH, mag, left=1.0, right=1.0)
    phi_i = np.interp(fr, fH, phi, left=phi[0], right=phi[-1])
    Hi = mag_i * np.exp(1j * phi_i)
    if zero_dc:
        Hi[0] = 0.0
        if Nfft % 2 == 0: Hi[-1] = 0.0
    y = np.fft.irfft(X * Hi, n=Nfft)[:N]
    return y

# --------- air properties and pressure‑invariant source equalizer (density‑only) ----------
R = 287.05  # J/kg/K
P_ATM = 101_325.0

def rho_of_psig(psig: float, TdegC: float = 20.0) -> float:
    T = 273.15 + float(TdegC)
    P = P_ATM + float(psig) * PSI_TO_PA
    return P / (R * T)

def c_of_T(TdegC: float = 20.0, gamma: float = 1.4) -> float:
    T = 273.15 + float(TdegC)
    return np.sqrt(gamma * R * T)

def z_run(psig: float, TdegC: float, z_model: str) -> float:
    rho = rho_of_psig(psig, TdegC)
    if z_model.lower() == "rho_c": return rho * c_of_T(TdegC)
    if z_model.lower() == "rho":   return rho
    raise ValueError("z_model must be 'rho_c' or 'rho'")

def source_equalizer_gain(psig: float, *, ref_psig: float, TdegC: float,
                          z_model: str, b_density: float) -> float:
    """
    Returns the constant *amplitude* gain that maps the source at 'psig'
    to the reference source at 'ref_psig', using ONLY the density exponent b.
        g = (z_ref / z)^b
    This keeps spectral shape intact (no frequency exponent applied).
    """
    z = z_run(psig, TdegC, z_model)
    z_ref = z_run(ref_psig, TdegC, z_model)
    return (z_ref / z) ** float(b_density)

# =====================================================================
# calibration builder for a label ('0psig', '50psig', ...)
# =====================================================================
def build_ph_to_nc_calibration(label: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns fused FRF H_nc/ph (f_cal, H_fused) for a given pressure label.
    Assumes: calib_{label}_1.mat carries PH1 (PH2 blank), _2.mat carries PH2 (PH1 blank).
    Both PH and NC are converted to Pa and corrected by -0.01 dB/kPa before FRF.
    """
    psig = _psig_to_float(label)
    f_high = F_CUT_BY_LABEL.get(label, 5_000.0)

    # run 1 (PH1–NC)
    dat1 = loadmat(CALIB_BASE / f"calib_{label}_1.mat")
    ch1, _, nc1, _ = dat1["channelData_WN"].T
    ph1_pa = volts_to_pa_corrected(ch1, channel="ph1", psig=psig, f_high=f_high)
    nc1_pa = volts_to_pa_corrected(nc1, channel="nc",  psig=psig, f_high=f_high)
    f1, H1, g2_1 = estimate_frf(ph1_pa, nc1_pa, fs=FS)

    # run 2 (PH2–NC)
    dat2 = loadmat(CALIB_BASE / f"calib_{label}_2.mat")
    _, ch2, nc2, _ = dat2["channelData_WN"].T
    ph2_pa = volts_to_pa_corrected(ch2, channel="ph2", psig=psig, f_high=f_high)
    nc2_pa = volts_to_pa_corrected(nc2, channel="nc",  psig=psig, f_high=f_high)
    f2, H2, g2_2 = estimate_frf(ph2_pa, nc2_pa, fs=FS)

    # simple coherence‑weighted fusion onto a log grid
    fmin = max(f1.min(), f2.min()); fmax = min(f1.max(), f2.max())
    f_common = np.geomspace(max(1.0, fmin), fmax, 512)

    def _interp_c(f_src, H_src, g2_src):
        mag = np.abs(H_src); ang = np.unwrap(np.angle(H_src))
        mag_i = np.interp(f_common, f_src, mag)
        ang_i = np.interp(f_common, f_src, ang)
        H_i = mag_i * np.exp(1j * ang_i)
        g2_i = np.clip(np.interp(f_common, f_src, g2_src), 0.0, 1.0)
        return H_i, g2_i

    H1i, G1 = _interp_c(f1, H1, g2_1)
    H2i, G2 = _interp_c(f2, H2, g2_2)
    eps = 1e-12
    w1, w2 = G1 + eps, G2 + eps
    H_fused = (w1 * H1i + w2 * H2i) / (w1 + w2)
    return f_common, H_fused

# =====================================================================
# main plot: raw vs corrected with pressure‑invariant source
# =====================================================================
def plot_raw_vs_corrected(label_pos: str,
                          *,
                          ref_label: str = "0psig",
                          TdegC_for_rho: float = 20.0,
                          use_saved_frf: bool = False,
                          apply_meas_sensor_corr: bool = True) -> None:
    """
    EXACT sequence:
      1) build (or load) H_nc/ph(f) for the label's pressure
      2) read cleaned PH1/PH2; optionally apply -0.01 dB/kPa to measurement channels
      3) map PH -> NC with H_nc/ph
      4) multiply each corrected trace by the constant source equalizer:
             g = (z_ref / z)^b     [ONLY b; no frequency exponent]
         so all runs correspond to the *same source* (the ref_label pressure)
      5) plot Welch spectra of raw vs corrected:
             f*Pyy / (rho^2 u_tau^4)  on 100–1000 Hz
    """
    if "_" not in label_pos:
        raise ValueError("label_pos must look like '0psig_close' or '50psig_far'")
    label, pos = label_pos.split("_", 1)
    psig = _psig_to_float(label)
    ref_psig = _psig_to_float(ref_label)

    # --- 1) calibration FRF
    if use_saved_frf:
        with h5py.File(CALIB_BASE / f"calibs_{label}.h5", "r") as hf:
            f_cal = np.asarray(hf["frequencies"][:], float)
            H_cal = np.asarray(hf["H_fused"][:])
    else:
        f_cal, H_cal = build_ph_to_nc_calibration(label)

    # --- 2) measurement (cleaned) + optional sensor corr
    with h5py.File(CLEANED_BASE / f"{label_pos}_cleaned.h5", "r") as hf:
        ph1 = np.asarray(hf["ph1_clean"][:], float)
        ph2 = np.asarray(hf["ph2_clean"][:], float)
        rho = float(hf.attrs["rho"]); u_tau = float(hf.attrs["u_tau"]); nu = float(hf.attrs["nu"])

    if apply_meas_sensor_corr:
        g_chain = pressure_sens_gain(psig)  # same -0.01 dB/kPa you applied in calibration
        ph1 *= g_chain; ph2 *= g_chain

    # --- 3) map PH -> NC
    nc1 = apply_frf(ph1, FS, f_cal, H_cal, demean=True, zero_dc=True)
    nc2 = apply_frf(ph2, FS, f_cal, H_cal, demean=True, zero_dc=True)

    # --- 4) pressure‑invariant source equalization (density‑only)
    g_eq = source_equalizer_gain(psig, ref_psig=ref_psig, TdegC=TdegC_for_rho,
                                 z_model=SRC_Z_MODEL, b_density=SRC_B_DENSITY)
    # (constant in f => multiply time series; spectral shape preserved)
    nc1 *= g_eq; nc2 *= g_eq

    # --- 5) spectra and normalization
    def pm_norm(f, P): return (f * P) / (rho**2 * u_tau**4)

    f1r, P1r = compute_spec(FS, ph1); f2r, P2r = compute_spec(FS, ph2)
    f1c, P1c = compute_spec(FS, nc1);  f2c, P2c = compute_spec(FS, nc2)

    # align grids (identical Welch; keep robust)
    if not np.allclose(f1r, f2r): P2r = np.interp(f1r, f2r, P2r); f2r = f1r
    if not np.allclose(f1c, f2c): P2c = np.interp(f1c, f2c, P2c); f2c = f1c

    f_raw = f1r; f_cor = f1c
    Y1_raw, Y2_raw = pm_norm(f_raw, P1r), pm_norm(f_raw, P2r)
    Y1_cor, Y2_cor = pm_norm(f_cor, P1c), pm_norm(f_cor, P2c)

    mraw = (f_raw >= FMIN) & (f_raw <= FMAX)
    mcor = (f_cor >= FMIN) & (f_cor <= FMAX)

    # --- plot
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.6), tight_layout=True)
    ax.semilogx(f_raw[mraw], Y1_raw[mraw], "C0--", lw=0.9, alpha=0.9, label="PH1 raw")
    ax.semilogx(f_raw[mraw], Y2_raw[mraw], "C1--", lw=0.9, alpha=0.9, label="PH2 raw")
    ax.semilogx(f_cor[mcor], Y1_cor[mcor], "C0-",  lw=1.2, alpha=0.95, label="PH1 corrected")
    ax.semilogx(f_cor[mcor], Y2_cor[mcor], "C1-",  lw=1.2, alpha=0.95, label="PH2 corrected")

    ax.set_xlim(FMIN, FMAX)
    ax.set_xlabel(r"$f$ [Hz]")
    ax.set_ylabel(r"$f\,\phi_{pp}^{+}$")
    ax.grid(True, which="major", ls="--", lw=0.4, alpha=0.7)
    ax.grid(True, which="minor", ls=":",  lw=0.25, alpha=0.6)
    ax.legend(ncol=2, fontsize=8, loc="best")

    ttl = (f"{label_pos} | H built at {label} | source equalized → {ref_label} "
           f"(b={SRC_B_DENSITY:+.3f}, z={SRC_Z_MODEL})")
    ax.set_title(ttl, fontsize=9)

    out = FIG_DIR / f"tf_corrected_spectra_roi_{label_pos}.png"
    fig.savefig(out, dpi=350)
    plt.close(fig)
    print(f"Saved: {out}")


# quick batch
if __name__ == "__main__":
    for lbl in ("0psig_close", "0psig_far", "50psig_close", "50psig_far",
                "100psig_close", "100psig_far"):
        plot_raw_vs_corrected(lbl, ref_label="0psig", TdegC_for_rho=20.0,
                              use_saved_frf=False, apply_meas_sensor_corr=True)
