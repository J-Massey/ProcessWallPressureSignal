import numpy as np
from pathlib import Path
from scipy.signal import welch
import scipy.io as sio

# from icecream import ic  # Optional for debugging
from plotting import plot_rawspectrum

# -------------------
# Constants & defaults
# -------------------
FS = 25_000.0
NPERSEG = 2**11
WINDOW = "hann"
DETREND = "constant"

R = 287.0         # J/kg/K
T = 293.0         # K (adjust if you have per-case temps)
P_ATM = 101_325.0 # Pa
PSI_TO_PA = 6_894.76

def load_mat(path: str, key: str = "channelData"):
    """Load an Nx2 array from a MATLAB .mat file under `key`."""
    mat = sio.loadmat(path, squeeze_me=True)
    if key not in mat:
        raise KeyError(f"Key '{key}' not found in {path}. Available: {list(mat.keys())}")
    data = np.asarray(mat[key])
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected Nx2 array under '{key}', got shape {data.shape} in {path}")
    # Let welch(detrend='constant') handle mean removal per segment
    x = data[:, 0].astype(float)
    y = data[:, 1].astype(float)
    return x, y

def compute_spec(fs: float, x: np.ndarray, nperseg: int = NPERSEG):
    """Welch PSD matching MATLAB pwelch-style settings."""
    nperseg = int(min(nperseg, len(x)))
    f, Pxx = welch(
        x,
        fs=fs,
        window=WINDOW,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        nfft=nperseg,
        detrend=DETREND,
        scaling="density",
        return_onesided=True,
    )
    return f, Pxx

def inner_scales(Re_taus, u_taus, nu_atm):
    """Return delta (from the atm case) and nu for each case via Re_tau relation."""
    Re_taus = np.asarray(Re_taus, dtype=float)
    u_taus = np.asarray(u_taus, dtype=float)
    delta = Re_taus[0] * nu_atm / u_taus[0]
    nus = delta * u_taus / Re_taus
    return float(delta), nus

def raw_data():
    """Compute premultiplied inner-scaled wall-pressure spectra and plot."""
    psi_labels = ['atm', '10psi', '30psi', '50psi', '70psi', '100psi']
    Re_taus = np.array([1500, 2500, 3500, 4500, 6000, 8000], dtype=float)
    u_taus  = np.array([0.571, 0.532, 0.492, 0.515, 0.433, 0.481], dtype=float)
    nu_atm  = 1.5e-5  # m^2/s

    # Inner scales
    delta, nu_s = inner_scales(Re_taus, u_taus, nu_atm)

    # Densities via ideal gas
    pressures = np.array([P_ATM] + [P_ATM + float(p[:-3]) * PSI_TO_PA for p in psi_labels[1:]], dtype=float)
    rhos = pressures / (R * T)

    # IO paths
    root = Path('data/14082025/flow/maxspeed')
    fn_sweep = [root / f"data_{p}.mat" for p in psi_labels]
    outdir = Path("figures/real/spectra")
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "Pxx.png"

    Tplus_list = []
    premult_list = []

    for idx, label in enumerate(psi_labels):
        # ic(f"Processing {label} …")
        x_r, _y_r = load_mat(str(fn_sweep[idx]))

        f, Pxx = compute_spec(FS, x_r)

        # Remove DC to avoid T^+ = inf
        if f[0] == 0.0:
            f = f[1:]
            Pxx = Pxx[1:]

        # Inner variables
        f_plus = f * nu_s[idx] / (u_taus[idx] ** 2)                        # f^+
        Pxx_plus = Pxx / ((u_taus[idx] ** 2 * rhos[idx]) ** 2)             # PSD of p^+ [1/Hz]
        premult = f * Pxx_plus                                             # == f^+ Φ^+(f^+) (dimensionless)
        T_plus = 1.0 / f_plus                                              # T^+

        Tplus_list.append(T_plus)
        premult_list.append(premult)

    # The plotting function should label:
    # x-axis: r"$T^+ = 1/f^+$"
    # y-axis: r"$f^+\Phi_{p^+p^+}$ (dimensionless)"
    plot_rawspectrum(Tplus_list, premult_list, str(outfile))

    return Tplus_list, premult_list, str(outfile)

if __name__ == "__main__":
    raw_data()
