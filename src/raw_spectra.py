import numpy as np
from scipy.signal import welch
import scipy.io as sio

from icecream import ic
import os

from plotting import plot_rawspectrum
from plotting import plot_psd_loglog  # new: PSD vs f (log-log) plot

# Some hard coded params
FS = 25000.0
NPERSEG = 2**11
WINDOW = "hann"
DETREND = "constant"


def load_mat(path):
    mat = sio.loadmat(path)
    ic(mat.keys())
    data = mat['channelData']
    x_r = np.array(data[:, 0])
    x_r = x_r - np.mean(x_r)
    y_r = np.array(data[:, 1])
    y_r = y_r - np.mean(y_r)
    return x_r, y_r


def compute_spec(fs: float, x: np.ndarray, nperseg: int = NPERSEG):
    """Welch PSD matching MATLAB pwelch settings.

    - window: hann(nperseg)
    - overlap: 50%
    - nfft: nperseg
    - detrend: 'constant' (subtract mean)
    """
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


def raw_data():
    psi = ['atm', '10psi', '30psi', '50psi', '70psi', '100psi']
    # V ~ 14 m/s (max speed in HPWT as of Aug. 2025)
    V = 14.0
    rhos = 
    Re_taus = [1500, 2500, 3500, 4500, 6000, 8000]
    u_taus = [0.571, 0.532, 0.492, 0.515, 0.433, 0.481]
    nu_atm =  1.5e-5  # m^2/s
    delta =  Re_taus[0] * nu_atm / u_taus[0]  # assume constant delta
    nu_s = [delta * u_tau / Re for u_tau, Re in zip(u_taus, Re_taus)]

    root = 'data/14082025/flow/maxspeed'
    fn_sweep = [f'{root}/data_{p}.mat' for p in psi]
    OUTPUT_DIR = "figures/real"
    PH_path = "figures/PH-NKD"
    NC_path = "figures/NC-NKD"

    freqs = []
    Pyys = []
    for idx in range(len(psi)):
        ic(f"Processing {psi[idx]}...")

        # Load data
        x_r, y_r = load_mat(fn_sweep[idx])
        ic(x_r.shape, y_r.shape)
        # Welch PSD (match MATLAB pwelch)
        f, Pxx = compute_spec(FS, x_r)
        f_plus = f * nu_s[idx] / u_taus[idx]**2
        Pxx_plus = Pxx * u_taus[idx]**2 / nu_s[idx]
        # ic(f.shape, Pxx.shape)
        freqs.append(f)
        Pyys.append(Pxx)
    # Ensure output directory exists (including nested 'spectra')
    outfile = f"{OUTPUT_DIR}/spectra/Pxx.png"
    # os.makedirs(os.path.dirname(outfile), exist_ok=True)

    Times = [(1/fi) for fi in freqs]
    spec_fphi = [fi * Pi for fi, Pi in zip(freqs, Pyys)]
    plot_rawspectrum(Times, spec_fphi, outfile)


if __name__ == "__main__":
    raw_data()
