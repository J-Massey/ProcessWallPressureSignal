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
        # plot_spectrum(f, f*Pxx, f"{OUTPUT_DIR}/Pxrxr_{psi[idx]}_raw")
        # f, Pyryr = compute_spec(25000.0, y_r)
        # Pyryrs.append(f*Pyryr)
        # plot_spectrum(f, f*Pxx, f*Pyy, f"{OUTPUT_DIR}/spectra/P_{psi[idx]}_raw")
        ###
        freqs.append(1/f)
        Pyys.append(f*Pxx)
    # Ensure output directory exists (including nested 'spectra')
    outfile = f"{OUTPUT_DIR}/spectra/Pxx.png"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    # Option A: Plot like MATLAB (PSD vs frequency)
    plot_psd_loglog(freqs, Pyys, outfile)

    # Option B: If you prefer plotting vs period T and f*Phi, use:
    # Times = [1/fi for fi in freqs]
    # spec_fphi = [fi * Pi for fi, Pi in zip(freqs, Pyys)]
    # plot_rawspectrum(Times, spec_fphi, f"{OUTPUT_DIR}/spectra/Pxx_T.png")


if __name__ == "__main__":
    raw_data()
