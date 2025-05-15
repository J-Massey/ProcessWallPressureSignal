import numpy as np
import scipy.io as sio

def load_wallpressure(path, var_name="wall_pressure_fluc_Pa"):
    """Load frequency and pressure from a .mat file."""
    data = sio.loadmat(path)
    fs = np.ravel(data.get("fs_pressure_fluc_Pa", data.get("fs", None)))
    p  = np.ravel(data.get(var_name))
    return fs, p