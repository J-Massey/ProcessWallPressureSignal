import torch
import scipy.io as sio

def load_stan_wallpressure(path, var_name="wall_pressure_fluc_Pa"):
    """Load frequency and pressure from a .mat file."""
    data = sio.loadmat(path)
    fs = torch.tensor(data.get("fs_pressure_fluc_Pa", data.get("fs", None))).ravel()
    p  = torch.tensor(data.get(var_name)).ravel()
    return fs, p

def load_melb_wallpressure(path, var_name="wall_pressure_fluc_Pa"):
    """Load frequency and pressure from a .mat file."""
    data = sio.loadmat(path)
    print(data.keys())
    fs = torch.tensor(data.get("fs_pressure_fluc_Pa", data.get("fs", None))).ravel()
    p  = torch.tensor(data.get(var_name)).ravel()
    return fs, p