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

def load_test_wallpressure(path):
    data = sio.loadmat(path)['channelData']
    fs = torch.tensor(data[:, 0])
    p = torch.tensor(data[:, 1])
    return fs, p


if __name__ == "__main__":
    from icecream import ic
    path = "data/testatm.mat"
    ic(sio.loadmat(path)['channelData'])