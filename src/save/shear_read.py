import h5py
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import scienceplots
from matplotlib.colors import to_rgba
plt.style.use(["science", "grid"])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")


fn = "/home/masseyj/Downloads/D_shear_SU_production_test.hdf5"

with h5py.File(fn, 'r') as f:
    ts = f['Experiment']
    print(ts.attrs.keys())
    print(ts.attrs['daq_type'])
