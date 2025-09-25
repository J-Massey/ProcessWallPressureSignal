import numpy as np
from icecream import ic
            
            
nu_utau = np.array([29.00,
                    7.97,
                    4.00]) * 1e-6
delta = np.array([0.030])
U_cl = 14

l = 100 * nu_utau

res = np.array([1500, 5000, 8200])
delta = 0.035
nu_utau = (1/(res/delta))*1e6
ic(nu_utau)