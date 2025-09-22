import numpy as np
from icecream import ic
            
            
nu_utau = np.array([29.00,
                    7.97,
                    4.00]) * 1e-6
delta = np.array([0.030])
U_cl = 14

l = 200 * nu_utau
ic(l*1000)  # mm