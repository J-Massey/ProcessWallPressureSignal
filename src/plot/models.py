import numpy as np

def cf_approx(Re_tau: float) -> float:
    return (np.log(Re_tau) / 0.384 + 4.127) ** -2

def bl_model(Tplus, Re_tau: float, cf_2: float) -> np.ndarray:
    A1 = 2.2
    sig1 = 3.9
    mean_Tplus = 20
    A2 = 1.4 * (np.log10(Re_tau) - 2.2)
    sig2 = 1.2
    mean_To = 0.82
    r1 = 0.5
    r2 = 7
    rv = np.exp(r1 * Tplus)/(np.exp(r1*r2) + np.exp(r1 * Tplus)) # correct
    rv = np.nan_to_num(rv, nan=1)  # replace NaNs with 0
    mean_To_plus = mean_To * Re_tau * np.sqrt(cf_2)
    g1 = A1 * np.exp(-sig1 * (np.log10(Tplus) - np.log10(mean_Tplus))**2)
    g2 = A2 * np.exp(-sig2 * (np.log10(Tplus) - np.log10(mean_To_plus))**2)
    # ic(g1[0], Tplus[0], rv[0])
    return g1, g2, rv

def pipe_model(Tplus, Re_tau: float, u_tau: float, u_cl) -> np.ndarray:
    A1 = 2.9*(1 - 1000/Re_tau)
    sig1 = 4.3
    mean_Tplus = 20
    A2 = 0.91 * (np.log10(Re_tau) - 2.2)
    sig2 = 1.0
    mean_To = 0.18
    r1 = 0.5
    r2 = 7
    rv = np.exp(r1 * Tplus)/(np.exp(r1*r2) + np.exp(r1 * Tplus)) # correct
    rv = np.nan_to_num(rv, nan=1)  # replace NaNs with 0
    mean_To_plus = mean_To * Re_tau * u_tau/u_cl
    g1 = A1 * np.exp(-sig1 * (np.log10(Tplus) - np.log10(mean_Tplus))**2)
    g2 = A2 * np.exp(-sig2 * (np.log10(Tplus) - np.log10(mean_To_plus))**2)
    return g1, g2, rv


def channel_model(Tplus, Re_tau: float, u_tau: float, u_cl) -> np.ndarray:
    A1 = 2.1*(1 - 100/Re_tau)
    sig1 = 4.4
    mean_Tplus = 12
    A2 = 0.9 * (np.log10(Re_tau) - 2.2)
    sig2 = 1.0
    mean_To = 0.6
    r1 = 0.5
    r2 = 3
    rv = np.exp(r1 * Tplus)/(np.exp(r1*r2) + np.exp(r1 * Tplus)) # correct
    rv = np.nan_to_num(rv, nan=1)  # replace NaNs with 0
    mean_To_plus = mean_To * Re_tau * u_tau/u_cl
    g1 = A1 * np.exp(-sig1 * (np.log10(Tplus) - np.log10(mean_Tplus))**2)
    g2 = A2 * np.exp(-sig2 * (np.log10(Tplus) - np.log10(mean_To_plus))**2)
    return g1, g2, rv
