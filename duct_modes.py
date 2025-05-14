import numpy as np

from icecream import ic
import numpy as np
from scipy.sparse import diags, eye, bmat
from scipy.sparse.linalg import eigs

###--- Stanford params ---###
# -- Fit values --#
t_w = np.array([0.086, 0.193, 0.629, 1.207, 3.033])  # (Pa)
Cf = np.array([1.98, 1.68, 1.47, 1.27, 1.12]) * 2 * 1e-3  # (C_f/2*10^3)
nu_u_tau = np.array([57.8, 38.2, 21.2, 7.97, 3.55]) * 1e-6

# -- Meassured values --#
Ue = np.array([6.04, 9.83, 18.95, 14.36, 17.15])
delta = np.array([31.25, 37.93, 35.87, 34.56, 35.58]) * 1e-3
rho = t_w / (np.sqrt(Cf * Ue**2 / 2) ** 2)
cs = np.sqrt(1.4*101325/rho)  # (m s⁻²)
ic(cs)

# -- Derived values --#
u_tau = np.sqrt(Cf * Ue**2 / 2)  # (m s⁻¹)
nu = nu_u_tau * u_tau
Re_tau = u_tau * delta / nu
ic(Re_tau)

from matplotlib import pyplot as plt
import seaborn as sns
import scienceplots

plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")

stanford_params = np.array([t_w, Cf, nu_u_tau, Ue, delta])


def compute_modes_closed_inlet(W, H, L, U, c, max_l=4):
    modes = []
    DeltaL = 0.6 * np.sqrt(W * H)  # characteristic length scale
    for l in range(max_l + 1):
        # analytic quarter-wave fundamental (m=n=0)
        f = (2*l + 1) * np.sqrt(c**2 - U**2) / (4 * L)  # simplifies to (2l+1)*sqrt(c^2-U^2)/(4L)
        # f = 1/(np.pi *2) * np.sqrt ((c**2 - U**2)**2/(4*c**2)*((2*l+1)*np.pi/L)**2)
        modes.append(f)
    return modes


# Example usage
# W, H, L = 0.30, 0.152, 3.0
# U, c = 20.0, 343.0

# # Analytic closed-inlet modes (m=0,n=0,l=0..4)
# analytic_freqs = compute_modes_closed_inlet(W, H, L, U, c, max_m=0, max_n=0, max_l=4)
# print("Analytic modes (closed inlet):", [f for f,_,_,_ in analytic_freqs])


def compute_quarter_wave_mode_frequency(W, H, L, U, c,
                                        m=0, n=0, l=0,
                                        Nx=400, tol=1e-6):
    # 1) Grid of interior points (drop x=L)
    x_full = np.linspace(0, L, Nx+1)
    x      = x_full[:-1]
    h      = x[1] - x[0]
    N      = len(x)

    # 2) Transverse wavenumber
    kperp2 = (m*np.pi/W)**2 + (n*np.pi/H)**2

    # 3) D1 (Neumann inlet, zero at outlet)
    e = np.ones(N)
    D1 = diags([-e[1:], e[:-1]], [-1,1])/(2*h)
    D1 = D1.tolil()
    D1[0,:] = 0; D1[-1,:] = 0
    D1 = D1.tocsc()

    # 4) D2 (Neumann inlet via ghost; Dirichlet outlet via ghost)
    D2 = diags([e[1:], -2*e, e[:-1]], [-1,0,1])/(h*h)
    D2 = D2.tolil()
    # inlet Neumann
    D2[0,:] = 0
    D2[0,0] = -2/(h*h); D2[0,1] = 2/(h*h)
    # outlet Dirichlet
    D2[-1,:]   = 0
    D2[-1, -2] =  1/(h*h)
    D2[-1, -1] = -2/(h*h)
    # all other entries on row -1 must remain zero

    D2 = D2.tocsc()

    # 5) Build QEP matrices
    M2 = eye(N, format='csc')
    M1 = 2j * U * D1
    M0 = (c**2 - U**2)*D2 - c**2*kperp2*eye(N, format='csc')

    # linearise and shift-invert as before
    I = eye(N, format='csc')
    A = bmat([[None,   I],
            [-M0,  -M1]], format='csc')
    B = bmat([[  I, None],
            [None,  M2]], format='csc')

    omega_guess = (2*l + 1) * np.pi * np.sqrt(c**2 - U**2) / (2 * L)
    vals, _ = eigs(A, M=B, k=3, sigma=omega_guess, which='LM', tol=tol)
    omegas = np.real(vals)

    # 9) Filter out near‐zero modes
    omegas = omegas[omegas > 1e-10]

    # 10) Pick the one closest to the guess
    omega_l = omegas[np.argmin(np.abs(omegas - omega_guess))]

    return omega_l / (2*np.pi)  # Hz

# Example usage:
# Fundamental mode (m=0,n=0,l=0)
# eigen_freqs = compute_quarter_wave_mode_frequency(W, H, L, U, c, m=0, n=0, l=0, Nz=200)
# print(f"Fundamental mode f_0 ≈ {eigen_freqs:.2f} Hz")


if __name__ == "__main__":
    # Plotting
    W, H, L = 0.30, 0.152, 3.0
    U, c = Ue[-1], cs[-1]
    N = 5  # number of modes

    # for l in range(3):
    #     f_analytic = (2*l+1)*np.sqrt(c**2 - U**2)/(4*L)
        # f_qep      = compute_quarter_wave_mode_frequency(W,H,L,U,c,l=l)
        # print(f"l={l}: analytic={f_analytic:.2f} Hz, QEP={f_qep:.2f} Hz")

    # Compute frequencies
    eigen_freqs    = [compute_quarter_wave_mode_frequency(W, H, L, U, c, Nx=200, l=i) for i in range(N)]

    fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.5), tight_layout=True)
    idx = np.arange(1, N+1)
    U, c = Ue[-1], cs[-1]
    analytic_freqs = np.array(compute_modes_closed_inlet(W, H, L, U, c, max_l=N-1))
    ic(analytic_freqs, analytic_freqs*nu[-1]/u_tau[-1]**2)
    ax[0].plot(idx, analytic_freqs, 'o-', label=r'$Re_{\tau}=10,022$')
    ax[1].plot(idx, analytic_freqs*nu[-1]/u_tau[-1]**2, 'o-', label='$Re_{\\tau}=10,022$')
    U, c = Ue[-3], cs[-3]

    analytic_freqs = np.array(compute_modes_closed_inlet(W, H, L, U, c, max_l=N-1))
    ic(analytic_freqs, analytic_freqs*nu[-3]/u_tau[-3]**2)
    ax[0].plot(idx, analytic_freqs, 'o-', label=r'$Re_{\tau}=1,692$')
    ax[1].plot(idx, analytic_freqs*nu[-3]/u_tau[-3]**2, 'o-', label='$Re_{\\tau}=1,692$')


    # ax.plot(idx, eigen_freqs, 's-', label='Eigenvalue')
    ax[0].set_xlabel('Sorted mode index')
    ax[1].set_xlabel('Sorted mode index')
    ax[0].set_ylabel('$f$ (Hz)')
    ax[1].set_ylabel('$f^+$')
    ax[0].legend()
    plt.savefig("figures/rectangular_modes.png", dpi=600)