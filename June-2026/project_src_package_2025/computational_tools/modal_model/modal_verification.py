"""
modal_verification.py -- compare the modal prototype against the discrete numba
scheme (DL field), and expose the structural cos/sin diagnostic.

Metrics per saved time t:
  * E_F(t)        : relative Frobenius error of the reconstructed DL field.
  * bc_res_rel(t) : interface-constraint residual / rhs-norm.
  * per-angular-index energy: discrete cos_n & sin_n vs modal E_n.
"""
from __future__ import annotations

import numpy as np

from . import modal_basis as mb, modal_matrices as mm, modal_solver as ms
from . import modal_initial_conditions as mic


def run_discrete_dl(rg_param, ry_param, v_param, w_param, N_LIST, times,
                    domain_radius=1.0):
    """Discrete DL field snapshots via launch.compute_ang_traj_mat.

    Returns (F_stack[nt, rg-1, ry], r_rings[rg-1], theta[ry], central[nt])."""
    from launch_functions import launch
    dm, central = launch.compute_ang_traj_mat(
        rg_param, ry_param, v_param, w_param, np.asarray(N_LIST),
        list(times), domain_radius=domain_radius, mass_checkpoint=10 ** 9)
    dRad = domain_radius / rg_param
    r_rings = (np.arange(rg_param - 1) + 1) * dRad
    theta = np.arange(ry_param) * (2 * np.pi / ry_param)
    return np.asarray(dm), r_rings, theta, np.asarray(central)


def discrete_harmonic_energy(F, theta, N, Nmax):
    """cos_n, sin_n angular energy (summed over radius) of a discrete field F[r,theta]."""
    cos_e = np.zeros(Nmax + 1); sin_e = np.zeros(Nmax + 1)
    for ri in range(F.shape[0]):
        H = np.fft.rfft(F[ri]) / F.shape[1]
        for n in range(Nmax + 1):
            k = n * N
            if k < len(H):
                # real fft: cos amp ~ 2*Re, sin amp ~ -2*Im (n>=1); n=0 special
                if n == 0:
                    cos_e[0] += np.real(H[0]) ** 2
                else:
                    cos_e[n] += (2 * np.real(H[k])) ** 2
                    sin_e[n] += (2 * np.imag(H[k])) ** 2
    return cos_e, sin_e


def compare(space, sol: ms.ReducedSolution, F_stack, r_rings, theta, central=None):
    out = []
    zeroB = np.zeros(space.nB)
    for k, t in enumerate(sol.times):
        Fd = F_stack[k].astype(float)
        Fm = ms.reconstruct_F(space, sol.A[k], sol.B[k], r_rings, theta)
        Fm0 = ms.reconstruct_F(space, sol.A[k], zeroB, r_rings, theta)   # B=0 baseline
        nrm = np.linalg.norm(Fd) + 1e-30
        EF = np.linalg.norm(Fm - Fd) / nrm
        EF_B0 = np.linalg.norm(Fm0 - Fd) / nrm
        bc_rel = sol.bc_res[k] / (sol.bc_rhs_norm[k] + 1e-30)
        cos_e, sin_e = discrete_harmonic_energy(Fd, theta, space.N, space.Nmax)
        # fraction of discrete angular energy in the modulation (n>=1) vs the mean
        mod_frac = cos_e[1:].sum() / (cos_e[0] + 1e-30)
        Emod = ms.modal_angular_energy(space, sol.A[k], sol.B[k])
        out.append(dict(t=t, E_F=EF, E_F_B0=EF_B0, bc_res=sol.bc_res[k], bc_rel=bc_rel,
                        disc_cos=cos_e, disc_sin=sin_e, modal_E=Emod, mod_frac=mod_frac,
                        Bnorm=np.linalg.norm(sol.B[k])))
    return out


def run_minimal_experiment(Nmax=3, Jmax=4, Qmax=8, N=4,
                           rg_param=32, ry_param=32, v_param=1.0, w_param=10.0,
                           times=(0.1, 0.2, 0.3, 0.4), domain_radius=1.0, rcond=1e-6):
    """End-to-end reduced-model verification on the benchmark discrete run."""
    N_LIST = np.linspace(0, ry_param - ry_param // N, N, dtype=int)   # evenly spaced
    space = mb.ModalSpace(N=N, Nmax=Nmax, Jmax=Jmax, Qmax=Qmax, domain_radius=domain_radius)
    M = mm.assemble(space)
    A0, D0 = mic.ic_centered_patch(space, rg_param)
    # a = b = w in the discrete scheme's scalar convention; v as given (inward).
    a = b = w_param
    sol = ms.evolve_reduced(M, A0, D0, times, a=a, b=b, v=v_param, rcond=rcond)
    F_stack, r_rings, theta, central = run_discrete_dl(
        rg_param, ry_param, v_param, w_param, N_LIST, times, domain_radius)
    report = compare(space, sol, F_stack, r_rings, theta, central)
    return dict(space=space, matrices=M, solution=sol, report=report,
                F_stack=F_stack, r_rings=r_rings, theta=theta,
                cond=mm.conditioning_report(M), N_LIST=N_LIST)
