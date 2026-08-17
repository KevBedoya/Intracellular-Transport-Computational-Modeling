"""
modal_solver.py -- Phase 3 reduced-model time evolution.

Reduced system (constraint-eliminated B):

    A'(t) = -Lam_A * A(t)                              (free Dirichlet-Bessel decay)
    M^R D'(t) = v T^R D + a P A(t) - b M^R D           (AL transport, projected)
    B(t) : solve   a CA A + CB B - b CR D = 0   (rank-truncated least squares)

At each output time we reconstruct B, the boundary residual, and (optionally) the
fields.  NOTE (documented limitation): with a symmetric centered IC only the n=0
cos-modes are excited and A'=-Lam_A A keeps A_{n>=1}=0 for all t, so the reduced
model can express angular structure only through the sin-modes B.  The boundary
residual and the DL error quantify how much this misses; see modal_verification.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from .modal_matrices import ModalMatrices


def solve_B(M: ModalMatrices, A, D, a, b, rcond=1e-10):
    """B = argmin || CB B - (b CR D - a CA A) ||  (rank-truncated LS).

    Returns (B, residual_2norm, effective_rank)."""
    rhs = b * (M.CR @ D) - a * (M.CA @ A)
    B, _, rank, _ = np.linalg.lstsq(M.CB, rhs, rcond=rcond)
    resid = a * (M.CA @ A) + M.CB @ B - b * (M.CR @ D)
    return B, float(np.linalg.norm(resid)), int(rank)


@dataclass
class ReducedSolution:
    times: np.ndarray
    A: np.ndarray               # (nt, nA)
    B: np.ndarray               # (nt, nB)
    D: np.ndarray               # (nt, Q)
    bc_res: np.ndarray          # (nt,)   interface-constraint residual norm
    bc_rhs_norm: np.ndarray     # (nt,)   ||b CR D - a CA A||  (for relative residual)
    rank_CB: int


def evolve_reduced(M: ModalMatrices, A0, D0, times, a, b, v, rcond=1e-10):
    """Integrate the reduced system and reconstruct B at each output time."""
    sp = M.space
    LamA = M.LamA
    Minv = np.linalg.inv(M.MR)
    L = Minv @ (v * M.TR - b * M.MR)               # linear part of D'
    aMinvP = a * (Minv @ M.P)                       # forcing operator on A(t)

    def A_of_t(t):
        return np.exp(-LamA * t) * A0

    def rhs(t, D):
        return L @ D + aMinvP @ A_of_t(t)

    times = np.asarray(times, float)
    sol = solve_ivp(rhs, (0.0, times.max()), D0, t_eval=times,
                    method="LSODA", rtol=1e-8, atol=1e-10)
    D_t = sol.y.T                                   # (nt, Q)

    A_t = np.array([A_of_t(t) for t in times])
    B_t = np.zeros((len(times), sp.nB))
    res = np.zeros(len(times)); rhsn = np.zeros(len(times)); rank = 0
    for k, t in enumerate(times):
        B, r, rank = solve_B(M, A_t[k], D_t[k], a, b, rcond=rcond)
        B_t[k] = B; res[k] = r
        rhsn[k] = np.linalg.norm(b * (M.CR @ D_t[k]) - a * (M.CA @ A_t[k]))
    return ReducedSolution(times, A_t, B_t, D_t, res, rhsn, rank)


# ---- field reconstruction ----
def reconstruct_F(space, A, B, r_grid, theta_grid):
    """F(r,theta) = sum A_{nj} J_{nN}(kr) cos(nN t) + sum B_{nj} J_{nN}(kr) sin(nN t)."""
    r_grid = np.asarray(r_grid, float); theta_grid = np.asarray(theta_grid, float)
    Jc = space.eval_Jcos(r_grid)                    # (nA, nr)
    Js = space.eval_Jsin(r_grid)                    # (nB, nr)
    F = np.zeros((len(r_grid), len(theta_grid)))
    for i, (n, j) in enumerate(space.cos_index):
        F += A[i] * np.outer(Jc[i], np.cos(n * space.N * theta_grid))
    for i, (n, j) in enumerate(space.sin_index):
        F += B[i] * np.outer(Js[i], np.sin(n * space.N * theta_grid))
    return F


def reconstruct_R(space, D, r_grid):
    CHI = space.eval_chi(np.asarray(r_grid, float))  # (Q, nr)
    return D @ CHI


def modal_angular_energy(space, A, B):
    """E_n = sum_j (A_{n,j}^2 + B_{n,j}^2) per angular index n."""
    E = np.zeros(space.Nmax + 1)
    for i, (n, j) in enumerate(space.cos_index):
        E[n] += A[i] ** 2
    for i, (n, j) in enumerate(space.sin_index):
        E[n] += B[i] ** 2
    return E
