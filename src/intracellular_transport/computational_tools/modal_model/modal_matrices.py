"""
modal_matrices.py -- Phase 1 projection matrices for the modal DL/AL prototype.

All matrices are BARE integrals (no a,b,v factors); the physical rates are applied
at solve time (modal_solver) so the same matrices serve every parameter set.

    M^R_pq  = int chi_p chi_q dr
    T^R_pq  = int chi_p chi_q' dr
    P_p,(nj)= int chi_p J_{nN}(kappa_{n,j} r) dr
    CA_(mL),(nj) = int r J_{mN}(kappa_{m,L} r) J_{nN}(kappa_{n,j} r) dr    (bare, x a later)
    CB_(mL),(nj) = -2 nN int J_{mN}(kappa_{m,L} r) J_{nN}(kappa_{n,j} r) dr (sin cols)
    CR_(mL),q    = int r J_{mN}(kappa_{m,L} r) chi_q(r) dr                 (x b later)
    Lam_A, Lam_B : diagonal (kappa/R)^2  (interior Dirichlet-Bessel decay rates)

Interface constraint (applied in the solver):  a*CA*A + CB*B - b*CR*D = 0.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .modal_basis import ModalSpace


@dataclass
class ModalMatrices:
    space: ModalSpace
    MR: np.ndarray
    TR: np.ndarray
    P: np.ndarray
    CA: np.ndarray
    CB: np.ndarray
    CR: np.ndarray
    LamA: np.ndarray            # 1-D array of decay rates for cos modes
    LamB: np.ndarray            # 1-D array of decay rates for sin modes


def assemble(space: ModalSpace) -> ModalMatrices:
    r, w = space.r, space.w
    Jcos, Jsin, CHI, dCHI = space.Jcos, space.Jsin, space.CHI, space.dCHI
    R = space.domain_radius

    MR = (CHI * w) @ CHI.T                       # (Q,Q)
    TR = (CHI * w) @ dCHI.T                       # (Q,Q)
    P = (CHI * w) @ Jcos.T                        # (Q, nA)
    CA = (Jcos * (w * r)) @ Jcos.T                # (nA, nA)   bare int r J J
    CR = (Jcos * (w * r)) @ CHI.T                 # (nA, Q)
    CB = ((Jcos * w) @ Jsin.T) * (-2.0 * space.nN_sin)[None, :]   # (nA, nB)

    LamA = (space.kappa_cos / R) ** 2
    LamB = (space.kappa_sin / R) ** 2
    return ModalMatrices(space, MR, TR, P, CA, CB, CR, LamA, LamB)


def conditioning_report(M: ModalMatrices) -> dict:
    """Diagnostics on matrix conditioning / rank (Phase 1-2 sanity)."""
    s_MR = np.linalg.svd(M.MR, compute_uv=False)
    s_CB = np.linalg.svd(M.CB, compute_uv=False)
    # orthogonality check: diagonal blocks of CA (same order m=n) should be
    # diagonal with value J_{nN+1}(kappa)^2 / 2 (Fourier-Bessel norm).
    sp = M.space
    diag_err = 0.0
    from scipy.special import jv
    for i, (n, j) in enumerate(sp.cos_index):
        expect = 0.5 * jv(n * sp.N + 1, sp.kappa_cos[i]) ** 2 * sp.domain_radius ** 2
        diag_err = max(diag_err, abs(M.CA[i, i] - expect) / (abs(expect) + 1e-30))
    return dict(
        cond_MR=s_MR[0] / s_MR[-1],
        cond_CB=(s_CB[0] / s_CB[-1]) if s_CB[-1] > 0 else np.inf,
        rank_CB=int(np.sum(s_CB > 1e-12 * s_CB[0])),
        shape_CB=M.CB.shape,
        CA_self_norm_relerr=diag_err,
    )
