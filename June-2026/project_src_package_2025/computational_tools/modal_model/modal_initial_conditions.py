"""
modal_initial_conditions.py -- project discrete data onto the modal basis.

Primary path (this prototype's benchmark): a CENTERED patch release, which is
angularly uniform, so only the n=0 cos-modes are excited at t=0 and the AL is
empty:  A_{0,j}(0) = Fourier-Bessel coefficients of the central patch,
A_{n>=1,j}(0)=0, B(0) from the interface constraint, D(0)=0.

A general Galerkin projection of an arbitrary discrete DL field is also provided.
"""
from __future__ import annotations

import numpy as np

from computational_tools import analytic_solution as asol
from .modal_basis import ModalSpace


def ic_centered_patch(space: ModalSpace, rg_param: int):
    """A(0), D(0) for the centered unit-patch release used by the numba solver.

    Uses asol.fourier_bessel_coeffs_uncoupled (order-0, 'patch' IC) for the n=0
    block; every n>=1 cos-mode and the whole AL start at zero.
    """
    c, kappa0 = asol.fourier_bessel_coeffs_uncoupled(
        rg_param, space.Jmax, domain_radius=space.domain_radius, ic="patch")
    A0 = np.zeros(space.nA)
    # first Jmax cos entries are the n=0 block (cos_index ordered n-major)
    A0[:space.Jmax] = c
    D0 = np.zeros(space.Qmax)
    return A0, D0


def project_dl_field(space: ModalSpace, phi, r_disc, theta_disc):
    """Least-squares Galerkin projection of a discrete DL field phi[r,theta].

    Returns (A0, B0) minimizing || phi - reconstruct(A0,B0) ||.  Uses the same
    cos/sin Fourier-Bessel basis evaluated on the discrete grid.
    """
    Jc = space.eval_Jcos(r_disc)                      # (nA, nr)
    Js = space.eval_Jsin(r_disc)                      # (nB, nr)
    cols = []
    for i, (n, j) in enumerate(space.cos_index):
        cols.append(np.outer(Jc[i], np.cos(n * space.N * theta_disc)).ravel())
    for i, (n, j) in enumerate(space.sin_index):
        cols.append(np.outer(Js[i], np.sin(n * space.N * theta_disc)).ravel())
    G = np.column_stack(cols)                          # (nr*ntheta, nA+nB)
    coef, *_ = np.linalg.lstsq(G, np.asarray(phi, float).ravel(), rcond=None)
    return coef[:space.nA], coef[space.nA:]


def project_al_field(space: ModalSpace, rho_r, r_disc):
    """Least-squares projection of a discrete AL radial profile onto chi_q."""
    CHI = space.eval_chi(r_disc)                       # (Q, nr)
    coef, *_ = np.linalg.lstsq(CHI.T, np.asarray(rho_r, float), rcond=None)
    return coef
