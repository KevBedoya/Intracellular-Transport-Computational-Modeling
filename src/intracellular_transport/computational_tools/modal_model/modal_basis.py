"""
modal_basis.py -- Phase 1 infrastructure for the coupled Fourier-Bessel modal
DL/AL prototype.

Provides a `ModalSpace` describing the truncated basis

    F(r,theta,t) = sum_{n=0..Nmax, j=1..Jmax} A_{n,j}(t) J_{nN}(kappa_{n,j} r) cos(nN theta)
                 + sum_{n=1..Nmax, j=1..Jmax} B_{n,j}(t) J_{nN}(kappa_{n,j} r) sin(nN theta)
    R(r,t)       = sum_{q=1..Qmax} D_q(t) chi_q(r),   chi_q(r) = r^{q-1}(1-r)

with kappa_{n,j} the j-th positive zero of J_{nN}, so F(1,theta,t)=0 and chi_q(1)=0.

Pure NumPy/SciPy (scipy.special not numba-safe); reuses helpers from
computational_tools.analytic_solution.  Grid conventions match numerical_tools
(cell-center radii r_m=(m+1)*dRad, absorbing rim at r=R).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.special import jv, eval_chebyt, eval_chebyu

from computational_tools import analytic_solution as asol


@dataclass
class ModalSpace:
    N: int                      # number of microtubules (angular fold)
    Nmax: int                   # highest angular index n (harmonic nN)
    Jmax: int                   # radial modes per angular index
    Qmax: int                   # AL radial basis size
    domain_radius: float = 1.0
    n_quad: int = 400           # Gauss-Legendre nodes on (0, R)

    # filled in __post_init__
    cos_index: list = field(default_factory=list)   # [(n,j)] n=0..Nmax
    sin_index: list = field(default_factory=list)   # [(n,j)] n=1..Nmax
    kappa_cos: np.ndarray = None
    kappa_sin: np.ndarray = None
    nN_cos: np.ndarray = None
    nN_sin: np.ndarray = None
    r: np.ndarray = None        # quad nodes
    w: np.ndarray = None        # quad weights
    Jcos: np.ndarray = None     # (len cos_index, n_quad)  J_{nN}(kappa r)
    Jsin: np.ndarray = None     # (len sin_index, n_quad)
    CHI: np.ndarray = None      # (Qmax, n_quad)  chi_q(r)
    dCHI: np.ndarray = None     # (Qmax, n_quad)  chi_q'(r)

    def __post_init__(self):
        R = self.domain_radius
        # zeros per order, cached
        zeros = {n: asol.fourier_bessel_zeros(n * self.N, self.Jmax)
                 for n in range(self.Nmax + 1)}
        self.cos_index = [(n, j) for n in range(self.Nmax + 1)
                          for j in range(1, self.Jmax + 1)]
        self.sin_index = [(n, j) for n in range(1, self.Nmax + 1)
                          for j in range(1, self.Jmax + 1)]
        self.kappa_cos = np.array([zeros[n][j - 1] for (n, j) in self.cos_index])
        self.kappa_sin = np.array([zeros[n][j - 1] for (n, j) in self.sin_index])
        self.nN_cos = np.array([n * self.N for (n, j) in self.cos_index], float)
        self.nN_sin = np.array([n * self.N for (n, j) in self.sin_index], float)

        self.r, self.w = asol.radial_quadrature(self.n_quad, R)
        self.Jcos = self.eval_Jcos(self.r)
        self.Jsin = self.eval_Jsin(self.r)
        self.CHI = self.eval_chi(self.r)
        self.dCHI = self.eval_dchi(self.r)

    # ---- radial DL basis on arbitrary r ----
    def eval_Jcos(self, r):
        r = np.asarray(r, float)
        R = self.domain_radius
        return np.array([jv(n * self.N, self.kappa_cos[i] * r / R)
                         for i, (n, j) in enumerate(self.cos_index)])

    def eval_Jsin(self, r):
        r = np.asarray(r, float)
        R = self.domain_radius
        return np.array([jv(n * self.N, self.kappa_sin[i] * r / R)
                         for i, (n, j) in enumerate(self.sin_index)])

    # ---- AL basis: chi_q(r) = (1-r) * T_{q-1}(2r-1), q=1..Qmax ----
    # Shifted-Chebyshev * (1-r): satisfies chi_q(1)=0, leaves R(0) free, and is
    # far better conditioned than the raw monomial basis r^{q-1}(1-r).
    def eval_chi(self, r):
        r = np.asarray(r, float)
        u = 2.0 * r - 1.0
        return np.array([(1.0 - r) * eval_chebyt(q - 1, u) for q in range(1, self.Qmax + 1)])

    def eval_dchi(self, r):
        r = np.asarray(r, float)
        u = 2.0 * r - 1.0
        out = []
        for q in range(1, self.Qmax + 1):
            k = q - 1
            T = eval_chebyt(k, u)
            # d/dr T_k(2r-1) = 2 k U_{k-1}(2r-1);  U_{-1}:=0
            dT = 2.0 * k * eval_chebyu(k - 1, u) if k >= 1 else np.zeros_like(r)
            out.append(-T + (1.0 - r) * dT)
        return np.array(out)

    # sizes
    @property
    def nA(self):
        return len(self.cos_index)

    @property
    def nB(self):
        return len(self.sin_index)
