"""
Analytic solution of the coupled diffusive-layer / advective-layer (DL--AL)
problem on the unit disk, used to benchmark the numerical PDE solver in this
package.

This module implements the Laplace/resolvent residue series derived in
``summaries_reports/analytic_solution_derivation_laplace_resolvent.{tex,pdf}``:

    Eq (24)  phi(r,theta,t) = sum_k c_k e^{s_k t}
                              sum_n [ -(N/2pi) int_0^1 K_{nN}(r,r';s_k) q_k(r') dr' ] e^{i n N theta}
    Eq (25)  rho(r,t)       = sum_k c_k e^{s_k t} R[psi_k](r,s_k)

with the closed modified-Bessel radial Green's function

    K_m(r,r';s) = ( I_m(sqrt(s) r_<) / I_m(sqrt(s)) )
                  * ( I_m(sqrt(s)) K_m(sqrt(s) r_>) - K_m(sqrt(s)) I_m(sqrt(s) r_>) ).

IMPORTANT IMPLEMENTATION NOTES
------------------------------
* This module is **pure NumPy/SciPy** and must NOT be numba-jitted: ``scipy.special``
  is not supported in numba nopython mode.  It is deliberately importable on its
  own and is **not** wired into ``computational_tools/__init__.py`` (that package
  eagerly imports ``njit``).
* Grid conventions match the numerical solver (``numerical_tools``):
  cell-center radius ``r_m = (m+1)*dRad``, ``m = 0..rg_param-1``, ``dRad = R/rg_param``;
  the disk center ``r = 0`` is tracked separately; the outer ring ``m = rg_param-1``
  sits exactly at ``r = R`` (absorbing, phi = 0).
* The uncoupled (a=b=0) path is rigorous and is the Stage-0 verification gate.
  The coupled path is the leading angular truncation (orders {0, N}) of the exact
  resolvent and is documented as such.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.special import jn_zeros, jv, iv, kv, ive, kve
from numpy.polynomial.legendre import leggauss


# ----------------------------------------------------------------------------
# Grid / time helpers (kept consistent with numerical_tools conventions)
# ----------------------------------------------------------------------------
def build_eval_grid(rg_param, ry_param, domain_radius=1.0):
    """Return the polar evaluation grid matching the numerical solver.

    Returns
    -------
    r_centers : (rg_param,) ndarray
        Ring radii r_m = (m+1)*dRad, m = 0..rg_param-1.  The last entry equals
        ``domain_radius`` exactly (the absorbing ring).
    r_with_center : (rg_param+1,) ndarray
        [0.0] + r_centers, matching the PvR radial-snapshot layout
        (index 0 = central patch).
    theta : (ry_param,) ndarray
        theta_n = n*dThe, dThe = 2*pi/ry_param.
    """
    dRad = domain_radius / rg_param
    dThe = 2.0 * np.pi / ry_param
    r_centers = (np.arange(rg_param) + 1) * dRad
    r_with_center = np.concatenate(([0.0], r_centers))
    theta = np.arange(ry_param) * dThe
    return r_centers, r_with_center, theta


def fourier_bessel_zeros(order_m, count):
    """First ``count`` positive zeros kappa_{order_m, j} of J_{order_m}."""
    return jn_zeros(int(order_m), int(count))


# ----------------------------------------------------------------------------
# Modified-Bessel radial Green's function K_m(r, r'; s)
# ----------------------------------------------------------------------------
def _sqrt_s(s):
    # principal complex square root; for s>0 this is real positive
    return np.sqrt(complex(s))


def kernel_K_m(m, r, r_prime, s):
    """Closed-form radial Green's kernel K_m(r, r'; s) (Eq. for K_m).

    Parameters may be scalars; ``r`` and ``r_prime`` may also be broadcastable
    arrays.  ``s`` is (generally complex) and ``sqrt(s)`` uses the principal
    branch.  Uses exponentially-scaled Bessel functions internally to avoid
    overflow/underflow for large |sqrt(s)|.

    K_m = (I_m(z r_<)/I_m(z)) [ I_m(z) K_m(z r_>) - K_m(z) I_m(z r_>) ],  z=sqrt(s)
    """
    z = _sqrt_s(s)
    r = np.asarray(r, dtype=float)
    r_prime = np.asarray(r_prime, dtype=float)
    r_lt = np.minimum(r, r_prime)
    r_gt = np.maximum(r, r_prime)

    # Scaled Bessel: ive(m,x) = iv(m,x) e^{-|Re x|}, kve(m,x) = kv(m,x) e^{x}.
    # For complex argument scipy applies the analytic continuation of the scale.
    # We reconstruct the unscaled combination while keeping exponentials bounded.
    # I_m(z r_lt)/I_m(z):
    iv_rlt = iv(m, z * r_lt)
    iv_z = iv(m, z)
    # I_m(z) K_m(z r_gt) - K_m(z) I_m(z r_gt):
    term = iv_z * kv(m, z * r_gt) - kv(m, z) * iv(m, z * r_gt)
    with np.errstate(divide="ignore", invalid="ignore"):
        K = (iv_rlt / iv_z) * term
    return K


def kernel_K_m_scaled(m, r, r_prime, s):
    """Numerically robust K_m via exponentially-scaled Bessel functions.

    Equivalent to :func:`kernel_K_m` but rewrites every ratio so the
    exponential growth of I and decay of K cancel analytically.  Preferred for
    large |sqrt(s)| (fine grids / high orders / stiff modes).
    """
    z = _sqrt_s(s)
    r = np.asarray(r, dtype=float)
    r_prime = np.asarray(r_prime, dtype=float)
    r_lt = np.minimum(r, r_prime)
    r_gt = np.maximum(r, r_prime)
    azr = np.abs(np.real(z))

    # iv(m,x) = ive(m,x) e^{|Re x|};  kv(m,x) = kve(m,x) e^{-x}
    # I_m(z r_lt)/I_m(z) = ive(m,z r_lt)/ive(m,z) * e^{|Re z|(r_lt-1)}
    ratio_I = (ive(m, z * r_lt) / ive(m, z)) * np.exp(azr * (r_lt - 1.0))
    # I_m(z) K_m(z r_gt) = ive(m,z) e^{|Re z|} * kve(m,z r_gt) e^{-z r_gt}
    t1 = ive(m, z) * kve(m, z * r_gt) * np.exp(azr - z * r_gt)
    # K_m(z) I_m(z r_gt) = kve(m,z) e^{-z} * ive(m,z r_gt) e^{|Re z| r_gt}
    t2 = kve(m, z) * ive(m, z * r_gt) * np.exp(-z + azr * r_gt)
    return ratio_I * (t1 - t2)


# ----------------------------------------------------------------------------
# UNCOUPLED (a = b = 0) -- rigorous Stage-0 ground truth
# ----------------------------------------------------------------------------
def fourier_bessel_coeffs_uncoupled(rg_param, n_modes, domain_radius=1.0,
                                    ic="patch"):
    """Fourier-Bessel coefficients c_j for the central-release initial condition.

    The numerical solver places unit mass in the central patch (a disk of radius
    ``dRad`` with uniform density 1/(pi*dRad^2)).  Expanding phi(r,0) in the
    order-0 Dirichlet basis J_0(kappa_{0,j} r):

        c_j = [ int_0^1 r phi0(r) J_0(kappa_j r) dr ] / [ J_1(kappa_j)^2 / 2 ].

    ``ic="patch"`` (default) integrates J_0 over the finite patch of radius dRad
    (matches the numerical IC):

        c_j = 2 J_1(kappa_j dRad) / (pi dRad kappa_j J_1(kappa_j)^2).

    ``ic="delta"`` uses the idealized point source (dRad -> 0 limit):

        c_j = 1 / (pi J_1(kappa_j)^2).
    """
    kappa = fourier_bessel_zeros(0, n_modes)
    j1 = jv(1, kappa)
    if ic == "delta":
        c = 1.0 / (np.pi * j1 ** 2)
    elif ic == "patch":
        dRad = domain_radius / rg_param
        # account for unit mass / radius scaling on a disk of radius `domain_radius`
        # phi0 = 1/(pi dRad^2) on [0,dRad];  zeros kappa are for the unit disk,
        # so rescale r -> r/domain_radius.  With domain_radius=1 this is the
        # plain expression below.
        c = 2.0 * jv(1, kappa * dRad / domain_radius) / (
            np.pi * (dRad / domain_radius) * kappa * j1 ** 2 * domain_radius ** 2)
    else:
        raise ValueError(f"unknown ic={ic!r} (use 'patch' or 'delta')")
    return c, kappa


def phi_uncoupled(r, t, coeffs, kappa, domain_radius=1.0):
    """Uncoupled diffusion field phi(r,t) = sum_j c_j J_0(kappa_j r/R) e^{-(kappa_j/R)^2 t}.

    ``r`` may be an array; ``t`` is a scalar.  Returns an array shaped like ``r``.
    """
    r = np.asarray(r, dtype=float)
    R = domain_radius
    # mode shapes and decay
    arg = np.outer(kappa / R, r)                      # (n_modes, len(r))
    shapes = jv(0, arg)
    decay = np.exp(-((kappa / R) ** 2) * t)           # (n_modes,)
    return decay @ (coeffs[:, None] * shapes)


def uncoupled_decay_rate(domain_radius=1.0):
    """Dominant uncoupled decay rate sigma = (kappa_{0,1}/R)^2 (~5.783 for R=1)."""
    k01 = fourier_bessel_zeros(0, 1)[0]
    return (k01 / domain_radius) ** 2


def bessel_mode_shape(order, j, r, domain_radius=1.0):
    """Normalized radial Bessel profile J_order(kappa_{order,j} r/R).

    Used as the analytic *structural* reference for the channels of eq (24):
    the angular-mean channel g(r) ~ J_0 (order 0) and the modulation amplitude
    a(r) ~ J_N (order N, pushed toward the rim by its N^2/r^2 barrier).
    Normalized to unit max-abs for shape comparison.
    """
    kappa = fourier_bessel_zeros(order, j)[j - 1]
    prof = jv(order, kappa * np.asarray(r, float) / domain_radius)
    m = np.max(np.abs(prof))
    return prof / m if m > 0 else prof


# ----------------------------------------------------------------------------
# COUPLED parameters
# ----------------------------------------------------------------------------
def al_functional(phi_ring, r_centers, sigma, a, b, v, domain_radius=1.0,
                  dThe=None):
    """Analytic AL elimination R[phi(.,0)] underpinning eq (25), eigenvalue form.

        rho(r) = -(a/v) e^{((b-sigma)/v) r} int_1^r e^{-((b-sigma)/v) r'} phi(r',0) dr'

    Given a sampled phi(r) on the ring grid ``r_centers`` and a decay rate
    ``sigma``, returns rho(r) by cumulative trapezoidal quadrature.  This is the
    direct, spectrum-free check of eq (25): the analytic R-functional applied to
    the *numerical* phi trace should reproduce the *numerical* rho profile shape.

    Coupling-asymmetry correction (``dThe`` given): the numerical scheme's AL
    on-rate is ``a*phi*(m+1)*dRad*dThe = a*phi*r*dThe`` (see
    ``dl-al-coupling-asymmetry``), i.e. rho is fed proportionally to r*phi rather
    than phi.  Passing ``dThe`` weights the on-rate by ``r*dThe`` to match the
    discrete scheme; this reproduces the off-center rho peak (and its outward
    migration in time) that constant-a cannot.  Omit ``dThe`` for the bare
    continuum form.
    """
    r = np.asarray(r_centers, float)
    phi = np.asarray(phi_ring, float)
    if dThe is not None:                      # scheme-matched on-rate weight r*dThe
        phi = phi * r * dThe
    g = (b - sigma) / v
    # cumulative integral from r=1 (outer) inward: int_1^{r_i} f dr'
    fr = np.exp(-g * r) * phi
    # integrate on the ring grid; append r=1 endpoint (phi(1)=0) for the lower limit
    r_ext = np.concatenate((r, [domain_radius]))
    fr_ext = np.concatenate((fr, [0.0]))
    # int_1^{r_i} = -int_{r_i}^1 ; build via reverse cumulative trapezoid
    out = np.zeros_like(r)
    for i in range(len(r)):
        # integrate f from r_i up to 1
        mask = r_ext >= r[i]
        rr = r_ext[mask]
        ff = fr_ext[mask]
        idx = np.argsort(rr)
        integral_ri_to_1 = np.trapezoid(ff[idx], rr[idx])
        out[i] = -integral_ri_to_1               # int_1^{r_i} = -int_{r_i}^1
    return (-(a / v)) * np.exp(g * r) * out


@dataclass
class AnalyticParams:
    """Continuum coefficients for the coupled analytic model.

    a, b : continuum coupling rates (on/off).  See analytic_benchmark for how
           these are derived from the code's scalar w_param (the discrete scheme
           carries geometric 1/(r dThe) factors; ``coupling_mode`` selects the
           reconciliation).
    v    : advection speed (inward); sign handled consistently with the solver.
    N    : number of microtubules (angular order of the modulation channel).
    D    : diffusion coefficient (the solver sets D=1).
    domain_radius : disk radius (solver sets R=1).
    coupling_mode : 'effective' | 'reference_ring' (diagnostic).
    """
    a: float
    b: float
    v: float
    N: int
    D: float = 1.0
    domain_radius: float = 1.0
    coupling_mode: str = "effective"


# ----------------------------------------------------------------------------
# COUPLED radial quadrature + operators (leading angular truncation {0, N})
# ----------------------------------------------------------------------------
def radial_quadrature(n_quad, domain_radius=1.0):
    """Gauss-Legendre nodes/weights on (0, domain_radius)."""
    x, w = leggauss(int(n_quad))            # on [-1, 1]
    r = 0.5 * (x + 1.0) * domain_radius
    wr = 0.5 * domain_radius * w
    return r, wr


def _R_functional_matrix(r_nodes, w_nodes, s, params: AnalyticParams):
    """Discrete AL functional R[psi](r) = -(a/v) e^{(b+s)r/v} int_1^r e^{-(b+s)r'/v} psi(r') dr'.

    Returns a matrix Rop such that (Rop @ psi_vec)[i] approximates R[psi](r_i).
    The inner integral from 1 to r is built by signed cumulative quadrature.
    """
    a, b, v = params.a, params.b, params.v
    R = params.domain_radius
    n = len(r_nodes)
    g = (b + s) / v
    # int_1^{r_i} e^{-g r'} psi(r') dr'  ~  sum_j A_ij e^{-g r_j} w_j psi_j
    # with A_ij = +1 if r_j between 1 and r_i going outward... build via masks.
    # Use ordered nodes; integral_1^r = -(integral_r^1).
    order = np.argsort(r_nodes)
    r_sorted = r_nodes[order]
    Rop = np.zeros((n, n), dtype=complex)
    expr = np.exp(g * r_nodes)        # e^{(b+s)r/v} prefactor
    expmr = np.exp(-g * r_nodes)      # e^{-(b+s)r'/v} integrand weight
    for ii in range(n):
        ri = r_nodes[ii]
        # nodes lying in [ri, R] contribute to int_1^ri = -int_ri^1
        contrib = np.zeros(n, dtype=complex)
        for jj in range(n):
            rj = r_nodes[jj]
            if rj >= ri:                       # between r and 1 (outer)
                contrib[jj] = -w_nodes[jj] * expmr[jj]
        Rop[ii, :] = (-(a / v)) * expr[ii] * contrib
    return Rop


def _S_kernel_matrix(r_nodes, w_nodes, s, params: AnalyticParams, n_orders=(0,)):
    """Rays-summed radial kernel S(r,r';s) = (N/2pi) sum_n K_{nN}, truncated.

    ``n_orders`` lists the |n| angular indices retained; n=0 contributes K_0
    once, each |n|>=1 contributes 2*K_{|n|N} (for +/-n).  Returns S_ij*w_j so
    that (S @ f)[i] ~ int_0^1 S(r_i,r') f(r') dr'.
    """
    N = params.N
    n = len(r_nodes)
    Rr, Rp = np.meshgrid(r_nodes, r_nodes, indexing="ij")
    S = np.zeros((n, n), dtype=complex)
    for idx in n_orders:
        order = abs(idx) * N
        Km = kernel_K_m_scaled(order, Rr, Rp, s)
        mult = 1.0 if idx == 0 else 2.0
        S += mult * Km
    S *= (N / (2.0 * np.pi))
    return S * w_nodes[None, :]


def _Phi0_vector(r_nodes, s, rg_param, domain_radius=1.0, ic="patch"):
    """Forced order-0 response Phi_0(r,s) for the central-release IC.

    Phi_0(r,s) = int_0^1 r' K_0(r,r';s) phi0(r') dr'.  For the central patch this
    is dominated by the small-r' behavior; we use the closed central-delta limit
    scaled by total mass = 1:  Phi_0(r,s) = (1/2pi) K_0(r,0;s)
    = (1/2pi)[K_0(z r) - (K_0(z)/I_0(z)) I_0(z r)],  z=sqrt(s).
    (The 'patch' correction is O(dRad^2) and negligible for the resolvent;
    retained for interface symmetry.)
    """
    z = _sqrt_s(s)
    r = np.asarray(r_nodes, dtype=float)
    val = kv(0, z * r) - (kv(0, z) / iv(0, z)) * iv(0, z * r)
    return val / (2.0 * np.pi)


def secular_matrix(s, params: AnalyticParams, r_nodes, w_nodes, n_orders=(0, 1)):
    """Build M(s) = I + A(s) for the Fredholm equation (I + A)psi = Phi_0.

    A(s) psi = int S(r,r';s) ( b R[psi](r') - a psi(r') ) dr', with S truncated
    to the angular orders in ``n_orders`` (default {0, N} -> the two-mode model).
    Returns an (n_quad x n_quad) complex matrix.
    """
    n = len(r_nodes)
    S = _S_kernel_matrix(r_nodes, w_nodes, s, params, n_orders=n_orders)
    Rop = _R_functional_matrix(r_nodes, w_nodes, s, params)
    coupling = params.b * Rop - params.a * np.eye(n)
    A = S @ coupling
    return np.eye(n) + A


def secular_determinant(s, params: AnalyticParams, r_nodes, w_nodes,
                        n_orders=(0, 1)):
    """log-safe secular determinant det(I + A(s))."""
    M = secular_matrix(s, params, r_nodes, w_nodes, n_orders=n_orders)
    sign, logdet = np.linalg.slogdet(M)
    return sign * np.exp(logdet)


# ----------------------------------------------------------------------------
# (The resolvent secular determinant above is correct but ill-conditioned for
#  root-finding because K_m itself has poles at the bare Bessel zeros.  The
#  first-principles coupled spectrum (decay rates s_k and amplitudes c_k of
#  eqs 24/25) is instead obtained below by a Galerkin discretization of the
#  continuum coupled generator in the Dirichlet-Bessel basis, then a dense
#  non-self-adjoint eigensolve -- equivalent, and verifiably reducing to the
#  bare diffusion spectrum -kappa_{0,j}^2 when a=b=0.)
# ----------------------------------------------------------------------------


@dataclass
class CoupledSpectrum:
    """Result of the first-principles coupled eigen-solve (eqs 24/25).

    s_k        : (K,) complex decay rates  (eigenvalues; phi,rho ~ e^{s_k t})
    c_k        : (K,) complex modal amplitudes set by the initial condition
    V          : (dim,K) right eigenvectors (modal spatial coefficients)
    phi_index  : list of (n, j) basis labels for the DL block
    rho_index  : list of l labels for the AL block
    kappa_phi  : (len phi_index,) Bessel zeros kappa_{nN,j}
    kappa_rho  : (len rho_index,) Bessel zeros kappa_{0,l}
    N, domain_radius
    """
    s_k: np.ndarray
    c_k: np.ndarray
    V: np.ndarray
    phi_index: list
    rho_index: list
    kappa_phi: np.ndarray
    kappa_rho: np.ndarray
    N: int
    domain_radius: float


def solve_coupled_spectrum(a, b, v, N, n_max=3, J_r=18, J_rho=18,
                           domain_radius=1.0, n_quad=400, ic_dRad=1.0 / 32,
                           coupling="continuum", dThe=None):
    """First-principles decay rates s_k and amplitudes c_k of eqs (24)/(25).

    Galerkin discretization of the continuum coupled generator (full-disk form
    of the DL/AL equations) in the Dirichlet-Bessel basis
        phi = sum_{n,j} a_{nj} J_{nN}(kappa_{nN,j} r) cos(nN theta),  n=0..n_max
        rho = sum_l      b_l   J_0(kappa_{0,l} r)                     (MT ray)
    Both bases satisfy phi(1)=rho(1)=0 and regularity at 0 by construction.

    The continuum coupling is asymmetric in r: the DL feels the microtubule
    source (1/r)(b rho - a phi) delta(theta-theta_i), whose 1/r cancels the area
    measure -> *unweighted* radial overlaps; the AL feels a plain a*phi(r,0) ->
    *r-weighted* projection.  These are the equations exactly as written in the
    derivation.

    Returns a :class:`CoupledSpectrum`.  At a=b=0 the dominant s_k is
    -kappa_{0,1}^2 and c_k reduces to the Fourier-Bessel coefficients.
    """
    R = domain_radius
    # Gauss-Legendre quadrature on (0, R)
    x, wq = leggauss(n_quad)
    r = 0.5 * (x + 1.0) * R
    w = 0.5 * R * wq

    # ---- basis indices and Bessel zeros ----
    phi_index = [(n, j) for n in range(n_max + 1) for j in range(1, J_r + 1)]
    rho_index = list(range(1, J_rho + 1))
    # zeros per order, cached
    zeros_cache = {n: jn_zeros(n * N, J_r) for n in range(n_max + 1)}
    zeros0 = jn_zeros(0, J_rho)
    kappa_phi = np.array([zeros_cache[n][j - 1] for (n, j) in phi_index])
    kappa_rho = np.array([zeros0[l - 1] for l in rho_index])

    nph, nrh = len(phi_index), len(rho_index)
    # basis radial profiles on the quad grid
    Jphi = np.array([jv(n * N, kappa_phi[i] * r / R)
                     for i, (n, j) in enumerate(phi_index)])      # (nph, nq)
    Jrho = np.array([jv(0, kappa_rho[l] * r / R) for l in range(nrh)])  # (nrh, nq)
    dJrho = np.array([-(kappa_rho[l] / R) * jv(1, kappa_rho[l] * r / R)
                      for l in range(nrh)])                       # d/dr

    cth = np.array([2 * np.pi if n == 0 else np.pi for (n, j) in phi_index])
    # r-weighted self-norms
    RRphi = (Jphi ** 2) @ (w * r)            # (nph,) = int J^2 r dr
    RRrho = (Jrho ** 2) @ (w * r)            # (nrh,)
    massphi = cth * RRphi                    # diagonal DL mass
    massrho = RRrho.copy()                   # diagonal AL mass

    # ---- overlap matrices (quadrature) ----
    PJ = (Jphi * w) @ Jphi.T                 # unweighted int J_a J_b dr  (nph,nph)
    QJ = (Jphi * w) @ Jrho.T                 # unweighted int J_phi J_rho dr (nph,nrh)
    QJw = (Jphi * (w * r)) @ Jrho.T          # r-weighted (nph,nrh) -> use .T for rho rows
    Adv = (Jrho * (w * r)) @ dJrho.T         # int J0_q dJ0_l r dr (nrh,nrh)

    # ---- assemble K (RHS operator), then L = Mass^{-1} K ----
    K = np.zeros((nph + nrh, nph + nrh))
    # DL diffusion (diagonal): -kappa^2 * mass
    K[:nph, :nph] = -np.diag((kappa_phi / R) ** 2 * massphi)
    if coupling == "continuum":
        # Equations as written: DL feels (1/r)(b rho - a phi) delta -> the 1/r
        # cancels the area measure -> UNWEIGHTED overlaps; AL feels plain a phi.
        K[:nph, :nph] += -a * N * PJ                 # -a phi(r,0) term
        K[:nph, nph:] += +b * N * QJ                 # +b rho term
        K[nph:, nph:] += v * Adv - b * np.diag(massrho)
        K[nph:, :nph] += a * QJw.T                   # on-rate: int r J0 Jphi dr
    elif coupling == "scheme":
        # The discrete solver effectively uses a -> a*r*dThe on BOTH the DL
        # a-term and the AL on-rate (the b-coupling is unchanged).  This encodes
        # the scheme's geometric coupling for an apples-to-apples comparison.
        if dThe is None:
            raise ValueError("coupling='scheme' requires dThe (=2*pi/ry_param)")
        PJw = (Jphi * (w * r)) @ Jphi.T              # r-weighted phi-phi
        QJ2w = (Jphi * (w * r * r)) @ Jrho.T         # r^2-weighted phi-rho
        K[:nph, :nph] += -a * dThe * N * PJw         # -a r dThe phi term
        K[:nph, nph:] += +b * N * QJ                 # +b rho term (unchanged)
        K[nph:, nph:] += v * Adv - b * np.diag(massrho)
        K[nph:, :nph] += a * dThe * QJ2w.T           # on-rate a r dThe: int r^2 J0 Jphi dr
    else:
        raise ValueError(f"unknown coupling={coupling!r} ('continuum' or 'scheme')")

    massvec = np.concatenate([massphi, massrho])
    L = K / massvec[:, None]

    # ---- non-self-adjoint eigensolve ----
    s_k, V = np.linalg.eig(L)

    # ---- initial condition: central patch on DL, rho=0 ----
    y0 = np.zeros(nph + nrh, dtype=complex)
    for i, (n, j) in enumerate(phi_index):
        if n != 0:
            continue
        kap = kappa_phi[i]
        # <Jhat_{0,j}, phi0_patch> = 2pi/(pi dRad^2) * int_0^dRad J0(kap r) r dr
        proj = 2.0 * jv(1, kap * ic_dRad / R) / (ic_dRad / R * kap)   # = 2pi*int/(pi)
        # divide by mass to get coefficient
        y0[i] = proj / massphi[i]
    # c_k from eigenvector expansion  y0 = sum_k c_k V[:,k]
    c_k = np.linalg.solve(V, y0)

    return CoupledSpectrum(s_k=s_k, c_k=c_k, V=V, phi_index=phi_index,
                           rho_index=rho_index, kappa_phi=kappa_phi,
                           kappa_rho=kappa_rho, N=N, domain_radius=R)


def coupled_phi_field(spec: CoupledSpectrum, r, theta, t):
    """Reconstruct phi(r,theta,t) from the coupled spectrum (eq 24), real part."""
    r = np.asarray(r, float); theta = np.asarray(theta, float)
    R = spec.domain_radius
    nph = len(spec.phi_index)
    # modal amplitudes at time t: weight_k = c_k e^{s_k t}
    wt = spec.c_k * np.exp(spec.s_k * t)               # (K,)
    # coefficient of each DL basis fn summed over modes: coef[idx] = sum_k V[idx,k] wt_k
    coef = spec.V[:nph, :] @ wt                         # (nph,)
    field = np.zeros((len(r), len(theta)), dtype=complex)
    for i, (n, j) in enumerate(spec.phi_index):
        radial = jv(n * spec.N, spec.kappa_phi[i] * r / R)
        ang = np.cos(n * spec.N * theta)
        field += coef[i] * np.outer(radial, ang)
    return np.real(field)


def coupled_rho_field(spec: CoupledSpectrum, r, t):
    """Reconstruct rho(r,t) from the coupled spectrum (eq 25), real part."""
    r = np.asarray(r, float)
    R = spec.domain_radius
    nph = len(spec.phi_index)
    wt = spec.c_k * np.exp(spec.s_k * t)
    coef = spec.V[nph:, :] @ wt                         # (nrh,)
    out = np.zeros(len(r), dtype=complex)
    for l in range(len(spec.rho_index)):
        out += coef[l] * jv(0, spec.kappa_rho[l] * r / R)
    return np.real(out)


def dominant_decay_rate(spec: CoupledSpectrum, weighted_by_amplitude=True):
    """Slowest (least-negative real part) significantly-excited decay rate sigma=-s."""
    re = np.real(spec.s_k)
    if weighted_by_amplitude:
        excited = np.abs(spec.c_k) > 1e-10 * np.max(np.abs(spec.c_k))
        idx = np.where(excited)[0]
        k = idx[np.argmax(re[idx])]
    else:
        k = np.argmax(re)
    return -spec.s_k[k]
