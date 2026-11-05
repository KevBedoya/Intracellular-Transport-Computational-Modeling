"""Mass analysis by eigen-modal expansion of the solver's one-step operator.

The time-stepper (analysis_tools.comp_mass_analysis_respect_to_time) is linear
and time-invariant: one step is x_{k+1} = P x_k, with the state x holding the
centre patch, the diffusive layer on rings 0 .. rg-2 and the advective layer on
the tube rays. The layer masses are linear functionals of x, so

    M(x_k) = w^T P^k x_0 = sum_n a_n beta_n^k,

where beta_n are the eigenvalues of P and a_n the modal amplitudes. This module
builds P from the solver's own step functions, diagonalises it once and
evaluates that sum at the sample steps, so it carries the same discretisation
as the time-stepper and differs from it only by round-off -- at a cost that does
not grow with T.

Two reductions keep the eigensolve small:

  * Symmetry. Rotations and reflections of the rays that map the tube set (and
    an off-centre seed) to themselves commute with P, so P^k x_0 stays in the
    subspace of states that are constant on each orbit of those symmetries.
    P is only ever applied to the orbit basis, and the restriction is checked:
    a residual above ``INVARIANCE_TOL`` drops the symmetry and solves on the
    full state instead.
  * Mode pruning. A mode is skipped over a block of samples once its largest
    possible contribution there is below ``PRUNE_TOL`` of the block's scale.

Every eigenmode of the restricted operator is kept (a dense eigensolve), not
the slowest few: the early decay of M(t) needs the fast modes.
"""
import time

import numpy as np
import scipy.linalg as la

from . import numerical_tools as num
from computational_tools import struct_init

INVARIANCE_TOL = 1e-12      # max |P Q - Q P_s|, relative to max |P Q|
PRUNE_TOL = 1e-17           # dropped-mode bound, relative to the block scale
BLOCK_ELEMENTS = 1 << 22    # samples x active modes evaluated per block


class _Layout:
    """Index map between the solver's layers and the flat state vector.

    x[0] is the centre patch, x[1 : 1 + nD] the diffusive layer on rings
    0 .. rg-2 (ring rg-1 is the absorbing boundary, always 0), and the rest the
    advective layer, one block of rg rings per tube.
    """

    def __init__(self, rg_param, ry_param, N_LIST):
        self.rg, self.ry = rg_param, ry_param
        self.N_LIST = np.asarray(N_LIST, dtype=np.int64)
        self.nD = (rg_param - 1) * ry_param
        self.nA = len(self.N_LIST) * rg_param
        self.n = 1 + self.nD + self.nA

    def unpack(self, x):
        D_LAYER = np.zeros((2, self.rg, self.ry))
        A_LAYER = np.zeros((2, self.rg, self.ry))
        D_LAYER[0, :self.rg - 1, :] = x[1:1 + self.nD].reshape(self.rg - 1, self.ry)
        A_LAYER[0][:, self.N_LIST] = x[1 + self.nD:].reshape(len(self.N_LIST), self.rg).T
        return x[0], D_LAYER, A_LAYER

    def pack_next(self, central_patch, D_LAYER, A_LAYER):
        return np.concatenate([[central_patch], D_LAYER[1, :self.rg - 1, :].ravel(),
                               A_LAYER[1][:, self.N_LIST].T.ravel()])

    def ray_permutation(self, g):
        """State permutation p, (S x)[p[j]] = x[j], for the ray map n -> g[n]."""
        rg, ry = self.rg, self.ry
        tube_of_ray = {int(r): i for i, r in enumerate(self.N_LIST)}
        p = np.empty(self.n, dtype=np.int64)
        p[0] = 0
        m, n = np.divmod(np.arange(self.nD), ry)
        p[1:1 + self.nD] = 1 + m * ry + g[n]
        for i, r in enumerate(self.N_LIST):
            i_img = tube_of_ray[int(g[r])]
            p[1 + self.nD + i * rg:1 + self.nD + (i + 1) * rg] = (
                1 + self.nD + i_img * rg + np.arange(rg))
        return p


class _Operator:
    """The solver's one-step map and mass functionals, on flat state vectors."""

    def __init__(self, layout, switch_param_a, switch_param_b, v_param, d_tube,
                 domain_radius, D):
        self.L = layout
        self.a, self.b = switch_param_a, switch_param_b
        self.v = -v_param                      # the drivers pass -v to the stencil
        self.d_tube = d_tube
        self.dRad = num.compute_dRad(layout.rg, domain_radius)
        self.dThe = num.compute_dThe(layout.ry)
        self.dT = num.compute_dT(layout.rg, layout.ry, domain_radius, D)
        self.d_map = struct_init.build_d_tube_map_dense(layout.rg, layout.ry, layout.N_LIST,
                                                        d_tube, domain_radius)

    def step(self, x):
        L = self.L
        central_patch, D_LAYER, A_LAYER = L.unpack(x)
        num.comp_DL_AL_kp1_2step(L.ry, L.rg, self.d_map, D_LAYER, central_patch, A_LAYER,
                                 L.N_LIST, self.dRad, self.dThe, self.dT, self.a, self.b,
                                 self.v, self.d_tube)
        central_next = num.u_center(D_LAYER, 0, self.dRad, self.dThe, self.dT, central_patch,
                                    A_LAYER, L.N_LIST, self.v)
        return L.pack_next(central_next, D_LAYER, A_LAYER)

    def masses(self, x):
        """(diffusive-layer mass, advective-layer mass) of state x."""
        L = self.L
        central_patch, D_LAYER, A_LAYER = L.unpack(x)
        return (num.calc_mass_diff(D_LAYER, 0, self.dRad, self.dThe, central_patch, L.rg, L.ry),
                num.calc_mass_adv(A_LAYER, 0, self.dRad, self.dThe, L.rg, L.N_LIST))


def initial_state(layout, center_init_cond=True, m_init=0, n_init=0, domain_radius=1.0):
    """The seed comp_mass_analysis_respect_to_time places, as a flat state."""
    x0 = np.zeros(layout.n)
    if center_init_cond:
        x0[0] = num.compute_init_cond_cent(layout.rg, domain_radius)
    else:
        if m_init >= layout.rg - 1:
            raise ValueError("an off-centre seed on the absorbing ring is never stepped")
        x0[1 + m_init * layout.ry + n_init] = num.compute_init_cond_patch(
            layout.rg, layout.ry, m_init, domain_radius)
    return x0


def symmetry_orbits(layout, d_map, center_init_cond=True, n_init=0):
    """Orbit label of every state index under the ray symmetries of the problem.

    The candidates are the dihedral maps n -> (s + sigma n) mod ry; one is kept
    when it maps the tube set to itself, maps the extraction map d_map onto
    itself and, for an off-centre seed, fixes the seed's ray. The kept maps form
    a group, so each index's orbit is the set of its images under them.
    """
    ry = layout.ry
    tubes = set(int(r) for r in layout.N_LIST)
    n = np.arange(ry)
    perms = []
    for sigma in (1, -1):
        for s in range(ry):
            g = (s + sigma * n) % ry
            if set(int(g[r]) for r in tubes) != tubes:
                continue
            if not center_init_cond and g[n_init] != n_init:
                continue
            g_map = np.where(d_map >= 0, g[np.maximum(d_map, 0)], -1)
            if not np.array_equal(d_map[:, g], g_map):
                continue
            perms.append(layout.ray_permutation(g))
    canonical = np.min(np.stack(perms), axis=0)
    _, labels = np.unique(canonical, return_inverse=True)
    return labels


def build_reduced_operator(op, x0, labels):
    """P restricted to the orbit basis Q (orthonormal orbit indicators).

    Returns P_s = Q^T P Q, the diffusive and advective mass weights Q^T w_D and
    Q^T w_A, x_s = Q^T x0, and the invariance residual max |P Q - Q P_s| relative
    to max |P Q|. Only the r basis vectors are stepped, never the full P.
    """
    r = int(labels.max()) + 1
    size = np.bincount(labels, minlength=r).astype(np.float64)
    scale = 1.0 / np.sqrt(size)
    members = np.argsort(labels, kind="stable")
    starts = np.concatenate([[0], np.cumsum(size.astype(np.int64))])

    Ps = np.empty((r, r))
    ws_D, ws_A = np.empty(r), np.empty(r)
    q = np.zeros(op.L.n)
    resid, pq_max = 0.0, 0.0
    for o in range(r):
        idx = members[starts[o]:starts[o + 1]]
        q[idx] = scale[o]
        y = op.step(q)
        ws_D[o], ws_A[o] = op.masses(q)
        q[idx] = 0.0
        Ps[:, o] = np.bincount(labels, weights=y, minlength=r) * scale
        resid = max(resid, float(np.max(np.abs(y - (Ps[:, o] * scale)[labels]))))
        pq_max = max(pq_max, float(np.max(np.abs(y))))
    xs = np.bincount(labels, weights=x0, minlength=r) * scale
    return Ps, ws_D, ws_A, xs, resid / pq_max


def modal_decomposition(Ps, dT, ws_D, ws_A, xs):
    """Eigenmodes of P_s = I + dT L_s and the layer-mass amplitudes of x_s.

    Returns log(beta_n) (complex), the diffusive and advective amplitudes, and
    cond(V), the eigenvector conditioning that bounds the round-off of the sum.
    The eigensolve is of L_s, and log(beta) = log1p(dT mu), so slow modes keep
    their relative precision. Modes with beta = 0 vanish after one step and are
    dropped (the series never samples k = 0 from the sum).
    """
    r = Ps.shape[0]
    mu, V = la.eig((Ps - np.eye(r)) / dT)
    c = la.solve(V, xs.astype(complex))
    amp_D, amp_A = (ws_D @ V) * c, (ws_A @ V) * c
    beta = 1 + dT * mu
    keep = beta != 0
    log_beta = np.log1p(dT * mu[keep].astype(complex))
    order = np.argsort(-log_beta.real)                     # slowest first
    return (log_beta[order], amp_D[keep][order], amp_A[keep][order],
            float(np.linalg.cond(V)))


def evaluate_modal_sum(k, log_beta, amps):
    """sum_n amp_n beta_n^k for each column of ``amps``, at the steps ``k``.

    ``k`` must be non-decreasing. Over each block of samples, a mode whose
    largest contribution there is below PRUNE_TOL of the block's scale (the sum
    of every mode's magnitude at the block's last step) is left out.
    """
    k = np.asarray(k, dtype=np.float64)
    amps = np.atleast_2d(np.asarray(amps).T).T                       # (modes, cols)
    out = np.empty((k.size, amps.shape[1]))
    mag = np.abs(amps).max(axis=1)
    rate = log_beta.real
    i, length = 0, 64
    while i < k.size:
        k0 = k[i]
        # Blocks grow geometrically, so the fast modes -- all active at k = 0 and
        # gone soon after -- are pruned within a few short blocks, rather than
        # carried through one long first block. The cap keeps memory bounded.
        with np.errstate(under="ignore"):
            bound0 = mag * np.exp(rate * k0)
        n_act = max(int(np.count_nonzero(bound0 >= PRUNE_TOL * bound0.sum())), 1)
        j = min(k.size, i + max(min(length, BLOCK_ELEMENTS // n_act), 1))
        length *= 2
        with np.errstate(under="ignore"):
            bound1 = mag * np.exp(rate * k[j - 1])
        # |beta| <= 1 makes bound0 the block maximum; max() also covers |beta| > 1
        act = np.maximum(bound0, bound1) >= PRUNE_TOL * bound1.sum()
        with np.errstate(under="ignore"):
            E = np.exp(k[i:j, None] * log_beta[None, act])
        out[i:j] = (E @ amps[act]).real
        i = j
    return out


def solve_modes(rg_param, ry_param, switch_param_a, switch_param_b, v_param, N_LIST, d_tube=0,
                domain_radius=1.0, D=1.0, center_init_cond=True, m_init=0, n_init=0,
                use_symmetry=True):
    """Every mode of the layer masses for one parameter point.

    ``v_param`` is the physical velocity (not negated), as for the time-stepper.

    Returns a dict: ``log_beta`` (complex, slowest first), the diffusive and
    advective amplitudes ``amp_D``/``amp_A`` and their sum ``amplitude``, the
    decay rates ``decay_rate`` = -log(beta)/dT, ``dT``, the state and reduced
    sizes, the invariance residual, cond(V), and the wall time of the build and
    of the eigensolve. M(k dT) = sum_n amplitude_n beta_n^k.
    """
    t_start = time.perf_counter()
    layout = _Layout(rg_param, ry_param, np.sort(np.asarray(N_LIST, dtype=np.int64)))
    op = _Operator(layout, switch_param_a, switch_param_b, v_param, d_tube, domain_radius, D)
    x0 = initial_state(layout, center_init_cond, m_init, n_init, domain_radius)

    labels = (symmetry_orbits(layout, op.d_map, center_init_cond, n_init) if use_symmetry
              else np.arange(layout.n))
    Ps, ws_D, ws_A, xs, resid = build_reduced_operator(op, x0, labels)
    if resid > INVARIANCE_TOL:
        print(f"Symmetry reduction not exact (residual {resid:.1e}); solving on the full state.")
        labels = np.arange(layout.n)
        Ps, ws_D, ws_A, xs, resid = build_reduced_operator(op, x0, labels)
    t_build = time.perf_counter()

    log_beta, amp_D, amp_A, cond_V = modal_decomposition(Ps, op.dT, ws_D, ws_A, xs)
    t_eig = time.perf_counter()

    return dict(log_beta=log_beta, amp_D=amp_D, amp_A=amp_A, amplitude=amp_D + amp_A,
                decay_rate=-log_beta / op.dT, dT=op.dT,
                n_state=layout.n, n_reduced=Ps.shape[0], invariance_residual=resid,
                cond_V=cond_V, wall_build=t_build - t_start, wall_eig=t_eig - t_build)


def ln_mass(modes, t):
    """ln M(t) from the modal sum, relative to the slowest mode so nothing underflows."""
    k = np.asarray(t, dtype=np.float64) / modes["dT"]
    log_beta = modes["log_beta"]
    S0 = evaluate_modal_sum(k, log_beta - log_beta[0].real, modes["amplitude"])[:, 0]
    return np.log(S0) + log_beta[0].real * k


def ln_mass_derivatives(modes, t):
    """M(t), y'(t), y''(t) of y = ln M, and a round-off bound on y'', from the modal sum.

    With p_n = a_n e^{-lambda_n t} / M, y' = -sum p_n lambda_n and
    y'' = sum p_n lambda_n^2 - (sum p_n lambda_n)^2, exactly -- no differencing.
    The sums are taken relative to the slowest mode, so nothing underflows at
    large t. ``y2_noise`` bounds the round-off in y'': the amplitudes carry a
    relative error of about eps * cond(V), and an error d a_n moves y'' by
    (d a_n e^{-lambda_n t} / M) ((lambda_n - lambda_bar)^2 - y''), with
    lambda_bar = -y' -- so the dominant mode, whose rate is lambda_bar, adds
    almost nothing: rescaling a single exponential leaves ln M straight.
    """
    t = np.asarray(t, dtype=np.float64)
    dT = modes["dT"]
    log_beta = modes["log_beta"]
    lam = modes["decay_rate"]
    a = modes["amplitude"]
    shift = log_beta - log_beta[0].real                   # e^{lambda_1 t} factored out
    k = t / dT
    S0, S1, S2 = evaluate_modal_sum(k, shift, np.column_stack([a, a * lam, a * lam ** 2])).T
    absa = np.abs(a)
    B0, BR, B2 = evaluate_modal_sum(k, shift.real.astype(complex),
                                    np.column_stack([absa, absa * lam.real,
                                                     absa * np.abs(lam) ** 2])).T
    y1 = -S1 / S0
    y2 = S2 / S0 - (S1 / S0) ** 2
    # sum |a_n| |lambda_n - lambda_bar|^2 e^{-Re lambda_n t}, lambda_bar = -y'
    spread = np.maximum(B2 + 2 * y1 * BR + y1 ** 2 * B0, 0.0)
    noise = np.finfo(float).eps * modes["cond_V"] * (spread + np.abs(y2) * B0) / np.abs(S0)
    M = S0 * np.exp(log_beta[0].real * k)
    return M, y1, y2, noise


def comp_mass_analysis_modal_expansion(rg_param, ry_param, switch_param_a, switch_param_b, v_param,
                                       T_param, N_LIST, MA_DL_timeseries, MA_AL_timeseries,
                                       MA_ALoI_timeseries, MA_ALoT_timeseries, MA_TM_timeseries,
                                       MA_collection_factor, relative_k, d_tube=0,
                                       domain_radius=1.0, D=1.0,
                                       center_init_cond=True, m_init=0, n_init=0,
                                       use_symmetry=True):
    """Fill the five mass series of comp_mass_analysis_respect_to_time from the modal sum.

    Same arguments and the same sampling as the time-stepper, which records at
    sample j the masses it computed one step earlier: slot 0 holds the initial
    (DL, AL) = (1, 0) and slot j >= 1 the masses of x_{j*MA_collection_factor - 1}.
    ``v_param`` is the physical velocity (not negated), as for the time-stepper.

    Returns the dict of solve_modes, plus the wall time of filling the series
    and of the whole call.
    """
    t_start = time.perf_counter()
    modes = solve_modes(rg_param, ry_param, switch_param_a, switch_param_b, v_param, N_LIST,
                        d_tube, domain_radius, D, center_init_cond, m_init, n_init, use_symmetry)
    t_eig = time.perf_counter()

    k = np.arange(relative_k) * MA_collection_factor - 1
    dl_al = evaluate_modal_sum(k[1:], modes["log_beta"],
                               np.column_stack([modes["amp_D"], modes["amp_A"]]))
    MA_DL_timeseries[0], MA_AL_timeseries[0] = 1.0, 0.0
    MA_DL_timeseries[1:relative_k], MA_AL_timeseries[1:relative_k] = dl_al[:, 0], dl_al[:, 1]
    MA_TM_timeseries[:relative_k] = MA_DL_timeseries[:relative_k] + MA_AL_timeseries[:relative_k]
    MA_ALoT_timeseries[:relative_k] = MA_AL_timeseries[:relative_k] / MA_TM_timeseries[:relative_k]
    MA_ALoI_timeseries[:relative_k] = MA_AL_timeseries[:relative_k] / D
    t_end = time.perf_counter()

    return dict(modes, wall_evaluate=t_end - t_eig, wall_total=t_end - t_start)
