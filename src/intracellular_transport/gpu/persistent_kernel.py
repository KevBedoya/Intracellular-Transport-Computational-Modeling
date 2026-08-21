"""Persistent cooperative kernel for the mass-analysis time loop.

One launch runs many timesteps, with a grid-wide barrier between them, because
per-step launches are dominated by host-side argument marshalling: an empty
kernel taking 17 arguments costs 184 us to launch on this machine, against ~3 us
for the physics itself. See docs/GPU_PLAN.md.

The per-step sequence mirrors the CPU loop exactly, including two details that
are easy to get wrong:

  * the stencil writes the *next* slot while the masses and the centre are
    computed from the *current* one, before the slots swap;
  * the sample recorded at step k holds the masses computed at the end of step
    k-1, not the ones about to be computed.

Reductions are parallel here and sequential on the CPU, so results are close but
not bit-identical -- floating-point addition is not associative. They are
deterministic for a fixed block count, so GPU runs reproduce each other exactly.
"""

import numpy as np

try:
    from numba import cuda
    CUDA_IMPORT_ERROR = None
except Exception as exc:                      # pragma: no cover
    cuda = None
    CUDA_IMPORT_ERROR = exc

# Fixed so shared-memory arrays can be sized statically, and so the reduction
# order -- and therefore the result -- is reproducible run to run.
THREADS_PER_BLOCK = 128
BLOCKS = 112          # 28 SMs x 4; cooperative launch fails above this here


def is_available():
    """True if a usable CUDA device is present."""
    if cuda is None:
        return False
    try:
        return cuda.is_available()
    except Exception:
        return False


def unavailable_reason():
    if cuda is None:
        return f"numba.cuda could not be imported: {CUDA_IMPORT_ERROR}"
    try:
        if not cuda.is_available():
            return "numba reports no CUDA device available"
    except Exception as exc:
        return f"probing CUDA failed: {exc}"
    return None


if cuda is not None:

    @cuda.jit(device=True, inline=True)
    def _patch_update(phi, rho, cur, dmap, tube, m, n, M, N,
                      dRad, dThe, dT, a, b, central):
        """One diffusive-layer patch, matching u_density / u_density_rect."""
        np1 = n + 1
        if np1 == N:
            np1 = 0
        nm1 = n - 1
        if nm1 < 0:
            nm1 = N - 1

        c = phi[cur, m, n]
        up = phi[cur, m + 1, n]
        dn = phi[cur, m - 1, n] if m > 0 else central
        e = phi[cur, m, np1]
        w = phi[cur, m, nm1]

        # radial: j_r_r and j_l_r
        j_r = -1.0 * ((up - c) / dRad)
        j_l = -1.0 * ((c - dn) / dRad)
        ca = (((m + 2) * j_r) - ((m + 1) * j_l)) * (dT / ((m + 1) * dRad))

        # angular: j_r_t and j_l_t
        r = (m + 1) * dRad
        jt_r = -1.0 * (e - c) / (r * dThe)
        jt_l = -1.0 * (c - w) / (r * dThe)
        cb = (jt_r - jt_l) * (dT / ((m + 1) * dRad * dThe))

        pos = dmap[m, n]
        if pos >= 0:
            # extraction region: d_tube == 0 gives j_max == 0, so the
            # (1 + 2*j_max) factor is 1 -- the only case this port supports.
            cc = a * c * dT - (b * rho[cur, m, pos] * dT) / ((m + 1) * dRad * dThe)
        elif tube[n] != 0:
            cc = (a * c) * dT - ((b * rho[cur, m, n]) * dT) / ((m + 1) * dRad * dThe)
        else:
            cc = 0.0
        return c - ca - cb - cc

    @cuda.jit
    def solve_chunk(phi, rho, dmap, tube, NL, partial, series, state,
                    M, N, nt, dRad, dThe, dT, a, b, v, Dcoef,
                    k_start, k_end, ma_factor, relative_k):
        """Advance from k_start to k_end inside a single cooperative launch.

        ``state`` is a small device array carrying what the CPU loop keeps in
        locals: [central_patch, dl_mass, al_mass, MA_k_step]. ``partial`` is
        (3, BLOCKS) scratch for the three per-step reductions.

        On entry the current field is in slot 0. On exit it is in slot 0 again --
        the caller does not have to reason about parity.
        """
        g = cuda.cg.this_grid()
        tid = cuda.grid(1)
        stride = cuda.gridsize(1)
        lane = cuda.threadIdx.x
        bid = cuda.blockIdx.x
        nblocks = cuda.gridDim.x

        sh = cuda.shared.array(shape=(3, THREADS_PER_BLOCK), dtype=np.float64)

        total = M * N
        adv_total = M * nt

        for k in range(k_start, k_end):
            cur = (k - k_start) & 1
            nxt = 1 - cur
            central = state[0]

            # ---- 1. stencil: read cur, write nxt ----
            i = tid
            while i < total:
                m = i // N
                n = i - m * N
                if m == M - 1:
                    phi[nxt, m, n] = 0.0
                else:
                    phi[nxt, m, n] = _patch_update(
                        phi, rho, cur, dmap, tube, m, n, M, N,
                        dRad, dThe, dT, a, b, central)
                i += stride

            t = tid
            while t < adv_total:
                m = t // nt
                n = NL[t - m * nt]
                j_l = v * rho[cur, m, n]
                j_r = 0.0 if m == M - 1 else v * rho[cur, m + 1, n]
                rho[nxt, m, n] = (rho[cur, m, n]
                                  - ((j_r - j_l) * (1.0 / dRad)) * dT) \
                    + phi[cur, m, n] * a * (m + 1) * (dRad * dThe * dT) \
                    - b * rho[cur, m, n] * dT
                t += stride
            g.sync()

            # ---- 2. record the PREVIOUS step's masses, as the CPU does ----
            if tid == 0:
                ma_k = int(state[3])
                if ma_k < relative_k and k % ma_factor == 0:
                    dl = state[1]
                    al = state[2]
                    tm = dl + al
                    series[0, ma_k] = dl
                    series[1, ma_k] = al
                    series[4, ma_k] = tm
                    series[3, ma_k] = al / tm
                    series[2, ma_k] = al / Dcoef
                    state[3] = ma_k + 1

            # ---- 3. reductions over the CURRENT slot ----
            acc0 = 0.0      # sum phi*(m+1)      -> diffusive mass
            acc1 = 0.0      # sum rho on tubes   -> advective mass
            acc2 = 0.0      # sum j_l_r at m=0   -> centre update
            i = tid
            while i < total:
                m = i // N
                n = i - m * N
                acc0 += phi[cur, m, n] * (m + 1)
                if m == 0:
                    acc2 += -1.0 * ((phi[cur, 0, n] - central) / dRad)
                i += stride
            t = tid
            while t < adv_total:
                m = t // nt
                n = NL[t - m * nt]
                acc1 += rho[cur, m, n] * dRad
                t += stride

            sh[0, lane] = acc0
            sh[1, lane] = acc1
            sh[2, lane] = acc2
            cuda.syncthreads()
            s = THREADS_PER_BLOCK // 2
            while s > 0:
                if lane < s:
                    sh[0, lane] += sh[0, lane + s]
                    sh[1, lane] += sh[1, lane + s]
                    sh[2, lane] += sh[2, lane + s]
                cuda.syncthreads()
                s //= 2
            if lane == 0:
                partial[0, bid] = sh[0, 0]
                partial[1, bid] = sh[1, 0]
                partial[2, bid] = sh[2, 0]
            g.sync()

            # ---- 4. combine and advance the carried scalars ----
            if tid == 0:
                s0 = 0.0
                s1 = 0.0
                s2 = 0.0
                for bb in range(nblocks):
                    s0 += partial[0, bb]
                    s1 += partial[1, bb]
                    s2 += partial[2, bb]

                # calc_mass_diff
                dl_mass = (central * 3.141592653589793 * dRad * dRad) \
                    + s0 * (dRad * dRad) * dThe
                # calc_mass_adv
                al_mass = s1
                # u_center: diffusive part, then the advective contributions
                tot = s2 * ((dThe * dT) / (3.141592653589793 * dRad))
                new_central = central - tot
                for ii in range(nt):
                    ang = NL[ii]
                    j_l = rho[cur, 0, ang] * v
                    new_central += (abs(j_l) * dT) \
                        / (3.141592653589793 * dRad * dRad)

                state[0] = new_central
                state[1] = dl_mass
                state[2] = al_mass
            g.sync()

        # ---- leave the current field in slot 0 ----
        if ((k_end - k_start) & 1) == 1:
            i = tid
            while i < total:
                m = i // N
                n = i - m * N
                phi[0, m, n] = phi[1, m, n]
                rho[0, m, n] = rho[1, m, n]
                i += stride
            g.sync()
