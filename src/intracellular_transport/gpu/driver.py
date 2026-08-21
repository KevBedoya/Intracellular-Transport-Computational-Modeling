"""GPU driver for the mass-analysis solve.

Presents the same shape as the CPU driver in analysis_tools: the caller supplies
the layers and the five timeseries and gets the carried scalars back. Setup, the
characteristic-time fit and all output stay on the CPU side, shared.

Supported configuration is deliberately narrow, and the driver refuses rather
than approximates anything outside it -- an unsupported case silently producing
plausible numbers is far worse than an exception.
"""

import numpy as np

from . import persistent_kernel as pk

# Chunk size: how many timesteps per cooperative launch. The launch cost is
# ~184 us of host-side marshalling, so it wants amortising over many steps;
# beyond a few thousand there is nothing left to gain, and smaller chunks give
# the checkpoint hook somewhere to run.
DEFAULT_CHUNK = 5000


class GpuUnsupported(RuntimeError):
    """The requested configuration is not covered by the GPU port."""


# Cached result of the cooperative-launch probe below.
_MAX_BLOCKS = None


def _max_cooperative_blocks(probe_args, candidates=(112, 84, 56, 42, 28, 14, 7)):
    """Largest block count this kernel can launch cooperatively.

    Cannot be assumed: a cooperative launch requires every block to be resident
    simultaneously, so the ceiling depends on the kernel's own register and
    shared-memory use, not just on the device. This kernel's shared-memory
    reduction buffer puts it below the 112 blocks a bare kernel manages on this
    card, and guessing produced CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE.

    Probed by launching with an empty step range, which executes no loop body.
    """
    global _MAX_BLOCKS
    if _MAX_BLOCKS is not None:
        return _MAX_BLOCKS

    from numba.cuda.cudadrv.driver import CudaAPIError

    for blocks in candidates:
        try:
            pk.solve_chunk[blocks, pk.THREADS_PER_BLOCK](*probe_args)
            from numba import cuda
            cuda.synchronize()
            _MAX_BLOCKS = blocks
            return blocks
        except CudaAPIError:
            continue
    raise GpuUnsupported(
        "no cooperative launch configuration worked; the device may not "
        "support cooperative groups")


def check_supported(rg_param, ry_param, d_tube, center_init_cond):
    """Raise GpuUnsupported unless this run is within the port's scope."""
    reason = pk.unavailable_reason()
    if reason:
        raise GpuUnsupported(reason)
    if d_tube != 0:
        raise GpuUnsupported(
            f"d_tube={d_tube} is not supported: the kernel assumes j_max == 0, "
            "which only holds for d_tube == 0. Wider extraction regions need "
            "the (1 + 2*j_max) factor and the multi-ray sum ported too.")
    if not center_init_cond:
        raise GpuUnsupported(
            "off-centre initial conditions are not supported; seed on the CPU "
            "path or extend the driver")
    if rg_param < 2 or ry_param < 2:
        raise GpuUnsupported("grid must be at least 2x2")


def solve(rg_param, ry_param, switch_param_a, switch_param_b, v_param, T_param,
          N_LIST, D_LAYER, A_LAYER, series, MA_collection_factor, relative_k,
          d_tube=0.0, domain_radius=1.0, D=1.0, center_init_cond=True,
          chunk_steps=DEFAULT_CHUNK, on_chunk=None, resume_state=None):
    """Run the mass-analysis solve on the GPU.

    ``series`` is the five timeseries in the same order the CPU driver takes
    them: (DL, AL, ALoI, ALoT, TM). ``resume_state`` optionally supplies
    ``(k, central_patch, dl_mass, al_mass, MA_k_step)`` to continue from.

    Returns ``(central_patch, dl_mass, al_mass, MA_k_step)``, and leaves the
    final field in D_LAYER / A_LAYER so the caller sees the same result shape as
    the CPU path.
    """
    from numba import cuda

    from computational_tools import numerical_tools as num
    from computational_tools import struct_init

    check_supported(rg_param, ry_param, d_tube, center_init_cond)

    dRad = num.compute_dRad(rg_param, domain_radius)
    dThe = num.compute_dThe(ry_param)
    dT = num.compute_dT(rg_param, ry_param, domain_radius, D)
    K = num.compute_K(rg_param, ry_param, T_param, domain_radius, D)

    NL = np.ascontiguousarray(np.asarray(N_LIST, dtype=np.int32))
    dmap = struct_init.build_d_tube_map_dense(
        rg_param, ry_param, np.asarray(N_LIST, dtype=np.int64),
        d_tube, domain_radius)
    tube = np.zeros(ry_param, dtype=np.int8)
    tube[np.asarray(N_LIST, dtype=np.int64)] = 1

    if resume_state is not None:
        k, central_patch, dl_mass, al_mass, MA_k_step = resume_state
    else:
        k = 0
        central_patch = num.compute_init_cond_cent(rg_param, domain_radius)
        dl_mass, al_mass, MA_k_step = 1.0, 0.0, 0

    # The CPU driver negates v once before the loop; do the same, once.
    v = -float(v_param)

    # Device state. phi/rho are (2, M, N): slot 0 current, slot 1 scratch.
    phi = np.zeros((2, rg_param, ry_param), dtype=np.float64)
    rho = np.zeros((2, rg_param, ry_param), dtype=np.float64)
    phi[0] = D_LAYER[0]
    rho[0] = A_LAYER[0]

    d_phi = cuda.to_device(phi)
    d_rho = cuda.to_device(rho)
    d_dmap = cuda.to_device(dmap)
    d_tube = cuda.to_device(tube)
    d_NL = cuda.to_device(NL)
    d_partial = cuda.device_array((3, pk.BLOCKS), dtype=np.float64)
    d_series = cuda.to_device(np.stack([np.asarray(s) for s in series]))
    d_state = cuda.to_device(
        np.array([central_patch, dl_mass, al_mass, float(MA_k_step)],
                 dtype=np.float64))

    # An empty step range: compiles the kernel and probes the launch geometry
    # without advancing the solution.
    probe = (d_phi, d_rho, d_dmap, d_tube, d_NL, d_partial, d_series, d_state,
             rg_param, ry_param, NL.size,
             dRad, dThe, dT, float(switch_param_a), float(switch_param_b), v,
             float(D), 0, 0, int(MA_collection_factor), int(relative_k))
    blocks = _max_cooperative_blocks(probe)

    step = K if chunk_steps <= 0 else min(int(chunk_steps), K)
    while k < K:
        k_end = min(k + step, K)
        pk.solve_chunk[blocks, pk.THREADS_PER_BLOCK](
            d_phi, d_rho, d_dmap, d_tube, d_NL, d_partial, d_series, d_state,
            rg_param, ry_param, NL.size,
            dRad, dThe, dT, float(switch_param_a), float(switch_param_b), v,
            float(D), k, k_end, int(MA_collection_factor), int(relative_k))
        cuda.synchronize()
        k = k_end
        st = d_state.copy_to_host()
        state = (float(st[0]), float(st[1]), float(st[2]), int(st[3]))
        if on_chunk is not None:
            # Copy back so a checkpoint written from the hook is consistent with
            # the reported timestep, not one chunk behind it.
            host_series = d_series.copy_to_host()
            for j, s in enumerate(series):
                s[:] = host_series[j]
            back_phi = d_phi.copy_to_host()
            back_rho = d_rho.copy_to_host()
            D_LAYER[0] = back_phi[0]
            D_LAYER[1] = back_phi[1]
            A_LAYER[0] = back_rho[0]
            A_LAYER[1] = back_rho[1]
            on_chunk(k, state)

    host_series = d_series.copy_to_host()
    for j, s in enumerate(series):
        s[:] = host_series[j]
    back_phi = d_phi.copy_to_host()
    back_rho = d_rho.copy_to_host()
    D_LAYER[0] = back_phi[0]
    D_LAYER[1] = back_phi[1]
    A_LAYER[0] = back_rho[0]
    A_LAYER[1] = back_rho[1]

    st = d_state.copy_to_host()
    return float(st[0]), float(st[1]), float(st[2]), int(st[3])
