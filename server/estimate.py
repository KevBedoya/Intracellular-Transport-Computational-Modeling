"""Wall-time estimates for queued and running jobs, per device.

Single source of truth for the cost model. It used to live in the browser, CPU
only, which made a GPU job's estimate wrong by an order of magnitude -- a 176x176
GPU run was quoted 5.6 days and finished in 22.5 hours.

The physics behind the shape: the scheme's stability limit is
``dT = 0.1 * min(dRad^2, dThe^2 * dRad^2) / (2D)``, so ``K ~ G^4`` timesteps over
``G^2`` patches and total work grows as ``G^6``. Rather than assume that, both
models below are ``us/step = fixed + slope * patches`` fitted to measured runs,
which captures the per-step overhead that pure ``G^6`` scaling misses.

All constants are measurements on this workstation (RTX 3060, 12 GB, 28 SMs),
recorded in docs/GPU_PLAN.md. Re-measure them if the hardware changes.
"""

import math

# --- GPU: fitted to the measured 128^2 (34.1 us/step) and 160^2 (43.5) points.
# Reproduces 96^2 to within 4%. Affine in patch count, because the per-step cost
# is a fixed reduction overhead plus a grid-stride pass over the patches.
GPU_FIXED_US = 17.39
GPU_NS_PER_PATCH = 1.020
# Below ~96^2 there are too few patches to fill the card and the affine fit
# undershoots; the measured floor is 27.2 us/step at 48^2 and 28.0 at 96^2.
GPU_FLOOR_US = 27.0

# --- CPU: a power law, not an affine fit. Cost per patch is not constant -- it
# rises as the working set falls out of cache (27.8 ns/patch at 96^2, 32.2 at
# 160^2). Fitting affine in patches gives a *negative* intercept, which then
# underestimates small grids by 5x. Fitted to the two uncontended anchors,
# 96^2 = 256.5 us/step and 160^2 = 825.0.
_CPU_POINTS = ((96, 256.5), (160, 825.0))
_p1, _c1 = float(_CPU_POINTS[0][0] ** 2), _CPU_POINTS[0][1]
_p2, _c2 = float(_CPU_POINTS[1][0] ** 2), _CPU_POINTS[1][1]
CPU_EXPONENT = math.log(_c2 / _c1) / math.log(_p2 / _p1)
CPU_COEFF = _c1 / (_p1 ** CPU_EXPONENT)

# numba compiles the kernels on every fresh process. Irrelevant against a
# multi-hour solve, but it is most of the runtime at 48^2, where ignoring it
# made the estimate read 45 s for a job that takes about 4 minutes.
STARTUP_SECONDS = 25.0

# Observed overshoot of the model against completed jobs. Kept explicit rather
# than folded into the constants so it stays visible and adjustable.
FUDGE = 1.10

# Computations that do not run the time-stepping loop, or whose cost is not a
# function of grid size in this way. Estimating them would be guessing.
_NO_ESTIMATE = ("Analytic Benchmark", "Modal")


def _grid(params):
    rg = params.get("rg_param")
    ry = params.get("ry_param")
    try:
        rg, ry = float(rg), float(ry)
    except (TypeError, ValueError):
        return None
    if rg < 2 or ry < 2:
        return None
    return rg, ry


def steps(rg, ry, T=1.0, domain_radius=1.0, D=1.0):
    """Timestep count K, matching numerical_tools.compute_K.

    Reimplemented here rather than imported so the API process does not have to
    load numba to answer an estimate request. Kept in step with the solver by
    tests/test_estimate.py, which asserts the two agree exactly.
    """
    dRad = float(domain_radius) / rg
    dThe = 2.0 * math.pi / ry
    dT = 0.1 * min(dRad ** 2, (dThe ** 2) * (dRad ** 2)) / (2.0 * D)
    return int(math.floor(float(T) / dT))


def us_per_step(rg, ry, device="cpu"):
    patches = float(rg) * float(ry)
    if str(device).lower() in ("gpu", "cuda"):
        return max(GPU_FLOOR_US,
                   GPU_FIXED_US + GPU_NS_PER_PATCH / 1000.0 * patches)
    return CPU_COEFF * (patches ** CPU_EXPONENT)


def seconds(computation, params, device=None):
    """Estimated solo wall-clock seconds, or None if not estimable.

    ``device`` overrides ``params['device']``; ``auto`` is treated as GPU, since
    that is what it will pick whenever the GPU is usable.
    """
    if computation and any(k in computation for k in _NO_ESTIMATE):
        return None
    g = _grid(params or {})
    if g is None:
        return None
    rg, ry = g

    dev = str(device or (params or {}).get("device") or "cpu").lower()
    if dev == "auto":
        dev = "gpu"

    T = params.get("T_param")
    try:
        T = float(T)
    except (TypeError, ValueError):
        T = 1.0
    if T <= 0:
        T = 1.0

    K = steps(rg, ry, T,
              float(params.get("domain_radius") or 1.0),
              float(params.get("D") or 1.0))

    # A velocity sweep solves once per velocity, sequentially.
    v_list = params.get("v_LIST")
    runs = len(v_list) if isinstance(v_list, (list, tuple)) and v_list else 1

    solve = K * us_per_step(rg, ry, dev) * 1e-6 * runs * FUDGE
    return solve + STARTUP_SECONDS


def describe(computation, params, device=None):
    """Estimate plus the inputs behind it, for display and for the tooltip."""
    s = seconds(computation, params, device)
    if s is None:
        return {"seconds": None, "device": None, "basis": None}
    g = _grid(params or {})
    dev = str(device or (params or {}).get("device") or "cpu").lower()
    resolved = "gpu" if dev in ("gpu", "auto", "cuda") else "cpu"
    return {
        "seconds": s,
        "device": resolved,
        "requested_device": dev,
        "basis": {
            "steps": steps(g[0], g[1],
                           float(params.get("T_param") or 1.0),
                           float(params.get("domain_radius") or 1.0),
                           float(params.get("D") or 1.0)),
            "us_per_step": us_per_step(g[0], g[1], resolved),
            "fudge": FUDGE,
        },
    }


def share_factor(device, gpu_running):
    """How much a solo estimate stretches when GPU jobs share the card.

    Cooperative kernels from separate processes time-slice one GPU, so N
    concurrent GPU jobs each run at roughly 1/N speed. Ignoring this is not a
    rounding error: three concurrent jobs turned a 6.6 hr estimate into a 22.5 hr
    actual. CPU jobs have their own core each and are left alone.
    """
    if str(device).lower() not in ("gpu", "cuda", "auto"):
        return 1.0
    return float(max(1, int(gpu_running or 1)))
