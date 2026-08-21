"""Measure how far the GPU solve diverges from the CPU one.

docs/GPU_PLAN.md explains why bit-identity is unreachable: the mass and centre
reductions are sequential on the CPU and parallel here, and floating-point
addition is not associative. So this does not assert equality -- it measures the
disagreement and fails only if it exceeds a stated tolerance.

TOLERANCE below is the acceptance criterion. It is a scientific judgement, not
an implementation detail: it says how much drift in t* and m* is acceptable in
exchange for a ~20x speedup. Change it deliberately.

Skips cleanly when no CUDA device is present.
"""
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src")
_PKG = os.path.join(_SRC, "intracellular_transport")
for _p in (_PKG, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from computational_tools import analysis_tools as ant       # noqa: E402
from computational_tools import numerical_tools as num      # noqa: E402
from computational_tools import supplements as sup          # noqa: E402
from gpu import driver as gpu_driver                        # noqa: E402
from gpu import persistent_kernel as pk                     # noqa: E402

# Relative agreement required between CPU and GPU on the recorded mass series
# and the final field. Set from measurement, not aspiration -- see the report
# printed by __main__.
TOLERANCE = 1e-9

MA_FACTOR = 5

CASES = [
    dict(rg=32, ry=32, T=0.02, tubes=8),
    dict(rg=48, ry=48, T=0.01, tubes=16),
    dict(rg=64, ry=64, T=0.004, tubes=16),
]
W = 100.0
V = 1e4


def _cpu(case):
    rg, ry, T = case["rg"], case["ry"], case["T"]
    NL = np.linspace(0, ry - (ry // case["tubes"]), case["tubes"], dtype=int)
    K = num.compute_K(rg, ry, T)
    rel = max(int(math.floor(K / MA_FACTOR)), 1)
    D, A = sup.initialize_layers(rg, ry)
    series = [np.zeros(rel, dtype=np.float64) for _ in range(5)]
    state = ant.comp_mass_analysis_respect_to_time(
        rg, ry, W, W, V, T, NL, D, A,
        series[0], series[1], series[2], series[3], series[4],
        MA_FACTOR, rel, 0.0, 1.0, 1.0, 10 ** 18, True, 0, 0)
    return D, A, series, state, K, rel, NL


def _gpu(case, NL, rel):
    rg, ry, T = case["rg"], case["ry"], case["T"]
    D, A = sup.initialize_layers(rg, ry)
    series = [np.zeros(rel, dtype=np.float64) for _ in range(5)]
    state = gpu_driver.solve(
        rg, ry, W, W, V, T, NL, D, A, series, MA_FACTOR, rel,
        d_tube=0.0, domain_radius=1.0, D=1.0, center_init_cond=True)
    return D, A, series, state


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.maximum(np.abs(a), 1e-300)
    return float(np.max(np.abs(a - b) / denom))


# Slot order of the five timeseries, shared by the CPU driver's parameter list
# and the GPU kernel's series[] indices. Compared by name below because an
# earlier version of this test checked only slots 0 and 4 -- and slots 2 and 3
# were in fact transposed between the two devices, which it could not see.
SERIES_NAMES = ("dl_mass", "al_mass", "al_over_D", "al_over_total", "total_mass")


def _measure(case):
    D_c, A_c, s_c, st_c, K, rel, NL = _cpu(case)
    D_g, A_g, s_g, st_g = _gpu(case, NL, rel)
    # Only compare samples that were actually written.
    n = min(int(st_c[3]), int(st_g[3]))
    out = {
        "K": K,
        "samples": n,
        "phi": _rel_err(D_c[0], D_g[0]),
        "rho": _rel_err(A_c[0], A_g[0]),
        "central": _rel_err([st_c[0]], [st_g[0]]),
        "cpu_state": st_c,
        "gpu_state": st_g,
    }
    for j, name in enumerate(SERIES_NAMES):
        out[name] = _rel_err(s_c[j][:n], s_g[j][:n]) if n else 0.0
    return out


def test_gpu_matches_cpu_within_tolerance():
    if not pk.is_available():
        print("no CUDA device; skipping")
        return
    failures = []
    for case in CASES:
        r = _measure(case)
        label = f"{case['rg']}x{case['ry']} T={case['T']} tubes={case['tubes']}"
        for field in ("phi", "rho", "central") + SERIES_NAMES:
            if r[field] > TOLERANCE:
                failures.append(f"{label}: {field} relative error "
                                f"{r[field]:.3e} exceeds {TOLERANCE:.0e}")
    assert not failures, "GPU disagrees with CPU beyond tolerance:\n  " + \
        "\n  ".join(failures)


def test_gpu_is_reproducible():
    """Two GPU runs of the same case must agree exactly.

    Bit-identity to the CPU is unreachable, but the GPU must at least be
    deterministic -- otherwise a result could not be reproduced even on the same
    machine, and the reduction's block count would be silently load-dependent.
    """
    if not pk.is_available():
        print("no CUDA device; skipping")
        return
    case = CASES[0]
    _, _, _, _, K, rel, NL = _cpu(case)
    D1, A1, s1, st1 = _gpu(case, NL, rel)
    D2, A2, s2, st2 = _gpu(case, NL, rel)
    assert np.array_equal(D1[0], D2[0]), "GPU phi is not reproducible"
    assert np.array_equal(A1[0], A2[0]), "GPU rho is not reproducible"
    for j, (a, b) in enumerate(zip(s1, s2)):
        assert np.array_equal(a, b), f"GPU timeseries {j} is not reproducible"
    assert st1 == st2, "GPU carried state is not reproducible"


def test_unsupported_configurations_are_refused():
    """Out-of-scope runs must raise, not silently approximate."""
    if not pk.is_available():
        print("no CUDA device; skipping")
        return
    for kwargs, expect in (
        (dict(rg_param=32, ry_param=32, d_tube=0.05, center_init_cond=True),
         "d_tube"),
        (dict(rg_param=32, ry_param=32, d_tube=0.0, center_init_cond=False),
         "off-centre"),
    ):
        try:
            gpu_driver.check_supported(**kwargs)
        except gpu_driver.GpuUnsupported as e:
            assert expect in str(e), f"unexpected refusal message: {e}"
        else:
            raise AssertionError(f"{kwargs} was accepted but should not be")


if __name__ == "__main__":
    if not pk.is_available():
        print("no CUDA device available:", pk.unavailable_reason())
        sys.exit(0)

    fields = ("phi", "rho", "central") + SERIES_NAMES
    print("Measured CPU-vs-GPU disagreement (max relative error)\n")
    header = f"{'case':>18} {'K':>9} {'samp':>6}" + \
        "".join(f" {f:>13}" for f in fields)
    print(header)
    worst = 0.0
    for case in CASES:
        r = _measure(case)
        label = f"{case['rg']}x{case['ry']} T={case['T']}"
        row = f"{label:>18} {r['K']:>9,} {r['samples']:>6,}" + \
            "".join(f" {r[f]:>13.3e}" for f in fields)
        print(row)
        worst = max(worst, *(r[f] for f in fields))
    print(f"\nworst relative error across all cases: {worst:.3e}")
    print(f"current TOLERANCE in this file:         {TOLERANCE:.0e}")
    print("verdict:", "within tolerance" if worst <= TOLERANCE
          else "EXCEEDS TOLERANCE -- do not accept GPU results yet")

    test_gpu_is_reproducible()
    print("\nOK: repeated GPU runs are bit-identical to each other")
    test_unsupported_configurations_are_refused()
    print("OK: unsupported configurations are refused")
