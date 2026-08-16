"""Golden-master test for the PDE stencil.

``fixtures/golden_stencil.npz`` was generated from the solver as it stood at
commit 2bb8e34, immediately BEFORE the hot-path performance work.  Every case
below must still reproduce that state bit-for-bit.

This is the regression net for optimisation: any change that is meant to be
purely a speed-up has to leave these arrays untouched.  A deliberate change to
the physics is expected to fail here -- when that happens, regenerate the
fixture with ``python tests/test_golden_stencil.py --regenerate`` in the SAME
commit that changes the behaviour, and say why in the commit message.

Run directly (``python tests/test_golden_stencil.py``) or under pytest.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.join(os.path.dirname(_HERE), "project_src_package_2025")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from computational_tools import numerical_tools as num          # noqa: E402
from computational_tools import supplements as sup              # noqa: E402
from computational_tools import struct_init                     # noqa: E402

FIXTURE = os.path.join(_HERE, "fixtures", "golden_stencil.npz")

# Chosen to exercise distinct code paths, not just the default configuration:
#   * d_tube = 0 collapses the extraction loop in u_tube_rect to one iteration;
#     the non-zero values exercise the multi-step extraction region.
#   * non-square grids separate the ring count from the ray count.
CASES = [
    (48, 48, 8, 0.0),
    (48, 48, 8, 0.01),
    (48, 48, 4, 0.05),
    (64, 48, 8, 0.0),     # non-square
    (64, 48, 4, 0.02),    # non-square
    (32, 32, 16, 0.0),
    (32, 32, 16, 0.03),
    (96, 96, 24, 0.0),    # production configuration
]

STEPS = 30
W = 100.0
V = -float(10 ** 4)


def _run_case(rg, ry, n_tubes, d_tube):
    """Advance the stencil STEPS times; return (D_LAYER, A_LAYER, central)."""
    N_LIST = np.linspace(0, ry - (ry // n_tubes), n_tubes, dtype=int)
    dRad = num.compute_dRad(rg, 1.0)
    dThe = num.compute_dThe(ry)
    dT = num.compute_dT(rg, ry, 1.0, 1.0)
    central = num.compute_init_cond_cent(rg, 1.0)
    d_map = struct_init.build_d_tube_map_dense(rg, ry, N_LIST, d_tube, 1.0)

    D_LAYER, A_LAYER = sup.initialize_layers(rg, ry)
    for _ in range(STEPS):
        num.comp_DL_AL_kp1_2step(ry, rg, d_map, D_LAYER, central, A_LAYER,
                                 N_LIST, dRad, dThe, dT, W, W, V, d_tube)
        central = num.u_center(D_LAYER, 0, dRad, dThe, dT, central,
                               A_LAYER, N_LIST, V)
        D_LAYER[0] = D_LAYER[1]
        A_LAYER[0] = A_LAYER[1]
    return D_LAYER, A_LAYER, central


def test_stencil_matches_golden():
    """Every case reproduces the recorded state exactly."""
    assert os.path.exists(FIXTURE), (
        f"missing golden fixture {FIXTURE}; regenerate with --regenerate")
    ref = np.load(FIXTURE)
    failures = []
    for idx, case in enumerate(CASES):
        D, A, c = _run_case(*case)
        label = f"{case[0]}x{case[1]} N={case[2]} d_tube={case[3]}"
        if not np.array_equal(ref[f"D{idx}"], D):
            failures.append(f"{label}: D_LAYER differs "
                            f"(max |delta| = {np.max(np.abs(ref[f'D{idx}'] - D)):.3e})")
        if not np.array_equal(ref[f"A{idx}"], A):
            failures.append(f"{label}: A_LAYER differs "
                            f"(max |delta| = {np.max(np.abs(ref[f'A{idx}'] - A)):.3e})")
        if ref[f"c{idx}"][0] != c:
            failures.append(f"{label}: central patch differs "
                            f"({ref[f'c{idx}'][0]!r} vs {c!r})")
    assert not failures, "stencil output changed:\n  " + "\n  ".join(failures)


def _regenerate():
    out = {}
    for idx, case in enumerate(CASES):
        D, A, c = _run_case(*case)
        out[f"D{idx}"] = D
        out[f"A{idx}"] = A
        out[f"c{idx}"] = np.array([c])
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    np.savez(FIXTURE, **out)
    print(f"regenerated {FIXTURE} ({len(CASES)} cases)")


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        _regenerate()
    else:
        test_stencil_matches_golden()
        print(f"OK: all {len(CASES)} cases bit-identical to the golden fixture")
