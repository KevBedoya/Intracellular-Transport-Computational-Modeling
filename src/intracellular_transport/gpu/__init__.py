"""GPU acceleration for the time-stepping loop.

Deliberately thin. This package contains one port of one loop -- the per-timestep
update that accounts for over 98% of runtime -- behind a driver whose interface
matches the CPU one. Setup, the characteristic-time fit, and all CSV and figure
output stay shared with the CPU path.

It is explicitly NOT a GPU version of the eleven njit loops in analysis_tools.py.
That road is how this repository previously acquired five divergent copies of the
solver and four different implementations of u_density; the CPU code remains the
reference and the authority on correctness.

See docs/GPU_PLAN.md for the measurements this design rests on, and in particular
for why results cannot be bit-identical to the CPU: the mass and centre
reductions are sequential on the CPU and parallel here, and floating-point
addition is not associative.
"""

from . import persistent_kernel  # noqa: F401
