"""Desktop-GUI entry to the computations: a thin alias of the shared router.

This module used to keep its own copy of the computation table and the result
shaping. The copy drifted -- the characteristic-time computations were listed
in the GUI's dropdown (built from PARAMETER_SCHEMAS) but missing here, so the
job queue failed them as "Unknown computation". The GUI's single-run path, the
HTTP API and the job worker all go through
multiprocessing_tools.computation_router already; the queue now does too, so
every caller runs the same functions and gets the same result shape.
"""
from multiprocessing_tools.computation_router import (   # noqa: F401
    COMPUTATION_FUNCTIONS,
    parse_input,
    run_selected_computation,
)
