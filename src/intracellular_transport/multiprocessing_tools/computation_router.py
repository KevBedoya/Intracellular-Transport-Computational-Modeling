import ast
import os
from launch_functions import launch


COMPUTATION_FUNCTIONS = {
    "Compute MFPT until mass %": launch.solve_mfpt_mass_,
    "MFPT point collection (Mass % dep.)": launch.collect_MFPT_snapshots_mass_dep,
    "Phi angular dependence (collection: Mass % dep.)": launch.collect_phi_ang_dep_mass_dep,
    "Phi/Rho radial dependence (collection: Mass % dep.)": launch.collect_density_rad_depend_mass_dep,
    "Static Heatplot Analysis (collection: Mass % dep.)": launch.heatmap_production_mass_dep,

    "Compute MFPT until time T": launch.solve_mfpt_time_,
    "MFPT point collection (time T dep.)": launch.collect_MFPT_snapshots_time_dep,
    "Phi angular dependence (collection: time T dep.)": launch.collect_phi_ang_dep_time_dep,
    "Phi/Rho radial dependence (collection: time T dep.)": launch.collect_density_rad_depend_time_dep,
    "Static Heatplot Analysis (collection: time T dep.)": launch.heatmap_production_time_dep,
    "Full Analysis (time t dep.)": launch.launch_super_comp_I,

    "Time Until Mass Depletion": launch.output_time_until_mass_depletion,
    "Mass Analysis": launch.collect_mass_analysis,

    "Characteristic Time (mass vs v)": launch.collect_char_time_mass,
    "Characteristic Time (a,b grid)": launch.collect_ab_grid_char_time,
    "Jrr Mass Sum Over Time": launch.collect_Jrr_mass_sum_over_time,
    "BC Parameter Dependence (w sweep)": launch.collect_BC_param_dependence,
    "BC Parameter Dependence (grid-size sweep)": launch.collect_BC_param_dependence_grid_size,
    "Angular Trajectory Matrix": launch.compute_ang_traj_mat,
}
# "Full Analysis": launch.launch_super_comp_I


def parse_input(value):
    """
    Attempts to convert a string input into its appropriate Python type.
    Supports float, int, bool, list, None, etc.

    Non-string values are returned unchanged.  The desktop GUI supplies every
    parameter as text from a form field, but a JSON caller (the HTTP API and
    the job worker) supplies real ints, floats, lists and bools -- those must
    pass through untouched rather than be re-parsed.
    """
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value.strip()


def _is_blank(value):
    """True for an omitted parameter: an empty/whitespace string, or None.

    Blank entries are dropped so the callee's own default applies.  Only
    strings are treated as blankable; 0 and False are real values.
    """
    if value is None:
        return True
    return isinstance(value, str) and value.strip() == ""


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------
# Submitting a job used to check only the computation *name*. A body missing a
# required parameter, carrying a wrong-typed value, or naming a parameter that
# does not exist was accepted, queued, and only failed once a worker had claimed
# a slot and started solving -- which on a long queue means a typo costs hours.
# These checks move that failure to submission time.
#
# Optional parameters get their expected type from their schema default. Required
# parameters have no default, so their types are listed explicitly rather than
# parsed out of the hint strings, which would silently stop working if a hint
# were reworded.
_REQUIRED_PARAM_TYPES = {
    "rg_param": int,
    "ry_param": int,
    "N_amount": int,
    "v_param": float,
    "w_param": float,
    "a_param": float,
    "b_param": float,
    "T_param": float,
    "checkpoint": float,
    "N_LIST": list,
    "grid_list": list,
    "v_LIST": list,
    "w_LIST": list,
    "a_list": list,
    "b_list": list,
    "checkpoint_collect_container": list,
}

# Optional parameters whose default does not imply the accepted type:
#   Timestamp_List defaults to None but takes a list of times
#   d_tube is written 0 in some schemas and 0.0 in others; it is a width
_OPTIONAL_TYPE_OVERRIDES = {
    "Timestamp_List": list,
    "d_tube": float,
}

# Characteristic-time computations. t* is the onset of steady exponential decay,
# which the criterion only accepts if the decay then holds for at least
# CHAR_TIME_MIN_HOLD before T -- so T must exceed that, and in practice the
# onset itself (typically 0.25-0.45).
_CHAR_TIME_MIN_T = launch.CHAR_TIME_MIN_HOLD
_CHAR_TIME_COMPUTATIONS = {"Characteristic Time (mass vs v)",
                            "Characteristic Time (a,b grid)"}


def _schema_for(computation_name):
    from gui_components.params_config import PARAMETER_SCHEMAS

    return PARAMETER_SCHEMAS.get(computation_name, {})


def _expected_type(param, default=None, has_default=False):
    if param in _OPTIONAL_TYPE_OVERRIDES:
        return _OPTIONAL_TYPE_OVERRIDES[param]
    if param in _REQUIRED_PARAM_TYPES:
        return _REQUIRED_PARAM_TYPES[param]
    if has_default and default is not None:
        t = type(default)
        return t if t in (int, float, bool, str, list) else None
    return None


def _coerce(value, expected):
    """Return (ok, coerced). Accepts the JSON forms a caller realistically sends."""
    if expected is bool:
        return (isinstance(value, bool), value)
    if expected is int:
        # bool is a subclass of int; a flag is not a grid size.
        if isinstance(value, bool):
            return (False, value)
        if isinstance(value, int):
            return (True, value)
        if isinstance(value, float) and float(value).is_integer():
            return (True, int(value))
        return (False, value)
    if expected is float:
        if isinstance(value, bool):
            return (False, value)
        if isinstance(value, (int, float)):
            return (True, float(value))
        return (False, value)
    if expected is list:
        if isinstance(value, (list, tuple)):
            return (all(isinstance(x, (int, float)) and not isinstance(x, bool)
                        for x in value), list(value))
        return (False, value)
    if expected is str:
        return (isinstance(value, str), value)
    return (True, value)


def d_tube_limit(rg_param, ry_param, N_LIST):
    """Largest d_tube the solver will use as given, for this grid and N_LIST.

    d_tube is the width of the diffusive-layer extraction region around each
    microtubule. The solver only supports regions that do not overlap, so it
    bounds d_tube by the tightest gap between neighbouring microtubules
    (``j_max_bef_overlap``) measured on the first ring -- and a value outside
    ``[0, limit]`` is not refused there but *silently replaced by the limit*
    (``struct_init.build_d_tube_mapping_no_overlap``). A job submitted with
    such a value would run, and report, a width nobody asked for.

    Computed with the same two functions and the same arguments as that
    solver code -- including its radius of 1, which it uses regardless of
    domain_radius -- so the bound here cannot drift from the one applied.
    """
    from computational_tools import supplements as sup

    positions = [int(x) for x in N_LIST]
    j_sup = sup.j_max_bef_overlap_no_JIT(int(ry_param), positions)
    return sup.solve_d_rect_no_JIT(1, int(rg_param), int(ry_param), j_sup, 0)


def check_d_tube(rg_param, ry_param, N_LIST, d_tube):
    """Whether ``d_tube`` is valid here, with the bound and a message to show.

    Returns ``{"valid": bool, "max_d_tube": float, "message": str}``. Assumes
    the grid and N_LIST have themselves been validated.
    """
    limit = d_tube_limit(rg_param, ry_param, N_LIST)
    # Exactly the solver's test, so "valid" means "used as given".
    valid = not (d_tube < 0 or d_tube > limit)
    span = (f"[0, {limit:.10g}] for a {int(rg_param)}x{int(ry_param)} grid "
            f"with {len(N_LIST)} microtubule(s)")
    if valid:
        message = f"valid: d_tube={d_tube:g} lies in {span}"
    else:
        message = (f"must lie in {span}; beyond that the extraction regions "
                   f"of neighbouring microtubules would overlap, and the "
                   f"solver would silently use {limit:.10g} instead")
    return {"valid": valid, "max_d_tube": float(limit), "message": message}


def validate_params(computation_name, params):
    """Check ``params`` against the computation's schema.

    Returns a mapping of field name to error message; empty means valid. The
    key ``"_"`` carries errors that are not about one field.
    """
    errors = {}

    if computation_name not in COMPUTATION_FUNCTIONS:
        return {"computation": f"unknown computation {computation_name!r}"}
    if not isinstance(params, dict):
        return {"_": f"params must be an object, got {type(params).__name__}"}

    schema = _schema_for(computation_name)
    required = [k for k, _ in schema.get("required", [])]
    optional = {k: v for k, v in schema.get("default", [])}
    known = set(required) | set(optional)

    for key in params:
        if key not in known:
            errors[key] = ("not a parameter of this computation; expected one "
                           "of: " + ", ".join(sorted(known)))

    for key in required:
        if key not in params or _is_blank(params.get(key)):
            errors[key] = "required"

    clean = {}
    for key, value in params.items():
        if key in errors or key not in known or _is_blank(value):
            continue
        has_default = key in optional
        expected = _expected_type(key, optional.get(key), has_default)
        if expected is None:
            clean[key] = value
            continue
        ok, coerced = _coerce(value, expected)
        if not ok:
            errors[key] = (f"expected {expected.__name__}, got "
                           f"{type(value).__name__}"
                           + (" of numbers" if expected is list else ""))
        else:
            clean[key] = coerced

    # --- semantic checks, only for cases that cannot be valid ---
    rg, ry = clean.get("rg_param"), clean.get("ry_param")
    for key in ("rg_param", "ry_param"):
        v = clean.get(key)
        if v is not None and v < 2:
            errors[key] = "must be at least 2"

    for key in ("T_param", "checkpoint"):
        v = clean.get(key)
        if v is not None and v <= 0:
            errors[key] = "must be positive"

    nlist = clean.get("N_LIST")
    if isinstance(nlist, list):
        if not nlist:
            errors["N_LIST"] = "must contain at least one microtubule position"
        elif any(float(x) != int(x) for x in nlist):
            errors["N_LIST"] = "positions must be whole numbers"
        elif len(set(int(x) for x in nlist)) != len(nlist):
            errors["N_LIST"] = "positions must be distinct"
        elif ry is not None and (min(nlist) < 0 or max(nlist) > ry - 1):
            errors["N_LIST"] = f"positions must lie in [0, {ry - 1}] for ry_param={ry}"

    # d_tube can only be judged against a valid grid and N_LIST; when either is
    # missing or already wrong, that error is the one worth reporting.
    d_tube = clean.get("d_tube")
    if (d_tube is not None and rg is not None and ry is not None
            and isinstance(nlist, list) and nlist
            and not {"rg_param", "ry_param", "N_LIST"} & set(errors)):
        verdict = check_d_tube(rg, ry, nlist, d_tube)
        if not verdict["valid"]:
            errors["d_tube"] = verdict["message"]

    # The (a, b) grid solves the cartesian product of these two, so an empty one
    # would produce no points at all -- and a negative switch rate is not a
    # slower switch, it is a source term the scheme was never derived for.
    for key in ("a_list", "b_list"):
        v = clean.get(key)
        if isinstance(v, list):
            if not v:
                errors[key] = "must contain at least one switch rate"
            elif any(x < 0 for x in v):
                errors[key] = "switch rates must be non-negative"

    # Same rule for the single (a, b) pair a mass analysis takes.
    for key in ("a_param", "b_param"):
        v = clean.get(key)
        if v is not None and v < 0:
            errors[key] = "switch rates must be non-negative"

    if computation_name in _CHAR_TIME_COMPUTATIONS:
        t = clean.get("T_param")
        if t is not None and t <= _CHAR_TIME_MIN_T:
            errors["T_param"] = (
                f"must exceed {_CHAR_TIME_MIN_T} for this computation: the "
                f"steady decay must hold for at least {_CHAR_TIME_MIN_T} after "
                f"t*, which itself is typically 0.25-0.45")
        tau = clean.get("tau")
        if tau is not None and not tau > 0:
            errors["tau"] = "must be a positive threshold"

    # Device selection. Checked here so a typo ("gpu1", "cuda:0") is rejected at
    # submission rather than after the solve has been queued -- and so that
    # asking for the GPU on a configuration the port refuses (d_tube != 0, an
    # off-centre seed) is caught before any compute time is spent.
    device = clean.get("device")
    if device is not None:
        from launch_functions.launch import DEVICES

        norm = str(device).strip().lower()
        if norm not in DEVICES:
            errors["device"] = f"must be one of {', '.join(DEVICES)}"
        else:
            clean["device"] = norm
            if norm == "gpu":
                if clean.get("d_tube"):
                    errors["device"] = (
                        "gpu does not support d_tube != 0; use cpu, or auto to "
                        "fall back automatically")
                elif clean.get("center_init_cond") is False:
                    errors["device"] = (
                        "gpu does not support off-centre initial conditions; "
                        "use cpu, or auto to fall back automatically")

    if clean.get("center_init_cond") is False:
        for key, limit, dim in (("m_init", rg, "rg_param"),
                                ("n_init", ry, "ry_param")):
            v = clean.get(key, 0)
            if limit is not None and not (0 <= v <= limit - 1):
                errors[key] = f"must lie in [0, {limit - 1}] for {dim}={limit}"

    return errors


def run_selected_computation(computation_name, param_dict):
    """
    Dispatches the selected computation function with the parsed parameters.
    Returns whatever the backend function returns, formatted into a consistent output.
    """
    if computation_name not in COMPUTATION_FUNCTIONS:
        raise ValueError(f"Unknown computation: {computation_name}")

    func = COMPUTATION_FUNCTIONS[computation_name]

    # Parse inputs
    parsed_inputs = {
        key: parse_input(val)
        for key, val in param_dict.items()
        if not _is_blank(val)
    }

    result = func(**parsed_inputs)

    # --- Case 1: Solve MFPT ---
    if computation_name == "Compute MFPT until time T":
        output = {}
        if isinstance(result, tuple):
            output["MFPT"] = result[0]
            output["duration"] = result[1]
        elif isinstance(result, dict):
            output["MFPT"] = result.get("MFPT")
            if "duration" in result and isinstance(result["duration"], (int, float)):
                output["duration"] = result["duration"]
        else:
            output["MFPT"] = result
        return output

    if computation_name == "Compute MFPT until mass %":
        output = {}
        if isinstance(result, tuple):
            output["MFPT"] = result[0]
            output["duration"] = result[1]
        else:
            output["MFPT"] = result
        return output

    # --- Case 2: Time Until Mass Depletion ---
    if computation_name == "Time Until Mass Depletion":
        return {"duration": result}

    # --- Case 3: Plot-producing functions (return list of paths) ---
    if computation_name in {
        "MFPT point collection (time T dep.)",
        "Phi angular dependence (collection: time T dep.)",
        "Phi/Rho radial dependence (collection: time T dep.)",
        "Static Heatplot Analysis (collection: time T dep.)",
        "Full Analysis (time t dep.)",

        "MFPT point collection (Mass % dep.)",
        "Phi angular dependence (collection: Mass % dep.)",
        "Phi/Rho radial dependence (collection: Mass % dep.)",
        "Static Heatplot Analysis (collection: Mass % dep.)",

        "Mass Analysis"

    }:

        if isinstance(result, list) and all(isinstance(p, str) for p in result):
            return {"output_dirs": [os.path.dirname(p) for p in result]}
        elif isinstance(result, dict):
            # Backward compatibility if returning a dict of lists
            output = {}
            for key, val in result.items():
                if isinstance(val, list) and all(isinstance(p, str) for p in val):
                    output["output_dirs"] = [os.path.dirname(p) for p in val]
                    break
            return output

    # --- Fallback: return as-is ---
    return result