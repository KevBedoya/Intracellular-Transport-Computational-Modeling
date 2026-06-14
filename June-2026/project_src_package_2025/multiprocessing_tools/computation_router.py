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
    "Mass Analysis": launch.collect_mass_analysis
}
# "Full Analysis": launch.launch_super_comp_I


def parse_input(value):
    """
    Attempts to convert a string input into its appropriate Python type.
    Supports float, int, bool, list, None, etc.
    """
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value.strip()


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
        if val.strip() != ""
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