
import os


def extract_csv_and_png_paths(destination_dirs):
    """
    Given a list of directory paths, return two lists:
    - csv_file_locations: paths to all .csv files found in the directories
    - png_file_locations: paths to all .png files found in the directories
    """
    csv_file_locations = []
    png_file_locations = []

    for dir_path in destination_dirs:
        if not os.path.isdir(dir_path):
            continue  # Skip if not a valid directory

        for filename in os.listdir(dir_path):
            full_path = os.path.join(dir_path, filename)
            if os.path.isfile(full_path):
                if filename.lower().endswith(".csv"):
                    csv_file_locations.append(full_path)
                elif filename.lower().endswith(".png"):
                    png_file_locations.append(full_path)

    return csv_file_locations, png_file_locations


def char_time_lines(result):
    """Output-panel lines for a characteristic-time result, one per point.

    ``result["t_star"]`` is the per-point table the router builds (see
    computation_router._char_time_summary): the swept parameter(s) plus t* and
    m* = M(t*), with None where |y''| had not settled below tau within T.
    Returns [] for any other result.
    """
    rows = result.get("t_star") if isinstance(result, dict) else None
    if not rows:
        return []
    tau = result.get("tau")
    lines = [f"Characteristic time (right-sweep |y''| <= tau"
             + (f", tau = {tau:g}" if tau is not None else "") + "):"]
    for r in rows:
        label = ", ".join(f"{k}={r[k]:g}" for k in ("v", "a", "b") if k in r)
        if r.get("t_star") is None:
            lines.append(f"    {label}: no t* within T (|y''| still above tau at the end)")
        else:
            lines.append(f"    {label}: t* = {r['t_star']:.6f}, m* = M(t*) = {r['m_star']:.6e}")
    return lines
