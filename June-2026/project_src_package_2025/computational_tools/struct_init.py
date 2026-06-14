from numba.typed import Dict, List
from typing import List as PyList
from numba.types import int64, ListType
from . import njit, np
import csv
import os
from computational_tools import supplements as sup, numerical_tools as num


# (****) (****)
@njit
def build_d_tube_mapping_no_overlap(rg_param, ry_param, N_LIST, d_tube=0.0, domain_radius=1.0):
    d_list = List()

    j_sup = sup.j_max_bef_overlap(ry_param, N_LIST)
    max_d_tube = sup.solve_d_rect(1, rg_param, ry_param, j_sup, 0)

    if d_tube < 0 or d_tube > max_d_tube:
        d_tube = max_d_tube

    dRad = num.compute_dRad(rg_param, domain_radius)
    dThe = num.compute_dThe(ry_param)

    for m in range(rg_param):
        j_max = np.ceil((d_tube / ((m + 1) * dRad * dThe)) - 0.5)
        keys = sup.mod_range_flat(N_LIST, j_max, ry_param, False)
        d = sup.dict_gen(keys, N_LIST)
        d_list.append(d)

    return d_list


# (****) (****)
@njit
def build_j_max_list(rg_param, ry_param, N_LIST, d_tube=0.0, domain_radius=1.0):

    j_max_list = []
    j_max_lim = sup.j_max_bef_overlap(ry_param, N_LIST)
    max_d_tube = sup.solve_d_rect(domain_radius, ry_param, rg_param, j_max_lim, 0)

    if d_tube < 0 or d_tube > max_d_tube:
        d_tube = max_d_tube

    dRad = num.compute_dRad(rg_param, domain_radius=domain_radius)
    dThe = num.compute_dThe(ry_param)

    for m in range(rg_param):
        j_max = int(np.ceil((d_tube / ((m + 1) * dRad * dThe)) - 0.5))
        j_max_list.append(j_max)

    return j_max_list


# <***> <***>
@njit
def build_rect_config_dict(RINGS, RAYS, N, D_THE, D_RAD, D_TUBE):
    """
    Title: "Build rectangular-configuration dictionary"

    Builds a list of dictionaries representing extraction regions
    on a polar grid, where each key in a ring-layer dictionary maps
    an angular location to microtubule sources within the tube radius.


    Args:
        RINGS: (int)  Number of rings in the domain
        RAYS: (int) Number of rays in the domain
        N: (List[int]) List of microtubule positions
        D_THE: (float) Delta theta
        D_RAD: (float) Delta radius
        D_TUBE: (float) desired tube width for microtubule extraction

    Returns:
        D (List[Dict[int, List[int]]]): Extraction region data structure.

    Each dictionary in the list corresponds to a radial ring in the domain beginning at the 1st ring, (the first ring enclosing the center patch)
    Each key in the dictionary (dict) of the mth ring ( m in [0, M-1] ), corresponds to angular ray on the diffusive layer (DL) at position {m,n}.
    The corresponding value to each key in the dict on ring m is the associated microtubule.

    The keys at the mth ring in the dict form an extraction region relative to the radius and extraction range (d_tube) at the mth ring.

    This DS serves as a reference table in the rectangular configuration of the particle diffusion code.

    ***The output DS allows for overlap of extraction regions to take place as well***
    """

    D = List()
    empty_int_list = List.empty_list(int64)

    for m in range(RINGS):
        d_m = Dict.empty(key_type=int64, value_type=empty_int_list)
        # calculate j_max at the mth ring
        Jm = int(np.ceil(D_TUBE / (D_RAD * D_THE * (m + 1)) - 0.5))

        for n in range(len(N)):

            center = N[n]
            for k in range(center - Jm, center + Jm + 1):
                key = k % RAYS
                if key in d_m:
                    already_in_list = False
                    for val in d_m[key]:
                        if val == center:
                            already_in_list = True
                            break
                    if not already_in_list:
                        d_m[key].append(center)
                else:
                    new_list = List.empty_list(int64)
                    new_list.append(center)
                    d_m[key] = new_list
        D.append(d_m)

    return D


def print_rect_config_dict_view(RINGS, RAYS, N_list, D_THE, D_RAD, D_TUBE):
    """
    Runs the build_rect_config_dict function and prints its output in
    a human-readable format showing each ring and its dictionary of mappings.
    """

    # Convert regular list to numba typed List
    typed_N = List()
    for val in N_list:
        typed_N.append(val)

    # Run the structure builder
    D = build_rect_config_dict(RINGS, RAYS, typed_N, D_THE, D_RAD, D_TUBE)

    # Pretty print
    print("[")
    for m, d in enumerate(D):
        ring_dict = {}
        for key in d:
            ring_dict[int(key)] = [int(v) for v in d[key]]
        print(f"Ring {m}: {ring_dict},")
    print("]")


def export_rect_config_structure_to_csv(
    RINGS: int,
    RAYS: int,
    N_microtubules: PyList[int],
    D_THE: float,
    D_RAD: float,
    D_TUBE: float,
    output_dir: str,
    filename: str = "dictionary_domain_result.csv"
):
    """
    Builds the rectangular config dictionary and exports its contents to a CSV file.

    Args:
        RINGS: Number of radial rings
        RAYS: Number of angular rays
        N_microtubules: List of microtubule angular positions
        D_THE: Delta theta
        D_RAD: Delta radius
        D_TUBE: Tube radius
        output_dir: Directory to save the CSV file
        filename: Name of the CSV file (default: 'dictionary_domain_result.csv')
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct full file path
    full_path = os.path.join(output_dir, filename)

    # Convert to numba typed list
    typed_N = List()
    for val in N_microtubules:
        typed_N.append(val)

    # Build structure
    structure = build_rect_config_dict(RINGS, RAYS, typed_N, D_THE, D_RAD, D_TUBE)

    # Write to CSV
    with open(full_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Ring", "DL_Angle_Index", "Microtubules"])

        for m, d in enumerate(structure):
            for angle_key in d:
                mt_list = list(d[angle_key])
                writer.writerow([m, int(angle_key), mt_list])

    print(f"Structure exported to: {full_path}")
