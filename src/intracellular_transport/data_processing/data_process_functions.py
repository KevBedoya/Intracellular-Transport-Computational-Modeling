from system_configuration import file_paths as fp
from auxiliary_tools import prints, tabulate_functions as tb
from data_visualization import plot_functions as plt, animation_functions as ani
from computational_tools import numerical_tools as num

import pandas as pd
import time
import os
import numpy as np


def process_BC_analysis_DL(PvT_DL_snapshots, BC_data_dict, v_param, w_LIST, N_LIST, T_fixed_ring_seg, save_png, show_plt,
                           checkpoint, approach, ry_param, rg_param, collect_radial_info=False):

    output_location_list = []

    timestamp = prints.return_timestamp()
    data_filepath = os.path.abspath(tb.create_directory(fp.phi_v_theta_output, timestamp))
    filename = f"BC_analysis_Dl_v={v_param}_N={len(N_LIST)}_Domain={rg_param}x{ry_param}.csv"
    output_location = os.path.join(data_filepath, filename)

    df_bc = pd.DataFrame(BC_data_dict)
    df_bc.to_csv(output_location, index=False)
    plt.plot_bc_analysis(output_location, w_LIST, v_param, checkpoint, T_fixed_ring_seg, data_filepath, save_png, show_plt)

    if collect_radial_info:
        checkpoint_collect_container = [checkpoint]
        radians_mesh = [i * (2 * np.pi)/ry_param for i in range(ry_param)]
        degrees_mesh = [ang * (180/np.pi) for ang in radians_mesh]

        data_dict = {
            'Disc-Pos': list(range(ry_param)),
            'Radians': radians_mesh,
            'Degrees': degrees_mesh,
        }

        if int(approach) == 1:
            col_name = f"M={checkpoint:.4}"
        elif int(approach) == 2:
            col_name = f"T={checkpoint:.4f}"
        else:
            raise ValueError(f"Approach: {approach} is invalid. Please use approach 1 or 2. (int)")

        for i in range(len(w_LIST)):

            data_dict[col_name] = PvT_DL_snapshots[i]

            filename = f"BC_analysis_Dl_v={v_param}_w_param={w_LIST[ i ]}_N={len(N_LIST)}_Domain={rg_param}x{ry_param}.csv"
            output_location_i = os.path.join(data_filepath, filename)

            df = pd.DataFrame(data_dict)
            df.to_csv(output_location_i, index=False)

            plt.plot_phi_v_theta(output_location_i, v_param, w_LIST[i], N_LIST, approach, T_fixed_ring_seg,
                                 data_filepath, checkpoint_collect_container, save_png=save_png, show_plt=show_plt,
                                 png_title=f"phi_v_theta_w={w_LIST[i]}.png")

            time.sleep(1)

    print("\n")

    output_location_list.append(output_location)

    return output_location_list


def process_BC_analysis_grid_size(BC_data_dict, grid_list, v_param, w_param, N_amount, T_fixed_ring_seg, save_png, show_plt, checkpoint):

    output_location_list = []
    timestamp = prints.return_timestamp()
    data_filepath = os.path.abspath(tb.create_directory(fp.phi_v_theta_output, timestamp))
    filename = f"BC_vs_Gsize_analysis_grid_v={v_param}_w={w_param}_N={N_amount}.csv"
    output_location = os.path.join(data_filepath, filename)

    df_bc = pd.DataFrame(BC_data_dict)
    df_bc.to_csv(output_location, index=False)
    plt.plot_bc_analysis(output_location, grid_list, v_param, checkpoint, T_fixed_ring_seg, data_filepath, save_png, show_plt, True)
    output_location_list.append(output_location)
    return output_location_list

def process_PvT_DL(PvT_DL_snapshots, v_param, w_param, N_LIST, T_fixed_ring_seg, save_png, show_plt, checkpoint_collect_container, approach,
                   ry_param, rg_param):

    output_location_list = []

    timestamp = prints.return_timestamp()
    data_filepath = os.path.abspath(tb.create_directory(fp.phi_v_theta_output, timestamp))
    filename = f"PvT_Dl_v={v_param}_w={w_param}_N={len(N_LIST)}_Domain={rg_param}x{ry_param}.csv"
    output_location = os.path.join(data_filepath, filename)

    radians_mesh = [i * (2 * np.pi)/ry_param for i in range(ry_param)]
    degrees_mesh = [ang * (180/np.pi) for ang in radians_mesh]

    data_dict = {
        'Disc-Pos': list(range(ry_param)),
        'Radians': radians_mesh,
        'Degrees': degrees_mesh,
    }

    for c_point, snapshot in zip(checkpoint_collect_container, PvT_DL_snapshots):
        if int(approach) == 1:
            col_name = f"M={c_point:.4}"
        elif int(approach) == 2:
            col_name = f"T={c_point:.4f}"
        else:
            raise ValueError(f"Approach: {approach} is invalid. Please use approach 1 or 2. (int)")
        data_dict[col_name] = snapshot

    df = pd.DataFrame(data_dict)
    df.to_csv(output_location, index=False)

    time.sleep(1)
    plt.plot_phi_v_theta(output_location, v_param, w_param, N_LIST, approach, T_fixed_ring_seg,
                         data_filepath, checkpoint_collect_container, save_png=save_png, show_plt=show_plt)
    print("\n")

    output_location_list.append(output_location)

    return output_location_list


def process_Jrr_sum_results(Jrr_sum_timeseries, v_param, w_param, N_LIST, rg_param,
                            ry_param, save_png, show_plt, collection_factor, domain_radius, D):

    output_location_list = []

    dt = num.compute_dT(rg_param, ry_param, domain_radius, D)

    T_mesh = [dt * collection_factor * t for t in range(len(Jrr_sum_timeseries))]

    # Diffusive mass analysis
    timestamp = prints.return_timestamp()
    data_filepath = os.path.abspath(tb.create_directory(fp.mass_analysis_diffusive, timestamp))

    filename = f"Jrr_mass_v={v_param}_w={w_param}_N={len(N_LIST)}_Domain={rg_param}x{ry_param}.csv"
    output_location = os.path.join(data_filepath, filename)

    data_dict = {
        'T': T_mesh,
        'Jrr_sum': Jrr_sum_timeseries
    }

    df = pd.DataFrame(data_dict)

    df.to_csv(output_location, index=False)

    peak_time, _ = num.find_max_with_time(output_location)

    plt.plot_Jrr_sum_analysis(output_location, v_param, w_param, N_LIST, rg_param, ry_param, data_filepath, save_png, show_plt)
    print("\n")
    output_location_list.append(output_location)
    return output_location_list, peak_time


def switch_rate_tag(a_param, b_param):
    """Filename fragment naming the two switch rates.

    Equal rates keep the ``w=<rate>`` spelling every mass-analysis file used
    before the two could be set separately, so results produced under the old
    single-w signature and under the new one sit in the same namespace. The
    rates are only spelled out individually when they actually differ.
    """
    return f"w={a_param}" if a_param == b_param else f"a={a_param}_b={b_param}"


def process_MA_results(MA_DL_timeseries, MA_AL_timeseries, MA_TM_timeseries, MA_ALoT_timeseries,
                       MA_ALoI_timeseries, v_param, a_param, b_param, N_LIST, T_param, rg_param, ry_param,
                       save_png, show_plt, mass_collection_factor, domain_radius, D):

    output_location_list = []

    w_tag = switch_rate_tag(a_param, b_param)

    dt = num.compute_dT(rg_param, ry_param, domain_radius, D)

    T_mesh = [dt * mass_collection_factor * t for t in range(len(MA_DL_timeseries))]

    # Diffusive mass analysis
    timestamp = prints.return_timestamp()
    data_filepath = os.path.abspath(tb.create_directory(fp.mass_analysis_diffusive, timestamp))

    filename = f"MA_DL_v={v_param}_{w_tag}_N={len(N_LIST)}_Domain={rg_param}x{ry_param}.csv"
    output_location = os.path.join(data_filepath, filename)

    data_dict = {
        'T': T_mesh,
        'M': MA_DL_timeseries
    }

    df = pd.DataFrame(data_dict)

    df.to_csv(output_location, index=False)
    plt.plot_mass_analysis(output_location, v_param, a_param, b_param, N_LIST, T_param, rg_param, ry_param,
                           "DL", "DL", data_filepath, save_png, show_plt)
    print("\n")
    output_location_list.append(output_location)

    # Advective mass analysis
    timestamp = prints.return_timestamp()
    data_filepath = os.path.abspath(tb.create_directory(fp.mass_analysis_advective, timestamp))
    filename = f"MA_AL_v={v_param}_{w_tag}_N={len(N_LIST)}_Domain={rg_param}x{ry_param}.csv"
    output_location = os.path.join(data_filepath, filename)

    data_dict = {
        'T': T_mesh,
        'M': MA_AL_timeseries
    }

    df = pd.DataFrame(data_dict)

    df.to_csv(output_location, index=False)
    plt.plot_mass_analysis(output_location, v_param, a_param, b_param, N_LIST, T_param, rg_param, ry_param,
                           "AL", "AL", data_filepath, save_png, show_plt)
    print("\n")
    output_location_list.append(output_location)

    # Total mass analysis
    timestamp = prints.return_timestamp()
    data_filepath = os.path.abspath(tb.create_directory(fp.mass_analysis_total, timestamp))
    filename = f"MA_total_v={v_param}_{w_tag}_N={len(N_LIST)}_Domain={rg_param}x{ry_param}.csv"
    output_location = os.path.join(data_filepath, filename)

    data_dict = {
        'T': T_mesh,
        'M': MA_TM_timeseries
    }

    df = pd.DataFrame(data_dict)

    df.to_csv(output_location, index=False)
    plt.plot_mass_analysis(output_location, v_param, a_param, b_param, N_LIST, T_param, rg_param, ry_param,
                           "Total", "Total", data_filepath, save_png, show_plt)
    print("\n")
    output_location_list.append(output_location)

    # Advective/running total mass analysis
    timestamp = prints.return_timestamp()
    data_filepath = os.path.abspath(tb.create_directory(fp.mass_analysis_advective_over_total, timestamp))
    filename = f"MA_AL_running_total_v={v_param}_{w_tag}_N={len(N_LIST)}_Domain={rg_param}x{ry_param}.csv"
    output_location = os.path.join(data_filepath, filename)

    data_dict = {
        'T': T_mesh,
        'M': MA_ALoT_timeseries
    }

    df = pd.DataFrame(data_dict)

    df.to_csv(output_location, index=False)
    plt.plot_mass_analysis(output_location, v_param, a_param, b_param, N_LIST, T_param, rg_param, ry_param,
                           "AL/running total", "Al_running_total", data_filepath, save_png, show_plt)
    print("\n")
    output_location_list.append(output_location)

    # Advective/initial total mass analysis
    timestamp = prints.return_timestamp()
    data_filepath = os.path.abspath(tb.create_directory(fp.mass_analysis_advective_over_initial, timestamp))
    filename = f"MA_AL_initial_total_v={v_param}_{w_tag}_N={len(N_LIST)}_Domain={rg_param}x{ry_param}.csv"
    output_location = os.path.join(data_filepath, filename)

    data_dict = {
        'T': T_mesh,
        'M': MA_ALoI_timeseries
    }

    df = pd.DataFrame(data_dict)

    df.to_csv(output_location, index=False)
    plt.plot_mass_analysis(output_location, v_param, a_param, b_param, N_LIST, T_param, rg_param, ry_param,
                           "AL/initial total", "AL_initial_total", data_filepath, save_png, show_plt)
    print("\n")
    output_location_list.append(output_location)

    return output_location_list


def process_DvR_results(PvR_DL_snapshots, RvR_AL_snapshots, v_param, w_param, N_LIST, rg_param,
                        ry_param, R_fixed_angle, checkpoint_collect_container, save_png, show_plt, approach, domain_radius):
    # Processing results for Phi v. Radius & Rho v. Radius

    output_location_list = []

    radial_mesh = [i * domain_radius * 1/rg_param for i in range(rg_param + 1)]

    data_dict = {
        'Disc-Pos': list(range(rg_param + 1)),
        'Radius': radial_mesh
    }

    for c_point, snapshot in zip(checkpoint_collect_container, PvR_DL_snapshots):
        if int(approach) == 1:
            col_name = f"M={c_point:.4}"
        elif int(approach) == 2:
            col_name = f"T={c_point:.4f}"
        else:
            raise ValueError(f"Approach: {approach} is invalid. Please use approach 1 or 2. (int)")
        data_dict[col_name] = snapshot

    timestamp = prints.return_timestamp()
    data_filepath = os.path.abspath(tb.create_directory(fp.radial_dependence_phi, timestamp))
    filename = f"PvR_DL_snapshots_@DiscAng={R_fixed_angle}_v={v_param}_w={w_param}_N={len(N_LIST)}_Domain={rg_param}x{ry_param}.csv"
    output_location = os.path.join(data_filepath, filename)
    df = pd.DataFrame(data_dict)
    # df = pd.DataFrame(PvR_DL_snapshots)
    df.to_csv(output_location, index=False)
    plt.plot_dense_v_rad("Phi", output_location, v_param, w_param,
                         len(N_LIST), rg_param, ry_param, R_fixed_angle, checkpoint_collect_container,
                         data_filepath, approach, save_png=save_png, show_plt=show_plt)
    print("\n")
    output_location_list.append(output_location)

    radial_mesh = [i * domain_radius * 1 / rg_param for i in range(1, rg_param+1)]

    data_dict = {
        'Disc-Pos': list(range(1, rg_param+1)),
        'Radius': radial_mesh
    }

    for c_point, snapshot in zip(checkpoint_collect_container, RvR_AL_snapshots):
        if int(approach) == 1:
            col_name = f"M={c_point:.4}"
        elif int(approach) == 2:
            col_name = f"T={c_point:.4f}"
        else:
            raise ValueError(f"Approach: {approach} is invalid. Please use approach 1 or 2. (int)")
        data_dict[col_name] = snapshot

    timestamp = prints.return_timestamp()
    data_filepath = os.path.abspath(tb.create_directory(fp.radial_dependence_rho, timestamp))
    filename = f"RvR_AL_snapshots_@DiscAng={R_fixed_angle}_v={v_param}_w={w_param}_N={len(N_LIST)}_Domain={rg_param}x{ry_param}.csv"
    output_location = os.path.join(data_filepath, filename)
    # df = pd.DataFrame(RvR_AL_snapshots)
    df = pd.DataFrame(data_dict)
    df.to_csv(output_location, index=False)
    plt.plot_dense_v_rad("Rho", output_location, v_param, w_param,
                         len(N_LIST), rg_param, ry_param, R_fixed_angle, checkpoint_collect_container,
                         data_filepath, approach, save_png=save_png, show_plt=show_plt)
    print("\n")
    output_location_list.append(output_location)

    return output_location_list


def process_static_HM_results(HM_DL_snapshots, HM_C_snapshots, MFPT_snapshots, checkpoint_collect_container,
                              heat_plot_border, w_param, v_param, N_LIST, heatplot_colorscheme,
                              save_png, show_plt, j_max_list, display_extraction, approach, rg_param, ry_param):

    output_location_list = []
    timestamp = prints.return_timestamp()
    data_filepath = tb.create_directory(fp.heatmap_output, timestamp)

    N_LIST_count = len(N_LIST)

    for t in range(len(checkpoint_collect_container)):

        MFPT = MFPT_snapshots[t]
        curr_DL_snapshot = HM_DL_snapshots[t]
        curr_C_snapshot = HM_C_snapshots[t]
        check_point = checkpoint_collect_container[t]

        csv_filename = f"HM_DL_snapshot_T={check_point}_v={v_param}_w={w_param}_N={len(N_LIST)}_Domain={rg_param}x{ry_param}.csv"
        output_csv_loc = os.path.join(data_filepath, csv_filename)
        # print(output_csv_loc)
        df = pd.DataFrame(curr_DL_snapshot)
        df.to_csv(output_csv_loc, header=False, index=False)

        ani.produce_heatmap_tool_rect(curr_DL_snapshot, curr_C_snapshot, w_param, v_param, N_LIST_count,
                                      approach, t, data_filepath, MFPT, check_point, N_LIST, j_max_list, display_extraction,
                                      save_png, show_plt, transparent=False, toggle_border=heat_plot_border, color_scheme=heatplot_colorscheme)
        print("\n")
        time.sleep(1)

    csv_filename = f"HM_C_snapshot_v={v_param}_w={w_param}_N={len(N_LIST)}_Domain={rg_param}x{ry_param}.csv"
    column_labels = ['T', 'Phi(center)']

    data_dict = {
        column_labels[0]: checkpoint_collect_container,
        column_labels[1]: HM_C_snapshots
    }

    output_csv_loc = os.path.join(data_filepath, csv_filename)
    df = pd.DataFrame(data_dict)
    df.to_csv(output_csv_loc, index=False)
    output_location_list.append(output_csv_loc)

    return output_location_list


def process_MFPT_results(MFPT_snapshots, checkpoint_collect_container, approach,
                         rg_param, ry_param, w_param, v_param, N_LIST, save_png=True,
                         show_plt=True):

    output_location_list = []

    timestamp = prints.return_timestamp()
    data_filepath = tb.create_directory(fp.mfpt_results_output, timestamp)

    csv_filename = f"MFPT_timestamp_v={v_param}_w={w_param}_N={len(N_LIST)}_Domain={rg_param}x{ry_param}.csv"

    if approach == 1:
        x_label = 'TM'
    elif approach == 2:
        x_label = 'T'
    else:
        raise ValueError(f'{approach} is not a valid argument, use either collection approach "1" or "2" (must be an int)')

    column_labels = [x_label, 'MFPT']

    data_dict = {
        column_labels[0]: checkpoint_collect_container,
        column_labels[1]: MFPT_snapshots
    }

    output_csv_loc = os.path.join(data_filepath, csv_filename)
    df = pd.DataFrame(data_dict)
    df.to_csv(output_csv_loc, index=False)
    plt.plot_mfpt_v_checkpoints(output_csv_loc, x_label, rg_param, ry_param, w_param, v_param, N_LIST, data_filepath,
                                save_png, show_plt)
    print("\n")
    output_location_list.append(output_csv_loc)

    return output_location_list
