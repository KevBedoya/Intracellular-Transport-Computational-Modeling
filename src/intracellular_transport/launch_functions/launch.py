from . import mfpt_comp, sup, ant, mp, np, num, super, os, pd, datetime, fp

from computational_tools import struct_init
from computational_tools import mass_analysis_modal as mam
from computational_tools import y2_spectrum as y2s
from data_processing import data_process_functions as pro
import matplotlib.pyplot as plt

# Accepted values for the `device` parameter, in the form the API and GUI use.
# Defined here rather than beside _solve_mass_analysis because they are default
# argument values, which are evaluated when the enclosing `def` executes.
DEVICE_CPU = "cpu"
DEVICE_GPU = "gpu"
DEVICE_AUTO = "auto"
DEVICES = (DEVICE_CPU, DEVICE_GPU, DEVICE_AUTO)


# (****) Validate an off-center initial-condition patch (m, n) (****)
def _validate_off_center_ic(center_init_cond, m_init, n_init, rg_param, ry_param):
    """
    Validate the discrete off-center initial-condition coordinates.

    Only enforced when center_init_cond is False (i.e. the user requested an
    off-center initial condition). The seeded ring m_init is limited to
    [0, rg_param-1] and the seeded ray n_init to [0, ry_param-1]. When
    center_init_cond is True the default centered scheme is used and no
    coordinates are required.
    """
    if center_init_cond:
        return
    if m_init < 0 or m_init > rg_param - 1:
        raise IndexError(
            f'Off-center initial condition ring m_init: {m_init} falls outside of the legal ring range [0, {rg_param - 1}].')
    if n_init < 0 or n_init > ry_param - 1:
        raise IndexError(
            f'Off-center initial condition ray n_init: {n_init} falls outside of the legal ray range [0, {ry_param - 1}].')


# v======================================== Mass dependent computations ========================================v


# (****) (****)
def solve_mfpt_mass_(rg_param, ry_param, N_LIST, v_param, w_param, domain_radius=1.0, D=1.0, mass_checkpoint=10**6, mass_retention_threshold=0.01, d_tube=0.0,
                     center_init_cond=True, m_init=0, n_init=0):
    if len(N_LIST) > ry_param:
        raise IndexError(
            f'Too many angular indices supplied for microtubule positions: {len(N_LIST)} > {ry_param} (number of angular positions in domain).'
        )

    for i in range(len(N_LIST)):
        if N_LIST[i] < 0 or N_LIST[i] > ry_param:
            raise IndexError(
                f'Angular index: {N_LIST[i]} falls outside of the legal index range: [0,{ry_param - 1}) under ry_param={ry_param}')

    N_LIST.sort()

    d_tube_max = sup.solve_d_rect(domain_radius, rg_param, ry_param, sup.j_max_bef_overlap(ry_param, N_LIST), 0)

    if d_tube < 0 or d_tube > d_tube_max:
        raise ValueError(f"d_tube: {d_tube} is outside of the legal range: [0, {d_tube_max}")

    _validate_off_center_ic(center_init_cond, m_init, n_init, rg_param, ry_param)

    D_LAYER, A_LAYER = sup.initialize_layers(rg_param, ry_param)
    MFPT, sim_time = mfpt_comp.comp_mfpt_by_mass_loss(rg_param, ry_param, w_param, w_param, v_param, N_LIST, D_LAYER, A_LAYER, mass_checkpoint, domain_radius, D, mass_retention_threshold, d_tube,
                                                      center_init_cond, m_init, n_init)
    print("\n\n")
    return MFPT, sim_time
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# (****) (****)
def collect_MFPT_snapshots_mass_dep(rg_param, ry_param, N_LIST, v_param, w_param,
                                    checkpoint_collect_container, mass_retention_threshold=0.01, domain_radius=1.0, D=1.0, mass_checkpoint=10**6,
                                    d_tube=0.0, save_png=True, show_plt=False,
                                    center_init_cond=True, m_init=0, n_init=0):
    if len(N_LIST) > ry_param:
        raise IndexError(
            f'Too many angular indices supplied for microtubule positions: {len(N_LIST)} > {ry_param} (number of angular positions in domain).'
        )

    for i in range(len(N_LIST)):
        if N_LIST[i] < 0 or N_LIST[i] >= ry_param:
            raise IndexError(
                f'Angular index: {N_LIST[i]} falls outside of the legal index range: [0,{ry_param - 1}) under ry_param={ry_param}')
    N_LIST.sort()

    checkpoint_collect_container.sort(reverse=True)

    checkpoint_collect_container = [float(item) for item in checkpoint_collect_container]

    checkpoint_enum = len(checkpoint_collect_container)

    d_tube_max = sup.solve_d_rect(domain_radius, rg_param, ry_param, sup.j_max_bef_overlap(ry_param, N_LIST), 0)

    if d_tube < 0 or d_tube > d_tube_max:
        raise ValueError(f"d_tube: {d_tube} is outside of the legal range: [0, {d_tube_max}")

    _validate_off_center_ic(center_init_cond, m_init, n_init, rg_param, ry_param)

    D_LAYER, A_LAYER = sup.initialize_layers(rg_param, ry_param)
    MFPT_snapshots = np.zeros([checkpoint_enum], dtype=np.float64)

    mfpt_comp.comp_mfpt_by_time_points_mass_dep(rg_param, ry_param, w_param, w_param, v_param, N_LIST,
                                                D_LAYER, A_LAYER, checkpoint_collect_container, MFPT_snapshots,
                                                mass_retention_threshold, mass_checkpoint, domain_radius, D, d_tube,
                                                center_init_cond, m_init, n_init)
    print("\n\n")
    return pro.process_MFPT_results(MFPT_snapshots, checkpoint_collect_container, 1, rg_param, ry_param, w_param, v_param, N_LIST,
                                    save_png, show_plt)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# (****) (****)
def collect_phi_ang_dep_mass_dep(rg_param, ry_param, v_param, w_param, N_LIST, checkpoint_collect_container,
                                 mass_retention_threshold=0.01, T_fixed_ring_seg=0.5, d_tube=0.0, domain_radius=1.0, D=1.0,
                                 mass_checkpoint=10**6, save_png=True, show_plt=False,
                                 center_init_cond=True, m_init=0, n_init=0):
    if len(N_LIST) > ry_param:
        raise IndexError(
            f'Too many angular indices supplied for microtubule positions: {len(N_LIST)} > {ry_param} (number of angular positions in domain).'
        )

    for i in range(len(N_LIST)):
        if N_LIST[i] < 0 or N_LIST[i] >= ry_param:
            raise IndexError(
                f'Angular index: {N_LIST[i]} falls outside of the legal index range: [0,{ry_param - 1}) under ry_param={ry_param}')
    N_LIST.sort()

    if T_fixed_ring_seg < 0 or T_fixed_ring_seg > 1:
        print("T_fixed_ring_seg automatically adjusted to legal range.")
        T_fixed_ring_seg = 0.5

    checkpoint_collect_container.sort(reverse=True)

    checkpoint_collect_container = [float(item) for item in checkpoint_collect_container]

    # d_tube_max is with respect to the non-overlapping microtubule extraction region implementation.
    d_tube_max = sup.solve_d_rect(domain_radius, rg_param, ry_param, sup.j_max_bef_overlap(ry_param, N_LIST), 0)

    if d_tube < 0 or d_tube > d_tube_max:
        raise ValueError(f"d_tube: {d_tube} is outside of the legal range: [0, {d_tube_max}")

    collection_stamp_enum = len(checkpoint_collect_container)

    PvT_DL_snapshots = np.zeros((collection_stamp_enum, ry_param), dtype=np.float64)

    _validate_off_center_ic(center_init_cond, m_init, n_init, rg_param, ry_param)

    D_LAYER, A_LAYER = sup.initialize_layers(rg_param, ry_param)

    ant.comp_diffusive_angle_snapshots_mass_dep(rg_param, ry_param, w_param, w_param, v_param, N_LIST,
                                                D_LAYER, A_LAYER, PvT_DL_snapshots, checkpoint_collect_container,
                                                mass_retention_threshold, T_fixed_ring_seg, d_tube, domain_radius, D, mass_checkpoint,
                                                center_init_cond, m_init, n_init)
    print("\n\n")
    return pro.process_PvT_DL(PvT_DL_snapshots, v_param, w_param, N_LIST, T_fixed_ring_seg, save_png, show_plt,
                              checkpoint_collect_container, 1, ry_param, rg_param)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


#
def compute_ang_traj_mat(rg_param, ry_param, v_param, w_param, N_LIST, checkpoint_collect_container,
                         mass_retention_threshold=0.01, d_tube=0.0, domain_radius=1.0, D=1.0, mass_checkpoint=10**6, save_png=True, show_plt=False,
                         center_init_cond=True, m_init=0, n_init=0):
    if len(N_LIST) > ry_param:
        raise IndexError(
            f'Too many angular indices supplied for microtubule positions: {len(N_LIST)} > {ry_param} (number of angular positions in domain).'
        )

    for i in range(len(N_LIST)):
        if N_LIST[i] < 0 or N_LIST[i] >= ry_param:
            raise IndexError(
                f'Angular index: {N_LIST[i]} falls outside of the legal index range: [0,{ry_param - 1}) under ry_param={ry_param}')
    N_LIST.sort()

    # checkpoint_collect_container.sort(reverse=True)

    checkpoint_collect_container = [float(item) for item in checkpoint_collect_container]

    # d_tube_max is with respect to the non-overlapping microtubule extraction region implementation.
    d_tube_max = sup.solve_d_rect(domain_radius, rg_param, ry_param, sup.j_max_bef_overlap(ry_param, N_LIST), 0)

    if d_tube < 0 or d_tube > d_tube_max:
        raise ValueError(f"d_tube: {d_tube} is outside of the legal range: [0, {d_tube_max}")

    collection_stamp_enum = len(checkpoint_collect_container)

    diffusion_matrix = np.zeros((collection_stamp_enum, rg_param-1, ry_param), dtype=np.float64)
    # ang_traj_mat = np.zeros((collection_stamp_enum, N_LIST[1]-1, ry_param), dtype=np.float64)
    central_vector = np.zeros(collection_stamp_enum, dtype=np.float64)

    D_LAYER, A_LAYER = sup.initialize_layers(rg_param, ry_param)

    T_param = checkpoint_collect_container[-1] + 0.001

    _validate_off_center_ic(center_init_cond, m_init, n_init, rg_param, ry_param)

    ant.comp_diffusive_angle_snapshots_time_dep_matrix(rg_param, ry_param, w_param, w_param, T_param, v_param, N_LIST,
                                                       D_LAYER, A_LAYER, diffusion_matrix, central_vector, checkpoint_collect_container, d_tube, domain_radius, D, mass_checkpoint,
                                                       center_init_cond, m_init, n_init)

    return diffusion_matrix, central_vector
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



# (****) (****)
def collect_density_rad_depend_mass_dep(rg_param, ry_param, v_param, w_param,  N_LIST, checkpoint_collect_container,
                                        R_fixed_angle=-1, domain_radius=1.0, D=1.0, d_tube=0.0,
                                        mass_retention_threshold=0.01, mass_checkpoint=10**6, save_png=True, show_plt=False,
                                        center_init_cond=True, m_init=0, n_init=0):

    if len(N_LIST) > ry_param:
        raise IndexError(
            f'Too many angular indices supplied for microtubule positions: {len(N_LIST)} > {ry_param} (number of angular positions in domain).'
        )

    for i in range(len(N_LIST)):
        if N_LIST[i] < 0 or N_LIST[i] >= ry_param:
            raise IndexError(
                f'Angular index: {N_LIST[i]} falls outside of the legal index range: [0,{ry_param - 1}) under ry_param={ry_param}')

    N_LIST.sort()

    checkpoint_collect_container.sort(reverse=True)

    checkpoint_collect_container = [float(item) for item in checkpoint_collect_container]

    d_tube_max = sup.solve_d_rect(domain_radius, rg_param, ry_param, sup.j_max_bef_overlap(ry_param, N_LIST), 0)

    if d_tube < 0 or d_tube > d_tube_max:
        raise ValueError(f"d_tube: {d_tube} is outside of the legal range: [0, {d_tube_max}")

    if R_fixed_angle < 0 or R_fixed_angle > ry_param - 1:
        print("R_fixed_angle automatically adjusted to legal range.")
        R_fixed_angle = N_LIST[0]

    collection_stamp_enum = len(checkpoint_collect_container)

    PvR_DL_snapshots = np.zeros((collection_stamp_enum, rg_param + 1), dtype=np.float64)
    RvR_AL_snapshots = np.zeros((collection_stamp_enum, rg_param), dtype=np.float64)

    _validate_off_center_ic(center_init_cond, m_init, n_init, rg_param, ry_param)

    D_LAYER, A_LAYER = sup.initialize_layers(rg_param, ry_param)

    ant.comp_diffusive_rad_snapshots_mass_dep(rg_param, ry_param, w_param, w_param, v_param, N_LIST,
                                              D_LAYER, A_LAYER, R_fixed_angle, PvR_DL_snapshots, RvR_AL_snapshots,
                                              checkpoint_collect_container, mass_retention_threshold, domain_radius, D, mass_checkpoint, d_tube,
                                              center_init_cond, m_init, n_init)

    print("\n\n")
    output_list = pro.process_DvR_results(PvR_DL_snapshots, RvR_AL_snapshots, v_param, w_param, N_LIST, rg_param,
                                          ry_param, R_fixed_angle,
                                          checkpoint_collect_container, save_png, show_plt, 1, domain_radius)

    return output_list
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# (****) (****)
def heatmap_production_mass_dep(rg_param, ry_param, v_param, w_param, N_LIST, checkpoint_collect_container,
                                mass_retention_threshold=0.01, domain_radius=1.0, D=1.0, mass_checkpoint=10**6, d_tube=0.0,
                                heatplot_border=False, heatplot_colorscheme='viridis', save_png=True, show_plt=True, display_extraction=True,
                                center_init_cond=True, m_init=0, n_init=0):

    if len(N_LIST) > ry_param:
        raise IndexError(
            f'Too many angular indices supplied for microtubule positions: {len(N_LIST)} > {ry_param} (number of angular positions in domain).'
        )

    for i in range(len(N_LIST)):
        if N_LIST[i] < 0 or N_LIST[i] >= ry_param:
            raise IndexError(
                f'Angular index: {N_LIST[i]} falls outside of the legal index range: [0,{ry_param - 1}) under ry_param={ry_param}')

    N_LIST.sort()

    d_tube_max = sup.solve_d_rect(domain_radius, rg_param, ry_param, sup.j_max_bef_overlap(ry_param, N_LIST), 0)

    if d_tube < 0 or d_tube > d_tube_max:
        raise ValueError(f"d_tube: {d_tube} is outside of the legal range: [0, {d_tube_max}")

    checkpoint_enum = len(checkpoint_collect_container)

    checkpoint_collect_container.sort(reverse=True)

    checkpoint_collect_container = [float(item) for item in checkpoint_collect_container]

    j_max_list = struct_init.build_j_max_list(rg_param, ry_param, N_LIST, d_tube, domain_radius)

    # Initialize layers
    D_LAYER, A_LAYER = sup.initialize_layers(rg_param, ry_param)

    HM_DL_snapshots = np.zeros((checkpoint_enum, rg_param, ry_param), dtype=np.float64)
    HM_C_snapshots = np.zeros([checkpoint_enum], dtype=np.float64)
    MFPT_snapshots = np.zeros([checkpoint_enum], dtype=np.float64)

    _validate_off_center_ic(center_init_cond, m_init, n_init, rg_param, ry_param)

    ant.comp_diffusive_snapshots_mass_dep(rg_param, ry_param, w_param, w_param, v_param, N_LIST,
                                          D_LAYER, A_LAYER, HM_DL_snapshots, HM_C_snapshots, MFPT_snapshots, checkpoint_collect_container,
                                          domain_radius, D, mass_retention_threshold, mass_checkpoint, d_tube,
                                          center_init_cond, m_init, n_init)

    print("\n\n")
    return pro.process_static_HM_results(HM_DL_snapshots, HM_C_snapshots, MFPT_snapshots, checkpoint_collect_container, heatplot_border, w_param, v_param,
                                         N_LIST, heatplot_colorscheme, save_png, show_plt, j_max_list, display_extraction, 1, rg_param, ry_param)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# ^======================================== Mass dependent computations ========================================^


# v======================================== Time dependent computations ========================================v


def collect_Jrr_mass_sum_over_time(rg_param, ry_param, v_param, w_param, T_param, N_LIST, collection_factor=5, domain_radius=1.0, D=1.0,
                                   mass_checkpoint=10 ** 6, d_tube=0.0, collection_factor_limit=10 ** 3, save_png=True, show_plt=False,
                                   center_init_cond=True, m_init=0, n_init=0):
    if len(N_LIST) > ry_param:
        raise IndexError(
            f'Too many angular indices supplied for microtubule positions: {len(N_LIST)} > {ry_param} (number of angular positions in domain).'
        )

    for i in range(len(N_LIST)):
        if N_LIST[i] < 0 or N_LIST[i] >= ry_param:
            raise IndexError(
                f'Angular index: {N_LIST[i]} falls outside of the legal index range: [0,{ry_param - 1}) under ry_param={ry_param}')
    N_LIST.sort()

    d_tube_max = sup.solve_d_rect(domain_radius, rg_param, ry_param, sup.j_max_bef_overlap(ry_param, N_LIST), 0)

    if d_tube < 0 or d_tube > d_tube_max:
        raise ValueError(f"d_tube: {d_tube} is outside of the legal range: [0, {d_tube_max}")

    if collection_factor < 1 or collection_factor > collection_factor_limit:
        print("MA_collection_factor automatically adjusted to legal range.")
        collection_factor = 100

    K = num.compute_K(rg_param, ry_param, T_param, domain_radius, D)
    relative_k = int(np.floor(K / collection_factor))

    _validate_off_center_ic(center_init_cond, m_init, n_init, rg_param, ry_param)

    D_LAYER, A_LAYER = sup.initialize_layers(rg_param, ry_param)

    Jrr_sum_timeseries = np.zeros([relative_k], dtype=np.float64)
    ant.comp_peak_time_mass_loss(rg_param, ry_param, w_param, w_param, T_param, v_param,
                                 Jrr_sum_timeseries, N_LIST, D_LAYER, A_LAYER, relative_k,
                                 collection_factor, d_tube, domain_radius, D, mass_checkpoint,
                                 center_init_cond, m_init, n_init)

    return pro.process_Jrr_sum_results(Jrr_sum_timeseries, v_param, w_param, N_LIST, rg_param, ry_param, save_png, show_plt, collection_factor, domain_radius, D)


def collect_BC_param_dependence(rg_param, ry_param, v_param, T_param, N_LIST, w_LIST,
                                 checkpoint, T_fixed_ring_seg=0.5, d_tube=0.0, domain_radius=1.0, D=1.0,
                                 mass_checkpoint=10**6, save_png=True, show_plt=False,
                                 center_init_cond=True, m_init=0, n_init=0):
    if len(N_LIST) > ry_param:
        raise IndexError(
            f'Too many angular indices supplied for microtubule positions: {len(N_LIST)} > {ry_param} (number of angular positions in domain).'
        )

    for i in range(len(N_LIST)):
        if N_LIST[i] < 0 or N_LIST[i] >= ry_param:
            raise IndexError(
                f'Angular index: {N_LIST[i]} falls outside of the legal index range: [0,{ry_param - 1}) under ry_param={ry_param}')
    N_LIST.sort()

    if T_fixed_ring_seg < 0 or T_fixed_ring_seg > 1:
        print("T_fixed_ring_seg automatically adjusted to legal range.")
        T_fixed_ring_seg = 0.5

    # d_tube_max is with respect to the non-overlapping microtubule extraction region implementation.
    d_tube_max = sup.solve_d_rect(domain_radius, rg_param, ry_param, sup.j_max_bef_overlap(ry_param, N_LIST), 0)

    if d_tube < 0 or d_tube > d_tube_max:
        raise ValueError(f"d_tube: {d_tube} is outside of the legal range: [0, {d_tube_max}")

    w_LIST_length = len(w_LIST)
    PvT_DL_snapshots = np.zeros((w_LIST_length, ry_param), dtype=np.float64)

    data_dict = {
        'BC_ratios': [],
        'mass_correspondence': []
    }

    # data_dict = {
    #     'RHS': [],
    #     'LHS': [],
    #     'mass_correspondence': []
    # }

    delta_R = num.compute_dRad(rg_param)
    delta_theta = num.compute_dThe(ry_param)
    fixed_ring_seg = int(np.floor(rg_param * T_fixed_ring_seg))

    _validate_off_center_ic(center_init_cond, m_init, n_init, rg_param, ry_param)

    for i in range(w_LIST_length):
        D_LAYER, A_LAYER = sup.initialize_layers(rg_param, ry_param)
        w_param = w_LIST[i]
        b_param = w_param
        a_param = b_param/(delta_R * delta_theta * (fixed_ring_seg+1))
        # a_param = w_param
        BC_ratio_snapshot, mass_retained = ant.comp_BC_analysis_snapshots_time_dep(rg_param, ry_param, a_param, b_param, T_param, v_param, N_LIST, D_LAYER,
                                                                                      A_LAYER, checkpoint, T_fixed_ring_seg=T_fixed_ring_seg, d_tube=d_tube, domain_radius=domain_radius,
                                                                                      D=D, mass_checkpoint=mass_checkpoint,
                                                                                      center_init_cond=center_init_cond, m_init=m_init, n_init=n_init)
        # RHS, LHS, mass_retained = ant.comp_BC_analysis_snapshots_time_dep(rg_param, ry_param, w_param, w_param, T_param, v_param, N_LIST, D_LAYER, A_LAYER, PvT_DL_snapshots, i, checkpoint)
        data_dict['BC_ratios'].append(BC_ratio_snapshot)
        # data_dict['RHS'].append(RHS)
        # data_dict['LHS'].append(LHS)
        data_dict['mass_correspondence'].append(mass_retained)

    data_dict["w"] = w_LIST
    print("\n\n")

    return pro.process_BC_analysis_DL(PvT_DL_snapshots, data_dict, v_param, w_LIST, N_LIST, T_fixed_ring_seg, save_png, show_plt, checkpoint, 2, ry_param, rg_param)


def collect_BC_param_dependence_grid_size(v_param, T_param, w_param, N_amount,
                                          checkpoint, grid_list, T_fixed_ring_seg=0.5, d_tube=0.0, domain_radius=1.0, D=1.0, mass_checkpoint=10**6, save_png=True, show_plt=False):

    if T_fixed_ring_seg < 0 or T_fixed_ring_seg > 1:
        print("T_fixed_ring_seg automatically adjusted to legal range.")
        T_fixed_ring_seg = 0.5

    grid_list_length = len(grid_list)

    data_dict = {
        'BC_ratios': [],
        'mass_correspondence': []
    }

    b_param = w_param
    a = b_param

    for i in range(grid_list_length):
        rg_param = grid_list[i]
        ry_param = rg_param

        drad = num.compute_dRad(rg_param, domain_radius)
        dthe = num.compute_dThe(ry_param)
        fixed_ring_seg = int(np.floor(rg_param * T_fixed_ring_seg))
        r = (fixed_ring_seg + 1) * drad
        a_param = b_param/(r * dthe)

        D_LAYER, A_LAYER = sup.initialize_layers(rg_param, ry_param)

        N_LIST = np.linspace(0, ry_param - (ry_param / N_amount), N_amount, dtype=int)
        BC_ratio_snapshot, mass_retained = ant.comp_BC_analysis_snapshots_time_dep_v2(rg_param, ry_param, a_param, b_param, T_param, v_param, N_LIST,
                                                                                      D_LAYER, A_LAYER, checkpoint, T_fixed_ring_seg, d_tube, domain_radius, D, mass_checkpoint)
        data_dict['BC_ratios'].append(BC_ratio_snapshot)
        data_dict['mass_correspondence'].append(mass_retained)

    data_dict["GS"] = grid_list
    print("\n\n")
    return pro.process_BC_analysis_grid_size(data_dict, grid_list, v_param, w_param, N_amount, T_fixed_ring_seg, save_png, show_plt, checkpoint)


# (****) (****)
def solve_mfpt_time_(rg_param, ry_param, N_LIST, v_param, w_param, T_param, domain_radius=1.0, D=1.0, mass_checkpoint=10 ** 6, d_tube=0.0,
                     center_init_cond=True, m_init=0, n_init=0):
    if len(N_LIST) > ry_param:
        raise IndexError(
            f'Too many angular indices supplied for microtubule positions: {len(N_LIST)} > {ry_param} (number of angular positions in domain).'
        )

    for i in range(len(N_LIST)):
        if N_LIST[i] < 0 or N_LIST[i] >= ry_param:
            raise IndexError(
                f'Angular index: {N_LIST[i]} falls outside of the legal index range: [0,{ry_param - 1}) under ry_param={ry_param}')

    N_LIST.sort()

    d_tube_max = sup.solve_d_rect(domain_radius, rg_param, ry_param, sup.j_max_bef_overlap(ry_param, N_LIST), 0)

    if d_tube < 0 or d_tube > d_tube_max:
        raise ValueError(f"d_tube: {d_tube} is outside of the legal range: [0, {d_tube_max}")

    _validate_off_center_ic(center_init_cond, m_init, n_init, rg_param, ry_param)

    D_LAYER, A_LAYER = sup.initialize_layers(rg_param, ry_param)
    MFPT = mfpt_comp.comp_mfpt_by_time(rg_param, ry_param, w_param, w_param, v_param, N_LIST,
                                       D_LAYER, A_LAYER, T_param, mass_checkpoint, domain_radius, D, d_tube,
                                       center_init_cond, m_init, n_init)
    print("\n\n")
    return MFPT
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# (****) (****)
def collect_MFPT_snapshots_time_dep(rg_param, ry_param, N_LIST, v_param, w_param, T_param, checkpoint_collect_container, domain_radius=1.0, D=1.0,
                                    mass_checkpoint=10 ** 6, d_tube=0.0, save_png=True, show_plt=False,
                                    center_init_cond=True, m_init=0, n_init=0):
    if len(N_LIST) > ry_param:
        raise IndexError(
            f'Too many angular indices supplied for microtubule positions: {len(N_LIST)} > {ry_param} (number of angular positions in domain).'
        )

    for i in range(len(N_LIST)):
        if N_LIST[i] < 0 or N_LIST[i] >= ry_param:
            raise IndexError(
                f'Angular index: {N_LIST[i]} falls outside of the legal index range: [0,{ry_param - 1}) under ry_param={ry_param}')
    N_LIST.sort()

    checkpoint_collect_container = [float(item) for item in checkpoint_collect_container]

    for t in range(len(checkpoint_collect_container)):
        if checkpoint_collect_container[t] < 0.0 or checkpoint_collect_container[t] > T_param:
            raise ValueError(
                f"Timestamp-point: {checkpoint_collect_container[t]} falls outside of the legal timestamp-point range: [0, {T_param} (T_param) ] (T_param = your input solution time duration)")
        dT = num.compute_dT(rg_param, ry_param, domain_radius=domain_radius, D=D)
        T_param += dT
    checkpoint_collect_container.sort()

    checkpoint_enum = len(checkpoint_collect_container)

    d_tube_max = sup.solve_d_rect(domain_radius, rg_param, ry_param, sup.j_max_bef_overlap(ry_param, N_LIST), 0)

    if d_tube < 0 or d_tube > d_tube_max:
        raise ValueError(f"d_tube: {d_tube} is outside of the legal range: [0, {d_tube_max}")

    T_param = float(T_param)

    _validate_off_center_ic(center_init_cond, m_init, n_init, rg_param, ry_param)

    D_LAYER, A_LAYER = sup.initialize_layers(rg_param, ry_param)
    MFPT_snapshots = np.zeros([checkpoint_enum], dtype=np.float64)

    mfpt_comp.comp_mfpt_by_time_points_time_dep(rg_param, ry_param, w_param, w_param, v_param, N_LIST,
                                                D_LAYER, A_LAYER, checkpoint_collect_container, MFPT_snapshots,
                                                T_param, mass_checkpoint, domain_radius, D, d_tube,
                                                center_init_cond, m_init, n_init)
    print("\n\n")

    return pro.process_MFPT_results(MFPT_snapshots, checkpoint_collect_container, 2, rg_param, ry_param, w_param, v_param, N_LIST,
                                    save_png, show_plt)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# (****) (****)
def collect_phi_ang_dep_time_dep(rg_param, ry_param, v_param, w_param, T_param, N_LIST,
                                 checkpoint_collect_container, T_fixed_ring_seg=0.5, d_tube=0.0, domain_radius=1.0, D=1.0,
                                 mass_checkpoint=10**6, save_png=True, show_plt=False,
                                 center_init_cond=True, m_init=0, n_init=0):
    if len(N_LIST) > ry_param:
        raise IndexError(
            f'Too many angular indices supplied for microtubule positions: {len(N_LIST)} > {ry_param} (number of angular positions in domain).'
        )

    for i in range(len(N_LIST)):
        if N_LIST[i] < 0 or N_LIST[i] >= ry_param:
            raise IndexError(
                f'Angular index: {N_LIST[ i ]} falls outside of the legal index range: [0,{ry_param - 1}) under ry_param={ry_param}')
    N_LIST.sort()

    if T_fixed_ring_seg < 0 or T_fixed_ring_seg > 1:
        print("T_fixed_ring_seg automatically adjusted to legal range.")
        T_fixed_ring_seg = 0.5

    checkpoint_collect_container = [float(item) for item in checkpoint_collect_container]

    for t in range(len(checkpoint_collect_container)):
        if checkpoint_collect_container[t] < 0.0 or checkpoint_collect_container[t] > T_param:
            raise ValueError(
                f"Timestamp-point: {checkpoint_collect_container[t]} falls outside of the legal timestamp-point range: [0, {T_param} (T_param) ] (T_param = your input solution time duration)")
        dT = num.compute_dT(rg_param, ry_param, domain_radius=domain_radius, D=D)
        T_param += dT
    checkpoint_collect_container.sort()

    # d_tube_max is with respect to the non-overlapping microtubule extraction region implementation.
    d_tube_max = sup.solve_d_rect(domain_radius, rg_param, ry_param, sup.j_max_bef_overlap(ry_param, N_LIST), 0)

    if d_tube < 0 or d_tube > d_tube_max:
        raise ValueError(f"d_tube: {d_tube} is outside of the legal range: [0, {d_tube_max}")

    collection_stamp_enum = len(checkpoint_collect_container)

    PvT_DL_snapshots = np.zeros((collection_stamp_enum, ry_param), dtype=np.float64)

    D_LAYER, A_LAYER = sup.initialize_layers(rg_param, ry_param)

    _validate_off_center_ic(center_init_cond, m_init, n_init, rg_param, ry_param)

    ant.comp_diffusive_angle_snapshots_time_dep(rg_param, ry_param, w_param, w_param, T_param, v_param, N_LIST,
                                                D_LAYER, A_LAYER, PvT_DL_snapshots,
                                                checkpoint_collect_container, T_fixed_ring_seg, d_tube, domain_radius, D, mass_checkpoint,
                                                center_init_cond, m_init, n_init)
    print("\n\n")

    return pro.process_PvT_DL(PvT_DL_snapshots, v_param, w_param, N_LIST, T_fixed_ring_seg, save_png, show_plt,
                              checkpoint_collect_container, 2, ry_param, rg_param)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# (****) (****)
def collect_density_rad_depend_time_dep(rg_param, ry_param, v_param, w_param, T_param, N_LIST, checkpoint_collect_container,
                                        R_fixed_angle=-1, domain_radius=1.0, D=1.0, d_tube=0.0,
                                        mass_checkpoint=10**6, save_png=True, show_plt=False,
                                        center_init_cond=True, m_init=0, n_init=0):

    if len(N_LIST) > ry_param:
        raise IndexError(
            f'Too many angular indices supplied for microtubule positions: {len(N_LIST)} > {ry_param} (number of angular positions in domain).'
        )

    for i in range(len(N_LIST)):
        if N_LIST[i] < 0 or N_LIST[i] >= ry_param:
            raise IndexError(
                f'Angular index: {N_LIST[i]} falls outside of the legal index range: [0,{ry_param - 1}) under ry_param={ry_param}')

    N_LIST.sort()

    for t in range(len(checkpoint_collect_container)):
        if checkpoint_collect_container[t] < 0.0 or checkpoint_collect_container[t] > T_param:
            raise ValueError(
                f"Timestamp-point: {checkpoint_collect_container[t]} falls outside of the legal timestamp-point range: [0, {T_param} (T_param) ] (T_param = your input solution time duration)")
        dT = num.compute_dT(rg_param, ry_param, domain_radius=domain_radius, D=D)
        T_param += dT
    checkpoint_collect_container.sort()

    checkpoint_collect_container = [float(item) for item in checkpoint_collect_container]

    d_tube_max = sup.solve_d_rect(domain_radius, rg_param, ry_param, sup.j_max_bef_overlap(ry_param, N_LIST), 0)

    if d_tube < 0 or d_tube > d_tube_max:
        raise ValueError(f"d_tube: {d_tube} is outside of the legal range: [0, {d_tube_max}")

    if R_fixed_angle < 0 or R_fixed_angle > ry_param - 1:
        print("R_fixed_angle automatically adjusted to legal range.")
        R_fixed_angle = N_LIST[0]

    collection_stamp_enum = len(checkpoint_collect_container)

    PvR_DL_snapshots = np.zeros((collection_stamp_enum, rg_param + 1), dtype=np.float64)
    RvR_AL_snapshots = np.zeros((collection_stamp_enum, rg_param), dtype=np.float64)

    D_LAYER, A_LAYER = sup.initialize_layers(rg_param, ry_param)

    _validate_off_center_ic(center_init_cond, m_init, n_init, rg_param, ry_param)

    ant.comp_diffusive_rad_snapshots_time_dep(rg_param, ry_param, w_param, w_param, v_param, T_param, N_LIST,
                                              D_LAYER, A_LAYER, R_fixed_angle, PvR_DL_snapshots, RvR_AL_snapshots,
                                              checkpoint_collect_container, domain_radius, D, mass_checkpoint, d_tube,
                                              center_init_cond, m_init, n_init)
    print("\n\n")

    output_list = pro.process_DvR_results(PvR_DL_snapshots, RvR_AL_snapshots, v_param, w_param, N_LIST, rg_param,
                                          ry_param, R_fixed_angle,
                                          checkpoint_collect_container, save_png, show_plt, 2, domain_radius)

    return output_list
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# (****) (****)
def heatmap_production_time_dep(rg_param, ry_param, v_param, w_param, N_LIST, T_param, checkpoint_collect_container,
                                domain_radius=1.0, D=1.0, mass_checkpoint=10**6, d_tube=0.0,
                                heatplot_border=False, heatplot_colorscheme='viridis', save_png=True, show_plt=True, display_extraction=True,
                                center_init_cond=True, m_init=0, n_init=0):

    if len(N_LIST) > ry_param:
        raise IndexError(
            f'Too many angular indices supplied for microtubule positions: {len(N_LIST)} > {ry_param} (number of angular positions in domain).'
        )

    for i in range(len(N_LIST)):
        if N_LIST[i] < 0 or N_LIST[i] >= ry_param:
            raise IndexError(
                f'Angular index: {N_LIST[i]} falls outside of the legal index range: [0,{ry_param - 1}) under ry_param={ry_param}')

    N_LIST.sort()

    d_tube_max = sup.solve_d_rect(domain_radius, rg_param, ry_param, sup.j_max_bef_overlap(ry_param, N_LIST), 0)

    if d_tube < 0 or d_tube > d_tube_max:
        raise ValueError(f"d_tube: {d_tube} is outside of the legal range: [0, {d_tube_max}")

    checkpoint_enum = len(checkpoint_collect_container)

    for t in range(len(checkpoint_collect_container)):
        if checkpoint_collect_container[t] < 0.0 or checkpoint_collect_container[t] > T_param:
            raise ValueError(
                f"Timestamp-point: {checkpoint_collect_container[t]} falls outside of the legal timestamp-point range: [0, {T_param} (T_param) ] (T_param = your input solution time duration)")
        dT = num.compute_dT(rg_param, ry_param, domain_radius=domain_radius, D=D)
        T_param += dT
    checkpoint_collect_container.sort()

    checkpoint_collect_container = [float(item) for item in checkpoint_collect_container]

    j_max_list = struct_init.build_j_max_list(rg_param, ry_param, N_LIST, d_tube, domain_radius)

    # Initialize layers
    D_LAYER, A_LAYER = sup.initialize_layers(rg_param, ry_param)

    HM_DL_snapshots = np.zeros((checkpoint_enum, rg_param, ry_param), dtype=np.float64)
    HM_C_snapshots = np.zeros([checkpoint_enum], dtype=np.float64)
    MFPT_snapshots = np.zeros([checkpoint_enum], dtype=np.float64)

    _validate_off_center_ic(center_init_cond, m_init, n_init, rg_param, ry_param)

    ant.comp_diffusive_snapshots_time_dep(rg_param, ry_param, w_param, w_param, v_param, T_param, N_LIST,
                                          D_LAYER, A_LAYER, HM_DL_snapshots, HM_C_snapshots, MFPT_snapshots,
                                          checkpoint_collect_container, domain_radius, D, mass_checkpoint, d_tube,
                                          center_init_cond, m_init, n_init)
    print("\n\n")

    return pro.process_static_HM_results(HM_DL_snapshots, HM_C_snapshots, MFPT_snapshots, checkpoint_collect_container, heatplot_border, w_param, v_param,
                                         N_LIST, heatplot_colorscheme, save_png, show_plt, j_max_list, display_extraction, 2, rg_param, ry_param)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# (****) (****)
def launch_super_comp_I(rg_param, ry_param, v_param, w_param, T_param, N_LIST, d_tube=0.0, Timestamp_List=None,
                        MA_collection_factor=5, MA_collection_factor_limit=10 ** 3,
                        D=1.0, domain_radius=1.0, mass_checkpoint=10 ** 6, T_fixed_ring_seg=0.5, R_fixed_angle=-1,
                        save_png=True, show_plt=False, heat_plot_border=False, heatplot_colorscheme='viridis',
                        display_extraction=True,
                        center_init_cond=True, m_init=0, n_init=0):
    if len(N_LIST) > ry_param:
        raise IndexError(
            f'Too many angular indices supplied for microtubule positions: {len(N_LIST)} > {ry_param} (number of angular positions in domain).'
        )

    for i in range(len(N_LIST)):
        if N_LIST[i] < 0 or N_LIST[i] >= ry_param:
            raise IndexError(
                f'Angular index: {N_LIST[i]} falls outside of the legal index range: [0,{ry_param - 1}) under ry_param={ry_param}')
    N_LIST.sort()

    if T_fixed_ring_seg < 0 or T_fixed_ring_seg > 1:
        print("T_fixed_ring_seg automatically adjusted to legal range.")
        T_fixed_ring_seg = 0.5

    if R_fixed_angle < 0 or R_fixed_angle > ry_param - 1:
        print("R_fixed_angle automatically adjusted to legal range.")
        R_fixed_angle = N_LIST[0]

    if MA_collection_factor < 1 or MA_collection_factor > MA_collection_factor_limit:
        print("MA_collection_factor automatically adjusted to legal range.")
        MA_collection_factor = 100

    d_tube_max = sup.solve_d_rect(domain_radius, rg_param, ry_param, sup.j_max_bef_overlap(ry_param, N_LIST), 0)

    if d_tube < 0 or d_tube > d_tube_max:
        raise ValueError(f"d_tube: {d_tube} is outside of the legal range: [0, {d_tube_max}")

    T_param = float(T_param)

    if Timestamp_List is None:
        # Default timestamps
        Timestamp_List = [T_param * 0.25, T_param * 0.5, T_param * 0.75, T_param]
    else:
        Timestamp_List.sort()

    Timestamp_List = [float(item) for item in Timestamp_List]

    for t in range(len(Timestamp_List)):
        if Timestamp_List[t] < 0.0 or Timestamp_List[t] > T_param:
            raise ValueError(
                f"Timestamp-point: {Timestamp_List[t]} falls outside of the legal timestamp-point range: [0, {T_param} (T_param) ] (T_param = your input solution time duration)")

    dT = num.compute_dT(rg_param, ry_param, domain_radius=domain_radius, D=D)
    T_param += dT
    Timestamp_enum = len(Timestamp_List)

    # Initialize mass analysis time series collection containers

    K = T_param / dT
    relative_k = int(np.floor(K / MA_collection_factor))

    MA_DL_timeseries = np.zeros([relative_k], dtype=np.float64)
    MA_AL_timeseries = np.zeros([relative_k], dtype=np.float64)
    MA_TM_timeseries = np.zeros([relative_k], dtype=np.float64)
    MA_ALoT_timeseries = np.zeros([relative_k], dtype=np.float64)
    MA_ALoI_timeseries = np.zeros([relative_k], dtype=np.float64)

    # Initialize Phi v. Theta diffusive layer snapshot container

    PvT_DL_snapshots = np.zeros((Timestamp_enum, ry_param), dtype=np.float64)

    # Initialize Density v. Radius snapshot containers

    PvR_DL_snapshots = np.zeros((Timestamp_enum, rg_param + 1), dtype=np.float64)
    RvR_AL_snapshots = np.zeros((Timestamp_enum, rg_param), dtype=np.float64)

    # Prepare for heatmap collection

    j_max_list = struct_init.build_j_max_list(rg_param, ry_param, N_LIST, d_tube, domain_radius)

    HM_DL_snapshots = np.zeros((Timestamp_enum, rg_param, ry_param), dtype=np.float64)
    HM_C_snapshots = np.zeros([Timestamp_enum], dtype=np.float64)
    MFPT_snapshots = np.zeros([Timestamp_enum], dtype=np.float64)

    _validate_off_center_ic(center_init_cond, m_init, n_init, rg_param, ry_param)

    # Initialize layers
    D_LAYER, A_LAYER = sup.initialize_layers(rg_param, ry_param)

    # Release the Kraken.. (execute super_comp_type_I)
    super.super_comp_type_I(rg_param, ry_param, w_param, w_param, T_param, v_param, N_LIST, D_LAYER, A_LAYER,
                            Timestamp_List,
                            HM_DL_snapshots, HM_C_snapshots, PvT_DL_snapshots, T_fixed_ring_seg, MA_DL_timeseries,
                            MA_AL_timeseries,
                            MA_ALoI_timeseries, MA_ALoT_timeseries, MA_TM_timeseries, MA_collection_factor, relative_k,
                            PvR_DL_snapshots,
                            RvR_AL_snapshots, R_fixed_angle, MFPT_snapshots, d_tube, D, domain_radius, mass_checkpoint,
                            center_init_cond, m_init, n_init)

    # Process results: Produce CSVs, PNGs (plots and heatmaps) (log results to filepath_log<timestamp>.txt)
    # Parameter chart (relative to the computation) is also included in the result
    # Desired output: a package containing all results categorized by type

    # Processing MFPT results

    MFPT_results = pro.process_MFPT_results(MFPT_snapshots, Timestamp_List, 2, rg_param, ry_param, w_param, v_param, N_LIST, save_png, show_plt)
    print("\n\n")

    # Processing results for Mass-Analysis

    # Diffusive mass analysis
    MA_results = pro.process_MA_results(MA_DL_timeseries, MA_AL_timeseries, MA_TM_timeseries, MA_ALoT_timeseries,
                                        MA_ALoI_timeseries, v_param, w_param, w_param, N_LIST, T_param, rg_param, ry_param,
                                        save_png, show_plt, MA_collection_factor, domain_radius, D)
    print("\n\n")
    # Processing results for Phi v. Theta

    PvT_DL_results = pro.process_PvT_DL(PvT_DL_snapshots, v_param, w_param, N_LIST, T_fixed_ring_seg, save_png,
                                        show_plt, Timestamp_List, 2, ry_param, rg_param)

    # Processing results for Phi v. Radius & Rho v. Radius

    DvR_results = pro.process_DvR_results(PvR_DL_snapshots, RvR_AL_snapshots, v_param, w_param, N_LIST, rg_param,
                                          ry_param, R_fixed_angle, Timestamp_List, save_png, show_plt, 2, domain_radius)
    print("\n\n")
    # Processing static heat-plots

    static_HM_results = pro.process_static_HM_results(HM_DL_snapshots, HM_C_snapshots, MFPT_snapshots, Timestamp_List,
                                                      heat_plot_border, w_param, v_param, N_LIST, heatplot_colorscheme,
                                                      save_png, show_plt, j_max_list, display_extraction, 2, rg_param, ry_param)
    print("\n\n")
    print("Successfully completed super-function. View results in intracellular_transport/data_output.")

    output_list = MFPT_results + MA_results + PvT_DL_results + DvR_results + static_HM_results
    return output_list
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# ^======================================== Time dependent computations ========================================^


# (****) (****)
def output_time_until_mass_depletion(rg_param, ry_param, N_LIST, v_param, w_param, domain_radius=1.0, D=1.0,
                                     d_tube=0, mass_threshold=0.01,
                                     center_init_cond=True, m_init=0, n_init=0):
    D_LAYER, A_LAYER = sup.initialize_layers(rg_param, ry_param)

    if len(N_LIST) > ry_param:
        raise IndexError(
            f'Too many microtubules requested: {len(N_LIST)}, within domain of {ry_param} angular rays.')

    for i in range(len(N_LIST)):
        if N_LIST[i] < 0 or N_LIST[i] >= ry_param:
            raise IndexError(f'Angle {N_LIST[i]} is out of bounds, your range should be [0, {ry_param - 1}]')
    N_LIST.sort()

    _validate_off_center_ic(center_init_cond, m_init, n_init, rg_param, ry_param)

    duration = ant.comp_until_mass_depletion(rg_param, ry_param, w_param, w_param,
                                             v_param, N_LIST, D_LAYER, A_LAYER, domain_radius, D, mass_threshold,
                                             d_tube, center_init_cond, m_init, n_init)
    return duration
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# The mass analysis keeps the time-stepper as its default engine; "modal" (the
# engines are listed with the characteristic time's, CHAR_TIME_ENGINES) fills
# the same series from mass_analysis_modal's eigen-expansion instead.
MASS_ANALYSIS_ENGINE_DEFAULT = "time_stepper"


# (****) (****)
def collect_mass_analysis(rg_param, ry_param, v_param, a_param, b_param, T_param, N_LIST, MA_collection_factor=5,
                          domain_radius=1.0, D=1.0,
                          mass_checkpoint=10 ** 6, d_tube=0.0, MA_collection_factor_limit=10 ** 3, save_png=True,
                          show_plt=False,
                          center_init_cond=True, m_init=0, n_init=0,
                          device=DEVICE_CPU, engine=MASS_ANALYSIS_ENGINE_DEFAULT,
                          compute_psd=False):
    """Mass on each layer over time, for one (a, b) pair.

    ``a_param`` is the switch rate onto the diffusive layer and ``b_param`` the
    rate onto the advective layer, matching the naming the solver and the
    (a, b) grid kernel already use. Passing the same value for both reproduces
    what this function did when it took a single ``w_param``, including the
    names of the files it writes.

    ``engine`` picks how the series are obtained: "time_stepper" (the default)
    runs the solve to T on ``device``; "modal" evaluates the eigen-expansion of
    the same one-step operator at the same samples, at a cost that barely grows
    with T. The two agree in total mass to ~1e-18 G^4.45
    (results/2026-11-04_modal-mass-analysis-validation/). ``device`` and
    ``mass_checkpoint`` apply to the time-stepper only.

    ``compute_psd`` also writes |F[y'']|^2 of y = ln(total mass), zoomed on its
    peak, with the peak and FWHM (computational_tools/y2_spectrum.py), under
    ``mass_analysis_results/psd_y2/``.
    """
    engine = str(engine or MASS_ANALYSIS_ENGINE_DEFAULT).strip().lower()
    if engine not in CHAR_TIME_ENGINES:
        raise ValueError(f"unknown engine {engine!r}; expected one of "
                         f"{', '.join(CHAR_TIME_ENGINES)}")

    if len(N_LIST) > ry_param:
        raise IndexError(
            f'Too many angular indices supplied for microtubule positions: {len(N_LIST)} > {ry_param} (number of angular positions in domain).'
        )

    for i in range(len(N_LIST)):
        if N_LIST[i] < 0 or N_LIST[i] >= ry_param:
            raise IndexError(
                f'Angular index: {N_LIST[i]} falls outside of the legal index range: [0,{ry_param - 1}) under ry_param={ry_param}')
    N_LIST.sort()

    d_tube_max = sup.solve_d_rect(domain_radius, rg_param, ry_param, sup.j_max_bef_overlap(ry_param, N_LIST), 0)

    if d_tube < 0 or d_tube > d_tube_max:
        raise ValueError(f"d_tube: {d_tube} is outside of the legal range: [0, {d_tube_max}")

    if MA_collection_factor < 1 or MA_collection_factor > MA_collection_factor_limit:
        print("MA_collection_factor automatically adjusted to legal range.")
        MA_collection_factor = 100

    K = num.compute_K(rg_param, ry_param, T_param, domain_radius, D)
    relative_k = int(np.floor(K / MA_collection_factor))

    D_LAYER, A_LAYER = sup.initialize_layers(rg_param, ry_param)

    MA_DL_timeseries = np.zeros([relative_k], dtype=np.float64)
    MA_AL_timeseries = np.zeros([relative_k], dtype=np.float64)
    MA_ALoI_timeseries = np.zeros([relative_k], dtype=np.float64)
    MA_ALoT_timeseries = np.zeros([relative_k], dtype=np.float64)
    MA_TM_timeseries = np.zeros([relative_k], dtype=np.float64)

    # Canonical slot order shared by the CPU driver's parameter list and the
    # GPU kernel's series[] indices.
    series = [MA_DL_timeseries, MA_AL_timeseries, MA_ALoI_timeseries,
              MA_ALoT_timeseries, MA_TM_timeseries]

    _validate_off_center_ic(center_init_cond, m_init, n_init, rg_param, ry_param)

    if engine == CHAR_TIME_ENGINE_MODAL:
        modes = mam.comp_mass_analysis_modal_expansion(
            rg_param, ry_param, a_param, b_param, v_param, T_param, N_LIST, *series,
            MA_collection_factor, relative_k, d_tube, domain_radius, D,
            center_init_cond, m_init, n_init)
        print(f"engine=modal: {modes['n_reduced']:,} modes (state {modes['n_state']:,}), "
              f"cond(V) = {modes['cond_V']:.1e}, build {modes['wall_build']:.1f} s, "
              f"eigensolve {modes['wall_eig']:.1f} s, {relative_k:,} samples "
              f"{modes['wall_evaluate']:.1f} s")
    else:
        _solve_mass_analysis(device, series, rg_param, ry_param, a_param, b_param,
                             v_param, T_param, N_LIST, D_LAYER, A_LAYER,
                             MA_collection_factor, relative_k, d_tube,
                             domain_radius, D, mass_checkpoint, center_init_cond,
                             m_init, n_init)

    outputs = pro.process_MA_results(MA_DL_timeseries, MA_AL_timeseries, MA_TM_timeseries, MA_ALoT_timeseries,
                                     MA_ALoI_timeseries,
                                     v_param, a_param, b_param, N_LIST, T_param, rg_param, ry_param, save_png, show_plt, MA_collection_factor, domain_radius, D)
    if compute_psd:
        sample_dt = MA_collection_factor * num.compute_dT(rg_param, ry_param, domain_radius, D)
        outputs.append(_mass_analysis_psd(MA_TM_timeseries, sample_dt, v_param, a_param, b_param,
                                          N_LIST, T_param, rg_param, ry_param, engine))
    return outputs


def _mass_analysis_psd(total_mass, sample_dt, v_param, a_param, b_param, N_LIST, T_param,
                       rg_param, ry_param, engine):
    """|F[y'']|^2 of y = ln(total mass) about its peak: a PNG and a one-row CSV.

    y is read every s samples, H = s * sample_dt ~ CHAR_TIME_DIFF_SPACING, the
    spacing the t* criterion differences at. Returns the CSV path.
    """
    s, H = y2s.spacing(sample_dt, CHAR_TIME_DIFF_SPACING)
    mass = np.asarray(total_mass[::s], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        spec = y2s.spectrum(np.log(np.where(mass > 0, mass, np.nan)), H)

    directory = _create_unique_timestamp_dir(fp.mass_analysis_psd)
    stem = (f"v={v_param}_{pro.switch_rate_tag(a_param, b_param)}_N={len(N_LIST)}"
            f"_Domain={rg_param}x{ry_param}")
    y2s.save_figure(os.path.join(directory, f"MA_psd_y2_{stem}.png"), [spec],
                    [f"v={v_param:g}, a={a_param:g}, b={b_param:g}, T={T_param:g}"], 1,
                    rf"$|\mathcal{{F}}[y'']|^2$ about its peak -- {rg_param}x{ry_param}, "
                    f"N={len(N_LIST)}, engine={engine}")
    csv = os.path.join(directory, f"MA_psd_y2_{stem}.csv")
    pd.DataFrame([dict(y2s.summary(spec), H=spec["H"], d_omega=spec["d_omega"],
                       pad_factor=y2s.PAD_FACTOR)]).to_csv(csv, index=False)
    print(f"PSD of y'': omega_peak = {spec['omega_peak']:.6g}, peak = {spec['psd_peak']:.6g}, "
          f"FWHM = {spec['fwhm']:.6g} ({spec['fwhm_side']}); saved to {directory}")
    return csv


# (****) Create a timestamped output subdirectory that never overwrites an existing one (****)
def _create_unique_timestamp_dir(parent_directory, timestamp_format="%Y-%m-%d_%H-%M"):
    """
    Create and return a subdirectory of parent_directory named after the time of
    its creation, resolved up to the minute (e.g. 2026-08-05_15-54).

    The parent directory is created on demand and seeded with a .gitkeep so the
    (otherwise empty) output directory is retained under version control.

    Edge case: a directory stamped with the exact same minute is never replaced.
    A distinguishing marker is appended instead -- _(2), _(3), ... -- until an
    unused name is found.
    """
    os.makedirs(parent_directory, exist_ok=True)

    gitkeep_location = os.path.join(parent_directory, '.gitkeep')
    if not os.path.exists(gitkeep_location):
        open(gitkeep_location, 'a').close()

    timestamp = datetime.now().strftime(timestamp_format)
    directory_path = os.path.join(parent_directory, timestamp)

    marker = 2
    while True:
        try:
            os.makedirs(directory_path, exist_ok=False)
            return directory_path
        except FileExistsError:
            directory_path = os.path.join(parent_directory, f'{timestamp}_({marker})')
            marker += 1
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def _setup_checkpoint(tag, n_samples, extend_from=None, **fp_params):
    """Prepare checkpointing for one solve, if it is enabled for this run.

    ``n_samples`` is the timeseries length; it is named distinctly from the
    fingerprint's own ``relative_k`` so the two cannot collide in **fp_params.

    Enabled by the ITCM_CHECKPOINT_DIR environment variable, which the job
    worker sets per job. Left unset -- direct scripts, the desktop GUI -- this
    returns no checkpoint and ordinary in-RAM timeseries, so nothing about the
    existing paths changes.

    ``extend_from`` continues an earlier, shorter run of the same point from
    its saved final state instead of starting at t = 0: a job id recorded in
    comp-state-ledger, or a checkpoints directory. Only T may differ from the
    earlier run (checkpointing.prepare_extension refuses anything else). See
    comp-state-ledger/README.md.

    Returns ``(checkpoint_or_None, five_timeseries)``.
    """
    import os
    from computational_tools import checkpointing as chk

    root = os.environ.get("ITCM_CHECKPOINT_DIR")
    if not root:
        if extend_from:
            raise ValueError(
                "extend_from needs checkpointing, which is on for jobs run "
                "through the compute server (or with ITCM_CHECKPOINT_DIR set)")
        return None, chk.allocate_timeseries(n_samples, 5)

    directory = os.path.join(root, tag)
    digest, params = chk.fingerprint(**fp_params)
    if extend_from and not os.path.exists(os.path.join(directory, chk.STATE_FILE)):
        from computational_tools import state_ledger

        src = state_ledger.resolve(extend_from, tag)
        meta = chk.prepare_extension(src, directory, digest, params, n_samples)
        print(f"extending {tag} from its saved state at t = {meta['t']:.6g} "
              f"(step {meta['k']:,}) in {src}")
    # Resume only if this exact configuration was already under way here.
    resume = os.path.exists(os.path.join(directory, chk.STATE_FILE))
    series = chk.allocate_timeseries(n_samples, 5, directory=directory,
                                     resume=resume)
    return chk.Checkpoint(directory, digest, params), series


def _solve_mass_analysis(device, series, rg_param, ry_param, a_param, b_param,
                         v_param, T_param, N_LIST, D_LAYER, A_LAYER,
                         MA_collection_factor, relative_k, d_tube,
                         domain_radius, D, mass_checkpoint, center_init_cond,
                         m_init, n_init, cp=None):
    """Run the time-stepping solve on the requested device.

    ``series`` is the five timeseries in canonical order (DL, AL, ALoI, ALoT,
    TM); both devices write those same slots.

    The CPU path is the reference implementation and the authority on
    correctness (docs/GPU_PLAN.md).  The GPU path agrees with it to ~1e-14
    relative rather than bit-identically, because its mass and centre
    reductions are parallel tree reductions where the CPU's are sequential
    accumulations, and floating-point addition is not associative.  That
    tolerance is an accepted trade for roughly 9x at 96^2 rising to 16x at
    160^2; it is asserted by tests/test_gpu_agreement.py.

    ``device`` semantics differ deliberately in how they fail:

      * ``cpu``  -- never touches CUDA.
      * ``gpu``  -- raises if the GPU cannot run this configuration.  An
                    explicit request for the GPU that silently ran on the CPU
                    would misreport what produced the numbers.
      * ``auto`` -- prefers the GPU, falls back to the CPU with a printed
                    reason.  Suitable for batch submission where either device
                    is acceptable.
    """
    dev = (device or DEVICE_CPU).strip().lower()
    if dev not in DEVICES:
        raise ValueError(
            f"unknown device {device!r}; expected one of {', '.join(DEVICES)}")

    if dev in (DEVICE_GPU, DEVICE_AUTO):
        from gpu import driver as gpu_driver

        try:
            gpu_driver.check_supported(rg_param, ry_param, d_tube,
                                       center_init_cond)
        except gpu_driver.GpuUnsupported as exc:
            if dev == DEVICE_GPU:
                raise
            print(f"device=auto: GPU unavailable, using CPU instead ({exc})")
        else:
            print(f"device={dev}: solving on GPU "
                  f"({rg_param}x{ry_param}, agreement ~1e-14 vs CPU)")
            resume_state = None
            if cp is not None:
                resumed = cp.try_resume(D_LAYER, A_LAYER)
                if resumed is not None:
                    resume_state = resumed
                    print(f"    resuming from checkpoint at step "
                          f"{resume_state[0]:,}")

            # Keep the final state at k_align, as the CPU path does, so a
            # later run can extend this one (Checkpoint.save_final).
            K = num.compute_K(rg_param, ry_param, T_param, domain_radius, D)
            k_align = min(MA_collection_factor * relative_k, K)
            dT = num.compute_dT(rg_param, ry_param, domain_radius, D)

            def on_chunk(k, state):
                cp.save(k, D_LAYER, A_LAYER, state, series=series)
                if k == k_align and k > 0:
                    cp.save_final(k, D_LAYER, A_LAYER, state, series,
                                  t=k * dT, dT=dT)

            result = gpu_driver.solve(
                rg_param, ry_param, a_param, b_param, v_param, T_param,
                N_LIST, D_LAYER, A_LAYER, series, MA_collection_factor,
                relative_k, d_tube=d_tube, domain_radius=domain_radius, D=D,
                center_init_cond=center_init_cond,
                on_chunk=None if cp is None else on_chunk,
                resume_state=resume_state,
                boundary=None if cp is None else k_align)
            if cp is not None:
                cp.clear()
            return result

    return ant.comp_mass_analysis_respect_to_time(
        rg_param, ry_param, a_param, b_param, v_param, T_param, N_LIST,
        D_LAYER, A_LAYER, series[0], series[1], series[2], series[3],
        series[4], MA_collection_factor, relative_k, d_tube, domain_radius, D,
        mass_checkpoint, center_init_cond, m_init, n_init, checkpoint=cp)


# v------------------------------ The t* criterion ------------------------------v
#
# t* is the onset of steady exponential decay: the time from which ln(total
# mass) stays a straight line to the end of the solve, judged by its second
# derivative.
#
# THE RULE. With y(t) = ln M(t) and y'' its second-order central difference,
# mark each sample
#
#     l(t) = 1  if |y''(t)| <= tau,   0 otherwise.
#
# Sweeping from the right, t^ is the last sample with l = 0 -- the latest time
# y'' is still larger than tau -- and t* is the sample immediately after it,
# so |y''| <= tau from t* to the end of the run. If the final judged sample
# already has l = 0, the run ended before the decay settled and there is no t*.
#
# Why from the right rather than "first time below tau": y'' can pass through
# zero on its way somewhere else -- at an inflection, or in the damped
# oscillation of the decay rate seen for slow switching onto the advective
# layer (b <= 1) -- so the first time it is small is not when it stays small.
# Only the tail tells you that. The study behind this choice is in
# results/2026-10-05_ab-grid-16x16_tau-mask/.
#
# THE STEP. y'' is (y[i+s] - 2 y[i] + y[i-s]) / H^2 with H = s * sample_dt as
# close as possible to CHAR_TIME_DIFF_SPACING = 1e-4, the double-precision
# rule of thumb (~eps^(1/4)) for a second difference. Rounding grows as 1/H^2
# and the sample spacing shrinks with the grid, so differencing adjacent
# samples on a fine grid would bury y'' in noise; at H ~ 1e-4 the noise in y''
# is ~1e-7 to 1e-6, far below any useful tau. t* is reported at that sample
# resolution, not interpolated.
#
# tau is the one knob, exposed on both kernels with CHAR_TIME_TAU as its
# default; the spacing is part of the definition and read from here by both.
CHAR_TIME_TAU = 1e-3
CHAR_TIME_DIFF_SPACING = 1e-4

# The per-point result columns both kernels write, in order, after their own
# leading parameter column(s). One list so the two CSVs stay concatenable.
# fit_window_t1/_t2 are the detected steady segment [t*, t_end] and
# total_mass_at_t1/_t2 the mass at its two ends.
CHAR_TIME_RESULT_COLUMNS = ("t_star", "m_star", "fit_slope", "fit_intercept",
                            "fit_window_t1", "fit_window_t2",
                            "total_mass_at_t1", "total_mass_at_t2", "tau",
                            "char_time_mult", "lambda_1", "lambda_2", "lambda_2_imag")

# m* is the mass at char_time_mult * t*. 1 (the default) is M(t*); the older
# definitions read the mass at an integer multiple of t* (10 t* before
# 2026-09-28), which this reproduces. Bounded for now to the range below.
CHAR_TIME_MULT_DEFAULT = 1.0
CHAR_TIME_MULT_RANGE = (1.0, 10.0)


def _char_time_preflight(T_param, MA_collection_factor, relative_k, tau,
                         char_time_mult=CHAR_TIME_MULT_DEFAULT,
                         engine=None):
    """Reject a run that cannot possibly yield a t*, before any solving.

    Pre-flight and cheap: it runs before the time-stepping loop, which at
    160x160 means days, so a hopeless parameter set fails in a second rather
    than after the solve. Where the onset falls is only known afterwards, so
    this rules out only what is certain to fail: a non-positive tau, or a
    record too short to take a second difference over. A run that passes may
    still find no t* within T -- that is reported as a degenerate point after
    the solve, not raised.
    """
    if not (np.isfinite(tau) and tau > 0):
        raise ValueError(f"tau must be a positive number (got tau={tau})")
    lo, hi = CHAR_TIME_MULT_RANGE
    if not (np.isfinite(char_time_mult) and lo <= char_time_mult <= hi):
        raise ValueError(f"char_time_mult must lie in [{lo:g}, {hi:g}] "
                         f"(got char_time_mult={char_time_mult})")
    if engine is not None and engine not in CHAR_TIME_ENGINES:
        raise ValueError(f"unknown engine {engine!r}; expected one of "
                         f"{', '.join(CHAR_TIME_ENGINES)}")
    if T_param <= 0 or relative_k < 8:
        raise ValueError(
            f"T_param={T_param} is too short for a characteristic time: the "
            f"mass timeseries has {relative_k} samples, fewer than the 8 a "
            f"second difference needs.")


def _char_time_onset(total_mass, sample_dt, tau=CHAR_TIME_TAU,
                     spacing=CHAR_TIME_DIFF_SPACING):
    """t* from a total-mass timeseries, by the right-sweep rule described above.

    ``total_mass[i]`` is the mass at time ``i * sample_dt``. Pure numpy and
    free of solver state, so it is tested directly against analytic curves in
    tests/test_char_time_criterion.py.

    Returns a dict of plain floats:

      * ``t_star`` -- the first sample after the last one with |y''| > tau;
      * ``m_star`` -- the total mass at t*;
      * ``fit_slope``, ``fit_intercept`` -- the least-squares line through
        log10(total mass) over the steady segment [t*, t_end];
      * ``fit_window_t1``, ``fit_window_t2`` -- that segment's ends;
      * ``total_mass_at_t1``, ``total_mass_at_t2`` -- the mass at each end.

    Every entry is NaN when there is no t* -- |y''| is still above tau at the
    last judged sample, i.e. the run ended before the decay settled. That is
    the degenerate case; the caller reports it.
    """
    nan = float("nan")
    degenerate = dict(t_star=nan, m_star=nan, fit_slope=nan, fit_intercept=nan,
                      fit_window_t1=nan, fit_window_t2=nan,
                      total_mass_at_t1=nan, total_mass_at_t2=nan)

    mass = np.asarray(total_mass, dtype=np.float64)
    # Only the leading run of positive, finite samples is usable: a log is
    # needed, and a sample past a bad one cannot be differenced across it.
    bad = np.nonzero(~(np.isfinite(mass) & (mass > 0)))[0]
    if bad.size:
        mass = mass[:bad[0]]

    s = max(1, int(round(spacing / sample_dt)))
    H = s * sample_dt
    n = mass.size
    if n < 2 * s + 2:
        return degenerate

    t = np.arange(n) * sample_dt
    y = np.log(mass)
    # Second difference at every sample with a neighbour s away on both sides;
    # the first and last s samples are not judged.
    d2y = (y[2 * s:] - 2.0 * y[s:-s] + y[:-2 * s]) / H ** 2
    judged = np.arange(s, n - s)

    over = ~(np.abs(d2y) <= tau)          # l = 0; a non-finite y'' counts too
    if over[-1]:
        return degenerate                 # still above tau when the run ends
    if over.any():
        onset = int(judged[np.nonzero(over)[0][-1] + 1])
    else:
        onset = int(judged[0])            # never above tau
    t_star = float(t[onset])

    m_star = float(mass[onset])
    slope, intercept = np.polyfit(t[onset:], np.log10(mass[onset:]), 1)

    return dict(t_star=t_star, m_star=m_star,
                fit_slope=float(slope), fit_intercept=float(intercept),
                fit_window_t1=t_star, fit_window_t2=float(t[-1]),
                total_mass_at_t1=m_star,
                total_mass_at_t2=float(mass[-1]))


# THE MODAL ENGINE. With engine="modal" (the default) there is no mass series to
# difference: mass_analysis_modal diagonalises the solver's one-step operator,
# and y'' comes exactly from the modal sum as the weighted variance of the decay
# rates still present. The rule is the same right sweep, applied to that exact
# y'' on a grid of spacing CHAR_TIME_DIFF_SPACING over [0, T], with the last
# crossing of |y''| = tau refined by Brent's method -- so t* is not tied to a
# sample. Near t = 0 the sum is dominated by fast modes and its round-off can
# exceed tau; only the trailing stretch where the round-off bound on y'' is
# below CHAR_TIME_MODAL_NOISE_FRACTION * tau is judged. At 96x96, tau = 0.01,
# this t* agrees with the time-stepper's mask t* to within 1e-5
# (results/2026-10-07_ab-grid-96x96_tau0.01/eigen_t_star_96x96_tau0.01.pdf).
CHAR_TIME_ENGINE_MODAL = "modal"
CHAR_TIME_ENGINE_STEPPER = "time_stepper"
CHAR_TIME_ENGINES = (CHAR_TIME_ENGINE_MODAL, CHAR_TIME_ENGINE_STEPPER)
CHAR_TIME_MODAL_NOISE_FRACTION = 1e-2
# A mode counts towards lambda_2 when |a_n| exceeds this fraction of |a_1|;
# below it the mode is not excited by the initial condition (round-off level).
CHAR_TIME_EXCITED_AMPLITUDE = 1e-9

# Columns only the modal engine fills (NaN from the time-stepper): the slowest
# decay rate, and the slowest excited rate after it (real part, and the
# magnitude of its imaginary part -- a complex lambda_2 comes with its conjugate).
CHAR_TIME_MODAL_COLUMNS = ("lambda_1", "lambda_2", "lambda_2_imag")


def _slowest_rates(modes):
    """(lambda_1, Re lambda_2, |Im lambda_2|) of the modes the seed excites."""
    lam = modes["decay_rate"]
    amp = np.abs(modes["amplitude"])
    lam1 = float(lam[0].real)
    excited = amp >= CHAR_TIME_EXCITED_AMPLITUDE * amp[0]
    others = excited & (np.abs(lam - lam[0]) > 1e-9 * abs(lam1))
    if not others.any():
        return lam1, float("nan"), float("nan")
    l2 = lam[np.nonzero(others)[0][0]]
    return lam1, float(l2.real), float(abs(l2.imag))


def _char_time_onset_modal(modes, T_param, tau=CHAR_TIME_TAU,
                           spacing=CHAR_TIME_DIFF_SPACING):
    """t* and the fit columns of _char_time_onset, from the exact modal y''.

    Same rule and same returned keys as _char_time_onset, plus
    CHAR_TIME_MODAL_COLUMNS. NaN when |y''(T)| > tau. ``m_star`` here is M(t*);
    _char_time_point moves it to M(char_time_mult * t*).
    """
    from scipy.optimize import brentq

    nan = float("nan")
    lam1, lam2, lam2_imag = _slowest_rates(modes)
    row = dict(t_star=nan, m_star=nan, fit_slope=nan, fit_intercept=nan,
               fit_window_t1=nan, fit_window_t2=nan,
               total_mass_at_t1=nan, total_mass_at_t2=nan,
               lambda_1=lam1, lambda_2=lam2, lambda_2_imag=lam2_imag)

    n = max(int(np.ceil(T_param / spacing)), 8) + 1
    t = np.linspace(0.0, T_param, n)
    M, _, y2, noise = mam.ln_mass_derivatives(modes, t)

    # judged: the trailing stretch where round-off cannot reach tau
    unreliable = np.nonzero(~(noise <= CHAR_TIME_MODAL_NOISE_FRACTION * tau))[0]
    j0 = int(unreliable[-1]) + 1 if unreliable.size else 0
    if j0 >= n:
        return row
    over = np.abs(y2[j0:]) > tau
    if over[-1]:
        return row                                # still above tau when the run ends
    if over.any():
        i = j0 + int(np.nonzero(over)[0][-1])
        f = lambda s: abs(mam.ln_mass_derivatives(modes, [s])[2][0]) - tau
        t_star = float(brentq(f, t[i], t[i + 1], xtol=1e-12))
    else:
        t_star = float(t[j0])                     # never above tau where judged
        if j0 > 0:
            print(f"*** WARNING: |y''| <= tau throughout the stretch the modal "
                  f"sum resolves, which starts at t = {t_star:.6g}; t* is "
                  f"reported there but may lie earlier (cond(V) = "
                  f"{modes['cond_V']:.1e}). engine='time_stepper' resolves it.")

    m_star = float(mam.ln_mass_derivatives(modes, [t_star])[0][0])
    tail = t > t_star
    t_fit = np.concatenate([[t_star], t[tail]])
    M_fit = np.concatenate([[m_star], M[tail]])
    if t_fit.size >= 2:
        slope, intercept = np.polyfit(t_fit, np.log10(M_fit), 1)
    else:
        slope = intercept = nan
    row.update(t_star=t_star, m_star=m_star,
               fit_slope=float(slope), fit_intercept=float(intercept),
               fit_window_t1=t_star, fit_window_t2=float(T_param),
               total_mass_at_t1=m_star, total_mass_at_t2=float(M[-1]))
    return row
# ^------------------------------ The t* criterion ------------------------------^


def _char_time_point(tag, rg_param, ry_param, a_param, b_param, v_param,
                     T_param, N_LIST, MA_collection_factor, relative_k, tau,
                     d_tube, domain_radius, D, mass_checkpoint,
                     center_init_cond, m_init, n_init, device, label="",
                     extend_from=None, engine=CHAR_TIME_ENGINE_MODAL,
                     char_time_mult=CHAR_TIME_MULT_DEFAULT, compute_psd=False):
    """One solve plus the steady-decay criterion that defines t*.

    This is the characteristic-time kernel proper: everything done per parameter
    point, with no plotting, no file writing and no knowledge of what is being
    swept. ``collect_char_time_mass`` calls it once per velocity;
    ``collect_ab_grid_char_time`` calls it once per (a, b) pair. Keeping it in
    one place is what makes the two kernels report the same t* for the same
    parameters, which is asserted by tests/test_ab_grid_char_time.py.

    ``engine`` picks how M(t) is obtained. "modal" (the default) diagonalises
    the solver's one-step operator once (mass_analysis_modal) and applies the
    right-sweep rule to the exact y'' of the modal sum; it costs seconds whatever
    T is, and also yields lambda_1 and lambda_2. "time_stepper" runs the solve to
    T and applies the rule to the central difference of the sampled series; it
    is the only engine that uses ``device``, ``mass_checkpoint`` and
    ``extend_from``.

    ``m_star`` is the mass at ``char_time_mult * t*`` (M(t*) itself is
    total_mass_at_t1). The modal engine evaluates it at any time; the
    time-stepper reads the nearest sample, and NaN past T.

    ``label`` names this point in the degenerate-fit warning below, since the
    caller knows what is being swept and this function does not.

    ``compute_psd`` adds y2_spectrum.PSD_COLUMNS -- the peak and FWHM of
    |F[y'']|^2, y = ln(total mass) sampled at the criterion's spacing over
    [0, T] -- and, under "_psd", the spectrum about its peak for plotting.

    Returns plain Python floats in a dict -- plain because the result has to
    survive being pickled back from a worker process.
    """
    _validate_off_center_ic(center_init_cond, m_init, n_init, rg_param, ry_param)

    if engine == CHAR_TIME_ENGINE_MODAL:
        modes = mam.solve_modes(rg_param, ry_param, a_param, b_param, v_param,
                                N_LIST, d_tube, domain_radius, D,
                                center_init_cond, m_init, n_init)
        row = _char_time_onset_modal(modes, T_param, tau=tau)
        if np.isfinite(row["t_star"]):
            row["m_star"] = float(mam.ln_mass_derivatives(
                modes, [char_time_mult * row["t_star"]])[0][0])
        if compute_psd:
            # the time-stepper's sample grid, so both engines transform the same t_n
            _, H = y2s.spacing(MA_collection_factor * modes["dT"], CHAR_TIME_DIFF_SPACING)
            spec = y2s.spectrum(mam.ln_mass(modes, np.arange(int(T_param / H) + 1) * H), H)
    else:
        D_LAYER, A_LAYER = sup.initialize_layers(rg_param, ry_param)

        # One checkpoint directory per point, since each is a separate solve;
        # without a distinguishing tag a sweep would resume the wrong trajectory.
        cp, series = _setup_checkpoint(
            tag, relative_k, extend_from=extend_from,
            rg_param=rg_param, ry_param=ry_param, a=a_param, b=b_param,
            v=v_param, T=T_param, N_LIST=N_LIST, d_tube=d_tube,
            domain_radius=domain_radius, D=D,
            MA_collection_factor=MA_collection_factor, relative_k=relative_k,
            center_init_cond=center_init_cond, m_init=m_init, n_init=n_init)
        # Canonical slot order, matching the parameter order of
        # comp_mass_analysis_respect_to_time: (DL, AL, ALoI, ALoT, TM).
        # The GPU kernel writes these same indices, so the order here is a
        # contract between the two devices, not a local naming choice --
        # swapping slots 2 and 3 makes the two paths disagree on which series
        # holds al/T and which holds al/D.
        (MA_DL_timeseries, MA_AL_timeseries, MA_ALoI_timeseries,
         MA_ALoT_timeseries, MA_TM_timeseries) = series

        _solve_mass_analysis(device, series, rg_param, ry_param, a_param, b_param,
                             v_param, T_param, N_LIST, D_LAYER, A_LAYER,
                             MA_collection_factor, relative_k, d_tube,
                             domain_radius, D, mass_checkpoint,
                             center_init_cond, m_init, n_init, cp=cp)

        # Sample i of the timeseries is the state after i * MA_collection_factor
        # steps, so the samples are MA_collection_factor * dT apart in time.
        sample_dt = MA_collection_factor * num.compute_dT(rg_param, ry_param,
                                                          domain_radius, D)
        row = _char_time_onset(MA_TM_timeseries, sample_dt, tau=tau)
        row.update({c: float("nan") for c in CHAR_TIME_MODAL_COLUMNS})
        if np.isfinite(row["t_star"]) and char_time_mult != 1:
            i = int(round(char_time_mult * row["t_star"] / sample_dt))
            row["m_star"] = (float(MA_TM_timeseries[i]) if i < relative_k
                             else float("nan"))
            if i >= relative_k:
                print(f"*** WARNING: {char_time_mult:g} t* = "
                      f"{char_time_mult * row['t_star']:.6g} lies past T_param="
                      f"{T_param}; recording m_star = NaN ({label or tag}). "
                      f"The modal engine has no such limit.")
        if compute_psd:
            s, H = y2s.spacing(sample_dt, CHAR_TIME_DIFF_SPACING)
            mass = np.asarray(MA_TM_timeseries[::s], dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                spec = y2s.spectrum(np.log(np.where(mass > 0, mass, np.nan)), H)

    if compute_psd:
        row.update(y2s.summary(spec))
        row["_psd"] = y2s.trim(spec)

    # The onset is only known after the solve, so a record that ends before
    # the decay settles cannot be caught in pre-flight. Report it loudly and
    # record NaN rather than raise: a raise here would discard the whole solve,
    # and the other points of a sweep are still good.
    if not np.isfinite(row["t_star"]):
        print("*** WARNING: no steady exponential decay found ***")
        print(f"    point      = {label or tag}")
        print(f"    tau        = {tau}")
        print(f"    |d2 ln(total mass)/dt2| was still above tau at the end "
              f"of the run (T_param={T_param}).")
        print("    recording t_star = m_star = NaN; raise T_param or tau.")

    # The fit over the steady segment is recorded, not only t*: its slope is
    # the asymptotic decay rate of log10(total mass), a result in its own right,
    # and the mass timeseries it came from is not retained -- a GPU run logs
    # nothing per step -- so once it is out of the CSV it is unrecoverable
    # short of re-solving.
    row["tau"] = float(tau)
    row["char_time_mult"] = float(char_time_mult)
    return row


def collect_char_time_mass(rg_param, ry_param, v_LIST, w_param, T_param, N_LIST, MA_collection_factor=5, domain_radius=1.0, D=1.0,
                           mass_checkpoint=10 ** 6, d_tube=0.0, center_init_cond=True, m_init=0, n_init=0, show_plt=True,
                           device=DEVICE_CPU, tau=CHAR_TIME_TAU, extend_from=None,
                           engine=CHAR_TIME_ENGINE_MODAL, char_time_mult=CHAR_TIME_MULT_DEFAULT,
                           compute_psd=False):
    """m* and t* against velocity, at a single mutual switch rate a = b = w.

    t* is the onset of steady exponential decay of the total mass and m* the
    mass remaining at ``char_time_mult * t*``; ``tau`` is the threshold on
    |d2 ln(total mass)/dt2| that decides it. See the block above
    _char_time_onset, and _char_time_point for ``engine`` and ``compute_psd``
    (which adds the PSD columns and a peak-zoom figure, char_t_psd_y2.png).

    ``extend_from`` continues each velocity's earlier, shorter run from its
    saved final state rather than starting at t = 0 -- a job id from
    comp-state-ledger, or that job's checkpoints directory. Only T may differ.
    """

    K = num.compute_K(rg_param, ry_param, T_param, domain_radius, D)
    print("deltaT = ", num.compute_dT(rg_param, ry_param))
    relative_k = int(np.floor(K / MA_collection_factor))

    b_param = w_param
    a_param = w_param

    tau = float(tau)
    char_time_mult = float(char_time_mult)
    engine = str(engine or CHAR_TIME_ENGINE_MODAL).strip().lower()
    _char_time_preflight(T_param, MA_collection_factor, relative_k, tau,
                         char_time_mult, engine)

    # One fit per velocity, keyed by it, in the order the sweep was requested.
    fits = {}
    for v_param in v_LIST:
        fits[v_param] = _char_time_point(
            f"char_time_v{v_param:g}", rg_param, ry_param, a_param, b_param,
            v_param, T_param, N_LIST, MA_collection_factor, relative_k, tau,
            d_tube, domain_radius, D, mass_checkpoint, center_init_cond,
            m_init, n_init, device, label=f"v = {v_param}",
            extend_from=extend_from, engine=engine,
            char_time_mult=char_time_mult, compute_psd=compute_psd)

    v_axis = list(fits.keys())
    m_axis = [fits[v_param]["m_star"] for v_param in v_axis]

    plt.scatter(v_axis, m_axis)
    plt.xscale('log')
    plt.xlabel(r"($v$) velocity")
    plt.ylabel("(m) Mass")
    plt.title(r"$m(v)$ at $t^*$, " + f"N={len(N_LIST)}, " + f"a=b={w_param}, " + f"grid={rg_param}x{ry_param}")

    # (****) Store the plot and the tabulated data under data_output/char_time_analysis (****)
    # Both files land in a subdirectory stamped with the time of creation (up to the minute).
    output_directory = _create_unique_timestamp_dir(fp.char_time_analysis_output)

    plot_location = os.path.join(output_directory, 'char_t_analysis_plot.png')
    plt.savefig(plot_location, bbox_inches='tight')

    data_location = os.path.join(output_directory, 'char_t_analysis_data.csv')
    # v, t_star and m_star stay in the leading columns so anything already
    # reading this file by position keeps working; the fit columns are appended.
    df = pd.DataFrame({'v': v_axis})
    for column in _char_time_columns(compute_psd):
        df[column] = [fits[v_param][column] for v_param in v_axis]
    df.to_csv(data_location, index=False)

    print(f'Characteristic time plot saved to {plot_location}')
    print(f'Characteristic time data saved to {data_location}')

    if show_plt:
        plt.show()

    plots = {"m_star_vs_v": plot_location}
    if compute_psd:
        plots["psd_y2"] = y2s.save_figure(
            os.path.join(output_directory, 'char_t_psd_y2.png'),
            [fits[v]["_psd"] for v in v_axis], [f"v = {v:g}" for v in v_axis],
            min(4, len(v_axis)),
            rf"$|\mathcal{{F}}[y'']|^2$ about its peak -- a=b={w_param:g}, N={len(N_LIST)}, "
            f"{rg_param}x{ry_param}, T={T_param:g}, engine={engine}")

    # Same shape as collect_ab_grid_char_time returns, so the router can hand
    # both to the desktop GUI and the API the same way. It used to return None,
    # which left the caller to find its files by searching the output tree.
    return {"output_dir": output_directory, "csv": data_location,
            "plots": plots, "points": len(v_axis),
            "engine": engine}


def _char_time_columns(compute_psd):
    """The per-point CSV columns: the result columns, then the PSD's if computed."""
    return CHAR_TIME_RESULT_COLUMNS + (y2s.PSD_COLUMNS if compute_psd else ())


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# v==================================== Characteristic time over an (a, b) grid ====================================v
#
# WHY THE (a, b) POINTS RUN AS CONCURRENT PROCESSES ON THE CPU
#
# Every (a, b) pair is an independent solve: the pairs share no state, and each
# one only reads the parameters it was given and returns a handful of floats. So the grid
# parallelises perfectly, and the reasons it is worth doing are the same three
# that made the microtubule sweep in main.py concurrent:
#
# 1. The solver is single-threaded. comp_mass_analysis_respect_to_time is an
#    @njit function with a serial time loop, measured at 0.99 cores busy. A
#    sequential 4x4 grid therefore leaves eleven of this machine's twelve cores
#    idle for the whole run.
#
# 2. The points cost the same. K depends only on the grid and T, never on a or
#    b, so every point runs an identical number of timesteps and they finish
#    together. There is no straggler to wait on.
#
# 3. Processes, not threads. numba compiles these kernels without nogil, so
#    threads would serialise on the GIL and buy nothing. Separate processes also
#    give each point its own matplotlib state, which is what stops one point's
#    figure from accumulating another's (the bug described in main.py).
#
# The GPU is the opposite case and is handled the opposite way: see the
# device-resolution block in collect_ab_grid_char_time.


def _ab_grid_worker_count(requested, n_points):
    """How many (a, b) points to solve at once on the CPU.

    Never more than there are points, and never more than one per core with one
    left over for the parent process and the OS. ``requested`` pins it; the job
    worker already runs up to four jobs at a time, so a grid submitted into a
    busy queue wants pinning rather than the default.
    """
    if requested and int(requested) > 0:
        return max(1, min(int(requested), n_points))
    return max(1, min(n_points, (os.cpu_count() or 2) - 1))


def _ab_grid_char_time_worker(job):
    """Solve one (a, b) point. Runs in its own process under the pool below.

    Must stay a module-level function taking a single picklable argument: the
    "spawn" start method (the only one on Windows) pickles the callable by
    module and qualified name and re-imports it in the child.
    """
    import time as _time

    a_param, b_param, cfg = job

    start = _time.perf_counter()
    row = _char_time_point(
        f"ab_grid_char_time_a{a_param:g}_b{b_param:g}",
        a_param=a_param, b_param=b_param,
        label=f"a = {a_param}, b = {b_param}", **cfg)
    row["a"] = a_param
    row["b"] = b_param
    row["seconds"] = _time.perf_counter() - start
    print(f"[a={a_param:g} b={b_param:g}] t* = {row['t_star']:.6g} "
          f"in {row['seconds']/60:.2f} min", flush=True)
    return row


def _ab_grid_stem(v_param, n_tubes, rg_param, ry_param):
    """Filename fragment naming the parameters held fixed across a grid.

    v, the microtubule count and the grid resolution do not vary within a run,
    so they are not columns in the CSV -- they go here, and every file the run
    writes carries them. A file that has been moved out of its timestamped
    directory then still says what produced it.

    ``%g`` is safe in a filename for every value that reaches it: it emits only
    digits, ``.``, ``-``, ``+`` and ``e``. Velocities are routinely 1e4, which
    is why the format is not ``%f``.
    """
    return f"v{v_param:g}_N{n_tubes}_{rg_param}x{ry_param}"


def _ab_grid_log_x(values):
    """Whether an axis over these switch rates should be logarithmic.

    Switch rates are swept by decade far more often than linearly (1, 10, 100),
    and on a linear axis that puts every point but the largest against the
    origin. But a log axis silently drops a non-positive value, and 0 is a legal
    rate -- it means "no switching in this direction" -- so this only applies
    when it cannot lose a point, and only when there is a real span to justify
    it.
    """
    if len(values) < 2 or any(v <= 0 for v in values):
        return False
    return max(values) / min(values) >= 10.0


def _ab_grid_slice_plot(path, x_values, series_values, grid, x_symbol,
                        x_label, series_symbol, subtitle, show_plt):
    """m* against one switch rate, one curve per value of the other.

    ``grid`` is indexed [x][series], so the caller passes the m* grid for the
    a-axis and its transpose for the b-axis.

    Drawn on its own Figure and closed afterwards rather than through the pyplot
    state machine: this is called three times per run, and pyplot would let each
    figure accumulate the previous one's artists -- the bug documented in
    main.py, which made a sweep's PNGs each show every earlier job's points.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for j, series in enumerate(series_values):
        ax.plot(x_values, grid[:, j], marker='o',
                label=f"{series_symbol} = {series:g}")
    if _ab_grid_log_x(x_values):
        ax.set_xscale('log')
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"($m^*$) mass retained at $t^*$")
    ax.set_title(f"$m^*({x_symbol})$, {subtitle}")
    # No legend title: each entry already names the parameter, and a "b" header
    # above three rows reading "b = 1" is noise.
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(path, bbox_inches='tight')
    if show_plt:
        plt.show()
    plt.close(fig)
    return path


def _ab_grid_intensity_plot(path, a_list, b_list, grid, subtitle, show_plt):
    """m* over the (a, b) grid: a vertical, b horizontal, colour by magnitude.

    One square per solved point, so the axes are categorical -- the tick labels
    are the requested rates, evenly spaced, not positioned by value. That is
    deliberate: the rates are usually swept by decade, and spacing the cells by
    value would make a 1-10-100 sweep three cells of wildly different widths.
    """
    fig, ax = plt.subplots(figsize=(max(4.0, 1.1 * len(b_list) + 2.5),
                                    max(3.5, 1.0 * len(a_list) + 2.0)))
    # origin='lower' so a increases upward, the way it reads on an axis.
    image = ax.imshow(grid, origin='lower', aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(b_list)))
    ax.set_xticklabels([f"{b:g}" for b in b_list])
    ax.set_yticks(range(len(a_list)))
    ax.set_yticklabels([f"{a:g}" for a in a_list])
    ax.set_xlabel(r"($b$) switch rate onto the advective layer")
    ax.set_ylabel(r"($a$) switch rate onto the diffusive layer")
    ax.set_title(r"$m^*(a, b)$, " + subtitle)
    fig.colorbar(image, ax=ax, label=r"($m^*$) mass retained at $t^*$")

    # Print the value in each cell on a grid small enough to read it. On a
    # coarse grid -- which is what these are, being one full solve per cell --
    # the number is the result and the colour is only the summary. A cell whose
    # fit degenerated is NaN: imshow leaves it blank, so it is labelled rather
    # than left looking like a missing tile.
    #
    # Text colour follows the cell rather than being fixed: viridis runs from
    # dark blue to bright yellow, so a single colour is illegible over half the
    # range whichever one is chosen.
    if len(a_list) * len(b_list) <= 64:
        finite = grid[np.isfinite(grid)]
        midpoint = (finite.min() + finite.max()) / 2 if finite.size else 0.0
        for i in range(len(a_list)):
            for j in range(len(b_list)):
                value = grid[i, j]
                ax.text(j, i, "--" if not np.isfinite(value) else f"{value:.4g}",
                        ha='center', va='center', fontsize=7,
                        color='k' if np.isfinite(value) and value > midpoint
                        else 'w')

    fig.savefig(path, bbox_inches='tight')
    if show_plt:
        plt.show()
    plt.close(fig)
    return path


def collect_ab_grid_char_time(rg_param, ry_param, a_list, b_list, v_param, T_param,
                              N_LIST, MA_collection_factor=5, domain_radius=1.0,
                              D=1.0, mass_checkpoint=10 ** 6, d_tube=0.0,
                              center_init_cond=True, m_init=0, n_init=0,
                              show_plt=False, device=DEVICE_CPU, workers=0,
                              tau=CHAR_TIME_TAU, extend_from=None,
                              engine=CHAR_TIME_ENGINE_MODAL,
                              char_time_mult=CHAR_TIME_MULT_DEFAULT,
                              compute_psd=False):
    """Characteristic time t* over the full grid of (a, b) switch rates.

    Where ``collect_char_time_mass`` sweeps velocity at a single switch rate
    (a = b = w), this sweeps the two switch rates independently at a single
    velocity: ``a_list`` supplies the rate onto the diffusive layer and
    ``b_list`` the rate onto the advective layer, and every pair in the cartesian
    product ``a_list x b_list`` is solved. ``v_param`` is one value here, not a
    list -- the grid is already two-dimensional, and a third axis would multiply
    an already expensive job by the length of a velocity list.

    Each point is the same solve-and-criterion as the velocity sweep, run
    through the shared ``_char_time_point`` kernel, so a 1x1 grid at (w, w)
    reproduces ``collect_char_time_mass`` for that velocity exactly. ``tau`` is
    the steady-decay threshold, ``char_time_mult`` places m* at that multiple
    of t*, and ``engine`` picks the modal expansion (default) or the
    time-stepper, as there. ``device``, ``workers`` and ``extend_from`` apply to
    the time-stepper only: the modal engine solves the points one after another
    in this process, each in seconds, with the eigensolve itself multithreaded.

    Outputs, under ``data_output/ab_grid_char_time/<timestamp>/``. Every
    filename carries the parameters held fixed across the run -- ``v``, the
    microtubule count and the grid -- so a file moved out of its directory still
    says what produced it (``<stem>`` below is e.g. ``v10000_N4_96x96``):

      * ``ab_grid_char_t_<stem>.csv`` -- one row per (a, b): ``a, b, t_star,
        m_star``, then the fit each was derived from
      * ``ab_grid_m_star_vs_a_<stem>.png`` -- m* against a, one curve per b
      * ``ab_grid_m_star_vs_b_<stem>.png`` -- m* against b, one curve per a
      * ``ab_grid_m_star_intensity_<stem>.png`` -- m* over the grid, a on the
        vertical axis and b on the horizontal, one square per solved point
        coloured by magnitude
      * ``ab_grid_psd_y2_<stem>.png`` -- with ``compute_psd``: |F[y'']|^2 about
        its peak for every (a, b), a down and b across, with the peak and FWHM
        (also appended to the CSV; see _char_time_point)

    Returns the paths it wrote, so a caller that is not reading the output tree
    still learns where the results went.
    """
    a_list = [float(a) for a in a_list]
    b_list = [float(b) for b in b_list]
    if not a_list or not b_list:
        raise ValueError(
            f"a_list and b_list must each hold at least one switch rate "
            f"(got {len(a_list)} and {len(b_list)})")

    K = num.compute_K(rg_param, ry_param, T_param, domain_radius, D)
    print("deltaT = ", num.compute_dT(rg_param, ry_param))
    relative_k = int(np.floor(K / MA_collection_factor))

    tau = float(tau)
    char_time_mult = float(char_time_mult)
    engine = str(engine or CHAR_TIME_ENGINE_MODAL).strip().lower()
    _char_time_preflight(T_param, MA_collection_factor, relative_k, tau,
                         char_time_mult, engine)

    # Normalised once, here rather than per point: N_LIST crosses a process
    # boundary on the CPU path and goes to the device on the GPU one, and the
    # two must be given the same thing for their results to be comparable.
    N_LIST = np.asarray(N_LIST, dtype=np.int64)

    _validate_off_center_ic(center_init_cond, m_init, n_init, rg_param, ry_param)

    points = [(a, b) for a in a_list for b in b_list]

    # --- device resolution, once for the whole grid ------------------------
    # Decided up front rather than per point so a grid cannot end up half on one
    # device and half on the other -- the two agree only to ~1e-14, so a mixed
    # grid would carry a discontinuity that belongs to the hardware and not to
    # the physics.
    dev = (device or DEVICE_CPU).strip().lower()
    if dev not in DEVICES:
        raise ValueError(
            f"unknown device {device!r}; expected one of {', '.join(DEVICES)}")

    on_gpu = False
    if engine == CHAR_TIME_ENGINE_MODAL:
        dev = DEVICE_CPU                  # no time-stepping loop to place
    if dev in (DEVICE_GPU, DEVICE_AUTO):
        from gpu import driver as gpu_driver

        try:
            gpu_driver.check_supported(rg_param, ry_param, d_tube,
                                       center_init_cond)
        except gpu_driver.GpuUnsupported as exc:
            if dev == DEVICE_GPU:
                raise
            print(f"device=auto: GPU unavailable, using CPU instead ({exc})")
            dev = DEVICE_CPU
        else:
            on_gpu = True

    cfg = dict(
        rg_param=rg_param, ry_param=ry_param, v_param=v_param, T_param=T_param,
        N_LIST=N_LIST, MA_collection_factor=MA_collection_factor,
        relative_k=relative_k, tau=tau, d_tube=d_tube, extend_from=extend_from,
        domain_radius=domain_radius, D=D, mass_checkpoint=mass_checkpoint,
        center_init_cond=center_init_cond, m_init=m_init, n_init=n_init,
        device=DEVICE_GPU if on_gpu else DEVICE_CPU, engine=engine,
        char_time_mult=char_time_mult, compute_psd=bool(compute_psd))

    print("=== characteristic time over an (a, b) grid ===")
    print(f"grid          : {rg_param}x{ry_param}")
    print(f"a list        : {a_list}")
    print(f"b list        : {b_list}")
    print(f"v             : {v_param}")
    print(f"T             : {T_param}")
    print(f"(a, b) points : {len(points)}")
    print(f"engine        : {engine}")

    if engine == CHAR_TIME_ENGINE_MODAL:
        print("execution     : modal expansion, one point at a time")
        print(flush=True)
        rows = [_ab_grid_char_time_worker((a, b, cfg)) for a, b in points]
    elif on_gpu:
        # Sequential, deliberately. The GPU solve is one cooperative kernel that
        # occupies every SM on the card for the whole run, so a second
        # concurrent point does not get its own hardware -- it time-slices with
        # the first. That is measured, not assumed: three concurrent GPU jobs
        # turned a 6.6 hr solo estimate into a 22.5 hr actual, which is where
        # estimate.share_factor's 1/N comes from. Running the points one after
        # another is therefore both the fastest option and the one whose
        # progress output is legible.
        print("execution     : GPU, one point at a time "
              "(a cooperative kernel already fills the card)")
        print(flush=True)
        rows = [_ab_grid_char_time_worker((a, b, cfg)) for a, b in points]
    else:
        n_workers = _ab_grid_worker_count(workers, len(points))
        print(f"execution     : CPU, {n_workers} point(s) at a time")
        print(flush=True)
        jobs = [(a, b, cfg) for a, b in points]
        if n_workers == 1:
            # In-process: a pool of one buys nothing and costs a numba compile
            # in the child plus an opaque traceback if a point raises.
            rows = [_ab_grid_char_time_worker(job) for job in jobs]
        else:
            # "spawn" is the only start method on Windows and keeps each
            # worker's numba and matplotlib state fully isolated.
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=n_workers) as pool:
                rows = pool.map(_ab_grid_char_time_worker, jobs)

    # --- tabulate ----------------------------------------------------------
    output_directory = _create_unique_timestamp_dir(fp.ab_grid_char_time_output)

    # v, N and the grid resolution are fixed across the whole run, so they are
    # not columns -- they name the file instead. That way a CSV lifted out of
    # its timestamped directory still says which run it came from, which the
    # velocity sweep's fixed 'char_t_analysis_data.csv' does not.
    stem = _ab_grid_stem(v_param, len(N_LIST), rg_param, ry_param)

    data_location = os.path.join(output_directory, f'ab_grid_char_t_{stem}.csv')
    # a, b, t_star, m_star are the results proper and lead the file.
    #
    # The fit that produced them is appended, under the same column names
    # collect_char_time_mass writes, so the two CSVs stay concatenable. Those
    # columns are kept rather than dropped for the reason they were added to the
    # velocity sweep in the first place: the slope over the steady segment is
    # the asymptotic decay rate, a result in its own right, the mass timeseries
    # it came from is not
    # retained, and on a GPU run nothing is logged per step -- so once they are
    # out of this file they are unrecoverable short of re-solving.
    df = pd.DataFrame({'a': [r["a"] for r in rows],
                       'b': [r["b"] for r in rows]})
    for column in _char_time_columns(compute_psd):
        df[column] = [r[column] for r in rows]
    df.to_csv(data_location, index=False)

    # --- figures -----------------------------------------------------------
    # Rows are a, columns are b, in the order they were supplied -- which is the
    # order `points` was built in, so the reshape is the inverse of that loop.
    m_star_grid = np.array([r["m_star"] for r in rows],
                           dtype=np.float64).reshape(len(a_list), len(b_list))

    subtitle = (f"N={len(N_LIST)}, v={v_param:g}, "
                f"grid={rg_param}x{ry_param}")

    plots = {
        # m* against a, one curve per b. A 2-D sweep read as 1-D slices: it is
        # the only way to see whether the two rates act independently, which the
        # intensity map can only suggest.
        "m_star_vs_a": _ab_grid_slice_plot(
            os.path.join(output_directory, f'ab_grid_m_star_vs_a_{stem}.png'),
            x_values=a_list, series_values=b_list, grid=m_star_grid,
            x_symbol="a",
            x_label=r"($a$) switch rate onto the diffusive layer",
            series_symbol="b", subtitle=subtitle, show_plt=show_plt),
        # m* against b, one curve per a: the same data, transposed.
        "m_star_vs_b": _ab_grid_slice_plot(
            os.path.join(output_directory, f'ab_grid_m_star_vs_b_{stem}.png'),
            x_values=b_list, series_values=a_list, grid=m_star_grid.T,
            x_symbol="b",
            x_label=r"($b$) switch rate onto the advective layer",
            series_symbol="a", subtitle=subtitle, show_plt=show_plt),
        "m_star_intensity": _ab_grid_intensity_plot(
            os.path.join(output_directory,
                         f'ab_grid_m_star_intensity_{stem}.png'),
            a_list=a_list, b_list=b_list, grid=m_star_grid,
            subtitle=subtitle, show_plt=show_plt),
    }
    if compute_psd:
        plots["psd_y2"] = y2s.save_figure(
            os.path.join(output_directory, f'ab_grid_psd_y2_{stem}.png'),
            [r["_psd"] for r in rows], [f"a = {r['a']:g}, b = {r['b']:g}" for r in rows],
            len(b_list),
            rf"$|\mathcal{{F}}[y'']|^2$ about its peak -- {subtitle}, T={T_param:g}, "
            f"engine={engine}")

    print(f'(a, b) grid characteristic-time data saved to {data_location}')
    for path in plots.values():
        print(f'(a, b) grid characteristic-time plot saved to {path}')

    return {"output_dir": output_directory, "csv": data_location,
            "plots": plots, "points": len(rows),
            "device": DEVICE_GPU if on_gpu else DEVICE_CPU, "engine": engine}
# ^==================================== Characteristic time over an (a, b) grid ====================================^
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# <***************************** UNDER DEVELOPMENT/REQUIRES MORE TESTING *****************************>

# MFPT as a function of W saturation analysis
# under construction
# def mfpt_of_W_sat_analysis(domain_list, N_param_list, v_param, w_param_list, T_param, r=1.0, d=1.0,
#                            mass_checkpoint=10 ** 6, d_tube=-1):
#     '''
#     Produces a visualization for a saturation of analysis as MFPT as a function of W (switching rate a=b) for varying domain sizes)
#     In addition, this current implementation only functions for a single microtubule configuration.
#
#     We are interested in saturation of MFPT across varying domain sizes under fixed number of microtubules, d_tube, velocity, and time.
#
#     Args:
#         domain_list: (2-Dimensional list: a list of domain dimensions, i.e. MxN, Rings x Rays)
#         N_param_list: (2-Dimensional list: a list of microtubule configurations corresponding to the dimensions in the above list)
#         v_param:
#         w_param_list: (1-dimensional list: a list of desired w_values in sorted order)
#         T_param:
#         r:
#         d:
#         mass_checkpoint:
#         d_tube:
#
#     Returns:
#     '''
#
#     # Throw an error if the sizes of the domain_list, N_param_list, and the w_param_list do not match
#
#     # In the case of non overlap, the current selection of d_tube (assuming it is non-zero) needs to be verified if it functions properly across all domain sizes and microtubule configurations.
#
#     output_file_list = []
#     for i in range(len(domain_list)):
#         MFPT_list = []
#         for j in range(len(w_param_list)):
#             MFPT = solve_mfpt_(domain_list[i][0], domain_list[i][1], N_param_list[i], v_param, w_param_list[j], T_param,
#                                r=r, d=d, mass_checkpoint=mass_checkpoint, d_tube=d_tube)
#             MFPT_list.append(MFPT)
#
#         # store the data and save the data to csv
#         # store a pointer to the csv data file
#         # append this pointer to a list of csv data files
#     # Configure a plot to display the saturation of results by overlapping the results contained in the list of csv data files.
#
#
# '''
#     Configure a launch function to produce an analysis on MFPT as a function of W for varying amounts of N
# '''

# <***************************** UNDER DEVELOPMENT/REQUIRES MORE TESTING *****************************>


# <**********************************************************> REQUIRES REVIEW (These functions have not been used since 2024)

# ind_param:  independent parameter: the value that remains static across all MFPT solutions
# dep_param:  dependent parameter(s) : value that is being tested for MFPT dependence (contained within a set)
#
#

# def solve_mfpt_multi_process(N_param, rg_param, ry_param, dep_type, ind_param, dep_param):
#     M = rg_param
#     N = ry_param
#
#     if dep_type == "W":
#         w_param = dep_param
#         v_param = ind_param
#     elif dep_type == "V":
#         v_param = dep_param
#         w_param = ind_param
#     else:
#         raise f"{dep_type} not yet defined, must use either V or W"
#
#     mfpt, duration = solve_mfpt(rg_param, ry_param, N_param, v_param, w_param, return_duration=True)
#
#     print(f"MxN={M}x{N}    Duration (sim time) : {duration}    Microtubule configuration: {N_param}"
#           f"    W: {w_param}    V: {v_param}")
#
#     if dep_type == "W":
#         return {f'W: {w_param}', f'MFPT: {mfpt}'}
#     elif dep_type == "V":
#         return {f'V: {w_param}', f'MFPT: {mfpt}'}
#
#
# # Will require further inspection
# def parallel_process_mfpt(N_list, rg_param, ry_param, dep_type, ind_type, dep_param, ind_list, cores=None):
#     dep_type = dep_type.upper()
#     if dep_type != "W" and dep_type != "V":
#         raise (f"{dep_type} is an undefined dependent parameter. The current available "
#                f"dependent parameters are Switch Rate (W), and Velocity (V)")
#
#     print(f"{ind_type} list: {ind_list}")
#
#     current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
#     data_filepath = tb.create_directory(fp.mfpt_results_output, current_time)
#
#     if cores is not None and cores > 0:
#         core_count = min(cores, config.core_amount)
#     else:
#         core_count = config.core_amount
#
#     for n in range(len(N_list)):
#         with mp.Pool(processes=core_count) as pool:
#             mfpt_results = pool.map(partial(solve_mfpt_multi_process, N_list[n],
#                                             rg_param, ry_param, dep_type, dep_param), ind_list)
#         print(mfpt_results)
#         tb.produce_csv_from_xy(mfpt_results, dep_type, "MFPT", data_filepath,
#                                f'MFPT_Results_N={len(N_list[n])}_{ind_type}={dep_param}_')
#
#
# # Will require further inspection
# def solve_mfpt(rg_param, ry_param, N_param, v_param, w_param, r=1.0, d=1.0, mass_checkpoint=10 ** 6,
#                mass_threshold=0.01, return_duration=False, mixed_config=False, mx_cn_rrange=1):
#     diff_layer, adv_layer = sup.initialize_layers(rg_param, ry_param)
#     mfpt, duration = mfpt_comp.comp_mfpt_by_mass_loss_rect(rg_param, ry_param, w_param, w_param, v_param,
#                                                            N_param, diff_layer, adv_layer, mass_checkpoint, r, d,
#                                                            mass_threshold, mixed_config, mx_cn_rrange)
#
#     if return_duration:
#         return mfpt, duration
#     else:
#         return mfpt

# <**********************************************************> REQUIRES REVIEW (These functions have not been used since 2024)
