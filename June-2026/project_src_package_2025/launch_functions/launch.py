from . import mfpt_comp, sup, ant, np, num, super, os, pd, datetime, fp

from computational_tools import struct_init
from data_processing import data_process_functions as pro
import matplotlib.pyplot as plt


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
                                        MA_ALoI_timeseries, v_param, w_param, N_LIST, T_param, rg_param, ry_param,
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
    print("Successfully completed super-function. View results in project_src_package_2025/data_output.")

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

# (****) (****)
def collect_mass_analysis(rg_param, ry_param, v_param, w_param, T_param, N_LIST, MA_collection_factor=5,
                          domain_radius=1.0, D=1.0,
                          mass_checkpoint=10 ** 6, d_tube=0.0, MA_collection_factor_limit=10 ** 3, save_png=True,
                          show_plt=False,
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

    if MA_collection_factor < 1 or MA_collection_factor > MA_collection_factor_limit:
        print("MA_collection_factor automatically adjusted to legal range.")
        MA_collection_factor = 100

    K = num.compute_K(rg_param, ry_param, T_param, domain_radius, D)
    relative_k = int(np.floor(K / MA_collection_factor))

    D_LAYER, A_LAYER = sup.initialize_layers(rg_param, ry_param)

    MA_DL_timeseries = np.zeros([relative_k], dtype=np.float64)
    MA_AL_timeseries = np.zeros([relative_k], dtype=np.float64)
    MA_ALoT_timeseries = np.zeros([relative_k], dtype=np.float64)
    MA_ALoI_timeseries = np.zeros([relative_k], dtype=np.float64)
    MA_TM_timeseries = np.zeros([relative_k], dtype=np.float64)

    # delta_R = num.compute_dRad(rg_param)
    # delta_theta = num.compute_dThe(ry_param)
    # fixed_ring_seg = int(np.floor(rg_param * 0.5))
    b_param = w_param
    # a_param = b_param/(delta_R * delta_theta * (fixed_ring_seg + 1))
    a_param = w_param

    _validate_off_center_ic(center_init_cond, m_init, n_init, rg_param, ry_param)

    ant.comp_mass_analysis_respect_to_time(rg_param, ry_param, a_param, b_param, v_param, T_param, N_LIST, D_LAYER,
                                           A_LAYER, MA_DL_timeseries, MA_AL_timeseries, MA_ALoI_timeseries,
                                           MA_ALoT_timeseries, MA_TM_timeseries, MA_collection_factor,
                                           relative_k, d_tube, domain_radius, D, mass_checkpoint,
                                           center_init_cond, m_init, n_init)

    return pro.process_MA_results(MA_DL_timeseries, MA_AL_timeseries, MA_TM_timeseries, MA_ALoT_timeseries,
                                  MA_ALoI_timeseries,
                                  v_param, w_param, N_LIST, T_param, rg_param, ry_param, save_png, show_plt, MA_collection_factor, domain_radius, D)


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


def collect_char_time_mass(rg_param, ry_param, v_LIST, w_param, T_param, N_LIST, MA_collection_factor=5, domain_radius=1.0, D=1.0,
                           mass_checkpoint=10 ** 6, d_tube=0.0, center_init_cond=True, m_init=0, n_init=0, show_plt=True):

    K = num.compute_K(rg_param, ry_param, T_param, domain_radius, D)
    print("deltaT = ", num.compute_dT(rg_param, ry_param))
    relative_k = int(np.floor(K / MA_collection_factor))

    b_param = w_param
    a_param = w_param

    m_star_dict = {}
    t_star_dict = {}

    # adjacent time points for the log scale line computation
    x1 = 0.4
    x2 = 0.5

    # convert the real time to discretized time
    t1_ = num.compute_K(rg_param, ry_param, x1)
    t1 = num.closest_multiple(t1_, MA_collection_factor) // MA_collection_factor

    t2_ = num.compute_K(rg_param, ry_param, x2)
    t2 = num.closest_multiple(t2_, MA_collection_factor) // MA_collection_factor

    # Pre-flight bounds check (cheap, runs before any solving).
    # t1 / t2 are the log-fit sample indices into MA_*_timeseries, which has
    # relative_k entries.  Because x1 / x2 are fixed at 0.4 / 0.5 above, this
    # function silently requires T_param > x2; otherwise the fit below would
    # index past the end of the series.  Validating here means a bad parameter
    # set fails immediately instead of after the full time-stepping loop.
    if not (0 <= t1 < relative_k and 0 <= t2 < relative_k):
        raise ValueError(
            f"characteristic-time fit indices out of range: "
            f"t1={t1}, t2={t2}, valid range [0, {relative_k}). "
            f"The fit window is hard-coded at x1={x1}, x2={x2}, so T_param "
            f"must exceed {x2} (got T_param={T_param}).")

    for v_param in v_LIST:

        D_LAYER, A_LAYER = sup.initialize_layers(rg_param, ry_param)
        MA_DL_timeseries = np.zeros([relative_k], dtype=np.float64)
        MA_AL_timeseries = np.zeros([relative_k], dtype=np.float64)
        MA_ALoT_timeseries = np.zeros([relative_k], dtype=np.float64)
        MA_ALoI_timeseries = np.zeros([relative_k], dtype=np.float64)
        MA_TM_timeseries = np.zeros([relative_k], dtype=np.float64)

        _validate_off_center_ic(center_init_cond, m_init, n_init, rg_param, ry_param)

        ant.comp_mass_analysis_respect_to_time(rg_param, ry_param, a_param, b_param, v_param, T_param, N_LIST, D_LAYER,
                                               A_LAYER, MA_DL_timeseries, MA_AL_timeseries, MA_ALoI_timeseries,
                                               MA_ALoT_timeseries, MA_TM_timeseries, MA_collection_factor,
                                               relative_k, d_tube, domain_radius, D, mass_checkpoint,
                                               center_init_cond, m_init, n_init)

        y1 = np.log10(MA_TM_timeseries[t1])
        y2 = np.log10(MA_TM_timeseries[t2])

        m = (y2 - y1) / (x2 - x1)
        b = y2 - m*x2

        # characteristic time
        t_star = -b/m
        t_star_dict[v_param] = t_star

        # compute the mass corresponding to t_star, and then record
        k_star_ = num.compute_K(rg_param, ry_param, 10 * t_star)
        k_star = num.closest_multiple(k_star_, MA_collection_factor) // MA_collection_factor

        # Guard the k_star lookup.
        # k_star is derived from the *fitted* t_star, so it cannot be validated
        # before the solve.  A degenerate fit (t_star non-finite, non-positive,
        # or large enough that 10 * t_star exceeds T_param) would index outside
        # MA_TM_timeseries and raise a bare IndexError only after the entire
        # time-stepping loop has finished -- discarding hours of computation for
        # a one-line lookup.  Instead, record m_star as NaN and report loudly:
        # t_star is unaffected and is still written to the CSV.
        if (not np.isfinite(t_star)) or t_star <= 0 or not (0 <= k_star < relative_k):
            print("*** WARNING: degenerate characteristic-time fit ***")
            print(f"    v          = {v_param}")
            print(f"    t_star     = {t_star}")
            print(f"    k_star     = {k_star}  (valid range [0, {relative_k}))")
            print(f"    m_star requires 10 * t_star < T_param; "
                  f"here T_param={T_param} and 10 * t_star={10 * t_star}")
            print("    recording m_star = NaN; t_star is still reported.")
            m_star = np.nan
        else:
            m_star = MA_TM_timeseries[k_star]
        m_star_dict[v_param] = m_star

    v_axis = list(m_star_dict.keys())
    m_axis = list(m_star_dict.values())

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
    df = pd.DataFrame({
        'v': v_axis,
        't_star': [t_star_dict[v_param] for v_param in v_axis],
        'm_star': m_axis
    })
    df.to_csv(data_location, index=False)

    print(f'Characteristic time plot saved to {plot_location}')
    print(f'Characteristic time data saved to {data_location}')

    if show_plt:
        plt.show()


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
