from . import mfpt_comp, sup, ant, mp, np, num, super, os, pd, datetime, fp

from computational_tools import struct_init
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

# (****) (****)
def collect_mass_analysis(rg_param, ry_param, v_param, w_param, T_param, N_LIST, MA_collection_factor=5,
                          domain_radius=1.0, D=1.0,
                          mass_checkpoint=10 ** 6, d_tube=0.0, MA_collection_factor_limit=10 ** 3, save_png=True,
                          show_plt=False,
                          center_init_cond=True, m_init=0, n_init=0,
                          device=DEVICE_CPU):

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

    # delta_R = num.compute_dRad(rg_param)
    # delta_theta = num.compute_dThe(ry_param)
    # fixed_ring_seg = int(np.floor(rg_param * 0.5))
    b_param = w_param
    # a_param = b_param/(delta_R * delta_theta * (fixed_ring_seg + 1))
    a_param = w_param

    _validate_off_center_ic(center_init_cond, m_init, n_init, rg_param, ry_param)

    _solve_mass_analysis(device, series, rg_param, ry_param, a_param, b_param,
                         v_param, T_param, N_LIST, D_LAYER, A_LAYER,
                         MA_collection_factor, relative_k, d_tube,
                         domain_radius, D, mass_checkpoint, center_init_cond,
                         m_init, n_init)

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


def _setup_checkpoint(tag, n_samples, **fp_params):
    """Prepare checkpointing for one solve, if it is enabled for this run.

    ``n_samples`` is the timeseries length; it is named distinctly from the
    fingerprint's own ``relative_k`` so the two cannot collide in **fp_params.

    Enabled by the ITCM_CHECKPOINT_DIR environment variable, which the job
    worker sets per job. Left unset -- direct scripts, the desktop GUI -- this
    returns no checkpoint and ordinary in-RAM timeseries, so nothing about the
    existing paths changes.

    Returns ``(checkpoint_or_None, five_timeseries)``.
    """
    import os
    from computational_tools import checkpointing as chk

    root = os.environ.get("ITCM_CHECKPOINT_DIR")
    if not root:
        return None, chk.allocate_timeseries(n_samples, 5)

    directory = os.path.join(root, tag)
    digest, params = chk.fingerprint(**fp_params)
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

            def on_chunk(k, state):
                cp.save(k, D_LAYER, A_LAYER, state, series=series)

            result = gpu_driver.solve(
                rg_param, ry_param, a_param, b_param, v_param, T_param,
                N_LIST, D_LAYER, A_LAYER, series, MA_collection_factor,
                relative_k, d_tube=d_tube, domain_radius=domain_radius, D=D,
                center_init_cond=center_init_cond,
                on_chunk=None if cp is None else on_chunk,
                resume_state=resume_state)
            if cp is not None:
                cp.clear()
            return result

    return ant.comp_mass_analysis_respect_to_time(
        rg_param, ry_param, a_param, b_param, v_param, T_param, N_LIST,
        D_LAYER, A_LAYER, series[0], series[1], series[2], series[3],
        series[4], MA_collection_factor, relative_k, d_tube, domain_radius, D,
        mass_checkpoint, center_init_cond, m_init, n_init, checkpoint=cp)


# The log-linear fit window behind t*, in dimensionless time. Fixed rather than
# exposed as a parameter: a characteristic time in this project is *defined* as
# the decay of log10(total mass) across these two points, so a kernel fitting a
# different window would report a t* not comparable with any number already
# published. Both characteristic-time kernels read the window from here, so the
# two cannot silently drift apart.
CHAR_TIME_FIT_X1 = 0.4
CHAR_TIME_FIT_X2 = 0.5


def _char_time_fit_indices(rg_param, ry_param, T_param, MA_collection_factor,
                           relative_k):
    """Sample indices into the mass timeseries for the two fit points.

    Pre-flight and cheap: this runs before any solving, so a parameter set whose
    fit window falls outside the requested T fails immediately instead of after
    the full time-stepping loop -- which at 160x160 means days.

    Because the window is fixed at CHAR_TIME_FIT_X1 / _X2, every
    characteristic-time computation silently requires ``T_param`` to exceed
    CHAR_TIME_FIT_X2; otherwise the fit would index past the end of a series
    that only holds ``relative_k`` samples.

    Note that ``compute_K`` is called here with the default domain_radius and D
    rather than the run's own, matching what collect_char_time_mass has always
    done. The two agree for the only values ever used (1.0 and 1.0); they would
    not for any other, so both kernels share this one definition rather than
    each growing its own.
    """
    t1_ = num.compute_K(rg_param, ry_param, CHAR_TIME_FIT_X1)
    t1 = num.closest_multiple(t1_, MA_collection_factor) // MA_collection_factor

    t2_ = num.compute_K(rg_param, ry_param, CHAR_TIME_FIT_X2)
    t2 = num.closest_multiple(t2_, MA_collection_factor) // MA_collection_factor

    if not (0 <= t1 < relative_k and 0 <= t2 < relative_k):
        raise ValueError(
            f"characteristic-time fit indices out of range: "
            f"t1={t1}, t2={t2}, valid range [0, {relative_k}). "
            f"The fit window is hard-coded at x1={CHAR_TIME_FIT_X1}, "
            f"x2={CHAR_TIME_FIT_X2}, so T_param must exceed "
            f"{CHAR_TIME_FIT_X2} (got T_param={T_param}).")
    return t1, t2


def _char_time_point(tag, rg_param, ry_param, a_param, b_param, v_param,
                     T_param, N_LIST, MA_collection_factor, relative_k, t1, t2,
                     d_tube, domain_radius, D, mass_checkpoint,
                     center_init_cond, m_init, n_init, device, label=""):
    """One solve plus the log-linear fit that defines t*.

    This is the characteristic-time kernel proper: everything done per parameter
    point, with no plotting, no file writing and no knowledge of what is being
    swept. ``collect_char_time_mass`` calls it once per velocity;
    ``collect_ab_grid_char_time`` calls it once per (a, b) pair, several at a
    time in separate processes. Keeping it in one place is what makes the two
    kernels report the same t* for the same parameters, which is asserted by
    tests/test_ab_grid_char_time.py.

    ``label`` names this point in the degenerate-fit warning below, since the
    caller knows what is being swept and this function does not.

    Returns plain Python floats in a dict -- plain because the result has to
    survive being pickled back from a worker process.
    """
    D_LAYER, A_LAYER = sup.initialize_layers(rg_param, ry_param)

    # One checkpoint directory per point, since each is a separate solve;
    # without a distinguishing tag a sweep would resume the wrong trajectory.
    cp, series = _setup_checkpoint(
        tag, relative_k,
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

    _validate_off_center_ic(center_init_cond, m_init, n_init, rg_param, ry_param)

    _solve_mass_analysis(device, series, rg_param, ry_param, a_param, b_param,
                         v_param, T_param, N_LIST, D_LAYER, A_LAYER,
                         MA_collection_factor, relative_k, d_tube,
                         domain_radius, D, mass_checkpoint,
                         center_init_cond, m_init, n_init, cp=cp)

    y1 = np.log10(MA_TM_timeseries[t1])
    y2 = np.log10(MA_TM_timeseries[t2])

    m = (y2 - y1) / (CHAR_TIME_FIT_X2 - CHAR_TIME_FIT_X1)
    b = y2 - m * CHAR_TIME_FIT_X2

    # characteristic time
    t_star = -b / m

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
        print(f"    point      = {label or tag}")
        print(f"    t_star     = {t_star}")
        print(f"    k_star     = {k_star}  (valid range [0, {relative_k}))")
        print(f"    m_star requires 10 * t_star < T_param; "
              f"here T_param={T_param} and 10 * t_star={10 * t_star}")
        print("    recording m_star = NaN; t_star is still reported.")
        m_star = np.nan
    else:
        m_star = MA_TM_timeseries[k_star]

    # Record the fit itself, not just what it implies.
    #
    # The slope is the decay rate of log10(total mass) across the window and
    # is a result in its own right -- it is what t* is derived from, and it
    # varies with grid size in a way t* alone does not show. It used to be a
    # local that went out of scope, which meant recovering it afterwards
    # needed the mass timeseries, and that is not retained: for the GPU runs
    # nothing is logged per step, so the slope was simply unrecoverable
    # short of re-solving. The two window masses are kept too, since they
    # are the entire input to the fit and cost two floats.
    return {
        "t_star": float(t_star),
        "m_star": float(m_star),
        "fit_slope": float(m),
        "fit_intercept": float(b),
        "total_mass_at_t1": float(MA_TM_timeseries[t1]),
        "total_mass_at_t2": float(MA_TM_timeseries[t2]),
    }


def collect_char_time_mass(rg_param, ry_param, v_LIST, w_param, T_param, N_LIST, MA_collection_factor=5, domain_radius=1.0, D=1.0,
                           mass_checkpoint=10 ** 6, d_tube=0.0, center_init_cond=True, m_init=0, n_init=0, show_plt=True,
                           device=DEVICE_CPU):

    K = num.compute_K(rg_param, ry_param, T_param, domain_radius, D)
    print("deltaT = ", num.compute_dT(rg_param, ry_param))
    relative_k = int(np.floor(K / MA_collection_factor))

    b_param = w_param
    a_param = w_param

    x1 = CHAR_TIME_FIT_X1
    x2 = CHAR_TIME_FIT_X2
    t1, t2 = _char_time_fit_indices(rg_param, ry_param, T_param,
                                    MA_collection_factor, relative_k)

    # One fit per velocity, keyed by it, in the order the sweep was requested.
    fits = {}
    for v_param in v_LIST:
        fits[v_param] = _char_time_point(
            f"char_time_v{v_param:g}", rg_param, ry_param, a_param, b_param,
            v_param, T_param, N_LIST, MA_collection_factor, relative_k, t1, t2,
            d_tube, domain_radius, D, mass_checkpoint, center_init_cond,
            m_init, n_init, device, label=f"v = {v_param}")

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
    df = pd.DataFrame({
        'v': v_axis,
        't_star': [fits[v_param]["t_star"] for v_param in v_axis],
        'm_star': m_axis,
        'fit_slope': [fits[v_param]["fit_slope"] for v_param in v_axis],
        'fit_intercept': [fits[v_param]["fit_intercept"] for v_param in v_axis],
        'fit_window_t1': [x1] * len(v_axis),
        'fit_window_t2': [x2] * len(v_axis),
        'total_mass_at_t1': [fits[v_param]["total_mass_at_t1"] for v_param in v_axis],
        'total_mass_at_t2': [fits[v_param]["total_mass_at_t2"] for v_param in v_axis],
    })
    df.to_csv(data_location, index=False)

    print(f'Characteristic time plot saved to {plot_location}')
    print(f'Characteristic time data saved to {data_location}')

    if show_plt:
        plt.show()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# v==================================== Characteristic time over an (a, b) grid ====================================v
#
# WHY THE (a, b) POINTS RUN AS CONCURRENT PROCESSES ON THE CPU
#
# Every (a, b) pair is an independent solve: the pairs share no state, and each
# one only reads the parameters it was given and returns six floats. So the grid
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


def collect_ab_grid_char_time(rg_param, ry_param, a_list, b_list, v_param, T_param,
                              N_LIST, MA_collection_factor=5, domain_radius=1.0,
                              D=1.0, mass_checkpoint=10 ** 6, d_tube=0.0,
                              center_init_cond=True, m_init=0, n_init=0,
                              show_plt=False, device=DEVICE_CPU, workers=0):
    """Characteristic time t* over the full grid of (a, b) switch rates.

    Where ``collect_char_time_mass`` sweeps velocity at a single switch rate
    (a = b = w), this sweeps the two switch rates independently at a single
    velocity: ``a_list`` supplies the rate onto the diffusive layer and
    ``b_list`` the rate onto the advective layer, and every pair in the cartesian
    product ``a_list x b_list`` is solved. ``v_param`` is one value here, not a
    list -- the grid is already two-dimensional, and a third axis would multiply
    an already expensive job by the length of a velocity list.

    Each point is the same solve-and-fit as the velocity sweep, run through the
    shared ``_char_time_point`` kernel, so a 1x1 grid at (w, w) reproduces
    ``collect_char_time_mass`` for that velocity exactly.

    Outputs, under ``data_output/ab_grid_char_time/<timestamp>/``:
      * ``ab_grid_char_t_data.csv``  -- one row per (a, b), with t*, m* and the
        fit each was derived from
      * ``ab_grid_char_t_heatmap.png`` -- t* over the grid

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

    x1 = CHAR_TIME_FIT_X1
    x2 = CHAR_TIME_FIT_X2
    t1, t2 = _char_time_fit_indices(rg_param, ry_param, T_param,
                                    MA_collection_factor, relative_k)

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
        relative_k=relative_k, t1=t1, t2=t2, d_tube=d_tube,
        domain_radius=domain_radius, D=D, mass_checkpoint=mass_checkpoint,
        center_init_cond=center_init_cond, m_init=m_init, n_init=n_init,
        device=DEVICE_GPU if on_gpu else DEVICE_CPU)

    print("=== characteristic time over an (a, b) grid ===")
    print(f"grid          : {rg_param}x{ry_param}")
    print(f"a list        : {a_list}")
    print(f"b list        : {b_list}")
    print(f"v             : {v_param}")
    print(f"T             : {T_param}")
    print(f"(a, b) points : {len(points)}")

    if on_gpu:
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

    data_location = os.path.join(output_directory, 'ab_grid_char_t_data.csv')
    # a, b, v, t_star and m_star lead; the fit that produced them follows, in
    # the same column names collect_char_time_mass writes, so the two CSVs can
    # be concatenated.
    df = pd.DataFrame({
        'a': [r["a"] for r in rows],
        'b': [r["b"] for r in rows],
        'v': [v_param] * len(rows),
        't_star': [r["t_star"] for r in rows],
        'm_star': [r["m_star"] for r in rows],
        'fit_slope': [r["fit_slope"] for r in rows],
        'fit_intercept': [r["fit_intercept"] for r in rows],
        'fit_window_t1': [x1] * len(rows),
        'fit_window_t2': [x2] * len(rows),
        'total_mass_at_t1': [r["total_mass_at_t1"] for r in rows],
        'total_mass_at_t2': [r["total_mass_at_t2"] for r in rows],
    })
    df.to_csv(data_location, index=False)

    # --- heatmap -----------------------------------------------------------
    # Rows are a, columns are b, in the order they were supplied. Drawn on its
    # own Figure and closed afterwards rather than through the pyplot state
    # machine, so two grids run back to back in one process do not share axes.
    t_star_grid = np.array([r["t_star"] for r in rows],
                           dtype=np.float64).reshape(len(a_list), len(b_list))

    fig, ax = plt.subplots(figsize=(max(4.0, 1.1 * len(b_list) + 2.5),
                                    max(3.5, 1.0 * len(a_list) + 2.0)))
    image = ax.imshow(t_star_grid, origin='lower', aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(b_list)))
    ax.set_xticklabels([f"{b:g}" for b in b_list])
    ax.set_yticks(range(len(a_list)))
    ax.set_yticklabels([f"{a:g}" for a in a_list])
    ax.set_xlabel(r"($b$) switch rate onto the advective layer")
    ax.set_ylabel(r"($a$) switch rate onto the diffusive layer")
    ax.set_title(r"$t^*(a, b)$, " + f"N={len(N_LIST)}, " + f"v={v_param:g}, "
                 + f"grid={rg_param}x{ry_param}")
    fig.colorbar(image, ax=ax, label=r"characteristic time $t^*$")

    # Print the value in each cell on a grid small enough to read it. On a
    # coarse grid -- which is what these are, being one full solve per cell --
    # the number is the result and the colour is only the summary.
    #
    # Text colour follows the cell rather than being fixed: viridis runs from
    # dark blue to bright yellow, so a single colour is illegible over half the
    # range whichever one is chosen.
    if len(a_list) * len(b_list) <= 64:
        finite = t_star_grid[np.isfinite(t_star_grid)]
        midpoint = (finite.min() + finite.max()) / 2 if finite.size else 0.0
        for i in range(len(a_list)):
            for j in range(len(b_list)):
                value = t_star_grid[i, j]
                ax.text(j, i, "--" if not np.isfinite(value) else f"{value:.4g}",
                        ha='center', va='center', fontsize=7,
                        color='k' if np.isfinite(value) and value > midpoint
                        else 'w')

    plot_location = os.path.join(output_directory, 'ab_grid_char_t_heatmap.png')
    fig.savefig(plot_location, bbox_inches='tight')

    print(f'(a, b) grid characteristic-time data saved to {data_location}')
    print(f'(a, b) grid characteristic-time heatmap saved to {plot_location}')

    if show_plt:
        plt.show()
    plt.close(fig)

    return {"output_dir": output_directory, "csv": data_location,
            "plot": plot_location, "points": len(rows),
            "device": DEVICE_GPU if on_gpu else DEVICE_CPU}
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
