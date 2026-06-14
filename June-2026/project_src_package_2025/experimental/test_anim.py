import numpy as np
from numba import njit
from project_src_package_2025.computational_tools import numerical_tools as num
from project_src_package_2025.computational_tools import supplements as sup
from project_src_package_2025.data_processing import data_process_functions as pro
from project_src_package_2025.system_configuration import sys_config

ENABLE_JIT = sys_config.ENABLE_NJIT


@njit(nopython=ENABLE_JIT)
def collect_stamps_for_animation_test(rings, rays, a, b, v, tube_placements, diffusive_layer, advective_layer,
                                      PvT_DL_snapshots, Timestamp_List, center, K,
                                      r=1.0, d=1.0, d_tube=-1, T_fixed_ring_seg=0.5):
    d_radius = r / rings
    d_theta = ((2 * np.pi) / rays)
    d_time = (0.1 * min(d_radius * d_radius, d_theta * d_theta * d_radius * d_radius)) / (2 * d)

    d_list = [ ]

    if d_tube < 0:
        d_tube = sup.solve_d_rect(1, rings, rays, sup.j_max_bef_overlap(rays, tube_placements), 0)

    for m in range(rings):
        j_max = np.ceil((d_tube / ((m + 1) * d_radius * d_theta)) - 0.5)
        keys = sup.mod_range_flat(tube_placements, j_max, rays, False)
        dict_ = sup.dict_gen(keys, tube_placements)
        d_list.append(dict_)

    stamp_iter = 0

    for k in range(K - 1):
        m = 0
        # aIdx = 0
        while m < rings:
            aIdx = 0
            n = 0
            while n < rays:
                if m == rings - 1:
                    diffusive_layer[ k + 1 ][ m ][ n ] = 0
                else:
                    if n in d_list[ m ]:
                        diffusive_layer[ k + 1 ][ m ][ n ] = num.u_density_rect(diffusive_layer, k, m, n, d_radius,
                                                                                d_theta,
                                                                                d_time, center[ k ], rings,
                                                                                advective_layer,
                                                                                int(d_list[ m ][ n ]), a, b, d_tube)
                    else:
                        diffusive_layer[ k + 1 ][ m ][ n ] = num.u_density(diffusive_layer, k, m, n, d_radius, d_theta,
                                                                           d_time, center[ k ], rings, advective_layer,
                                                                           aIdx, a, b, tube_placements)
                    if n == tube_placements[ aIdx ]:
                        advective_layer[ k + 1 ][ m ][ n ] = num.u_tube_rect(advective_layer, diffusive_layer, k, m, n,
                                                                             a,
                                                                             b, v, d_time, d_radius, d_theta, d_tube)
                        if aIdx < len(tube_placements) - 1:
                            aIdx += 1
                n += 1
            m += 1

        center[ k + 1 ] = num.u_center(diffusive_layer, k, d_radius, d_theta, d_time, center[ k ],
                                       advective_layer, tube_placements, v)

        if stamp_iter < len(Timestamp_List):
            curr_stamp = int(np.floor(Timestamp_List[ stamp_iter ] / d_time))
            if curr_stamp == k:
                print(curr_stamp)
                PvT_DL_snapshots[ stamp_iter ] = diffusive_layer[ k ][ int(np.floor(rings * T_fixed_ring_seg)) ]
                stamp_iter += 1


def test_batch_method_1_iter(rg_param, ry_param, w_param, v_param, N_LIST, T_LIST, T_param, T_fixed_ring_seg, D=1.0,
                             domain_radius=1.0,
                             save_png=True, show_plt=False):
    PvT_DL_snapshots = np.zeros((len(T_LIST), ry_param), dtype=np.float64)

    K = int((T_param / num.compute_dT(rg_param, ry_param, domain_radius, D))) + 2

    v_param *= -1

    init_cond_center = num.compute_init_cond_cent(rg_param)

    D_LAYER = np.zeros((K, rg_param, ry_param), dtype=np.float64)
    A_LAYER = np.zeros((K, rg_param, ry_param), dtype=np.float64)
    C_list = np.zeros([K], dtype=np.float64)
    C_list[ 0 ] = init_cond_center

    collect_stamps_for_animation_test(rg_param, ry_param, w_param, w_param, v_param, N_LIST, D_LAYER, A_LAYER,
                                      PvT_DL_snapshots,
                                      T_LIST, C_list, K, d_tube=0, T_fixed_ring_seg=T_fixed_ring_seg)

    pro.process_PvT_DL(PvT_DL_snapshots, v_param, w_param, N_LIST, T_fixed_ring_seg, save_png, show_plt, T_LIST)
