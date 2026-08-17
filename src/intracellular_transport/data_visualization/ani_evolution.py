from . import np
from . import sup
from . import num
from . import plt
from . import njit
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import cm
from matplotlib.colors import Normalize

GLOBAL_ani = None
GLOBAL_fig = None


class BatchManager:
    def __init__(self, rg_param, ry_param, w_param, v_param, N_param, K_param, T_param, d_tube, r=1, d=1):
        self.rg_param = rg_param
        self.ry_param = ry_param
        self.w_param = w_param
        self.v_param = v_param * -1
        self.N_param = N_param
        self.K_param = K_param
        self.T_param = T_param
        self.d_tube = d_tube

        self.d_radius = r / self.rg_param
        self.d_theta = ((2 * np.pi) / self.ry_param)
        self.d_time = (0.1 * min(self.d_radius ** 2, self.d_theta ** 2 * self.d_radius ** 2)) / (2 * d)

        self.batch_counter = 0
        self.diff_batch = np.zeros((K_param, rg_param, ry_param), dtype=np.float64)
        self.adv_batch = np.zeros((K_param, rg_param, ry_param), dtype=np.float64)
        self.cen_batch = np.zeros(K_param, dtype=np.float64)

        d_radius = r / self.rg_param

        self.cen_batch[0] = 1 / (np.pi * (d_radius * d_radius))

        self.diff_batch, self.adv_batch, self.cen_batch = collect_stamps_for_animation(
            rg_param, ry_param,
            w_param, w_param,
            v_param, N_param,
            self.diff_batch, self.adv_batch, self.cen_batch,
            K_param, r=r, d=d,  d_tube=d_tube
        )

    def get_current_batch(self):
        return self.diff_batch, self.adv_batch, self.cen_batch

    def compute_next_batch(self):

        self.batch_counter += 1

        diff_next = np.zeros((self.K_param, self.rg_param, self.ry_param), dtype=np.float64)
        adv_next = np.zeros((self.K_param, self.rg_param, self.ry_param), dtype=np.float64)
        cen_next = np.zeros(self.K_param, dtype=np.float64)

        diff_next[0] = self.diff_batch[-1]
        adv_next[0] = self.adv_batch[-1]
        cen_next[0] = self.cen_batch[-1]

        self.diff_batch, self.adv_batch, self.cen_batch = collect_stamps_for_animation(
            self.rg_param, self.ry_param,
            self.w_param, self.w_param,
            self.v_param, self.N_param,
            diff_next, adv_next, cen_next,
            self.K_param, d_tube=self.d_tube
        )

def compute_batches_in_background(rg, ry, w, v, N, K_param, T, d_tube, result_queue):
    batch_mgr = BatchManager(rg, ry, w, v, N, K_param, T, d_tube)
    total_batches = int(np.ceil(T / batch_mgr.d_time / K_param))

    for _ in range(total_batches):
        batch_mgr.compute_next_batch()
        diff_batch, adv_batch, cen_batch = batch_mgr.get_current_batch()
        result_queue.put((diff_batch, adv_batch, cen_batch))

    result_queue.put("DONE")


# Implement the following into analysis tools
@njit
def collect_stamps_for_animation(rings, rays, a, b, v, tube_placements, diffusive_layer, advective_layer, center, K,
                                 r=1.0, d=1.0, d_tube=-1):
    d_radius = r / rings
    d_theta = ((2 * np.pi) / rays)
    d_time = (0.1 * min(d_radius * d_radius, d_theta * d_theta * d_radius * d_radius)) / (2 * d)

    d_list = []

    if d_tube < 0:
        d_tube = sup.solve_d_rect(1, rings, rays, sup.j_max_bef_overlap(rays, tube_placements), 0)

    for m in range(rings):
        j_max = np.ceil((d_tube / ((m + 1) * d_radius * d_theta)) - 0.5)
        keys = sup.mod_range_flat(tube_placements, j_max, rays, False)
        dict_ = sup.dict_gen(keys, tube_placements)
        d_list.append(dict_)

    for k in range(K - 1):
        m = 0
        # aIdx = 0

        while m < rings:
            aIdx = 0
            n = 0
            while n < rays:
                if m == rings - 1:
                    diffusive_layer[k + 1][m][n] = 0
                else:
                    if n in d_list[m]:
                        diffusive_layer[k + 1][m][n] = num.u_density_rect(diffusive_layer, k, m, n, d_radius, d_theta,
                                                                          d_time, center[k], rings, advective_layer,
                                                                          int(d_list[m][n]), a, b, d_tube)
                    else:
                        diffusive_layer[k + 1][m][n] = num.u_density(diffusive_layer, k, m, n, d_radius, d_theta,
                                                                     d_time, center[k], rings, advective_layer,
                                                                     aIdx, a, b, tube_placements)
                    if n == tube_placements[aIdx]:
                        advective_layer[k + 1][m][n] = num.u_tube_rect(advective_layer, diffusive_layer, k, m, n, a,
                                                                       b, v, d_time, d_radius, d_theta, d_tube)
                        if aIdx < len(tube_placements) - 1:
                            aIdx += 1
                n += 1
            m += 1

        center[k+1] = num.u_center(diffusive_layer, k, d_radius, d_theta, d_time, center[k],
                                   advective_layer, tube_placements, v)

    return diffusive_layer, advective_layer, center

def animate_diffusion(
        rg_param, ry_param, w_param, v_param, N_param, K_param, T_param, d_tube,
        steps_per_frame=10, interval_ms=10, color_scheme='viridis', epsilon=0.001, border=True
):
    batch_mgr = BatchManager(rg_param, ry_param, w_param, v_param, N_param, K_param, T_param, d_tube)
    diff_batch, adv_batch, cen_batch = batch_mgr.get_current_batch()

    frame_in_batch = 0

    rings = rg_param + 1
    rays = ry_param

    r = np.linspace(0, 1, rings + 1)
    theta = np.linspace(0, 2 * np.pi, rays + 1)
    R, Theta = np.meshgrid(r, theta)
    X, Y = R * np.cos(Theta), R * np.sin(Theta)

    fig, ax = plt.subplots(figsize=(10, 10))
    canvas = FigureCanvas(fig)

    cmap = cm.get_cmap(color_scheme, 512)
    norm = Normalize(vmin=0, vmax=1)
    initial = np.vstack([np.full((1, ry_param), cen_batch[0]), diff_batch[0]])
    if border:
        heatmap = ax.pcolormesh(X, Y, initial.T, shading='flat', cmap=cmap, edgecolors='k', linewidth=0.01, norm=norm)
    else:
        heatmap = ax.pcolormesh(X, Y, initial.T, shading='flat', cmap=cmap, norm=norm)
    ax.axis('off')
    sim_time = 0

    def update(frame):
        nonlocal diff_batch, cen_batch, frame_in_batch, heatmap, sim_time

        #
        # Check if current batch is exhausted
        if frame_in_batch >= K_param:
            batch_mgr.compute_next_batch()
            diff_batch, _, cen_batch = batch_mgr.get_current_batch()
            frame_in_batch = 0
            print(f"Loaded new batch at frame {frame}")

        # Get current density data
        diff = diff_batch[frame_in_batch]
        center = cen_batch[frame_in_batch]

        # Reconstruct full layer with center
        full_layer = np.vstack([np.full((1, ry_param), center), diff])  # (rg_param+1, ry_param)

        # Remove old mesh and draw new one
        heatmap.remove()

        full_layer_T = full_layer.T

        if border:
            heatmap = ax.pcolormesh(X, Y, full_layer_T, shading='flat', cmap=cmap, edgecolors='k', linewidth=0.01, norm=norm)
        else:
            heatmap = ax.pcolormesh(X, Y, full_layer_T, shading='flat', cmap=cmap, norm=norm)

        sim_time += steps_per_frame * batch_mgr.d_time
        ax.set_title(f"Simulation time: {sim_time:.4f}", fontsize=14)

        if batch_mgr.T_param * (1 - epsilon) <= sim_time <= batch_mgr.T_param * (1 + epsilon):
            print(f"Simulation paused at time T ~ {sim_time:.3f}")
            ani.event_source.stop()

        frame_in_batch += steps_per_frame
        # plt.show()
        return [heatmap]

    ani = FuncAnimation(
        fig, update,
        frames=100000,  # effectively infinite
        interval=interval_ms,
        blit=False
    )

    fig.set_tight_layout(True)
    plt.subplots_adjust(top=0.9)

    canvas.ani = ani
    canvas.ani_paused = False
    canvas.fig = fig

    return canvas


def run_realtime_simulation(rg_param, ry_param, w_param, v_param, N_param, K_param, T_param, d_tube, steps_per_frame=10, interval_ms=10):

    estimated_memory = K_param * rg_param * ry_param * 8

    print(f"Memory expenditure/footprint: {estimated_memory} bytes = {(estimated_memory / 10**7)}% of a GB")

    animate_diffusion(rg_param, ry_param, w_param, -1*v_param, N_param, K_param, T_param, d_tube, steps_per_frame=steps_per_frame, interval_ms=interval_ms)

# ========================== To be re-implemented in a future update ==========================

# def animate_diffusion_mp(
#     rg_param, ry_param, w_param, v_param, N_param, K_param, T_param, d_tube,
#     steps_per_frame=10, interval_ms=10, color_scheme='viridis', result_queue=None, epsilon=0.001
# ):
#     rings = rg_param + 1
#     rays = ry_param
#
#     r = np.linspace(0, 1, rings + 1)
#     theta = np.linspace(0, 2 * np.pi, rays + 1)
#     R, Theta = np.meshgrid(r, theta)
#     X, Y = R * np.cos(Theta), R * np.sin(Theta)
#
#     fig, ax = plt.subplots(figsize=(10, 10))
#     canvas = FigureCanvas(fig)
#
#     cmap = cm.get_cmap(color_scheme, 512)
#     norm = Normalize(vmin=0, vmax=1)
#
#     # Initial dummy full_layer until first batch arrives
#     dummy_layer = np.zeros((rings + 1, rays))
#     # heatmap = ax.pcolormesh(X, Y, dummy_layer.T, shading='flat', cmap=cmap, edgecolors='k', linewidth=0.01, norm=norm)
#     heatmap = ax.pcolormesh(
#         X[:-1, :-1], Y[:-1, :-1], dummy_layer.T,
#         shading='flat', cmap=cmap, edgecolors='k', linewidth=0.01, norm=norm
#     )
#
#     ax.axis('off')
#     fig.set_tight_layout(True)
#     plt.subplots_adjust(top=0.9)
#
#     frame_idx = 0
#     sim_time = 0
#
#     # Estimate d_time roughly to use for stopping
#     d_radius = 1.0 / rings
#     d_theta = (2 * np.pi) / rays
#     d_time_est = (0.1 * min(d_radius**2, (d_theta**2) * d_radius**2)) / (2 * 1.0)  # Assume d = 1
#
#     # Wait for first real batch
#     while result_queue.empty():
#         time.sleep(0.05)
#
#     first_loaded = False
#     while not first_loaded:
#         batch = result_queue.get()
#         if batch == "DONE":
#             print("Received DONE before first batch")
#             return canvas
#         else:
#             curr_diff, curr_cen = batch
#             first_loaded = True
#
#     # Preload the next batch if available
#     preload_next = None
#     if not result_queue.empty():
#         preload_next = result_queue.get()
#
#     def update(frame):
#         nonlocal frame_idx, sim_time, heatmap, curr_diff, curr_cen, preload_next
#
#         if frame_idx >= K_param:
#             if preload_next is not None:
#                 if preload_next == "DONE":
#                     print(f"Simulation ended at T ~ {sim_time:.3f}")
#                     ani.event_source.stop()
#                     return
#                 curr_diff, curr_cen = preload_next
#                 preload_next = None
#                 frame_idx = 0
#             else:
#                 return  # Wait until next batch is ready
#
#             # Try to preload next again
#             if not result_queue.empty():
#                 preload_next = result_queue.get()
#
#         # Draw current frame
#         center = curr_cen[frame_idx]
#         diff = curr_diff[frame_idx]
#         full_layer = np.vstack([np.full((1, ry_param), center), diff])
#
#         heatmap.remove()
#         # heatmap = ax.pcolormesh(X, Y, full_layer.T, shading='flat',
#         #                         cmap=cmap, edgecolors='k', linewidth=0.01, norm=norm)
#
#         heatmap = ax.pcolormesh(
#             X[:-1, :-1], Y[:-1, :-1], full_layer.T,
#             shading='flat', cmap=cmap, edgecolors='k', linewidth=0.01, norm=norm
#         )
#
#         sim_time += steps_per_frame * d_time_est
#         ax.set_title(f"Simulation time: {sim_time:.4f}", fontsize=14)
#
#         if T_param * (1 - epsilon) <= sim_time <= T_param * (1 + epsilon):
#             print(f"Simulation paused at time T ~ {sim_time:.3f}")
#             ani.event_source.stop()
#
#         frame_idx += steps_per_frame
#         return [heatmap]
#
#     ani = FuncAnimation(
#         fig, update,
#         frames=100000,
#         interval=interval_ms,
#         blit=False
#     )
#
#     canvas.ani = ani
#     canvas.ani_paused = False
#     canvas.fig = fig
#
#     return canvas
#
