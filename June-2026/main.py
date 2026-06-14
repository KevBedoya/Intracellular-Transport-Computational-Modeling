import multiprocessing
from project_src_package_2025.gui_components import main_gui as gui
from project_src_package_2025.launch_functions import launch
from project_src_package_2025.data_visualization import plot_functions as plt
from project_src_package_2025.computational_tools import numerical_tools as num, analysis_tools as ant
from project_src_package_2025.computational_tools import time_analysis as tim
from project_src_package_2025.computational_tools import supplements as sup
from project_src_package_2025.auxiliary_tools import unit_conversion_functions
from computational_tools import struct_init
import time
from project_src_package_2025.computational_tools import mat_computations
import numpy as np

"""
Version 2.1  (August 8 2025)

personal notes/tasks:
    - Include version information onto documentation
    - Read upon professional/software engineer version control tips/how to name or properly archive previous versions of the codebase
    - Write a script to automatically delete all data contents [ def delete_ALL_output_() ]
    - Write a script to save all data contents onto the cloud (Google Drive API), then call delete_ALL_output_ [ def upload_to_cloud_() ] 
    - Incorporate delete_ALL_output_() and def upload_to_cloud_() onto GUI
    - Include custom plot options onto GUI (first by selecting the appropriate data file [.csv] to analyze ) 
@Kevin
"""

def run_main():
    gui.run_app()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    run_main()

    # v_param = 1
    # w_param = 10
    # rg_param = 32
    # ry_param = 32
    # T_param = 2
    #
    # num = 4
    # N_LIST = np.linspace(0, ry_param - (ry_param / num), num, dtype=int)
    # N_LIST = [0, 5, 11, 17]
    #
    # diffusion_matrix, central_vector = launch.compute_ang_traj_mat(rg_param, ry_param, v_param, w_param, N_LIST, [T_param])
    #
    # A = diffusion_matrix[-1]
    # center = central_vector[-1]
    # Radial_trajectory_matrix = A
    # Angular_trajectory_matrix = Radial_trajectory_matrix.T

    # Gram_Radial = Radial_trajectory_matrix.T @ Radial_trajectory_matrix
    # Gran_Angular = Angular_trajectory_matrix.T @ Angular_trajectory_matrix

    # print("Frobenius Cosine Similarity Metric (phi v. theta): ", mat_computations.max_alignment_error(Gran_Angular))
    # print("Max Deviation (phi v. theta): ", mat_computations.max_alignment_error(Gran_Angular))
    #
    # print("Frobenius Cosine Similarity Metric (phi v. radius): ", mat_computations.mean_alignment_error(Gram_Radial))
    # print("Max Deviation (phi v. radius): ", mat_computations.max_alignment_error(Gram_Radial))
    #
    # print("Rank-1 dominance ratio (general): ", mat_computations.rank1_energy_ratio(A))
    #
    # plt.plot_cosine_similarity_heatmap(Gram_Radial, "Gram Radial")
    # plt.plot_cosine_similarity_heatmap(Gran_Angular, "Gram Angular")
    # plt.plot_normalized_separable_trajectories(A, v_param, w_param, T_param, len(N_LIST), center, [0, 4, 5, 9, 11, 17, 18],[0, 5, 10, 15, 20, 25, 30], include_central_patch=False)
    # print(Gran_Angular)
# today_str = datetime.now().strftime("%Y-%m-%d")
#
# log_dir = os.path.join(os.getcwd(), fp.rect_logs)
#
# os.makedirs(log_dir,  exist_ok=True)
#
# output_filename = os.path.join(log_dir, f"output_{today_str}.txt")
#
# class Tee:
#     def __init__(self, *streams):
#         self.streams = streams
#
#     def write(self, data):
#         for s in self.streams:
#             s.write(data)
#
#     def flush(self):
#         for s in self.streams:
#             s.flush()
#
#
# with open(output_filename, "w") as f:
#     tee = Tee(sys.stdout, f)
#     with redirect_stdout(tee):
#         run_main()

