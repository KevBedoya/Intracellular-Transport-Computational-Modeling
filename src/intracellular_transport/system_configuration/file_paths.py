import os
import sys
from pathlib import Path

# Determine base directory (for both source and frozen cases)
#
# In a source checkout this resolves to the repository root, NOT the package
# directory: this file sits at src/intracellular_transport/system_configuration/,
# so the root is four levels up. Generated output therefore lands in
# <repo>/data_output rather than inside the source tree, which keeps code and
# generated data cleanly separated.
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(os.path.dirname(sys.executable)).resolve()
else:
    BASE_DIR = Path(__file__).resolve().parents[3]

DATA_OUTPUT = Path.home() / 'BiophysicsAppData'
DATA_OUTPUT.mkdir(exist_ok=True)

# Create absolute paths using BASE_DIR
#
# ITCM_OUTPUT_ROOT overrides the output tree for a single process. The job
# worker sets it per job, so concurrent jobs write into disjoint directories
# and each job's outputs are attributable to it. Without the override every
# job writes into one shared tree, and with several running at once there is
# no way to tell afterwards which run produced which directory. Unset (the
# desktop GUI, direct scripts) the historical location is used unchanged.
general_output = Path(os.environ.get("ITCM_OUTPUT_ROOT") or (BASE_DIR / "data_output"))
mfpt_results_output = general_output / "mfpt-results"
heatmap_output = general_output / "heatmaps"
phi_v_theta_output = general_output / "diffusive-v-theta"
# phi_v_theta_output = "/Users/kbedoya88/Desktop"

# Density outputs
angular_dependence_phi = general_output / "density-results/angular-dependence/diffusive"
# angular_dependence_phi = "/Users/kbedoya88/Desktop"
# angular_dependence_rho = general_output / "density-results/angular-dependence/rho"
radial_dependence_phi = general_output / "density-results/radial-dependence/diffusive"
radial_dependence_rho = general_output / "density-results/radial-dependence/rho"

# Mass analysis results
mass_analysis_advective = general_output / "mass_analysis_results/advective"
mass_analysis_diffusive = general_output / "mass_analysis_results/diffusive"
mass_analysis_advective_over_total = general_output / "mass_analysis_results/advective_over_total"
mass_analysis_total = general_output / "mass_analysis_results/total"
mass_analysis_advective_over_initial = general_output / "mass_analysis_results/advective_over_initial"

# Characteristic time analysis results
char_time_analysis_output = general_output / "char_time_analysis"
# Kept separate from char_time_analysis rather than sharing it: the (a, b) grid
# writes a differently-shaped CSV (one row per switch-rate pair, not per
# velocity), and anything globbing the velocity-sweep directory would otherwise
# pick these up and mis-read them.
ab_grid_char_time_output = general_output / "ab_grid_char_time"

rect_logs = general_output / "output_logs"
# styles_location = BASE_DIR / "gui_components/styles/style.qss"

json_output = general_output / "json_output"


