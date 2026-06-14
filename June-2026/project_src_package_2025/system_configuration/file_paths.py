import os
import sys
from pathlib import Path

# Determine base directory (for both source and frozen cases)
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(os.path.dirname(sys.executable)).resolve()
else:
    BASE_DIR = Path(__file__).parent.parent.resolve()

DATA_OUTPUT = Path.home() / 'BiophysicsAppData'
DATA_OUTPUT.mkdir(exist_ok=True)

# Create absolute paths using BASE_DIR
general_output = BASE_DIR / "data_output"
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

rect_logs = general_output / "output_logs"
# styles_location = BASE_DIR / "gui_components/styles/style.qss"

json_output = general_output / "json_output"


