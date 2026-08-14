"""
ui_builder.py
=============

:class:`UIBuilderMixin` -- constructs the entire control-panel widget tree.

This is the "view" half of the control panel: it creates every widget, wires
signals to the handler methods provided by the behavioural mixins, and arranges
everything into the three-column layout (central tabs | left controls | right
output/queue). It deliberately contains **no business logic** -- only widget
construction and signal wiring.

The build is split into small, ordered ``_build_*`` steps that
:class:`ControlPanel` calls in sequence. The *order of side effects*
(``addWidget``/``addLayout``/``connect``) is identical to the original
monolithic ``__init__`` so the resulting layout is unchanged; the steps merely
make that long constructor navigable.

Styling note
------------
Where the original code used per-widget ``setStyleSheet(...)`` / ``QPalette``
calls (which fight the global stylesheet and look inconsistent), this builder
instead assigns ``objectName``\\ s (see :mod:`theme`) so all appearance lives in
``style.qss``. The loading spinner is loaded from the package via
:func:`theme.spinner_gif_path` rather than a hard-coded absolute path.
"""

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QFormLayout,
    QTextEdit, QCheckBox, QGroupBox, QStackedWidget, QTabWidget, QWidget,
    QListWidgetItem, QSpacerItem, QSizePolicy,
)
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import Qt

from project_src_package_2025.job_queuing_system import job_queue

from .. import params_config
from .. import history_cache
from .. import output_display_widget
from .. import theme
from .toggle_select_list import ToggleSelectListWidget


class UIBuilderMixin:
    """Widget-tree construction and signal wiring for :class:`ControlPanel`."""

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def build_ui(self, main_window):
        """Assemble the whole panel. Call once from ``__init__``.

        The sub-steps run in the same order as the original constructor so the
        layout and tab state are byte-for-byte equivalent (aside from styling).
        """
        self._init_runtime_state(main_window)
        self._build_core_layout()
        self._build_parameter_toggle()
        self._build_central_tabs()
        self._build_left_panel()
        self._build_action_buttons()
        self._attach_left_panel()
        self._build_right_panel()
        self._build_control_button_row()
        self._build_results_widgets()
        self._init_parameter_fields()
        self._build_visualization_panel()
        self._finalize_ui()

    # ------------------------------------------------------------------ #
    # Steps
    # ------------------------------------------------------------------ #
    def _init_runtime_state(self, main_window):
        """Initialise the non-widget runtime attributes (timers, processes...)."""
        self.ani_queue = None
        self.jit_timer = None
        self.jit_compile_process = None
        self.job_timer = None
        self.job_process = None
        self.job_queue = None
        self.current_job = None
        self.pending_jobs = None
        self.process = None
        self.poll_timer = None
        self.mp_queue = None
        self.current_canvas = None
        self._d_tube_validation_msg_index = None

        self.d_tube_valid = True

        self.main_window = main_window

    def _build_core_layout(self):
        """Create the top-level horizontal layout and the two side columns."""
        self.setContentsMargins(20, 20, 20, 20)  # GUI margin inward padding

        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)

        self.left_panel = QVBoxLayout()
        self.right_panel = QVBoxLayout()

        # Parameter form
        self.param_form = QFormLayout()
        self.param_inputs = {}

    def _build_parameter_toggle(self):
        """Create the "Show Advanced Parameters" checkbox."""
        self.advanced_toggle = QCheckBox("Show Advanced Parameters")
        self.advanced_toggle.stateChanged.connect(self.toggle_advanced_fields)
        self.advanced_widgets = []

    def _build_central_tabs(self):
        """Create the central Results/Visualization tab widget."""
        # Central Display area; to swap between results and visualization
        self.tab_widget = QTabWidget()

        self.result_display_tab = QWidget()
        self.result_layout = QVBoxLayout()
        self.result_display_tab.setLayout(self.result_layout)

        self.visualization_tab = QWidget()
        self.visualization_layout = QVBoxLayout()
        self.visualization_tab.setLayout(self.visualization_layout)

        self.tab_widget.addTab(self.result_display_tab, "Results")
        self.tab_widget.addTab(self.visualization_tab, "Visualization")
        # Open on the primary "Results" tab. (The original set index 0 then 1,
        # landing on an empty Visualization tab -- a leftover debug artifact.)
        self.tab_widget.setCurrentIndex(0)

        self.main_layout.addWidget(self.tab_widget)

    def _build_left_panel(self):
        """Build the left column: history, resets, and the computation block."""
        # History management
        self.history_dropdown = QComboBox()
        self.history_dropdown.addItem("Select Previous Computation: ")
        self.history_dropdown.currentIndexChanged.connect(self.load_history_entry)
        self.left_panel.addWidget(self.history_dropdown)

        for label in history_cache.cache.get_labels():
            self.history_dropdown.addItem(label)

        self.reset_params_button = QPushButton("Reset Parameters")
        self.reset_params_button.clicked.connect(self.clear_parameter_fields)
        self.left_panel.addWidget(self.reset_params_button)

        self.clear_hist_button = QPushButton("Clear History")
        self.clear_hist_button.clicked.connect(self.clear_history)
        self.left_panel.addWidget(self.clear_hist_button)

        self.clear_output_button = QPushButton("Clear Outputs")
        self.clear_output_button.clicked.connect(self.clear_displayed_results)
        self.left_panel.addWidget(self.clear_output_button)

        # -- Adjustable computation block wrapper --
        self.comp_select = QComboBox()
        self.comp_select.addItems(params_config.PARAMETER_SCHEMAS.keys())
        # Connect AFTER addItems so populating the box does not fire the slots.
        self.comp_select.currentTextChanged.connect(self.update_parameter_fields)
        self.comp_select.currentTextChanged.connect(self.validate_computation)

        self.computation_block_layout = QVBoxLayout()
        self.computation_block_layout.setContentsMargins(0, 10, 0, 0)  # (L, T, R, B)

        self.computation_block_layout.addWidget(self.advanced_toggle)
        self.computation_block_layout.addWidget(self.comp_select)
        self.computation_block_layout.addLayout(self.param_form)

        self.left_panel.addLayout(self.computation_block_layout)

    def _build_action_buttons(self):
        """Create the spinner and the Launch / enqueue / execute buttons."""
        self.spinner_label = QLabel()
        self.spinner_label.setVisible(False)
        self.spinner_movie = QMovie(theme.spinner_gif_path(100))
        self.spinner_label.setMovie(self.spinner_movie)

        self.launch_button = QPushButton("Launch")
        self.launch_button.setObjectName(theme.LAUNCH_BUTTON)
        self.launch_button.clicked.connect(self.run_computation_mp)
        self.set_launch_color("idle")

        self.enqueue_button = QPushButton("+")
        self.enqueue_button.setObjectName(theme.ENQUEUE_BUTTON)
        self.enqueue_button.setToolTip("Add this computation to the job queue")
        self.enqueue_button.clicked.connect(self.enqueue_job)

        self.run_queue_button = QPushButton("Execute Queue")
        self.run_queue_button.setObjectName(theme.RUN_QUEUE_BUTTON)
        self.run_queue_button.setToolTip("Start running enqueued computations")
        self.run_queue_button.clicked.connect(self.run_job_queue_mp)

    def _attach_left_panel(self):
        """Add the assembled left column to the main layout."""
        self.main_layout.addLayout(self.left_panel, stretch=2)

    def _build_right_panel(self):
        """Build the right column: output console, plot area and job queue."""
        # Output and Plot Display
        self.output_display = QTextEdit()
        self.output_display.setObjectName(theme.OUTPUT_CONSOLE)
        self.output_display.setReadOnly(True)
        self.right_panel.addWidget(self.output_display)

        self.restored_label = QLabel("")
        self.restored_label.setObjectName(theme.RESTORED_LABEL)
        self.restored_label.hide()
        self.right_panel.addWidget(self.restored_label, alignment=Qt.AlignRight)

        self.plot_layout = QVBoxLayout()
        self.right_panel.addLayout(self.plot_layout)

        # Queue Panel (relocated under output)
        self.queue_list_label = QLabel("Queued Jobs:")
        self.queue_list_label.setObjectName(theme.SECTION_LABEL)
        self.queue_list_widget = ToggleSelectListWidget()
        self.queue_list_widget.setObjectName(theme.QUEUE_LIST)
        self.queue_list_widget.setFixedWidth(300)
        self.queue_list_widget.itemSelectionChanged.connect(self.toggle_move_buttons_visibility)

        for job in job_queue.global_queue.jobs:
            item = QListWidgetItem(job.display_name())
            item.setData(Qt.UserRole, job)
            self.queue_list_widget.addItem(item)

        self.queue_panel = QVBoxLayout()
        self.queue_panel.setContentsMargins(5, 10, 5, 10)
        self.queue_panel.addWidget(self.queue_list_label)
        self.queue_panel.addWidget(self.queue_list_widget)

        self.edit_job_button = QPushButton("Edit Job")
        self.move_up_button = QPushButton("Move Up")
        self.move_down_button = QPushButton("Move Down")
        self.remove_job_button = QPushButton("Remove Job")
        self.clear_queue_button = QPushButton("Clear Queue")

        self.edit_job_button.clicked.connect(self.edit_selected_job)
        self.move_up_button.clicked.connect(self.move_selected_job_up)
        self.move_down_button.clicked.connect(self.move_selected_job_down)
        self.remove_job_button.clicked.connect(self.remove_selected_job)
        self.clear_queue_button.clicked.connect(self.clear_entire_queue)

        queue_button_layout = QHBoxLayout()
        queue_button_layout.addWidget(self.edit_job_button)
        queue_button_layout.addWidget(self.move_up_button)
        queue_button_layout.addWidget(self.move_down_button)
        queue_button_layout.addWidget(self.remove_job_button)
        queue_button_layout.addWidget(self.clear_queue_button)
        self.queue_panel.addLayout(queue_button_layout)

        self.right_panel.addLayout(self.queue_panel)
        self.right_panel.addSpacerItem(QSpacerItem(10, 30, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.main_layout.addLayout(self.right_panel, stretch=3)

    def _build_control_button_row(self):
        """Add the fixed Launch/spinner/enqueue/execute row beneath the queue."""
        self.control_button_row = QHBoxLayout()
        self.control_button_row.addWidget(self.launch_button)
        self.control_button_row.addWidget(self.spinner_label)
        self.control_button_row.addWidget(self.enqueue_button)
        self.control_button_row.addWidget(self.run_queue_button)

        self.right_panel.addLayout(self.control_button_row)

    def _build_results_widgets(self):
        """Create the PNG-preview and output-files widgets in the Results tab."""
        self.png_preview_widget = output_display_widget.PNGPreviewWidget()
        self.result_layout.addWidget(self.png_preview_widget)
        self.png_preview_widget.hide()

        self.output_files_widget = output_display_widget.OutputFilesWidget()
        self.result_layout.addWidget(self.output_files_widget)
        self.output_files_widget.hide()

    def _init_parameter_fields(self):
        """Populate the parameter form for the default computation."""
        self.update_parameter_fields(self.comp_select.currentText())
        for line_edit in self.param_inputs.values():
            line_edit.installEventFilter(self)

    def _build_visualization_panel(self):
        """Build the Visualization tab: selector, domain and animation controls."""
        # Visualization Section
        viz_group = QGroupBox()
        viz_layout = QVBoxLayout()
        viz_group.setLayout(viz_layout)
        viz_label = QLabel("Select Visualization: ")
        viz_label.setObjectName(theme.SECTION_LABEL)
        self.visualization_select = QComboBox()
        self.visualization_select.addItems(["Show Domain", "Animate Diffusion"])
        self.visualization_select.currentTextChanged.connect(self.handle_visualization_mode_change)
        viz_layout.addWidget(viz_label)
        viz_layout.addWidget(self.visualization_select)

        self.visualization_mode = "Show Domain"

        self.domain_checkbox_group = QGroupBox()
        domain_group_layout = QVBoxLayout()
        domain_checks_row = QHBoxLayout()
        domain_buttons_row = QHBoxLayout()
        self.display_extract_checkbox = QCheckBox("Display Extraction Region")
        self.toggle_border_checkbox = QCheckBox("Display Internal Borders")
        self.toggle_border_checkbox.setChecked(True)
        self.display_domain_button = QPushButton("Preview Domain")
        self.display_domain_button.clicked.connect(self.handle_display_domain)
        self.close_domain_button = QPushButton("Close Domain")
        self.close_domain_button.clicked.connect(self.close_domain)
        self.close_domain_button.hide()
        # Two rows (checkboxes / buttons) so the long labels are never clipped.
        domain_checks_row.addWidget(self.display_extract_checkbox)
        domain_checks_row.addWidget(self.toggle_border_checkbox)
        domain_checks_row.addStretch(1)
        domain_buttons_row.addWidget(self.display_domain_button)
        domain_buttons_row.addWidget(self.close_domain_button)
        domain_buttons_row.addStretch(1)
        domain_group_layout.addLayout(domain_checks_row)
        domain_group_layout.addLayout(domain_buttons_row)
        self.domain_checkbox_group.setLayout(domain_group_layout)

        self.animation_controls_group = QGroupBox()
        self.animation_controls_group.setVisible(False)
        self.anim_layout = QVBoxLayout()
        anim_buttons_row = QHBoxLayout()
        anim_options_row = QHBoxLayout()
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_animation)
        self.clear_ani_button = QPushButton("Clear Animation")
        self.clear_ani_button.clicked.connect(self.clear_animation)
        self.launch_animation_button = QPushButton("Launch Animation")
        self.launch_animation_button.setObjectName(theme.LAUNCH_ANIM_BUTTON)
        self.launch_animation_button.setEnabled(False)
        self.anim_border_select_checkbox = QCheckBox("Display domain border")
        self.anim_border_select_checkbox.setChecked(False)
        for key in ["rg_param", "ry_param", "w_param", "v_param", "N_LIST", "T_param"]:
            if key in self.param_inputs:
                self.param_inputs[key].textChanged.connect(self.validate_animation)

        self.launch_animation_button.clicked.connect(self.handle_launch_animation)

        # Two rows (buttons / options) for a clean, uncramped layout.
        anim_buttons_row.addWidget(self.pause_button)
        anim_buttons_row.addWidget(self.clear_ani_button)
        anim_buttons_row.addWidget(self.launch_animation_button)
        anim_buttons_row.addStretch(1)
        anim_options_row.addWidget(self.anim_border_select_checkbox)
        anim_options_row.addStretch(1)
        self.anim_layout.addLayout(anim_buttons_row)
        self.anim_layout.addLayout(anim_options_row)
        self.animation_controls_group.setLayout(self.anim_layout)

        self.visualization_area = QStackedWidget()
        self.visualization_area.setMinimumSize(600, 600)

        self.viz_controls_layout = QVBoxLayout()
        self.viz_controls_layout.addWidget(viz_group)
        self.viz_controls_layout.addWidget(self.domain_checkbox_group)
        self.viz_controls_layout.addWidget(self.animation_controls_group)

        self.visualization_layout.addLayout(self.viz_controls_layout)
        self.visualization_layout.addWidget(self.visualization_area)

    def _finalize_ui(self):
        """Apply the initial visibility/enabled state once everything exists."""
        self.toggle_move_buttons_visibility()
        self.update_queue_controls_visibility()
        self.update_history_dropdown_visibility()
        self.update_queue_execute_button_visibility()
        self.setup_d_tube_live_check()
        self._set_computation_controls_enabled(False)
