"""
visualization.py
=================

:class:`VisualizationMixin` -- the "Visualization" tab: static domain previews
and live diffusion animations.

There are two visualization modes, selected from a combo box and switched by
:meth:`handle_visualization_mode_change`:

* **Show Domain** -- renders a static grid of the cellular domain
  (:meth:`handle_display_domain` via ``animation_functions.display_domain_grid``),
  shown inside the ``visualization_area`` stacked widget.
* **Animate Diffusion** -- builds a Matplotlib ``FuncAnimation`` canvas
  (:meth:`handle_launch_animation` via ``ani_evolution.animate_diffusion``) with
  pause/resume and clear controls.

The numerical/plot generation lives entirely in the ``data_visualization``
package and is **not** modified. This mixin only manages which canvas is shown
and tears the previous one down cleanly.

GUI-only improvement: canvas replacement is funneled through
:meth:`clear_animation`, which now stops the previous animation's timer, removes
the canvas from the *correct* container (the visualization stacked widget, not
the right panel), and schedules it for deletion. The original code detached the
old canvas without deleting it, so repeatedly launching animations leaked
canvases whose ``FuncAnimation`` timers kept firing.
"""

import re
import ast

from PyQt5.QtWidgets import QMessageBox, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from intracellular_transport.data_visualization import animation_functions as ani
from intracellular_transport.data_visualization import ani_evolution as evo


class VisualizationMixin:
    """Domain preview and diffusion-animation management."""

    # --------------------------- animation ---------------------------- #
    def handle_launch_animation(self):
        """Build and show a diffusion animation canvas from the current inputs."""
        try:
            rg = int(self.param_inputs["rg_param"].text())
            ry = int(self.param_inputs["ry_param"].text())
            w = float(self.param_inputs["w_param"].text())
            v = float(self.param_inputs["v_param"].text())
            N_raw = self.param_inputs["N_LIST"].text()
            N = list(map(int, re.findall(r'\d+', N_raw))) if N_raw else []
            T = float(self.param_inputs["T_param"].text())
            d_tube = float(self.param_inputs["d_tube"].text())
            border = self.anim_border_select_checkbox.isChecked()

            steps_per_frame = 10
            fps = 50

            interval_ms = int(1000 / fps)
            K_param = 1000
            color_scheme = 'viridis'
            canvas = evo.animate_diffusion(rg, ry, w, v, N, K_param,
                                           T, d_tube, steps_per_frame=steps_per_frame,
                                           interval_ms=interval_ms,
                                           color_scheme=color_scheme,
                                           border=border)

            # Cleanly remove/stop any existing canvas before showing the new one.
            self.clear_animation()

            self.current_canvas = canvas
            self.visualization_area.addWidget(canvas)
            self.visualization_area.setCurrentWidget(canvas)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch animation: \n{str(e)}")

    def pause_animation(self):
        """Toggle pause/resume on the active animation canvas."""
        canvas = self.current_canvas
        if not hasattr(canvas, 'ani') or canvas.ani is None:
            return

        if canvas.ani_paused:
            canvas.ani.event_source.start()
            canvas.ani_paused = False
            self.pause_button.setText("Pause Animation")
        else:
            canvas.ani.event_source.stop()
            canvas.ani_paused = True
            self.pause_button.setText("Resume Animation")

    def clear_animation(self):
        """Stop and remove the current canvas, freeing its resources."""
        canvas = self.current_canvas
        if canvas is not None:
            # Stop the animation timer if this canvas is animated.
            if hasattr(canvas, 'ani') and canvas.ani is not None:
                canvas.ani.event_source.stop()
                canvas.ani = None

            # Canvas lives in the visualization stacked widget; remove it there,
            # detach it, and schedule deletion so its figure/timers are freed.
            self.visualization_area.removeWidget(canvas)
            canvas.setParent(None)
            canvas.deleteLater()

            self.current_canvas = None

    def validate_animation(self):
        """Enable the Launch-Animation button only when all inputs are present."""
        required_keys = ["rg_param", "ry_param", "w_param", "v_param", "N_LIST", "T_param", "d_tube"]
        inputs_filled = all(
            key in self.param_inputs and self.param_inputs[key].text().strip() != ""
            for key in required_keys
        )
        self._set_launch_animation_button_enabled(inputs_filled)

    def show_animation_ui(self):
        """Switch the visualization controls to animation mode."""
        self.domain_checkbox_group.hide()
        self.display_domain_button.hide()
        self.close_domain_button.hide()

        self.animation_controls_group.show()
        self.validate_animation()

    def connect_QLINE_editfields(self):
        """Wire animation-relevant fields to re-validate the animation button."""
        for key in ["rg_param", "ry_param", "w_param", "v_param", "N_LIST", "T_param"]:
            if key in self.param_inputs:
                self.param_inputs[key].textChanged.connect(self.validate_animation)

    # ----------------------------- domain ----------------------------- #
    def show_domain_ui(self):
        """Switch the visualization controls to domain-preview mode."""
        self.domain_checkbox_group.show()
        self.display_domain_button.show()
        self.close_domain_button.show()

        self.animation_controls_group.hide()

    def handle_display_domain(self):
        """Render a static domain grid from the current ring/ray/tube inputs."""
        try:
            self.clear_animation()  # Automatically stop animation if active

            rings = int(self.param_inputs["rg_param"].text())
            rays = int(self.param_inputs["ry_param"].text())
            d_tube = float(self.param_inputs["d_tube"].text())

            try:
                microtubules_input = self.param_inputs["N_LIST"].text()
                parsed = ast.literal_eval(microtubules_input)
                microtubules = list(parsed) if isinstance(parsed, (list, tuple)) else [int(parsed)]
            except (ValueError, SyntaxError):
                print("[Error] Invalid microtubule input. Please enter a list like [0,1,2] or comma-separated values.")
                microtubules = []

            display_extract = self.display_extract_checkbox.isChecked()
            toggle_border = self.toggle_border_checkbox.isChecked()

            # Get domain figure
            fig = ani.display_domain_grid(
                rings=rings,
                rays=rays,
                microtubules=microtubules,
                d_tube=d_tube,
                display_extract=display_extract,
                toggle_border=toggle_border
            )

            # Clear previous canvas if any
            if hasattr(self, 'current_canvas') and self.current_canvas:
                self.visualization_area.removeWidget(self.current_canvas)
                self.current_canvas.setParent(None)

            # Add new domain canvas
            canvas = FigureCanvas(fig)
            self.current_canvas = canvas
            self.visualization_area.addWidget(canvas)
            self.visualization_area.setCurrentWidget(canvas)

            self.close_domain_button.show()
            self.display_domain_button.setText("Update Domain")

        except Exception as e:
            print(f"[Error] Failed to display domain grid: {e}")

    def close_domain(self):
        """Remove the domain canvas and reset the domain buttons."""
        if hasattr(self, 'current_canvas') and self.current_canvas:
            self.visualization_area.removeWidget(self.current_canvas)
            self.current_canvas.setParent(None)
            self.current_canvas.deleteLater()
            self.current_canvas = None  # Reset reference

        self.close_domain_button.hide()
        self.display_domain_button.setText("Display Domain")

    # ------------------------- mode switching -------------------------- #
    def handle_visualization_mode_change(self, mode):
        """Switch between "Show Domain" and "Animate Diffusion" control sets."""
        self.visualization_mode = mode

        if mode == "Show Domain":
            self.show_domain_ui()
        elif mode == "Animate Diffusion":
            self.show_animation_ui()

    def display_matplotlib_figure(self, fig):
        """Render ``fig`` into the right-panel plot area, replacing any prior one."""
        for i in reversed(range(self.plot_layout.count())):
            widget_to_remove = self.plot_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)
                widget_to_remove.deleteLater()
        canvas = FigureCanvas(fig)
        canvas.setParent(self)

        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.updateGeometry()

        self.plot_layout.addWidget(canvas)
        canvas.draw()
        canvas.show()
