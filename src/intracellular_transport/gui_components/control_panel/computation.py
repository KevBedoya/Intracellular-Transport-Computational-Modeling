"""
computation.py
==============

:class:`ComputationMixin` -- launches a *single* computation in a separate
process and reports the result back to the GUI.

Flow
----
1. :meth:`validate_computation` enables the action buttons only when every
   required/defaulted field for the selected schema is filled (and ``d_tube``,
   if present, is valid).
2. :meth:`run_computation_mp` serialises the inputs to JSON and spawns
   ``python -m multiprocessing_tools.subprocess_launcher`` so the heavy numerical
   work never blocks the Qt event loop.
3. A :class:`QTimer` polls :meth:`check_for_computation_results`; once the child
   process exits it reads ``result.json``, then hands a successful payload to
   :meth:`process_result`, which records history and populates the result
   widgets.

The numerical backend (``subprocess_launcher`` / ``controller``) is **not**
modified here -- this mixin only orchestrates the GUI side.

GUI-only fixes relative to the original ``views.py``:

* the spinner is now hidden (``setVisible(False)``) when a run finishes, instead
  of being left visible in a stopped state;
* ``computation_name`` is resolved before the ``try`` block so the error path
  can never raise ``NameError``;
* the Launch button colour is driven by a QSS ``state`` property
  (:func:`theme.apply_button_state`) rather than a per-widget ``QPalette`` that
  fought the global stylesheet.
"""

import sys

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox

from .. import params_config
from .. import aux_gui_funcs
from .. import computation_history_entry
from .. import history_cache
from .. import theme


class ComputationMixin:
    """Single-computation launch, polling and result handling."""

    def set_launch_color(self, state):
        """Set the Launch button visual state.

        ``state`` is one of ``"idle"``, ``"running"``, ``"success"`` or
        ``"error"``; anything else falls back to idle. Colours are defined in
        ``style.qss`` against the ``#LaunchButton[state="..."]`` selectors.
        """
        valid = {
            theme.STATE_IDLE,
            theme.STATE_RUNNING,
            theme.STATE_SUCCESS,
            theme.STATE_ERROR,
        }
        theme.apply_button_state(
            self.launch_button, state if state in valid else theme.STATE_IDLE
        )

    def validate_computation(self):
        """Enable computation controls only when the schema's fields are valid."""
        computation_type = self.comp_select.currentText()
        schema = params_config.PARAMETER_SCHEMAS.get(computation_type)
        if schema is None:
            self._set_computation_controls_enabled(False)
            return

        # Get expected fields (both required and default)
        expected_fields = [param for param, _ in schema.get("default", [])] + \
                          [param for param, _ in schema.get("required", [])]

        # Check that all are filled
        inputs_filled = all(
            param in self.param_inputs and self.param_inputs[param].text().strip() != ""
            for param in expected_fields
        )

        # If d_tube is in the expected fields, combine check with its validity
        if "d_tube" in expected_fields:
            is_valid = inputs_filled and self.d_tube_valid
        else:
            is_valid = inputs_filled

        self._set_computation_controls_enabled(is_valid)

    def check_for_computation_results(self, computation_name):
        """Poll the worker process; on exit, load and dispatch ``result.json``."""
        import os
        import json
        if self.process.poll() is None:
            return
        self.poll_timer.stop()

        from intracellular_transport.multiprocessing_tools import compute_worker

        result_file = compute_worker.result_path(self.job_id)

        try:
            if not os.path.join(result_file) or os.path.getsize(result_file) == 0:
                raise FileNotFoundError("Result file missing or empty.")

            with open(result_file, "r") as f:
                response = json.load(f)
            os.remove(result_file)

            if response["status"] == "ok":
                self.output_display.append(
                    f"Computation {computation_name} executed successfully. [{self.produce_timestamp()}]"
                )
                self.process_result(response["result"])
            else:
                print("Computation failed:", response.get("message", "Unknown error"))
        except Exception as e:
            self.output_display.append(
                f"{e} [{self.produce_timestamp()}]"
            )

        self.launch_button.setEnabled(True)
        self.spinner_movie.stop()
        self.spinner_label.setVisible(False)

    def run_computation_mp(self):
        """Spawn the worker subprocess for the selected computation."""
        import json

        self.launch_button.setEnabled(False)
        self.spinner_label.setVisible(True)
        self.spinner_movie.start()

        # Resolved up front so the error path below always has a valid name.
        computation_name = self.comp_select.currentText()

        try:
            inputs = {param: field.text() for param, field in self.param_inputs.items()}
            if not any(inputs.values()):
                raise ValueError("No input parameters were provided.")
            if not computation_name:
                raise ValueError("No computation type selected.")
            self.output_display.append(
                f"Launching computation: {computation_name}...      [{self.produce_timestamp()}]"
            )

            from intracellular_transport.multiprocessing_tools import subprocess_launcher
            from intracellular_transport.multiprocessing_tools import compute_worker
            # Scope this run's result file to its own job id so a second
            # computation cannot overwrite the result we are waiting on.
            self.job_id = compute_worker.new_job_id()
            args = subprocess_launcher.build_worker_args(
                computation_name, json.dumps(inputs), self.job_id)
            self.process = subprocess_launcher.launch_subprocess(args)

            self.poll_timer = QTimer()
            self.poll_timer.timeout.connect(lambda: self.check_for_computation_results(computation_name))
            self.poll_timer.start(100)

        except Exception as e:
            self.set_launch_color("error")
            self.output_display.append(f"Computation {computation_name} failed.     [{self.produce_timestamp()}]")
            QMessageBox.critical(self, "Input Error", str(e))
            self.spinner_label.setVisible(False)
            self.spinner_movie.stop()
            self.launch_button.setEnabled(True)

    def process_result(self, result):
        """Display a successful result and append it to the history cache."""
        self.set_launch_color("success")

        if "MFPT" in result:
            self.output_display.append(
                f"Computation returned MFPT = {result['MFPT']:.6f}       [{self.produce_timestamp()}]")

        if "duration" in result:
            self.output_display.append(
                f"Dimensionless time duration: {result['duration']:.6f}      [{self.produce_timestamp()}]")

        for line in aux_gui_funcs.char_time_lines(result):
            self.output_display.append(line)

        csv_paths = []
        png_paths = []

        if "output_dirs" in result:
            csv_paths, png_paths = aux_gui_funcs.extract_csv_and_png_paths(result["output_dirs"])
            if csv_paths or png_paths:
                self.output_files_widget.update_display(csv_paths, png_paths)
                self.output_files_widget.show()
                self.png_preview_widget.update_png_list(png_paths)
                self.png_preview_widget.show()
            else:
                self.output_files_widget.hide()
                self.png_preview_widget.hide()
        else:
            self.output_files_widget.hide()

        record = computation_history_entry.ComputationRecord(
            comp_type=self.comp_select.currentText(),
            params={param: field.text() for param, field in self.param_inputs.items()},
            mfpt=result.get("MFPT"),
            duration=result.get("duration"),
            csv_files=csv_paths,
            png_files=png_paths,
            status="completed",
            error_msg=None
        )

        history_cache.cache.add_entry(record)
        self.history_dropdown.addItem(record.display_name())
        self.update_history_dropdown_visibility()
