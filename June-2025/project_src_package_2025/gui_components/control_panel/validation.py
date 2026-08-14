"""
validation.py
=============

:class:`ValidationMixin` -- live input validation and action-control gating.

The headline feature is the live ``d_tube`` check (:meth:`validate_d_tube_range`):
as the user types, it parses ``rg_param``/``ry_param``/``N_LIST``/``d_tube`` and
computes the maximum legal tube extraction distance via the
``computational_tools.supplements`` helpers, writing a single, in-place status
line to the output console (:meth:`_set_d_tube_validation_message`).

The validity result feeds :attr:`d_tube_valid`, which
:meth:`~project_src_package_2025.gui_components.control_panel.computation.ComputationMixin.validate_computation`
combines with the "all fields filled" check to enable or disable the run/queue/
preview/animation controls (:meth:`_set_computation_controls_enabled`).

The supplement functions (``j_max_bef_overlap_no_JIT``, ``solve_d_rect_no_JIT``)
are numerical and are **called, not modified**.
"""

import re

from project_src_package_2025.computational_tools import supplements as sup

from .. import params_config


class ValidationMixin:
    """Live ``d_tube`` validation and enable/disable gating of action controls."""

    def setup_d_tube_live_check(self):
        """Connect the ``d_tube`` field to live range validation (idempotent)."""
        if "d_tube" not in self.param_inputs:
            return

        d_tube_input = self.param_inputs["d_tube"]
        d_tube_input.textChanged.connect(self.validate_d_tube_range)

    def _set_d_tube_validation_message(self, message: str):
        """Write/replace a single ``d_tube`` validation line in the console."""
        cursor = self.output_display.textCursor()
        document = self.output_display.document()

        # Try to replace old validation line
        if self._d_tube_validation_msg_index is not None:
            block = document.findBlockByNumber(self._d_tube_validation_msg_index)
            if block.isValid():
                cursor.setPosition(block.position())
                cursor.select(cursor.LineUnderCursor)
                cursor.removeSelectedText()
                cursor.insertText(message)
                return  # Done
            else:
                # Block is invalid, fall back to appending
                self._d_tube_validation_msg_index = None

        # Append new message and track index
        self.output_display.append(message)
        self._d_tube_validation_msg_index = document.blockCount() - 1

    def validate_d_tube_range(self):
        """Validate ``d_tube`` against the geometry-derived maximum, live."""
        required_keys = ["rg_param", "ry_param", "N_LIST", "d_tube"]
        for key in required_keys:
            if key not in self.param_inputs:
                return
            if not self.param_inputs[key].text().strip():
                return  # Wait for user input

        # Extract raw values for inspection
        d_tube_raw = self.param_inputs["d_tube"].text().strip()
        rg_raw = self.param_inputs["rg_param"].text().strip()
        ry_raw = self.param_inputs["ry_param"].text().strip()
        N_raw = self.param_inputs["N_LIST"].text().strip()

        # Check if d_tube is non-numeric or empty after all inputs are filled
        if not d_tube_raw or not re.fullmatch(r"-?\d+(\.\d+)?", d_tube_raw):
            self._set_d_tube_validation_message("[Validation] d_tube is invalid or empty — must be a number.")
            self.d_tube_valid = False
            return

        # Now try parsing all values
        try:
            d_tube = float(d_tube_raw)
            rg = int(rg_raw)
            ry = int(ry_raw)
            N = list(map(int, re.findall(r"\d+", N_raw)))
            if not N:
                return  # Wait until N_list is valid
        except ValueError:
            return  # Likely mid-input; don't lock controls

        # Respect mixed_config if and only if it's part of the schema
        current_comp = self.comp_select.currentText()
        schema = params_config.PARAMETER_SCHEMAS.get(current_comp)
        if schema is None:
            return
        default_params = dict(schema.get("default", []))

        # Estimate maximum legal d_tube
        try:
            j_max_lim = sup.j_max_bef_overlap_no_JIT(ry, N)
            max_d_tube = sup.solve_d_rect_no_JIT(1, ry, rg, j_max_lim, 0)
        except Exception as e:
            self._set_d_tube_validation_message(f"[Validation Error] Internal calculation failed: {e}")
            self.d_tube_valid = False
            return

        # Final bounds check
        if 0 <= d_tube <= max_d_tube:
            self._set_d_tube_validation_message(
                f"[Validation] d_tube = {d_tube} is valid (max allowed: {max_d_tube - 10 ** -6:.6f})")
            self.d_tube_valid = True
            self.validate_computation()
        else:
            self._set_d_tube_validation_message(
                f"[Validation] d_tube = {d_tube} is invalid! Must be in range (0, {max_d_tube:.4f})")
            self.d_tube_valid = False
            self.validate_computation()

    def _set_computation_controls_enabled(self, enabled):
        """Enable/disable the run, enqueue, execute, preview and animate controls."""
        self.launch_button.setEnabled(enabled)
        self.enqueue_button.setEnabled(enabled)
        self.run_queue_button.setEnabled(enabled)
        self.display_domain_button.setEnabled(enabled)
        self._set_launch_animation_button_enabled(enabled)

    def _set_launch_animation_button_enabled(self, enabled):
        """Enable/disable the Launch-Animation button.

        The green (ready) / grey (disabled) appearance is handled by the
        ``#LaunchAnimationButton:enabled`` / ``:disabled`` rules in ``style.qss``,
        so no per-widget stylesheet is needed here.
        """
        self.launch_animation_button.setEnabled(enabled)
