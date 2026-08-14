"""
parameters.py
=============

:class:`ParametersMixin` -- builds and maintains the dynamic parameter form.

Responsibilities
----------------
* Rebuild the parameter ``QFormLayout`` whenever the selected computation
  changes (:meth:`update_parameter_fields`), wiring tooltips from
  ``params_config.PARAMETER_HINTS`` and live-validation signals.
* Show/hide the "advanced" (defaulted) parameters (:meth:`toggle_advanced_fields`).
* Reset the form and clear the result widgets (:meth:`clear_parameter_fields`,
  :meth:`clear_displayed_results`).
* Provide keyboard up/down navigation between fields (:meth:`eventFilter`).

Behaviour is preserved from the original ``views.py`` with two GUI-only fixes:

* ``update_parameter_fields`` now clears the form with
  ``while rowCount(): removeRow(0)`` instead of iterating ``count()`` (which
  counts *items*, not *rows*, and produced ``QFormLayout::takeRow: Invalid row``
  warnings on every rebuild).
* ``eventFilter`` guards ``list.index`` so a stray ``QLineEdit`` can never raise
  ``ValueError``.
"""

from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtWidgets import QLineEdit, QLabel

from .. import params_config


class ParametersMixin:
    """Parameter-form construction, resets and keyboard navigation."""

    def update_parameter_fields(self, computation_name):
        """Rebuild the parameter form for the given computation schema."""
        # Clear previous inputs. ``rowCount`` (not ``count``) is the number of
        # label/field pairs; removing row 0 repeatedly avoids invalid-row warnings.
        while self.param_form.rowCount() > 0:
            self.param_form.removeRow(0)

        self.param_inputs.clear()
        self.advanced_widgets.clear()

        schema = params_config.PARAMETER_SCHEMAS[computation_name]
        hints = params_config.PARAMETER_HINTS  # <-- Hints dictionary

        # === Required parameters ===
        for param, _ in schema.get("required", []):
            input_field = QLineEdit()
            input_field.installEventFilter(self)

            label = QLabel(f"{param}:")
            self.param_inputs[param] = input_field

            # Set hover-based tooltip if hint is available
            if param in hints:
                label.setToolTip(hints[param])
                input_field.setToolTip(hints[param])

            self.param_form.addRow(label, input_field)

        # === Default (advanced) parameters ===
        for param, value in schema.get("default", []):
            input_field = QLineEdit(str(value))
            input_field.installEventFilter(self)
            input_field.setVisible(False)

            label = QLabel(f"{param}:")
            label.setVisible(False)

            self.param_inputs[param] = input_field

            # Set hover-based tooltip if hint is available
            if param in hints:
                label.setToolTip(hints[param])
                input_field.setToolTip(hints[param])

            self.param_form.addRow(label, input_field)
            self.advanced_widgets.append((label, input_field))

        self.setup_d_tube_live_check()

        for field in self.param_inputs.values():
            field.textChanged.connect(self.validate_computation)

    def toggle_advanced_fields(self, state):
        """Show or hide the advanced (defaulted) parameter rows."""
        show = state == Qt.Checked
        for label, field in self.advanced_widgets:
            label.setVisible(show)
            field.setVisible(show)

    def clear_parameter_fields(self):
        """Reset the parameter form to defaults and clear the output console."""
        self.update_parameter_fields(self.comp_select.currentText())
        self.output_display.clear()

    def clear_displayed_results(self):
        """Hide the file/preview result widgets and the restored-state label."""
        self.output_files_widget.hide()
        self.png_preview_widget.hide()
        self.restored_label.hide()

    def eventFilter(self, source, event):
        """Up/Down arrow keys move focus between parameter fields (wrapping)."""
        if event.type() == QEvent.KeyPress and isinstance(source, QLineEdit):
            keys = list(self.param_inputs.values())
            if source in keys:
                idx = keys.index(source)
                if event.key() == Qt.Key_Up:
                    keys[(idx - 1) % len(keys)].setFocus()
                    return True
                elif event.key() == Qt.Key_Down:
                    keys[(idx + 1) % len(keys)].setFocus()
                    return True
        return super().eventFilter(source, event)

    def get_line_edit(self):
        """Install this panel as the event filter on every parameter field."""
        for line_edit in self.param_inputs.values():
            line_edit.installEventFilter(self)
