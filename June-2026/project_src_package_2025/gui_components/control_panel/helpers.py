"""
helpers.py
==========

:class:`HelpersMixin` -- small, dependency-light utility methods for the
control panel.

These are stateless helpers (timestamp formatting, parameter-hint pop-ups, and
widget-geometry introspection used during layout debugging). They were grouped
together because none of them touch the computation/queue/visualisation state;
keeping them isolated makes the behavioural mixins easier to read.

The mixin is a plain ``object`` subclass and is combined into
:class:`~project_src_package_2025.gui_components.control_panel.panel.ControlPanel`
via multiple inheritance, so ``self`` is always the live ``ControlPanel`` widget.
"""

from datetime import datetime

from PyQt5.QtWidgets import QMessageBox

from .. import params_config


class HelpersMixin:
    """Stateless utility methods mixed into :class:`ControlPanel`."""

    def show_param_hint(self, param_name):
        """Show a modal hint describing ``param_name`` (from ``PARAMETER_HINTS``)."""
        hint = getattr(params_config, "PARAMETER_HINTS", {}).get(
            param_name, "No hint available for this parameter."
        )
        QMessageBox.information(self, f"Hint: {param_name}", hint)

    # --- Widget geometry introspection (layout debugging utilities) -------- #
    @staticmethod
    def get_widget_center_global(widget):
        """Return the widget's centre point in global screen coordinates."""
        geom = widget.geometry()
        center_local = geom.center()
        center_global = widget.mapToGlobal(center_local)
        return {"Center of widget in global coorindates: ": center_global}

    @staticmethod
    def get_widget_corners_global(widget):
        """Return the widget's four corners in global screen coordinates."""
        rect = widget.rect()

        top_left = widget.mapToGlobal(rect.topLeft())
        top_right = widget.mapToGlobal(rect.topRight())
        bottom_left = widget.mapToGlobal(rect.bottomLeft())
        bottom_right = widget.mapToGlobal(rect.bottomRight())

        return {
            "top_left": top_left,
            "top_right": top_right,
            "bottom_left": bottom_left,
            "bottom_right": bottom_right,
        }

    @staticmethod
    def get_widget_dimensions(widget):
        """Return the widget's pixel width/height."""
        width = widget.width()
        height = widget.height()
        return {"width:": width, "height": height}

    @staticmethod
    def produce_timestamp():
        """Return a human-friendly ``hh:mm AM/PM mm/dd/yyyy`` timestamp.

        Leading zeros are stripped from the hour, month and day for a cleaner
        look in the output console.
        """
        timestamp = datetime.now().strftime("%I:%M %p %m/%d/%Y")
        timestamp = timestamp.replace(" 0", " ").replace("/0", "/")
        return timestamp
