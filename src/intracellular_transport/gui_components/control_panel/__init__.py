"""
control_panel
=============

The Biophysics GUI control panel, decomposed into focused modules.

This package replaces the former monolithic ``views.py``. The single
``ControlPanel`` class is now assembled from one mixin per concern (see
:mod:`~intracellular_transport.gui_components.control_panel.panel`), but its
public API is unchanged: import :class:`ControlPanel` from here (or, for
backward compatibility, from ``gui_components.views``).

    from intracellular_transport.gui_components.control_panel import ControlPanel
"""

from .panel import ControlPanel
from .toggle_select_list import ToggleSelectListWidget

__all__ = ["ControlPanel", "ToggleSelectListWidget"]
