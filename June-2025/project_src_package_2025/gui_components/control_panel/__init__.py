"""
control_panel
=============

The Biophysics GUI control panel, decomposed into focused modules.

This package replaces the former monolithic ``views.py``. The single
``ControlPanel`` class is now assembled from one mixin per concern (see
:mod:`~project_src_package_2025.gui_components.control_panel.panel`), but its
public API is unchanged: import :class:`ControlPanel` from here (or, for
backward compatibility, from ``gui_components.views``).

    from project_src_package_2025.gui_components.control_panel import ControlPanel
"""

from .panel import ControlPanel
from .toggle_select_list import ToggleSelectListWidget

__all__ = ["ControlPanel", "ToggleSelectListWidget"]
