"""
views.py  (compatibility shim)
==============================

The control panel that used to live here as one ~1600-line class has been
refactored into the :mod:`intracellular_transport.gui_components.control_panel`
package, with one module per concern (UI construction, parameters, computation,
queue, history, visualization, validation, helpers) plus a custom list widget.

This module is kept as a thin re-export so existing imports keep working:

    from intracellular_transport.gui_components import views
    panel = views.ControlPanel(main_window)          # still works
    views.ToggleSelectListWidget                      # still works

New code should prefer importing from the package directly::

    from intracellular_transport.gui_components.control_panel import ControlPanel

Nothing else needs to change -- ``ControlPanel`` exposes the exact same public
methods and behaviour as before.
"""

from .control_panel import ControlPanel
from .control_panel.toggle_select_list import ToggleSelectListWidget

__all__ = ["ControlPanel", "ToggleSelectListWidget"]
