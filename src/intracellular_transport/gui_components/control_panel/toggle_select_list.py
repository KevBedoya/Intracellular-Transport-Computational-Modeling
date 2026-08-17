"""
toggle_select_list.py
======================

Custom ``QListWidget`` used for the job-queue panel of the control panel.

The widget adds two affordances on top of the standard list:

* **Click-to-toggle selection** -- clicking an already-selected row clears the
  selection (instead of keeping it selected, which is the Qt default). This lets
  the user dismiss the move/remove/edit buttons simply by clicking the active
  row again.
* **Escape-to-clear** -- pressing :kbd:`Esc` clears the current selection and
  focus.

After either interaction the widget notifies its parent (the
:class:`~intracellular_transport.gui_components.control_panel.panel.ControlPanel`)
so it can show or hide the contextual queue-management buttons via
``toggle_move_buttons_visibility``.

This class was extracted verbatim (behaviour-preserving) from the original
``views.py`` monolith. The only addition is a defensive guard around the parent
callback so the widget can be instantiated/tested in isolation without a fully
wired parent.
"""

from PyQt5.QtWidgets import QListWidget
from PyQt5.QtCore import Qt


class ToggleSelectListWidget(QListWidget):
    """A ``QListWidget`` whose selection can be toggled off by re-clicking."""

    def _notify_parent(self):
        """Tell the owning panel to refresh queue-button visibility.

        Guarded so the widget is safe to use before it has been parented to a
        :class:`ControlPanel` (e.g. in unit tests or standalone previews).
        """
        parent = self.parent()
        if parent is not None and hasattr(parent, "toggle_move_buttons_visibility"):
            parent.toggle_move_buttons_visibility()

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        current_item = self.currentItem()

        # Clicking the row that is already current clears the selection.
        if item and item == current_item:
            self.clearSelection()
            self.clearFocus()
        else:
            super().mousePressEvent(event)

        self._notify_parent()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.clearSelection()
            self.clearFocus()
        else:
            super().keyPressEvent(event)

        self._notify_parent()
