"""
panel.py
========

:class:`ControlPanel` -- the main control widget of the Biophysics GUI.

This is the public face of the ``control_panel`` package. It is intentionally
*thin*: all behaviour is provided by focused mixins (one per file) and all widget
construction lives in :class:`~.ui_builder.UIBuilderMixin`. ``ControlPanel``
simply composes them and kicks off the build.

Composition / MRO
------------------
``ControlPanel`` inherits from every mixin plus :class:`~PyQt5.QtWidgets.QWidget`
(which must come **last** so cooperative ``super().__init__`` reaches Qt, and so
that the :meth:`eventFilter` override in
:class:`~.parameters.ParametersMixin` resolves up to ``QWidget.eventFilter``).
The mixins are plain ``object`` subclasses with no ``__init__`` of their own, so
they never interfere with Qt's initialisation. Because everything composes into
one instance, methods defined in different files all resolve on the same
``self`` exactly as they did in the original monolithic class -- no public method
name, signature, or behaviour changed.

Responsibilities by mixin
-------------------------
* :class:`~.ui_builder.UIBuilderMixin`     -- widget tree + signal wiring
* :class:`~.helpers.HelpersMixin`          -- timestamps, hints, geometry utils
* :class:`~.parameters.ParametersMixin`    -- dynamic parameter form + key nav
* :class:`~.computation.ComputationMixin`  -- single-run worker launch
* :class:`~.queue_ops.QueueMixin`          -- job queue + batch run
* :class:`~.history.HistoryMixin`          -- history dropdown + restoration
* :class:`~.visualization.VisualizationMixin` -- domain preview + animation
* :class:`~.validation.ValidationMixin`    -- live ``d_tube`` validation + gating
"""

from PyQt5.QtWidgets import QWidget

from .ui_builder import UIBuilderMixin
from .helpers import HelpersMixin
from .parameters import ParametersMixin
from .computation import ComputationMixin
from .queue_ops import QueueMixin
from .history import HistoryMixin
from .visualization import VisualizationMixin
from .validation import ValidationMixin


class ControlPanel(
    UIBuilderMixin,
    HelpersMixin,
    ParametersMixin,
    ComputationMixin,
    QueueMixin,
    HistoryMixin,
    VisualizationMixin,
    ValidationMixin,
    QWidget,
):
    """Top-level control panel widget, composed from the behavioural mixins."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.build_ui(main_window)
