"""
theme.py
========

Centralised look-and-feel utilities for the Biophysics GUI.

Everything that controls *appearance* (as opposed to behaviour or layout) lives
here so the rest of the GUI never has to hard-code colours, asset paths, or
stylesheet-loading logic:

* :func:`enable_high_dpi`   -- crisp rendering on Retina/HiDPI displays.
* :func:`load_stylesheet`   -- read the application QSS, with a filesystem
                               fallback for frozen (PyInstaller) builds.
* :func:`spinner_gif_path`  -- resolve the loading-spinner asset for both source
                               and frozen runs (replaces the old hard-coded
                               Windows path that never resolved on macOS/Linux).
* :func:`apply_button_state`-- drive the Launch button's idle/running/success/
                               error colours through a dynamic QSS property
                               instead of fighting the global stylesheet with a
                               per-widget :class:`QPalette`.

Object-name constants are exported so the Python widgets and the ``style.qss``
selectors stay in sync from a single source of truth.
"""

import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Object names / dynamic-property keys shared between Python and style.qss
# --------------------------------------------------------------------------- #
#: ``objectName`` of the primary Launch button (styled via ``#LaunchButton``).
LAUNCH_BUTTON = "LaunchButton"
#: ``objectName`` of the secondary "+" enqueue button.
ENQUEUE_BUTTON = "EnqueueButton"
#: ``objectName`` of the "Execute Queue" button.
RUN_QUEUE_BUTTON = "RunQueueButton"
#: ``objectName`` of the "Launch Animation" button (green when enabled).
LAUNCH_ANIM_BUTTON = "LaunchAnimationButton"
#: ``objectName`` of the read-only output console (``QTextEdit``).
OUTPUT_CONSOLE = "OutputConsole"
#: ``objectName`` of the job-queue list widget.
QUEUE_LIST = "QueueList"
#: ``objectName`` of the restored-state status label (kept camelCase for the
#: pre-existing ``#restoredLabel`` selector).
RESTORED_LABEL = "restoredLabel"
#: ``objectName`` applied to small section-heading labels.
SECTION_LABEL = "SectionLabel"

#: Dynamic property name carrying the Launch button state. style.qss reacts to
#: ``#LaunchButton[state="running"]`` etc.
STATE_PROPERTY = "state"
#: Valid values for :data:`STATE_PROPERTY`.
STATE_IDLE = "idle"
STATE_RUNNING = "running"
STATE_SUCCESS = "success"
STATE_ERROR = "error"


# --------------------------------------------------------------------------- #
# HiDPI
# --------------------------------------------------------------------------- #
def enable_high_dpi():
    """Enable HiDPI scaling and pixmaps.

    Must be called *before* the :class:`QApplication` is constructed, otherwise
    the attributes have no effect. Safe to call more than once.
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    for attr in ("AA_EnableHighDpiScaling", "AA_UseHighDpiPixmaps"):
        flag = getattr(Qt, attr, None)
        if flag is not None:
            QApplication.setAttribute(flag, True)


#: Preferred UI font families, most-desirable first.
_FONT_CANDIDATES = ("Helvetica Neue", "Segoe UI", "Arial")


def apply_base_font(app, point_size=11):
    """Set the application's base UI font and return the chosen family name.

    Setting the font on the :class:`QApplication` *before* widgets are created
    ensures every widget computes its size hint with the final metrics. (Doing
    this purely in the stylesheet enlarges text *after* size hints are computed,
    which clips tab labels and tight button rows.) Falls back to the platform
    default family at ``point_size`` if none of the preferred families exist.
    """
    from PyQt5.QtGui import QFont, QFontDatabase

    available = set(QFontDatabase().families())
    for family in _FONT_CANDIDATES:
        if family in available:
            app.setFont(QFont(family, point_size))
            return family

    font = app.font()
    font.setPointSize(point_size)
    app.setFont(font)
    return font.family()


# --------------------------------------------------------------------------- #
# Asset / stylesheet resolution
# --------------------------------------------------------------------------- #
def _package_root():
    """Filesystem directory of the ``gui_components`` package (this file's dir)."""
    return Path(__file__).resolve().parent


def _frozen_base():
    """Base directory PyInstaller extracts bundled data into, or ``None``."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return None


def load_stylesheet():
    """Return the application QSS as a string.

    Tries :mod:`importlib.resources` first (works for normal installs and source
    runs), then falls back to reading the file relative to this module, which is
    the robust path for frozen/bundled builds.
    """
    # Primary: importlib.resources against the styles sub-package.
    try:
        from importlib import resources
        return (
            resources.files("project_src_package_2025.gui_components.styles")
            .joinpath("style.qss")
            .read_text(encoding="utf-8")
        )
    except Exception:
        pass

    # Fallback: direct filesystem read (source tree or _MEIPASS).
    candidates = [_package_root() / "styles" / "style.qss"]
    frozen = _frozen_base()
    if frozen is not None:
        candidates.append(
            frozen / "project_src_package_2025" / "gui_components" / "styles" / "style.qss"
        )
    for path in candidates:
        try:
            if path.is_file():
                return path.read_text(encoding="utf-8")
        except OSError:
            continue
    return ""  # Render with Qt defaults rather than crashing.


def spinner_gif_path(size=100):
    """Return a filesystem path (str) to the loading-spinner GIF, or ``""``.

    ``size`` selects one of the bundled variants (40, 60, 100). Resolution works
    in source trees and in frozen builds; if nothing is found an empty string is
    returned so :class:`~PyQt5.QtGui.QMovie` simply renders nothing instead of
    raising.
    """
    rel = Path("visual_materials") / "gifs" / f"spinner_{size}x{size}.gif"
    candidates = [_package_root() / rel]
    frozen = _frozen_base()
    if frozen is not None:
        candidates.append(frozen / "project_src_package_2025" / "gui_components" / rel)
    for path in candidates:
        if path.is_file():
            return str(path)
    return ""


# --------------------------------------------------------------------------- #
# Dynamic stylesheet helpers
# --------------------------------------------------------------------------- #
def repolish(widget):
    """Force a widget to re-evaluate its stylesheet after a property change.

    Qt does not automatically restyle a widget when a dynamic property used in a
    selector changes, so we unpolish/polish and request a repaint.
    """
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def apply_button_state(button, state):
    """Set the Launch button ``state`` property and restyle it.

    ``state`` is one of :data:`STATE_IDLE`, :data:`STATE_RUNNING`,
    :data:`STATE_SUCCESS`, :data:`STATE_ERROR`.
    """
    button.setProperty(STATE_PROPERTY, state)
    repolish(button)
