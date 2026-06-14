# -*- mode: python ; coding: utf-8 -*-
#
# Cross-platform PyInstaller spec for the Biophysics GUI.
#
#   macOS    -> dist/Bedoya-Kogan.app   (a one-dir app bundle)
#   Windows  -> dist/Project2025App/    (a one-dir folder containing the .exe)
#
# Build with:   pyinstaller Project2025App.spec --noconfirm
# (see build_macos.sh / build_windows.bat and README.md)

import os
import sys

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# SPECPATH is the directory containing this .spec file (set by PyInstaller).
ROOT = os.path.abspath(SPECPATH)
PKG = os.path.join(ROOT, "project_src_package_2025")

# --------------------------------------------------------------------------- #
# Import resolution
# --------------------------------------------------------------------------- #
# The codebase uses a dual import scheme: some modules import via the
# `project_src_package_2025.<sub>` path (needs ROOT on the search path) while
# others import the sub-packages bare, e.g. `from computational_tools import ...`
# (needs PKG on the search path). Both directories must be searchable so the
# module graph resolves fully.
pathex = [ROOT, PKG]

# Bare-imported sub-packages. collect_submodules pulls in modules that are only
# reached through dynamic / conditional imports the static analysis can miss.
_bare_packages = [
    "computational_tools",
    "data_processing",
    "data_visualization",
    "auxiliary_tools",
    "system_configuration",
    "launch_functions",
    "job_queuing_system",
    "multiprocessing_tools",
    "gui_components",
]

hiddenimports = []
for _name in _bare_packages:
    hiddenimports += collect_submodules(_name)
hiddenimports += collect_submodules("project_src_package_2025")
# Matplotlib's Qt5 backend is imported as a string and must be forced in.
hiddenimports += ["matplotlib.backends.backend_qt5agg"]

# --------------------------------------------------------------------------- #
# Bundled data assets (resolved at runtime by theme.py against _MEIPASS)
# --------------------------------------------------------------------------- #
datas = [
    ("project_src_package_2025/gui_components/styles/style.qss",
     "project_src_package_2025/gui_components/styles"),
]
# All loading-spinner GIF variants.
datas += collect_data_files(
    "project_src_package_2025.gui_components",
    includes=["visual_materials/gifs/*.gif"],
)

# --------------------------------------------------------------------------- #
# Per-platform icon
# --------------------------------------------------------------------------- #
if sys.platform == "darwin":
    icon_file = os.path.join(ROOT, "TransparentApp.icns")
elif sys.platform.startswith("win"):
    icon_file = os.path.join(ROOT, "TransparentApp.ico")
else:
    icon_file = None


a = Analysis(
    ["main.py"],
    pathex=pathex,
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Project2025App",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Project2025App",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Bedoya-Kogan.app",
        icon=icon_file,
        bundle_identifier="edu.cuny.qc.biophysics.bedoya-kogan",
        info_plist={
            "CFBundleName": "Bedoya-Kogan",
            "CFBundleDisplayName": "Bedoya-Kogan Biophysics",
            "CFBundleShortVersionString": "2.2",
            "CFBundleVersion": "2.2",
            "NSHighResolutionCapable": True,
        },
    )
