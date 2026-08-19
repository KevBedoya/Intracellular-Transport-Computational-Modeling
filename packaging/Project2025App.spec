# -*- mode: python ; coding: utf-8 -*-
#
# Cross-platform PyInstaller spec for the Biophysics GUI.
#
#   macOS    -> dist/Bedoya-Kogan.app   (a one-dir app bundle)
#   Windows  -> dist/Project2025App/    (a one-dir folder containing the .exe)
#
# Build with:   pyinstaller packaging/Project2025App.spec --noconfirm
# (see build_macos.sh / build_windows.bat and README.md)

import os
import sys

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# SPECPATH is the directory containing this .spec file (set by PyInstaller).
# This spec lives in packaging/, so the repository root is one level up -- it is
# NOT SPECPATH. The package itself sits under src/ (a src layout), so neither
# path can be assumed to be the spec's own directory.
SPEC_DIR = os.path.abspath(SPECPATH)
ROOT = os.path.dirname(SPEC_DIR)
SRC = os.path.join(ROOT, "src")
PKG = os.path.join(SRC, "intracellular_transport")

# --------------------------------------------------------------------------- #
# Import resolution
# --------------------------------------------------------------------------- #
# The codebase uses a dual import scheme: some modules import via the
# `intracellular_transport.<sub>` path (needs ROOT on the search path) while
# others import the sub-packages bare, e.g. `from computational_tools import ...`
# (needs PKG on the search path). Both directories must be searchable so the
# module graph resolves fully.
pathex = [PKG, SRC, ROOT]

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
hiddenimports += collect_submodules("intracellular_transport")
# Matplotlib's Qt5 backend is imported as a string and must be forced in.
hiddenimports += ["matplotlib.backends.backend_qt5agg"]

# --------------------------------------------------------------------------- #
# Bundled data assets (resolved at runtime by theme.py against _MEIPASS)
# --------------------------------------------------------------------------- #
datas = [
    (os.path.join(PKG, "gui_components", "styles", "style.qss"),
     "intracellular_transport/gui_components/styles"),
]
# All loading-spinner GIF variants.
datas += collect_data_files(
    "intracellular_transport.gui_components",
    includes=["visual_materials/gifs/*.gif"],
)

# --------------------------------------------------------------------------- #
# Per-platform icon
# --------------------------------------------------------------------------- #
if sys.platform == "darwin":
    icon_file = os.path.join(SPEC_DIR, "App.icns")
elif sys.platform.startswith("win"):
    icon_file = os.path.join(SPEC_DIR, "TransparentApp.ico")
else:
    icon_file = None


a = Analysis(
    [os.path.join(ROOT, "main.py")],
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
