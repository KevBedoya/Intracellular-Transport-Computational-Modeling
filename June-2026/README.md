# Bedoya-Kogan Biophysics — Installation Guide

A desktop application for **intracellular-transport / MFPT computational
modeling**. It provides a graphical interface for running the project's
numerical routines (mean first-passage time, density and mass analyses,
heatmaps, etc.) without touching the command line.

This guide is for **installing and running the prebuilt application**. If you
want to build it yourself, jump to [Build from source](#build-from-source).

---

## Table of contents

- [What you receive](#what-you-receive)
- [System requirements](#system-requirements)
- [Install on macOS](#install-on-macos)
- [Install on Windows](#install-on-windows)
- [Verify it works](#verify-it-works)
- [Where your results are saved](#where-your-results-are-saved)
- [Uninstall](#uninstall)
- [Troubleshooting](#troubleshooting)
- [Build from source](#build-from-source)
- [Version & credits](#version--credits)

---

## What you receive

**Download:** [**Releases → latest**](https://github.com/KevinB88/Intracellular-Transport-Computational-Modeling/releases/latest)
(current: [**v2.2**](https://github.com/KevinB88/Intracellular-Transport-Computational-Modeling/releases/tag/v2.2)).

| Platform | File you get | What it is |
|----------|--------------|------------|
| **macOS** | [`Bedoya-Kogan.dmg`](https://github.com/KevinB88/Intracellular-Transport-Computational-Modeling/releases/latest) (~116 MB) | A disk image. Open it and drag the app to Applications. |
| **Windows** | `Project2025App` folder (zipped) | A folder containing `Project2025App.exe` and its support files. **Not yet attached to a release** — [build from source](#build-from-source) with `build_windows.bat`. |

> The application is **not code-signed / notarized**. This is normal for
> in-house research software, but it means the operating system will show a
> one-time security warning the first time you open it. The steps below explain
> exactly how to get past it.

---

## System requirements

**macOS**
- A Mac with **Apple Silicon** (M1, M2, M3, or newer).
- A recent version of macOS (Big Sur 11 or later).
- ⚠️ **Intel Macs are not supported by the prebuilt app.** If you have an Intel
  Mac, you must [build from source](#build-from-source) on that machine.

**Windows**
- **64-bit Windows 10 or Windows 11.**
- No Python installation is required — everything is bundled.

About **300 MB** of free disk space (the app bundles Python, NumPy, SciPy,
Numba, matplotlib, and Qt).

---

## Install on macOS

1. **Download & open the disk image.** Get **`Bedoya-Kogan.dmg`** from the
   [latest release](https://github.com/KevinB88/Intracellular-Transport-Computational-Modeling/releases/latest)
   and double-click it. A window opens showing the app icon next to an
   **Applications** shortcut.
2. **Install.** Drag **`Bedoya-Kogan.app`** onto the **Applications** shortcut in
   that same window.
3. **Eject the disk image.** In Finder, click the ⏏ eject button next to
   "Bedoya-Kogan" in the sidebar (you can delete the `.dmg` afterward).
4. **First launch (important — unsigned app).** Opening it normally the first
   time will say *"…cannot be opened because Apple cannot check it for malicious
   software."* Use **one** of these:

   - **Right-click method (easiest):** Open the **Applications** folder,
     **right-click** (or Control-click) **Bedoya-Kogan**, choose **Open**, then
     click **Open** in the dialog.
   - **Settings method:** Try to open the app once (it gets blocked), then open
      **System Settings → Privacy & Security**, scroll to the Security section,
      and click **Open Anyway** next to the Bedoya-Kogan message. Confirm with
      **Open**.

   You only need to do this **once**. After that, open it like any normal app
   (Launchpad, Spotlight, or double-click).

> If you instead see *"Bedoya-Kogan is damaged and can't be opened,"* that's the
> macOS quarantine flag, not actual damage — see
> [Troubleshooting](#macos-app-is-damaged-and-cant-be-opened).

---

## Install on Windows

1. **Unzip the whole folder.** Right-click the downloaded `Project2025App.zip`
   → **Extract All…** → choose a location (e.g. your Desktop or
   `C:\Program Files`). You should end up with a **`Project2025App`** folder.
2. **Run the app.** Open that folder and double-click **`Project2025App.exe`**.
3. **First launch (important — unsigned app).** Windows SmartScreen may show
   *"Windows protected your PC."* Click **More info**, then **Run anyway**.
   You only need to do this once.
4. **(Optional) Make a shortcut.** Right-click **`Project2025App.exe`** →
   **Send to → Desktop (create shortcut)**. Launch from the shortcut afterward.

> ⚠️ **Keep the folder intact.** `Project2025App.exe` depends on all the files
> next to it. **Do not move the `.exe` out of the folder** — move or copy the
> *entire* `Project2025App` folder instead. (A Desktop *shortcut* is fine; the
> shortcut points back into the folder.)

---

## Verify it works

A quick end-to-end check so you know the install is healthy:

1. Launch the app. The main window with the control panel should appear.
2. In the computation selector, choose **"Compute MFPT until time T."**
3. Enter small test parameters, for example:
   - `rg_param = 8`
   - `ry_param = 8`
   - `N_LIST = [0, 2, 4, 6]`
   - `v_param = 1`
   - `w_param = 10`
   - `T_param = 1`
4. Click **Launch**. After a short moment the output console should report a
   result such as **`MFPT = 0.2952…`** (your value depends on the parameters).

If you see a numeric MFPT result, the app — including its background
computation engine — is working correctly.

---

## Where your results are saved

Computation outputs (CSV data, PNG plots, JSON results) are written to a
**`data_output/`** folder located next to the application binary:

- **macOS:** inside the app bundle, at
  `Bedoya-Kogan.app/Contents/MacOS/data_output/`.
  (In Finder: right-click the app → **Show Package Contents** →
  `Contents/MacOS/data_output`.)
- **Windows:** inside the `Project2025App` folder, at
  `Project2025App\data_output\`.

The folder is created automatically the first time you run a computation.

---

## Uninstall

- **macOS:** Drag **Bedoya-Kogan.app** from Applications to the Trash. To also
  remove saved results, delete the `data_output` folder inside the bundle first
  (see above).
- **Windows:** Delete the entire **`Project2025App`** folder (and any Desktop
  shortcut you created).

---

## Troubleshooting

### macOS: "app is damaged and can't be opened"
This is the macOS quarantine attribute on unsigned apps downloaded from the
internet — the app is not actually damaged. Remove the flag from Terminal:

```bash
xattr -dr com.apple.quarantine /Applications/Bedoya-Kogan.app
```

Then open the app normally. (If you installed it somewhere other than
Applications, adjust the path accordingly.)

### macOS: "cannot be opened because Apple cannot check it…"
Expected on first launch — use the right-click → **Open** method described in
[Install on macOS](#install-on-macos).

### macOS: app bounces in the Dock then quits / won't open at all
Confirm your Mac is **Apple Silicon** (Apple menu → About This Mac →
"Chip" should read Apple M-series). The prebuilt app does **not** run on Intel
Macs — those must [build from source](#build-from-source).

### Windows: app won't start, or reports a missing DLL/file
This almost always means the `.exe` was moved out of its folder, or the folder
wasn't fully extracted. Re-extract the **entire** `Project2025App` folder and
run the `.exe` from inside it. Don't copy the `.exe` on its own.

### Windows: SmartScreen or antivirus blocks the app
Because the build is unsigned, SmartScreen may warn (**More info → Run anyway**)
and some antivirus tools may flag PyInstaller executables as a false positive.
If your antivirus quarantines it, allow/whitelist the `Project2025App` folder.

### A computation seems stuck
Heavy computations run in a background process and can take time for large grid
sizes. Try the small test parameters in
[Verify it works](#verify-it-works) first to confirm the engine runs, then scale
up.

---

## Build from source

You only need this if you're a developer, or you're on an **Intel Mac** /
another platform without a prebuilt download.

**Prerequisites:** Python 3.9+ (64-bit). PyInstaller **cannot cross-compile** —
build the macOS app on a Mac and the Windows executable on Windows.

The repository includes everything needed:

| File | Purpose |
|------|---------|
| `requirements.txt` | Runtime + build dependencies |
| `Project2025App.spec` | PyInstaller build definition (cross-platform) |
| `build_macos.sh` | One-command macOS build → `dist/Bedoya-Kogan.app` + `.dmg` |
| `build_windows.bat` | One-command Windows build → `dist\Project2025App\` |

**macOS:**
```bash
./build_macos.sh
```

**Windows** (from a Command Prompt in the project folder):
```bat
build_windows.bat
```

**Manual build (either platform):**
```bash
python -m venv .venv
# macOS/Linux:  source .venv/bin/activate
# Windows:      .venv\Scripts\activate.bat
pip install -r requirements.txt
pyinstaller Project2025App.spec --noconfirm --clean
```

The macOS build produces an **arm64** app by default (matching the build
machine). To target Intel Macs, build on an Intel Mac, or produce a `universal2`
build.

---

## Version & credits

- **Application:** Bedoya-Kogan Biophysics, **version 2.2**
- **Bundle identifier (macOS):** `edu.cuny.qc.biophysics.bedoya-kogan`
- **Author:** Kevin Bedoya

For deeper packaging internals (how the worker subprocess and import paths are
handled in frozen builds), see the comments in `Project2025App.spec`, `main.py`,
and `project_src_package_2025/multiprocessing_tools/subprocess_launcher.py`.
