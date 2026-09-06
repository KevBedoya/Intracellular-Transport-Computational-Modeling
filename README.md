# Bedoya-Kogan Biophysics

Computational modeling of **intracellular transport / MFPT**: a numerical PDE
solver for a two-layer (diffusive + advective) polar domain, with mean
first-passage time, density, mass and characteristic-time analyses.

There are **two ways to run it**, and which one you want depends on how big the
computation is:

| | Use it when | Start here |
|---|---|---|
| **Desktop app** | You want a local GUI on your own machine, for small to moderate runs. | This document |
| **Compute server** | You want to queue long runs on the lab workstation from your own laptop, and collect results later. | [`server/docs/CONNECTING.md`](server/docs/CONNECTING.md) |

The compute server is the better fit for anything large: it runs up to four jobs
concurrently on the workstation, survives you closing your laptop, and is driven
from a browser. Cost scales as the sixth power of grid size, so a 96×96 run takes
about 3.3 hours and a 112×112 run about 9 — not something to hold a laptop open
for.

Most of this document is about **installing and running the prebuilt desktop
application**. To build it yourself, jump to
[Build from source](#build-from-source).

## Other documentation

| | |
|---|---|
| [`server/docs/CONNECTING.md`](server/docs/CONNECTING.md) | Getting on the tailnet and reaching the compute server and its web UI |
| [`server/docs/COMPUTATION_MENU.md`](server/docs/COMPUTATION_MENU.md) | All 19 computations: parameters, outputs, and what each costs |
| [`server/docs/UI_PLAN.md`](server/docs/UI_PLAN.md) | Design of the browser front end |
| [`docs/REPOSITORY_LAYOUT.md`](docs/REPOSITORY_LAYOUT.md) | How the repository is organised |
| [`docs/COMPUTATIONAL_WORKFLOW.md`](docs/COMPUTATIONAL_WORKFLOW.md) | The numerical workflow |

---

## Table of contents

- [What you receive](#what-you-receive)
- [System requirements](#system-requirements)
- [Install on macOS](#install-on-macos)
- [Install on Windows](#install-on-windows)
- [Verify it works](#verify-it-works)
- [Where your results are saved](#where-your-results-are-saved)
- [Running on the compute server instead](#running-on-the-compute-server-instead)
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

Computation outputs (CSV data, PNG plots, JSON results) go to a
**`data_output/`** folder. Where that folder lives depends on how you are
running:

**Prebuilt application** — next to the application binary:

- **macOS:** inside the app bundle, at
  `Bedoya-Kogan.app/Contents/MacOS/data_output/`.
  (In Finder: right-click the app → **Show Package Contents** →
  `Contents/MacOS/data_output`.)
- **Windows:** inside the `Project2025App` folder, at
  `Project2025App\data_output\`.

**From a source checkout** — at the **repository root**, `<repo>/data_output/`,
deliberately outside `src/` so generated data never mixes with code. Set
`ITCM_OUTPUT_ROOT` to redirect it; the job worker uses that to give each job its
own directory.

Either way the folder is created on the first run. `data_output/` is scratch and
is not committed — results worth keeping get promoted into `results/` under a
descriptive name.

---

## Running on the compute server instead

If a run is going to take hours, queue it on the workstation rather than your own
machine. The server exposes the same computations over HTTP with a browser front
end, runs four jobs at once, and keeps going while your laptop is closed.

Full instructions are in [`server/docs/CONNECTING.md`](server/docs/CONNECTING.md);
the short version:

1. Join the tailnet (invite from the admin, then install Tailscale and sign in).
2. Find the server's address — on the workstation:
   ```powershell
   tailscale ip -4
   ```
3. Open `http://<that-address>:8000/ui` in a browser.

Pick a computation, fill the form, queue it, and collect the CSVs and plots from
the same page when it finishes. Available computations and their costs are
catalogued in
[`server/docs/COMPUTATION_MENU.md`](server/docs/COMPUTATION_MENU.md).

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
| `packaging/Project2025App.spec` | PyInstaller build definition (cross-platform) |
| `packaging/build_macos.sh` | One-command macOS build → `dist/Bedoya-Kogan.app` + `.dmg` |
| `packaging/build_windows.bat` | One-command Windows build → `dist\Project2025App\` |
| `packaging/setup_py2app.py` | py2app configuration (alternative macOS route) |

**macOS:**
```bash
./packaging/build_macos.sh
```

**Windows** (from a Command Prompt in the project folder):
```bat
packaging\build_windows.bat
```

**Manual build (either platform), from the repository root:**
```bash
python -m venv .venv
# macOS/Linux:  source .venv/bin/activate
# Windows:      .venv\Scripts\activate.bat
pip install -r requirements.txt
pyinstaller packaging/Project2025App.spec --noconfirm --clean
```

> Run the build from the **repository root**, not from `packaging/`. The spec
> resolves the repository root as its own parent directory and the package as
> `src/intracellular_transport`, so it does not care where you invoke it from —
> but `requirements.txt` and the virtualenv above assume the root.

The macOS build produces an **arm64** app by default (matching the build
machine). To target Intel Macs, build on an Intel Mac, or produce a `universal2`
build.

---

## Version & credits

- **Application:** Bedoya-Kogan Biophysics, **version 2.2**
- **Bundle identifier (macOS):** `edu.cuny.qc.biophysics.bedoya-kogan`
- **Author:** Kevin Bedoya

For deeper packaging internals (how the worker subprocess and import paths are
handled in frozen builds), see the comments in `packaging/Project2025App.spec`, `main.py`,
and `src/intracellular_transport/multiprocessing_tools/subprocess_launcher.py`.
