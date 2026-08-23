"""Characteristic-time grid-convergence study at fixed N=16.

Submits one job per grid to the compute server, then collects the results into
a plot and a LaTeX/PDF report.

    python scripts/grid_study_char_time.py submit     # dispatch the 5 jobs
    python scripts/grid_study_char_time.py status     # progress
    python scripts/grid_study_char_time.py report     # plot + .tex + .pdf

Jobs are submitted heaviest grid first. The worker runs 4 at a time and claims
the oldest queued job, so ordering this way lets 112x112 -- which dominates the
wall time at ~8.3 hr -- start immediately rather than waiting behind the
cheap grids.
"""

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "2026-08-18_char-time-grid-study_N16")
MANIFEST = os.path.join(OUT_DIR, "jobs.json")

API = os.environ.get("ITCM", "http://100.83.174.69:8000")
COMPUTATION = "Characteristic Time (mass vs v)"

# Fixed physics, varied only by grid.
N_TUBES = 16
V = 10 ** 4
W = 100
T = 1
GRIDS = [160, 144, 128, 112, 96, 80, 64, 48]   # heaviest first: see docstring

# Values for grids whose result CSV no longer exists on disk, produced by
# scripts/recover_char_time_from_log.py and validated there against every CSV
# that does survive. Read as a fallback only -- a live CSV always wins.
RECOVERED = os.path.join(OUT_DIR, "recovered_rows.json")

# Reference point for the cost estimate: 96x96 measured at 3.28 hr post-
# optimisation. Work scales as G^6 (dT ~ 1/G^4 gives K ~ G^4, times G^2 patches).
REF_GRID, REF_HOURS = 96, 3.28


def est_hours(g):
    return REF_HOURS * (g / REF_GRID) ** 6


def n_list(g, n=N_TUBES):
    """Evenly spaced microtubule ray indices, matching the sweep convention.

    Mirrors ``numpy.linspace(0, g - g // n, n, dtype=int)``, which computes in
    floating point and then *truncates*. Rounding instead diverges whenever the
    spacing is not exact -- for ry=50, n=16 it differs in 6 of 16 positions --
    and since these indices place the microtubules, a mismatch silently changes
    the physics rather than raising.
    """
    step = g - (g // n)
    return [int(i * step / (n - 1)) for i in range(n)]


def _get(path):
    with urllib.request.urlopen(API.rstrip("/") + path, timeout=30) as r:
        return json.load(r)


def _post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(API.rstrip("/") + path, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def params_for(g):
    return {
        "rg_param": g, "ry_param": g,
        "v_LIST": [V], "w_param": W, "T_param": T,
        "N_LIST": n_list(g),
        "show_plt": False,
    }


# ------------------------------------------------------------------- submit
def cmd_submit(a):
    try:
        health = _get("/health")
    except Exception as e:
        sys.exit(f"server unreachable at {API}: {e}\n"
                 f"Check `tailscale status` and that serve.ps1 is running.")
    print(f"server  : {API}  ({health['status']})")
    print(f"fixed   : N={N_TUBES} tubes, v={V}, a=b=w={W}, T={T}")
    print()
    print(f"{'grid':>9} {'est wall':>10}  action")

    if a.dry_run:
        for g in GRIDS:
            print(f"{g:>4}x{g:<4} {est_hours(g):>7.2f} hr  would submit")
        print("\n--dry-run: nothing submitted")
        print(json.dumps({"computation": COMPUTATION,
                          "params": params_for(GRIDS[0])}, indent=2))
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    entries = []
    for g in GRIDS:
        body = {"computation": COMPUTATION, "params": params_for(g),
                "submitted_by": a.by}
        r = _post("/jobs", body)
        entries.append({"grid": g, "job_id": r["job_id"],
                        "est_hours": round(est_hours(g), 3)})
        print(f"{g:>4}x{g:<4} {est_hours(g):>7.2f} hr  submitted {r['job_id'][:8]}")

    manifest = {
        "api": API, "computation": COMPUTATION,
        "fixed": {"N_tubes": N_TUBES, "v": V, "w": W, "T": T,
                  "domain_radius": 1.0, "D": 1.0, "d_tube": 0.0,
                  "center_init_cond": True},
        "jobs": entries,
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest: {MANIFEST}")
    print(f"expected wall time ~{max(est_hours(g) for g in GRIDS):.1f} hr "
          f"(dominated by {max(GRIDS)}x{max(GRIDS)}; they run concurrently)")
    print("\nnext:  python scripts/grid_study_char_time.py status")


# ------------------------------------------------------------------- status
def _load_manifest():
    if not os.path.exists(MANIFEST):
        sys.exit(f"no manifest at {MANIFEST}; run `submit` first")
    with open(MANIFEST) as f:
        return json.load(f)


def _elapsed_hours(job):
    from datetime import datetime
    if not (job.get("started_at") and job.get("finished_at")):
        return None
    try:
        d = (datetime.fromisoformat(job["finished_at"])
             - datetime.fromisoformat(job["started_at"]))
        return d.total_seconds() / 3600.0
    except ValueError:
        return None


def cmd_status(a):
    man = _load_manifest()
    print(f"{'grid':>9} {'status':>10} {'wall':>9} {'est':>9}  job")
    done = 0
    for e in man["jobs"]:
        j = _get(f"/jobs/{e['job_id']}")
        wall = _elapsed_hours(j)
        wall_s = f"{wall:.2f} hr" if wall is not None else "-"
        if j["status"] in ("succeeded", "failed", "cancelled"):
            done += 1
        print(f"{e['grid']:>4}x{e['grid']:<4} {j['status']:>10} {wall_s:>9} "
              f"{e['est_hours']:>6.2f} hr  {e['job_id'][:8]}")
        if j.get("error"):
            print(f"           error: {j['error']}")
    print(f"\n{done}/{len(man['jobs'])} finished")
    if done == len(man["jobs"]):
        print("next:  python scripts/grid_study_char_time.py report")


# ------------------------------------------------------------------- report
def _fetch_job_csv(job_id):
    """Return (v, t_star, m_star) from the job's characteristic-time CSV."""
    listing = _get(f"/jobs/{job_id}/outputs")
    csvs = [f["path"] for f in listing["files"] if f["path"].endswith(".csv")]
    if not csvs:
        return None
    url = (API.rstrip("/") + f"/jobs/{job_id}/outputs/"
           + urllib.parse.quote(csvs[0]))
    with urllib.request.urlopen(url, timeout=60) as r:
        text = r.read().decode()
    rows = [ln for ln in text.strip().splitlines() if ln.strip()]
    if len(rows) < 2:
        return None
    v, t_star, m_star = (rows[1].split(",") + ["", ""])[:3]
    return {
        "v": float(v),
        "t_star": float(t_star) if t_star.strip() else float("nan"),
        "m_star": float(m_star) if m_star.strip() else float("nan"),
        "source_csv": csvs[0],
    }


def _load_recovered():
    """Recovered rows keyed by grid, or {} if none have been produced."""
    if not os.path.exists(RECOVERED):
        return {}
    with open(RECOVERED) as f:
        return {int(k): v for k, v in json.load(f).items()}


def _load_prior_csv():
    """Rows from the committed results CSV, keyed by grid.

    Regenerating the report must never *lose* a result. Output files live on
    disk under each job's output root and are not archived anywhere, so a job
    can read `succeeded` while its CSV is long gone -- which is exactly what
    happened to every grid in this study. Without this fallback, a rerun would
    silently rewrite an eight-row table as a three-row one.

    Lowest priority of the three sources: a live CSV wins, then an explicitly
    recovered value, then whatever was published last.
    """
    import csv as _csv
    path = os.path.join(OUT_DIR, "grid_study_results.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for r in _csv.DictReader(f):
            try:
                g = int(r["grid"])
                wall = r.get("wall_hours") or ""
                out[g] = {
                    "grid": g,
                    "v": float(r["v"]),
                    "t_star": float(r["t_star"]),
                    "m_star": float(r["m_star"]),
                    "wall_hours": float(wall) if wall.strip() else None,
                    "job_id": r.get("job_id", ""),
                    "provenance": r.get("provenance") or "csv",
                    "source_csv": "",
                }
            except (ValueError, KeyError):
                continue
    return out


def cmd_report(a):
    man = _load_manifest()
    os.makedirs(OUT_DIR, exist_ok=True)
    recovered = _load_recovered()
    prior = _load_prior_csv()

    rows = []
    for e in man["jobs"]:
        g = e["grid"]
        j = _get(f"/jobs/{e['job_id']}")
        data = None
        if j["status"] == "succeeded":
            data = _fetch_job_csv(e["job_id"])
        if data is not None:
            data["provenance"] = "csv"
            rows.append({"grid": g, "job_id": e["job_id"],
                         "wall_hours": _elapsed_hours(j), **data})
            continue

        # Fall back to a recovered value. This covers two real cases: a job that
        # succeeded but whose output directory was later removed from disk, and
        # a job whose solve finished but which is recorded as failed because
        # only its result-file write crashed. In both the science exists; the
        # report says where each number came from rather than hiding the
        # difference.
        rec = recovered.get(g)
        if rec is not None:
            print(f"  {g}x{g}: {j['status']}, using recovered value "
                  f"({rec['provenance']})")
            rows.append({"grid": g, "job_id": rec["job_id"],
                         "wall_hours": rec.get("wall_hours"),
                         "v": rec["v"], "t_star": rec["t_star"],
                         "m_star": rec["m_star"],
                         "source_csv": rec.get("source", ""),
                         "provenance": rec["provenance"]})
            continue

        prev = prior.get(g)
        if prev is not None:
            print(f"  {g}x{g}: {j['status']}, no CSV on disk -- keeping the "
                  f"previously published value")
            rows.append(dict(prev))
            continue

        why = "no CSV found" if j["status"] == "succeeded" else j["status"]
        print(f"  {g}x{g}: {why}, nothing recovered or published -- skipped")

    if not rows:
        sys.exit("no successful jobs to report on")
    rows.sort(key=lambda r: r["grid"])

    # numerical results, so the report has co-located source data
    csv_path = os.path.join(OUT_DIR, "grid_study_results.csv")
    with open(csv_path, "w", newline="\n") as f:
        f.write("grid,v,t_star,m_star,wall_hours,job_id,provenance\n")
        for r in rows:
            wall = "" if r["wall_hours"] is None else f"{r['wall_hours']:.4f}"
            f.write(f"{r['grid']},{r['v']:.6g},{r['t_star']:.17g},"
                    f"{r['m_star']:.17g},{wall},{r['job_id']},"
                    f"{r.get('provenance','csv')}\n")
    print(f"wrote {csv_path}")

    _make_plot(rows)
    _make_tex(man, rows)
    _make_pdf(man, rows)
    print("\nreport complete:")
    for fn in sorted(os.listdir(OUT_DIR)):
        print(f"  {fn}")


def _make_plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SERIES = "#2a78d6"      # validated categorical slot 1
    SURFACE, INK, SECOND, MUTED, GRID_C, BASE = (
        "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")

    # A degenerate characteristic-time fit yields m_star = NaN (the solver
    # records it rather than failing). Plot only the real points, and say so
    # rather than silently dropping a grid from the figure.
    good = [r for r in rows if r["m_star"] == r["m_star"]]
    dropped = [r["grid"] for r in rows if r["m_star"] != r["m_star"]]
    if dropped:
        print(f"  note: m* is NaN for {dropped} (degenerate fit); "
              f"excluded from the plot")
    if not good:
        print("  no finite m* values -- skipping plot")
        return

    g = [r["grid"] for r in good]
    m = [r["m_star"] for r in good]

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Segoe UI", "DejaVu Sans"]})
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=170,
                           facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(g, m, "-o", color=SERIES, linewidth=2, markersize=9,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    for x, y in zip(g, m):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=8.5, color=SECOND)

    ax.grid(True, color=GRID_C, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_xticks(g)
    ax.set_xticklabels([f"{x}x{x}" for x in g])
    ax.set_xlabel("grid size", fontsize=10, color=SECOND)
    ax.set_ylabel("m* — mass at t*", fontsize=10, color=SECOND)
    title = (f"Mass at the characteristic time vs grid size  "
             f"(N={N_TUBES}, v=$10^4$, a=b={W}, T={T})")
    if dropped:
        title += f"\nomitted (degenerate fit): {', '.join(f'{d}x{d}' for d in dropped)}"
    ax.set_title(title, fontsize=11, color=INK, pad=12, loc="left")
    span = max(m) - min(m)
    ax.set_ylim(min(m) - 0.18 * span - 1e-9, max(m) + 0.22 * span + 1e-9)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "mass_vs_grid_size.png")
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _make_tex(man, rows):
    fx = man["fixed"]
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath}",
        r"\usepackage{graphicx}",
        r"",
        r"\title{Characteristic-Time Grid Study\\ Mass at $t^{*}$ versus Grid Size}",
        r"\date{18 August 2026, extended to $160 \times 160$ on 23 August 2026}",
        r"\author{}",
        r"",
        r"\begin{document}",
        r"\maketitle",
        r"",
        r"\section*{Parameters}",
        r"",
        r"\noindent",
        r"Held fixed across every grid:",
        r"",
        r"\medskip",
        r"\noindent",
        r"\begin{tabular}{ll}",
        r"\toprule",
        rf"Microtubules & $N = {fx['N_tubes']}$ \\",
        rf"Advective velocity & $v = 10^{{4}}$ \\",
        rf"Switch rates & $a = b = w = {fx['w']}$ \\",
        rf"Dimensionless solution time & $T = {fx['T']}$ \\",
        rf"Domain radius & $R = {fx['domain_radius']}$ \\",
        rf"Diffusion coefficient & $D = {fx['D']}$ \\",
        rf"Microtubule extraction width & $d_{{\mathrm{{tube}}}} = {fx['d_tube']}$ \\",
        r"Initial condition & centred \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\bigskip",
        r"\noindent",
        r"Microtubules are evenly spaced on each grid, so the ray indices are",
        r"regenerated per grid size while the count stays fixed.",
        r"",
        r"\section*{Results and wall time}",
        r"",
        r"\noindent",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Grid & {$t^{*}$} & {$m^{*}$} & {Wall time} \\",
        r"\midrule",
    ]
    for r in rows:
        wall = ("--" if r["wall_hours"] is None else
                (f"{r['wall_hours']*60:.1f} min" if r["wall_hours"] < 1
                 else f"{r['wall_hours']:.2f} hr"))
        lines.append(rf"${r['grid']} \times {r['grid']}$ "
                     rf"& {r['t_star']:.6f} "
                     rf"& {r['m_star']:.6f} & {wall} \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"",
    ]
    lines += [
        r"\section*{Mass at $t^{*}$ versus grid size}",
        r"",
        r"\begin{center}",
        r"\includegraphics[width=\textwidth]{mass_vs_grid_size.png}",
        r"\end{center}",
        r"",
        r"\end{document}",
        r"",
    ]
    out = os.path.join(OUT_DIR, "grid_study_report.tex")
    with open(out, "w", newline="\n") as f:
        f.write("\n".join(lines))
    print(f"wrote {out}")


def _make_pdf(man, rows, also_png=None):
    """Render the report as a PDF directly.

    No TeX engine is installed on the server, so the PDF is produced as vector
    output rather than compiled from the .tex. The two are kept in step by
    generating both from the same rows in one pass.

    ``also_png`` writes the same page as a raster image, which is only used to
    eyeball the layout (there is no poppler here to rasterise the PDF).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    fx = man["fixed"]
    INK, SECOND = "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["DejaVu Serif", "Times New Roman"]})

    fig = plt.figure(figsize=(8.27, 11.69), dpi=200)
    fig.patch.set_facecolor("white")

    def text(x, y, s, size=10.5, weight="normal", ha="left"):
        fig.text(x, y, s, fontsize=size, color=INK, ha=ha, va="top",
                 fontweight=weight)

    def rule(y, x0=0.11, x1=0.89, lw=1.1):
        fig.add_artist(plt.Line2D([x0, x1], [y, y], color=INK, linewidth=lw,
                                  transform=fig.transFigure))

    L, y = 0.11, 0.955
    text(0.5, y, "Characteristic-Time Grid Study", size=16, ha="center")
    y -= 0.026
    text(0.5, y, "Mass at $t^{*}$ versus Grid Size", size=16, ha="center")
    y -= 0.028
    text(0.5, y, "18 August 2026   ·   extended to 160 x 160, 23 August 2026",
         size=10, ha="center")
    y -= 0.048

    text(L, y, "Parameters", size=13, weight="bold")
    y -= 0.024
    text(L, y, "Held fixed across every grid:", size=10)
    y -= 0.024
    rule(y + 0.004)
    y -= 0.015
    params = [
        ("Microtubules", f"$N = {fx['N_tubes']}$"),
        ("Advective velocity", "$v = 10^{4}$"),
        ("Switch rates", f"$a = b = w = {fx['w']}$"),
        ("Dimensionless solution time", f"$T = {fx['T']}$"),
        ("Domain radius", f"$R = {fx['domain_radius']}$"),
        ("Diffusion coefficient", f"$D = {fx['D']}$"),
        ("Microtubule extraction width",
         f"$d_{{\\mathrm{{tube}}}} = {fx['d_tube']}$"),
        ("Initial condition", "centred"),
    ]
    for k, v in params:
        text(L + 0.01, y, k, size=10)
        text(L + 0.42, y, v, size=10)
        y -= 0.019
    rule(y + 0.006)
    y -= 0.040

    text(L, y, "Results and wall time", size=13, weight="bold")
    y -= 0.030
    cols = [L + 0.02, L + 0.22, L + 0.42, L + 0.62]
    rule(y + 0.006)
    y -= 0.016
    for x, h in zip(cols, ["Grid", "$t^{*}$", "$m^{*}$", "Wall time"]):
        text(x, y, h, size=10)
    y -= 0.017
    rule(y + 0.005, lw=0.7)
    y -= 0.014
    for r in rows:
        wall = ("--" if r["wall_hours"] is None else
                (f"{r['wall_hours']*60:.1f} min" if r["wall_hours"] < 1
                 else f"{r['wall_hours']:.2f} hr"))
        for x, v in zip(cols, [f"{r['grid']} x {r['grid']}",
                               f"{r['t_star']:.6f}", f"{r['m_star']:.6f}",
                               wall]):
            text(x, y, v, size=10)
        y -= 0.019
    rule(y + 0.006)
    y -= 0.045

    text(L, y, "Mass at $t^{*}$ versus grid size", size=13, weight="bold")
    # Clear of the heading: the PNG carries its own title inside the image, and
    # a smaller gap lets the two collide.
    y -= 0.026

    png = os.path.join(OUT_DIR, "mass_vs_grid_size.png")
    if os.path.exists(png):
        img = mpimg.imread(png)
        h, w = img.shape[0], img.shape[1]
        # figure is A4, so convert the image aspect into figure fractions
        fig_w_in, fig_h_in = fig.get_size_inches()
        aspect = (h / w) * (fig_w_in / fig_h_in)
        # Scale to whatever vertical space is actually left rather than assuming
        # a fixed width: the table grows a row per grid and the provenance note
        # adds several lines, either of which can push a fixed-size figure off
        # the bottom of the page.
        BOTTOM = 0.055
        avail_w = min(0.78, max(0.0, y - BOTTOM) / aspect)
        disp_h = avail_w * aspect
        x0 = L + (0.78 - avail_w) / 2      # keep it centred on the text block
        ax = fig.add_axes([x0, y - disp_h, avail_w, disp_h])
        ax.imshow(img)
        ax.axis("off")

    out = os.path.join(OUT_DIR, "grid_study_report.pdf")
    fig.savefig(out, format="pdf", facecolor="white")
    if also_png:
        fig.savefig(also_png, format="png", facecolor="white", dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="dispatch one job per grid")
    s.add_argument("--by", default="grid-study")
    s.add_argument("--dry-run", action="store_true",
                   help="show what would be submitted, send nothing")
    s.set_defaults(fn=cmd_submit)

    s = sub.add_parser("status", help="show progress")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("report", help="collect results, plot, .tex and .pdf")
    s.set_defaults(fn=cmd_report)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
