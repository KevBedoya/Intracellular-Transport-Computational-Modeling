"""Build results/2026-08-15_char-time-sweep/char_time_sweep.{tex,pdf}.

Page 1 is the original study: retained mass against microtubule count, at two
fixed grids. Page 2 adds the grid-size dependence at fixed N=16, which is the
same quantity swept along the other axis -- its 48x48 and 96x96 rows reproduce
the N=16 rows of page 1 exactly, so the two pages join up there.

No TeX engine is installed on this machine, so the PDF is rendered directly with
matplotlib rather than compiled from the .tex. Both come out of this one script
so they cannot drift apart.

    python scripts/build_char_time_sweep_report.py
"""

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "2026-08-15_char-time-sweep")
STUDY_DIR = os.path.join(ROOT, "results",
                         "2026-08-18_char-time-grid-study_N16")

# Page 1: the original N-sweep, transcribed from the committed .tex.
PARAMS = [
    ("Advective velocity", "$v = 10^{4}$"),
    ("Switch rates", "$a = b = w = 100$"),
    ("Dimensionless solution time", "$T = 1$"),
    ("Domain radius", "$R = 1$"),
    ("Diffusion coefficient", "$D = 1$"),
    ("Microtubule extraction width", "$d_{\\mathrm{tube}} = 0$"),
    ("Initial condition", "centred"),
]

SWEEP = {
    "96 \\times 96": ("Wall time (hr)", [
        (4, 0.073186, 0.057734, "10.61"),
        (8, 0.067073, 0.147390, "10.72"),
        (16, 0.059530, 0.377297, "10.88"),
        (24, 0.054723, 0.578565, "11.09"),
    ]),
    "48 \\times 48": ("Wall time (min)", [
        (4, 0.070148, 0.096793, "10.5"),
        (8, 0.061825, 0.295470, "10.7"),
        (16, 0.052357, 0.670035, "11.1"),
        (24, 0.046611, 0.858021, "11.6"),
    ]),
}


def load_grid_study():
    """Rows of the N=16 grid study, with the fit slope where it is known."""
    results = os.path.join(STUDY_DIR, "grid_study_results.csv")
    if not os.path.exists(results):
        sys.exit(f"missing {results}; run grid_study_char_time.py report first")

    slopes = {}
    slope_csv = os.path.join(STUDY_DIR, "fit_slopes.csv")
    if os.path.exists(slope_csv):
        for r in csv.DictReader(open(slope_csv)):
            try:
                slopes[int(r["grid"])] = float(r["fit_slope"])
            except (ValueError, KeyError):
                continue

    rows = []
    for r in csv.DictReader(open(results)):
        g = int(r["grid"])
        wall = r.get("wall_hours") or ""
        rows.append({
            "grid": g,
            "t_star": float(r["t_star"]),
            "m_star": float(r["m_star"]),
            "slope": slopes.get(g),
            "device": (r.get("device") or "cpu").upper(),
            "wall_hours": float(wall) if wall.strip() else None,
        })
    rows.sort(key=lambda x: x["grid"])
    return rows


def fmt_wall(h):
    if h is None:
        return "--"
    return f"{h * 60:.1f} min" if h < 1 else f"{h:.2f} hr"


# --------------------------------------------------------------------- tex
def build_tex(rows):
    L = [
        r"\documentclass[11pt]{article}",
        r"",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath}",
        r"\usepackage{siunitx}",
        r"\sisetup{round-mode=places, round-precision=6, table-format=1.6}",
        r"",
        r"\title{Characteristic-Time Analysis:\\ Retained Mass versus "
        r"Microtubule Count}",
        r"\date{15 August 2026, grid dependence added 27 August 2026}",
        r"\author{}",
        r"",
        r"\begin{document}",
        r"\maketitle",
        r"",
        r"\section*{Parameters}",
        r"",
        r"\noindent",
        r"Common to every run in both tables:",
        r"",
        r"\medskip",
        r"\noindent",
        r"\begin{tabular}{ll}",
        r"\toprule",
    ]
    for k, v in PARAMS:
        L.append(rf"{k} & {v} \\")
    L += [
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\bigskip",
        r"\noindent",
        r"$N$ denotes the number of evenly spaced microtubules, $t^{*}$ the",
        r"characteristic time, and $m^{*}$ the retained mass evaluated at "
        r"$t^{*}$.",
        r"",
    ]
    for grid, (wall_head, table) in SWEEP.items():
        L += [
            rf"\section*{{Domain grid: ${grid}$}}",
            r"",
            r"\noindent",
            r"\begin{tabular}{",
            r"    S[table-format=2.0]",
            r"    S[table-format=1.6]",
            r"    S[table-format=1.6]",
            r"    S[table-format=2.2]",
            r"  }",
            r"\toprule",
            rf"{{$N$}} & {{$t^{{*}}$}} & {{$m^{{*}}$}} & {{{wall_head}}} \\",
            r"\midrule",
        ]
        for n, t, m, w in table:
            L.append(rf"{n:2d} & {t:.6f} & {m:.6f} & {w} \\")
        L += [r"\bottomrule", r"\end{tabular}", r""]

    # Page 2.
    L += [
        r"\clearpage",
        r"",
        r"\section*{Grid dependence at $N = 16$}",
        r"",
        r"\noindent",
        r"The same parameters, with the microtubule count held at $N = 16$ and",
        r"the grid varied. The $48 \times 48$ and $96 \times 96$ rows reproduce",
        r"the $N = 16$ rows of the tables above. $m$ is the slope of",
        r"$\log_{10}$ of total mass across the fit window $t \in [0.4, 0.5]$,",
        r"from which $t^{*}$ is derived.",
        r"",
        r"\medskip",
        r"\noindent",
        r"\begin{tabular}{lrrrlr}",
        r"\toprule",
        r"Grid & {$t^{*}$} & {$m^{*}$} & {$m$} & {Device} & {Wall time} \\",
        r"\midrule",
    ]
    for r in rows:
        slope = "--" if r["slope"] is None else f"{r['slope']:.6f}"
        L.append(rf"${r['grid']} \times {r['grid']}$ & {r['t_star']:.6f} "
                 rf"& {r['m_star']:.6f} & {slope} & {r['device']} "
                 rf"& {fmt_wall(r['wall_hours'])} \\")
    L += [
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\end{document}",
        r"",
    ]
    path = os.path.join(OUT_DIR, "char_time_sweep.tex")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(L))
    print(f"wrote {os.path.relpath(path, ROOT)}")


# --------------------------------------------------------------------- pdf
def build_pdf(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    INK = "#0b0b0b"
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["DejaVu Serif", "Times New Roman"]})
    L = 0.11

    def page(pdf, draw):
        fig = plt.figure(figsize=(8.27, 11.69), dpi=200)
        fig.patch.set_facecolor("white")

        def text(x, y, s, size=10.5, weight="normal", ha="left"):
            fig.text(x, y, s, fontsize=size, color=INK, ha=ha, va="top",
                     fontweight=weight)

        def rule(y, x0=L, x1=0.89, lw=1.1):
            fig.add_artist(plt.Line2D([x0, x1], [y, y], color=INK, linewidth=lw,
                                      transform=fig.transFigure))
        draw(text, rule)
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

    def page1(text, rule):
        y = 0.955
        text(0.5, y, "Characteristic-Time Analysis:", size=16, ha="center")
        y -= 0.026
        text(0.5, y, "Retained Mass versus Microtubule Count", size=16,
             ha="center")
        y -= 0.028
        text(0.5, y, "15 August 2026   ·   grid dependence added "
                     "27 August 2026", size=10, ha="center")
        y -= 0.048

        text(L, y, "Parameters", size=13, weight="bold")
        y -= 0.024
        text(L, y, "Common to every run in both tables:", size=10)
        y -= 0.024
        rule(y + 0.004)
        y -= 0.015
        for k, v in PARAMS:
            text(L + 0.01, y, k, size=10)
            text(L + 0.42, y, _tex_to_mpl(v), size=10)
            y -= 0.019
        rule(y + 0.006)
        y -= 0.030
        text(L, y, "N is the number of evenly spaced microtubules, t* the "
                   "characteristic time,", size=10)
        y -= 0.019
        text(L, y, "and m* the retained mass evaluated at t*.", size=10)
        y -= 0.042

        for grid, (wall_head, table) in SWEEP.items():
            text(L, y, f"Domain grid: {grid.replace(chr(92)+'times', 'x')}",
                 size=13, weight="bold")
            y -= 0.030
            cols = [L + 0.02, L + 0.20, L + 0.40, L + 0.62]
            rule(y + 0.006)
            y -= 0.016
            for x, h in zip(cols, ["$N$", "$t^{*}$", "$m^{*}$", wall_head]):
                text(x, y, h, size=10)
            y -= 0.017
            rule(y + 0.005, lw=0.7)
            y -= 0.014
            for n, t, m, w in table:
                for x, v in zip(cols, [str(n), f"{t:.6f}", f"{m:.6f}", w]):
                    text(x, y, v, size=10)
                y -= 0.019
            rule(y + 0.006)
            y -= 0.045

    def page2(text, rule):
        y = 0.955
        text(L, y, "Grid dependence at N = 16", size=14, weight="bold")
        y -= 0.034
        for ln in ("The same parameters, with the microtubule count held at "
                   "N = 16 and the grid varied.",
                   "The 48 x 48 and 96 x 96 rows reproduce the N = 16 rows of "
                   "the tables on the previous",
                   "page. m is the slope of log10 of total mass across the fit "
                   "window t in [0.4, 0.5],",
                   "from which t* is derived."):
            text(L, y, ln, size=10)
            y -= 0.019
        y -= 0.024

        cols = [L + 0.02, L + 0.17, L + 0.32, L + 0.47, L + 0.615, L + 0.70]
        rule(y + 0.006)
        y -= 0.016
        for x, h in zip(cols, ["Grid", "$t^{*}$", "$m^{*}$", "$m$", "Device",
                               "Wall time"]):
            text(x, y, h, size=10)
        y -= 0.017
        rule(y + 0.005, lw=0.7)
        y -= 0.014
        for r in rows:
            slope = "--" if r["slope"] is None else f"{r['slope']:.6f}"
            for x, v in zip(cols, [f"{r['grid']} x {r['grid']}",
                                   f"{r['t_star']:.6f}", f"{r['m_star']:.6f}",
                                   slope, r["device"],
                                   fmt_wall(r["wall_hours"])]):
                text(x, y, v, size=10)
            y -= 0.019
        rule(y + 0.006)

    path = os.path.join(OUT_DIR, "char_time_sweep.pdf")
    with PdfPages(path) as pdf:
        page(pdf, page1)
        page(pdf, page2)
    print(f"wrote {os.path.relpath(path, ROOT)}")

    # There is no poppler here to rasterise the PDF, so page previews are
    # rendered from the same draw functions when asked for.
    if PNG_PREFIX:
        for name, draw in (("p1", page1), ("p2", page2)):
            fig = plt.figure(figsize=(8.27, 11.69), dpi=110)
            fig.patch.set_facecolor("white")

            def text(x, y, s, size=10.5, weight="normal", ha="left", _f=fig):
                _f.text(x, y, s, fontsize=size, color=INK, ha=ha, va="top",
                        fontweight=weight)

            def rule(y, x0=L, x1=0.89, lw=1.1, _f=fig):
                _f.add_artist(plt.Line2D([x0, x1], [y, y], color=INK,
                                         linewidth=lw,
                                         transform=_f.transFigure))
            draw(text, rule)
            out = f"{PNG_PREFIX}_{name}.png"
            fig.savefig(out, facecolor="white")
            plt.close(fig)
            print(f"preview: {out}")


def _tex_to_mpl(s):
    """The parameter values are already mathtext-compatible."""
    return s


PNG_PREFIX = None


def main():
    global PNG_PREFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=None,
                    help="also write <prefix>_p1.png / _p2.png for inspection")
    PNG_PREFIX = ap.parse_args().png

    rows = load_grid_study()
    missing = [r["grid"] for r in rows if r["slope"] is None]
    if missing:
        print(f"note: no fit slope yet for {missing} "
              f"(shown as -- ; re-run recover_char_time_from_log.py once those "
              f"jobs finish)")
    build_tex(rows)
    build_pdf(rows)


if __name__ == "__main__":
    main()
