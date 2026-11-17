"""Appendix A of eigen_t_star_96x96_tau0.01.pdf: what the semi-discrete operator L looks like.

    python eigen_appendix_L.py            # compute, write the appendix inputs, rebuild the PDF
    python eigen_appendix_L.py --no-pdf   # compute and write the inputs only

L = (P - I)/dT is built two ways and the two are compared entry by entry:

  extracted  P column by column from the solver's own step
             (mass_analysis_modal._Operator.step = comp_DL_AL_kp1_2step + u_center,
             the drivers' -v convention included), as in section 2.3 of the note;
  assembled  from the stencil written out in appendix A, with no solver code.

State order is mass_analysis_modal._Layout's: x = (c, phi ring by ring for rings
0 .. rg-2, rho tube by tube for rings 0 .. rg-1).

Writes to ../eigen_appendix/:
  L_example_matrix.tex   the 15 x 15 example (rg = 3, ry = 4, tubes on rays 0 and 2), symbolic
  L_example_values.tex   the symbols' values there at (a, b, v) = (10, 0.1, 1)
  L_sparsity_96x96.pdf   nonzero pattern of the 96 x 96 L (four tubes at rays 0, 24, 48, 72)
  L_check.txt            extracted vs assembled, and nonzero counts
and, unless --no-pdf, compiles appendix_L.tex with tectonic and rebuilds
../eigen_t_star_96x96_tau0.01.pdf as the note's first four pages followed by the
appendix pages. The first four pages are taken from that same file each time, so
rerunning replaces the appendix instead of appending a second copy.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
OUT = os.path.join(STUDY, "eigen_appendix")
NOTE = os.path.join(STUDY, "eigen_t_star_96x96_tau0.01.pdf")
MAIN_PAGES = 4
REPO = os.path.abspath(os.path.join(STUDY, "..", ".."))
for _p in (os.path.join(REPO, "src", "intracellular_transport"), os.path.join(REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from computational_tools import mass_analysis_modal as mam  # noqa: E402

EX = dict(rg=3, ry=4, N_LIST=[0, 2], a=10.0, b=0.1, v=1.0)
G96 = dict(rg=96, ry=96, N_LIST=[0, 24, 48, 72])
AB = [0.1, 1.0, 10.0, 100.0]


def extract_L(rg, ry, N_LIST, a, b, v):
    """(P - I)/dT from one solver step per unit vector."""
    lay = mam._Layout(rg, ry, N_LIST)
    op = mam._Operator(lay, a, b, v, 0, 1.0, 1.0)  # physical v; _Operator negates it, as the drivers do
    rows, cols, vals = [], [], []
    e = np.zeros(lay.n)
    for j in range(lay.n):
        e[j] = 1.0
        y = op.step(e)
        e[j] = 0.0
        nz = np.flatnonzero(y)
        rows.append(nz)
        cols.append(np.full(nz.size, j))
        vals.append(y[nz])
    P = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(lay.n, lay.n))
    return ((P - sp.identity(lay.n, format="csc")) / op.dT).tocsc(), op.dT


def assemble_L(rg, ry, N_LIST, a, b, v):
    """L from the appendix's stencil. Returns the matrix and a {(i, j): LaTeX label} map."""
    dr, dth = 1.0 / rg, 2 * np.pi / ry
    dT = 0.1 * min(dr ** 2, dth ** 2 * dr ** 2) / 2.0
    nD = (rg - 1) * ry
    n = 1 + nD + len(N_LIST) * rg
    C = 0
    PHI = lambda m, j: 1 + m * ry + (j % ry)
    RHO = lambda t, m: 1 + nD + t * rg + m
    kappa = 1 / dr ** 2
    gamma = dth / (np.pi * dr ** 2)
    u = v / (np.pi * dr ** 2)
    s = v / dr
    theta = lambda m: 1 / ((m + 1) ** 2 * dr ** 2 * dth ** 2)
    alpha = lambda m: a * (m + 1) * dr * dth
    beta = lambda m: b / ((m + 1) * dr * dth)
    d = lambda m: (2 * m + 3) / ((m + 1) * dr ** 2) + 2 * theta(m)
    tube_of = {r: t for t, r in enumerate(N_LIST)}

    ent, lab = {}, {}

    def put(i, j, val, tex):
        assert (i, j) not in ent, (i, j)
        ent[i, j], lab[i, j] = val, tex

    # centre patch
    put(C, C, -ry * gamma, rf"-{ry}\gamma")
    for j in range(ry):
        put(C, PHI(0, j), gamma, r"\gamma")
    for t in range(len(N_LIST)):
        put(C, RHO(t, 0), u, "u")
    # diffusive layer, rings 0 .. rg-2 (ring rg-1 is the rim, held at 0)
    for m in range(rg - 1):
        for j in range(ry):
            i = PHI(m, j)
            on_tube = j in tube_of
            put(i, i, -d(m) - (a if on_tube else 0.0), rf"-d_{m}" + ("-a" if on_tube else ""))
            put(i, C if m == 0 else PHI(m - 1, j), kappa, r"\kappa")
            if m + 1 <= rg - 2:
                k = (m + 2) / (m + 1)
                put(i, PHI(m + 1, j), k * kappa,
                    (rf"{k:g}\kappa" if float(k).is_integer() else rf"\tfrac{{{m + 2}}}{{{m + 1}}}\kappa"))
            put(i, PHI(m, j + 1), theta(m), rf"\theta_{m}")
            put(i, PHI(m, j - 1), theta(m), rf"\theta_{m}")
            if on_tube:
                put(i, RHO(tube_of[j], m), beta(m), rf"\beta_{m}")
    # advective layer, one block of rg rings per tube
    for t, jt in enumerate(N_LIST):
        for m in range(rg):
            i = RHO(t, m)
            if m == rg - 1:
                put(i, i, -1 / dT, r"-\frac{1}{\Delta t}")  # never written by the solver
                continue
            put(i, i, -s - b, "-s-b")
            put(i, RHO(t, m + 1), s, "s")
            put(i, PHI(m, jt), alpha(m), rf"\alpha_{m}")
    ij = np.array(list(ent.keys()))
    L = sp.csc_matrix((np.array(list(ent.values())), (ij[:, 0], ij[:, 1])), shape=(n, n))
    syms = {"const": {r"\kappa": kappa, r"\gamma": gamma, "u": u, "s": s, r"1/\Delta t": 1 / dT},
            "ring": [{r"\theta_m": theta(m), "d_m": d(m), r"\alpha_m": alpha(m), r"\beta_m": beta(m)}
                     for m in range(rg - 1)]}
    return L, lab, syms, dT


def rel_diff(A, B):
    """(max |A - B| / max |B|, max over B's nonzeros of |A - B| / |B|), same pattern assumed."""
    D = (A - B).tocsc()
    big = (np.max(np.abs(D.data)) if D.nnz else 0.0) / np.max(np.abs(B.data))
    Bc = B.tocoo()
    entry = np.max(np.abs(np.asarray(A.tocsc()[Bc.row, Bc.col]).ravel() - Bc.data) / np.abs(Bc.data))
    return big, entry


def state_labels(rg, ry, N_LIST):
    out = ["c"]
    out += [rf"\phi_{{{m},{j}}}" for m in range(rg - 1) for j in range(ry)]
    out += [rf"\rho_{{{m},{jt}}}" for jt in N_LIST for m in range(rg)]
    return out


def write_example_tex(lab, rg, ry, N_LIST):
    names = state_labels(rg, ry, N_LIST)
    n = len(names)
    nD = (rg - 1) * ry
    # block boundaries: after c, after each diffusive ring, after each tube
    cuts = {1} | {1 + (m + 1) * ry for m in range(rg - 1)} | {1 + nD + (t + 1) * rg for t in range(len(N_LIST))}
    cuts.discard(n)
    spec = "r|" + "".join("c" + ("|" if (j + 1) in cuts else "") for j in range(n))
    lines = [r"\begin{array}{" + spec + "}",
             " & " + " & ".join(rf"\scriptstyle {s}" for s in names) + r" \\ \hline"]
    for i in range(n):
        cells = [lab.get((i, j), r"\color{zero}\cdot") for j in range(n)]
        end = r" \\ \hline" if (i + 1) in cuts else r" \\"
        lines.append(rf"\scriptstyle {names[i]} & " + " & ".join(cells) + end)
    lines.append(r"\end{array}")
    with open(os.path.join(OUT, "L_example_matrix.tex"), "w") as f:
        f.write("% written by scripts/eigen_appendix_L.py\n" + "\n".join(lines) + "\n")


def write_values_tex(syms):
    ring, const = syms["ring"], syms["const"]
    rk = list(ring[0])
    t1 = ([r"\begin{tabular}{@{}l" + "r" * len(rk) + "@{}}", r"\toprule",
           " & " + " & ".join(f"${k}$" for k in rk) + r" \\ \midrule"]
          + [f"$m = {m}$ & " + " & ".join(f"{ring[m][k]:.6g}" for k in rk) + r" \\"
             for m in range(len(ring))]
          + [r"\bottomrule", r"\end{tabular}"])
    t2 = [r"\begin{tabular}{@{}" + "r" * len(const) + "@{}}", r"\toprule",
          " & ".join(f"${k}$" for k in const) + r" \\ \midrule",
          " & ".join(f"{v:.6g}" for v in const.values()) + r" \\",
          r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT, "L_example_values.tex"), "w") as f:
        f.write("% written by scripts/eigen_appendix_L.py\n" + "\n".join(t1) + "\n\\qquad\n"
                + "\n".join(t2) + "\n")


def sparsity_figure(L, rg, ry, N_LIST):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"],
                         "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
                         "font.size": 8, "axes.unicode_minus": False})
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    INK, MUTED, RULE, POS, NEG = "#0b0b0b", "#52514e", "#b8b7b2", "#2a78d6", "#e34948"
    n = L.shape[0]
    nD = (rg - 1) * ry
    Lr = L.tocsr()
    Lc = L.tocoo()
    r, c, val = Lc.row, Lc.col, Lc.data
    cmap = ListedColormap(["#ffffff", POS, NEG])

    def style(ax, title):
        ax.set_title(title, fontsize=8, color=INK)
        ax.tick_params(labelsize=6.5, length=2, colors=MUTED)
        for sp_ in ax.spines.values():
            sp_.set_color(MUTED)
            sp_.set_linewidth(0.5)

    def rules(ax, bounds, lo, hi, rows=True, cols=True):
        for bnd in bounds:
            if lo < bnd < hi:
                if rows:
                    ax.axhline(bnd - 0.5, color=RULE, lw=0.5, ls=(0, (3, 2)), zorder=0)
                if cols:
                    ax.axvline(bnd - 0.5, color=RULE, lw=0.5, ls=(0, (3, 2)), zorder=0)

    def cells(ax, rlo, rhi, clo, chi):
        """Every entry of the window as one cell: white zero, blue positive, red negative."""
        S = np.sign(Lr[rlo:rhi, clo:chi].toarray())
        img = np.where(S > 0, 1, np.where(S < 0, 2, 0))
        ax.imshow(img, cmap=cmap, vmin=0, vmax=2, interpolation="nearest", aspect="auto",
                  extent=(clo - 0.5, chi - 0.5, rhi - 0.5, rlo - 0.5))

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.2), constrained_layout=True)
    (a0, a1), (a2, a3) = axes

    # all of L: markers, since 9,505 x 9,505 cells cannot be drawn one by one
    for sel, col in ((val > 0, POS), (val < 0, NEG)):
        a0.plot(c[sel], r[sel], ls="none", marker="s", ms=0.6, mew=0, color=col)
    rules(a0, [1 + nD], 0, n)
    a0.set_xlim(-0.5, n - 0.5)
    a0.set_ylim(n - 0.5, -0.5)
    a0.set_aspect("equal")
    a0.text(0.30 * nD, 0.72 * nD, r"$\phi\phi$", ha="center", fontsize=9, color=INK)
    a0.annotate(r"$\rho\rho$", xy=(n - 200, n - 200), xytext=(0.62 * n, 0.93 * n),
                fontsize=9, color=INK, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    a0.annotate(r"$\phi\rho$", xy=(1 + nD + 40, 0.40 * nD), xytext=(0.78 * n, 0.30 * nD),
                fontsize=9, color=INK, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    a0.annotate(r"$\rho\phi$", xy=(0.40 * nD, 1 + nD + 40), xytext=(0.22 * nD, 0.88 * n),
                fontsize=9, color=INK, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    style(a0, "all of $L$")

    # top left: c, rings 0 and 1
    hi = 1 + 2 * ry
    cells(a1, 0, hi, 0, hi)
    rules(a1, [1, 1 + ry], 0, hi)
    a1.set_aspect("equal")
    style(a1, r"$c$ and rings 0, 1 of $\phi$")

    # ring 94 of phi and tube 0
    lo, hi = 1 + nD - ry, 1 + nD + rg
    cells(a2, lo, hi, lo, hi)
    rules(a2, [1 + nD], lo, hi)
    a2.set_aspect("equal")
    j0 = N_LIST[0]
    for (yy, xx) in ((lo + j0, 1 + nD + rg - 2), (1 + nD + rg - 2, lo + j0), (hi - 1, hi - 1)):
        a2.plot(xx, yy, ls="none", marker="o", ms=7, mfc="none", mec=INK, mew=0.6, clip_on=False)
    a2.annotate(r"$\beta_{94}$", xy=(1 + nD + rg - 2, lo + j0), xytext=(1 + nD + rg - 40, lo + 30),
                fontsize=7.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    a2.annotate(r"$\alpha_{94}$", xy=(lo + j0, 1 + nD + rg - 2), xytext=(lo + 22, 1 + nD + rg - 40),
                fontsize=7.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    a2.annotate(r"$-1/\Delta t$", xy=(hi - 1, hi - 1), xytext=(hi - 70, hi - 12),
                fontsize=7.5, color=INK, ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    style(a2, r"ring 94 of $\phi$ and tube 0")

    # phi rows x rho columns: the exchange beta_m, one cell per ring on each tube's ray
    # (markers: 9,120 rows are too many to draw as cells without dropping single entries)
    k = (r >= 1) & (r < 1 + nD) & (c >= 1 + nD)
    a3.plot(c[k], r[k], ls="none", marker="s", ms=1.2, mew=0, color=POS)
    assert np.all(val[k] > 0)
    rules(a3, [1 + nD + t * rg for t in range(1, len(N_LIST))], 1 + nD, n, rows=False)
    a3.set_xlim(1 + nD - 0.5, n - 0.5)
    a3.set_ylim(1 + nD - 0.5, 0.5)
    for t, jt in enumerate(N_LIST):
        a3.text(1 + nD + t * rg + rg / 2, 0.985 * nD, f"tube {t}\n(ray {jt})", ha="center",
                va="bottom", fontsize=6.5, color=MUTED)
    style(a3, r"$\phi$ rows $\times$ $\rho$ columns")

    for ax in (a0, a1, a2, a3):
        ax.set_xlabel("column", color=MUTED, fontsize=7.5)
        ax.set_ylabel("row", color=MUTED, fontsize=7.5)
    fig.legend(handles=[Patch(color=POS, label="positive"), Patch(color=NEG, label="negative")],
               loc="outside lower center", ncol=2, fontsize=7.5, frameon=False)
    fig.savefig(os.path.join(OUT, "L_sparsity_96x96.pdf"), dpi=600)
    plt.close(fig)


def nnz_report(L):
    col = np.diff(L.tocsc().indptr)
    row = np.diff(L.tocsr().indptr)
    vals, cnt = np.unique(col, return_counts=True)
    return (f"nnz = {L.nnz}, n = {L.shape[0]}, mean per column = {L.nnz / L.shape[0]:.3f}\n"
            f"  per-column counts: " + ", ".join(f"{v}: {c}" for v, c in zip(vals, cnt)) + "\n"
            f"  centre column {col[0]}, centre row {row[0]}\n")


def build_pdf():
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["tectonic", "--keep-logs", "-o", tmp, os.path.join(OUT, "appendix_L.tex")],
                       check=True, cwd=OUT)
        appendix = os.path.join(tmp, "appendix_L.pdf")
        subprocess.run(["pdfseparate", "-f", "1", "-l", str(MAIN_PAGES), NOTE,
                        os.path.join(tmp, "main_%d.pdf")], check=True)
        pages = [os.path.join(tmp, f"main_{k}.pdf") for k in range(1, MAIN_PAGES + 1)]
        merged = os.path.join(tmp, "merged.pdf")
        subprocess.run(["pdfunite", *pages, appendix, merged], check=True)
        os.replace(merged, NOTE)


def main():
    os.makedirs(OUT, exist_ok=True)
    log = []

    # the 15 x 15 example
    Le, dTe = extract_L(**EX)
    La, lab, syms, dTa = assemble_L(**EX)
    assert np.isclose(dTe, dTa, rtol=1e-15, atol=0)
    log.append(f"example rg = {EX['rg']}, ry = {EX['ry']}, tubes {EX['N_LIST']}, "
               f"(a, b, v) = ({EX['a']}, {EX['b']}, {EX['v']}): n = {La.shape[0]}, dT = {dTe:.6e}\n"
               f"  pattern equal: {np.array_equal((Le != 0).toarray(), (La != 0).toarray())}, "
               "max |dL| / max |L| = %.2e, max entrywise |dL| / |L| = %.2e\n" % rel_diff(Le, La))
    write_example_tex(lab, EX["rg"], EX["ry"], EX["N_LIST"])
    write_values_tex(syms)

    # 96 x 96, every (a, b) of the study
    worst, worst_e, L11 = 0.0, 0.0, None
    for a in AB:
        for b in AB:
            Lx, _ = extract_L(**G96, a=a, b=b, v=1.0)
            Ls, *_ = assemble_L(**G96, a=a, b=b, v=1.0)
            same = np.array_equal(Lx.indices, Ls.indices) and np.array_equal(Lx.indptr, Ls.indptr)
            rd, re_ = rel_diff(Lx, Ls)
            worst, worst_e = max(worst, rd), max(worst_e, re_)
            log.append(f"96x96 (a, b) = ({a:g}, {b:g}): pattern equal {same}, "
                       f"max |dL| / max |L| = {rd:.2e}, max entrywise |dL| / |L| = {re_:.2e}\n")
            if (a, b) == (1.0, 1.0):
                L11 = Lx
    log.append(f"96x96 worst over the 16 pairs: max |dL| / max |L| = {worst:.2e}, "
               f"entrywise {worst_e:.2e}\n")
    log.append("96x96 L at (a, b) = (1, 1): " + nnz_report(L11))
    sparsity_figure(L11, **G96)

    with open(os.path.join(OUT, "L_check.txt"), "w") as f:
        f.write("".join(log))
    print("".join(log))
    if "--no-pdf" not in sys.argv:
        build_pdf()
        print("rebuilt", NOTE)


if __name__ == "__main__":
    main()
