"""t* and M(10 t*) per (a, b) as a one-page PDF table, from summary_96x96_modal.csv.

    python scripts/table_pdf.py        (run from the results directory)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

d = pd.read_csv("summary_96x96_modal.csv")
rows = [[f"{r.a:g}", f"{r.b:g}", f"{r.t_star:.6f}", f"{r.ten_t_star:.5f}", f"{r.m_at_10_t_star:.6e}"]
        for r in d.itertuples()]
fig, ax = plt.subplots(figsize=(8.5, 7))
ax.axis("off")
tab = ax.table(cellText=rows, colLabels=["a", "b", "t*", "10 t*", "M(10 t*)"],
               loc="center", cellLoc="center")
tab.auto_set_font_size(False)
tab.set_fontsize(10)
tab.scale(1, 1.5)
for c in range(5):
    tab[0, c].set_text_props(weight="bold")
ax.set_title("96x96, v = 1, N = 4 evenly spaced, tau = 0.01, modal engine\n"
             "t* from the exact modal y''; M(10 t*) evaluated from the modal series", fontsize=11)
fig.tight_layout()
fig.savefig("t_star_m_10t_star_96x96_modal.pdf")
