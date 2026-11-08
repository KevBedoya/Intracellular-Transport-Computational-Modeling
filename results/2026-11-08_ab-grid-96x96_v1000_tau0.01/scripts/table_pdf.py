"""t* and M(10 t*) per (a, b) as a one-page PDF table, from summary_96x96_v1000.csv.

    June-2026/.venv/bin/python results/2026-11-08_ab-grid-96x96_v1000_tau0.01/scripts/table_pdf.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(os.path.join(OUT, "summary_96x96_v1000.csv"))
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
ax.set_title("96x96, v = 1000, N = 4 evenly spaced, tau = 0.01, power engine\n"
             "t* from the reference y''; M(10 t*) by exact powers of the one-step operator", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "t_star_m_10t_star_96x96_v1000.pdf"))
