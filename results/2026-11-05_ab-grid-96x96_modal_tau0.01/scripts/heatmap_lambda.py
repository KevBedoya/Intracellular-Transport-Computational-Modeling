"""lambda_1 and lambda_2 over the (a, b) grid, one heatmap each.

    June-2026/.venv/bin/python results/2026-11-05_ab-grid-96x96_modal_tau0.01/scripts/heatmap_lambda.py

Both are the char-time job's (summary_96x96_modal.csv), as launch._slowest_rates
returns them: lambda_1 the slowest decay rate of M (M ~ e^{-lambda_1 t} late),
lambda_2 the slowest rate after it among the modes the seed excites. lambda_2 can
be complex (with its conjugate): the heatmap colours Re(lambda_2), and each cell
shows Re +- i|Im| and |lambda_2|. heatmap_m_t_tilde.py reproduces lambda_2 from
a local solve to 6e-9.
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heatmap_m_5t_star as h

import numpy as np
import pandas as pd

d = pd.read_csv(os.path.join(h.OUT, "summary_96x96_modal.csv")).set_index(["a", "b"])


def arr(col):
    return np.array([[d.loc[(a, b), col] for b in h.AB] for a in h.AB])


l1, re2, im2 = arr("lambda_1"), arr("lambda_2"), arr("lambda_2_imag")
src = r"from the char-time job's modal solve"

h.grid(l1, [[(f"{l1[i, j]:.4f}", "") for j in range(4)] for i in range(4)],
       np.floor(l1.min()), np.ceil(l1.max()), r"$\lambda_1$",
       r"$\lambda_1$ per $(a, b)$: the slowest decay rate of $M$" + "\n" + h.HEAD + src,
       "heatmap_lambda_1_96x96_modal")

cells = [[(rf"{re2[i, j]:.3f} $\pm$ {im2[i, j]:.3f}$i$", rf"$|\lambda_2|$ = {np.hypot(re2[i, j], im2[i, j]):.4f}")
          if im2[i, j] > 0 else (f"{re2[i, j]:.4f}", "real")
          for j in range(4)] for i in range(4)]
h.grid(re2, cells, np.floor(re2.min()), np.ceil(re2.max()), r"$\mathrm{Re}(\lambda_2)$",
       r"$\lambda_2$ per $(a, b)$: the next excited decay rate; colour = $\mathrm{Re}(\lambda_2)$"
       + "\n" + h.HEAD + src, "heatmap_lambda_2_96x96_modal")

print(pd.DataFrame({"lambda_1": l1.ravel(), "Re lambda_2": re2.ravel(), "|Im lambda_2|": im2.ravel()},
                   index=pd.MultiIndex.from_product([h.AB, h.AB], names=["a", "b"])).to_string())
