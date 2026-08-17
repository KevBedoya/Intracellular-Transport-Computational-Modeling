"""Scale the N=8 and N=16 characteristic-time trajectories onto the N=4 one.

The scaling constant is the ratio of the masses at the largest velocity,

    c_N = m*_N(v_max) / m*_4(v_max),

so that m*_N / c_N lands on the N=4 curve. Writes the overlay figure.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent.parent / "intracellular_transport" / "data_output" / "char_time_analysis"
FIG_DIR = HERE.parent / "figures" / "char_time"

RUNS = {
    4: "2026-08-05_18-14-N=4",
    8: "2026-08-06_10-07_N=8",
    16: "2026-08-06_10-56_N=16",
}

STYLES = {
    4: ("#1f77b4", "o"),
    8: ("#d62728", "s"),
    16: ("#2ca02c", "^"),
}


def load(run_dir):
    arr = np.loadtxt(DATA_ROOT / run_dir / "char_t_analysis_data.csv", delimiter=",", skiprows=1)
    return arr[:, 0], arr[:, 2]  # v, m_star


data = {N: load(d) for N, d in RUNS.items()}
v4, m4 = data[4]
for N, (v, _) in data.items():
    assert np.allclose(v, v4), f"velocity sweep for N={N} differs from N=4"

# one constant per N: the ratio of the masses at the largest velocity
c = {N: data[N][1][-1] / m4[-1] for N in RUNS}
for N in (4, 8, 16):
    print(f"N={N:<3d} m*(1e4) = {data[N][1][-1]:.6f}   c_N = {c[N]:.4f}")

FIG_DIR.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(7.0, 5.0))

for N in (4, 8, 16):
    v, m = data[N]
    color, marker = STYLES[N]
    label = "N=4 (reference)" if N == 4 else rf"N={N}, $m^{{*}}/c_{{{N}}}$, $c_{{{N}}}={c[N]:.3f}$"
    ax.plot(v, m / c[N], marker=marker, color=color, ms=8, lw=1.5,
            mfc="none" if N != 4 else color, mew=1.6, label=label)

ax.set_xscale("log")
ax.set_xlabel(r"filament velocity $v$")
ax.set_ylabel(r"$m^{*}/c_{N}$")
ax.set_title(r"$m^{*}(v)$ for $N=4,8,16$ scaled onto a common curve"
             "\n" r"$a=b=100$, grid $48\times48$")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.grid(alpha=0.25)
fig.tight_layout()
out = FIG_DIR / "char_t_collapse_N4_N8_N16.png"
fig.savefig(out, dpi=200)
print(f"wrote {out}")
