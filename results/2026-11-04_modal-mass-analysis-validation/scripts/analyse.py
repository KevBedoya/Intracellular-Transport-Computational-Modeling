"""Worst cases per grid size and a fitted error model, from runs.jsonl.

Writes
  runs.csv               every run
  worst_cases.csv        per (G, reference, T): the run with the largest L_inf
                         and the run with the largest L_2
  error_model.json       least-squares fits of log10 L_inf and log10 L_2 on
                         log10 of G, T, a, b, v (all CPU-reference runs), the
                         hold-out check (fit on G <= 48, predict G = 64 and the
                         G = 96 GPU-reference runs) and the offset that turns
                         each fit into an upper bound on every run
  error_model.png        predicted vs measured
"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.DataFrame([json.loads(l) for l in open(os.path.join(OUT, "runs.jsonl"))])
df = df.sort_values(["reference", "G", "T", "v", "a", "b"]).reset_index(drop=True)
df.to_csv(os.path.join(OUT, "runs.csv"), index=False)

worst = []
for (G, ref, T), g in df.groupby(["G", "reference", "T"]):
    for metric in ("L_inf", "L_2"):
        r = g.loc[g[metric].idxmax()]
        worst.append(dict(G=G, reference=ref, T=T, metric=metric, value=r[metric],
                          a=r.a, b=r.b, v=r.v, runs=len(g),
                          median_over_runs=g[metric].median(), cond_V=r.cond_V,
                          t_at_L_inf=r.t_at_L_inf))
worst = pd.DataFrame(worst)
worst.to_csv(os.path.join(OUT, "worst_cases.csv"), index=False)
print(worst.to_string(index=False))

FEATURES = ["G", "T", "a", "b", "v"]


def design(d):
    return np.column_stack([np.ones(len(d))] + [np.log10(d[f].to_numpy()) for f in FEATURES])


model = {}
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
cpu = df[df.reference == "cpu"]
for ax, metric in zip(axes, ("L_inf", "L_2")):
    y = np.log10(cpu[metric].to_numpy())
    X = design(cpu)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    r2 = 1 - res.var() / y.var()
    # hold-out: fit on G <= 48, predict the larger grids
    tr = cpu.G <= 48
    c_tr, *_ = np.linalg.lstsq(design(cpu[tr]), np.log10(cpu[tr][metric]), rcond=None)
    held = df[(df.G > 48)]
    pred_held = design(held) @ c_tr
    err_held = np.log10(held[metric].to_numpy()) - pred_held
    model[metric] = dict(
        form=f"log10 {metric} = c0 + cG log10 G + cT log10 T + ca log10 a + cb log10 b + cv log10 v",
        coefficients=dict(zip(["c0"] + ["c" + f for f in FEATURES], map(float, coef))),
        r2=float(r2), rms_residual_decades=float(np.sqrt(np.mean(res ** 2))),
        max_residual_decades=float(res.max()),
        upper_bound_offset_decades=float(res.max()),
        holdout=dict(trained_on="G <= 48 (CPU reference)",
                     predicted=[dict(G=int(G), reference=ref,
                                     rms_decades=float(np.sqrt(np.mean(err_held[(held.G == G).to_numpy()] ** 2))),
                                     max_under_prediction_decades=float(err_held[(held.G == G).to_numpy()].max()))
                                for G, ref in held[["G", "reference"]].drop_duplicates().itertuples(index=False)]),
        n_runs=int(len(cpu)))
    for (G, ref), g in df.groupby(["G", "reference"]):
        ax.scatter(design(g) @ coef, np.log10(g[metric]), s=12,
                   marker="x" if ref != "cpu" else "o", label=f"G={G}" + (" (GPU ref)" if ref != "cpu" else ""))
    lo, hi = ax.get_xlim()
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
    ax.plot([lo, hi], [lo + res.max(), hi + res.max()], "k--", lw=0.8, label="fit + max residual")
    ax.set_xlabel(f"predicted log10 {metric}")
    ax.set_ylabel(f"measured log10 {metric}")
    ax.set_title(f"{metric}: R$^2$ = {r2:.3f}, rms residual {model[metric]['rms_residual_decades']:.2f} dec")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=7)
fig.suptitle("time-stepper vs modal expansion, total mass M(t): fitted error model")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "error_model.png"), dpi=120)
json.dump(model, open(os.path.join(OUT, "error_model.json"), "w"), indent=2)
print(json.dumps(model, indent=2))
