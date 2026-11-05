"""Time-stepper vs modal-expansion mass analysis over grid size and parameters.

For every run, both kernels fill the total-mass series TM (same sampling,
MA_collection_factor = 5) and the difference e_j = TM_step[j] - TM_modal[j] is
reduced to

    L_inf = max_j |e_j|                         (largest absolute difference)
    L_2   = sqrt(h sum_j e_j^2),  h = 5 dT      (~ the L2 norm over [0, T])

Runs (rg = ry = G, N_LIST = [0, G/4, G/2, 3G/4], centred seed, d_tube = 0):
  * G in 16, 24, 32, 48; (a, b) in {0.1, 1, 10, 100}^2; v in 0.1, 1, 10; T = 1
  * G = 64: the 16 (a, b) pairs at v = 1, T = 1
  * G in 16, 32: the 16 pairs at v = 1, T in 0.25, 4
  * G = 96: the 16 pairs at v = 1, T = 1, against the stored GPU series of job
    0b8d4022 (the CPU time-stepper costs ~5 h per pair there)
The time-stepper is the CPU reference (analysis_tools); everything else is
fresh. Rows are appended to runs.jsonl as they finish.

    python sweep.py [workers]
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
R = "N:/work/Intracellular-Transport-Computational-Modeling"
sys.path[:0] = [R + "/src/intracellular_transport", R + "/src"]
T1_96 = ("C:/Users/PopDaJuice/AppData/Local/Temp/claude/N--work-Intracellular-Transport-"
         "Computational-Modeling/859dba09-aa0f-491c-8ab4-3c47d703993a/scratchpad/run96")
MCF = 5
AB = [0.1, 1.0, 10.0, 100.0]
LOG = os.path.join(OUT, "runs.jsonl")


def tasks():
    t = []
    for G in (16, 24, 32, 48):
        t += [(G, a, b, v, 1.0, "cpu") for a in AB for b in AB for v in (0.1, 1.0, 10.0)]
    t += [(64, a, b, 1.0, 1.0, "cpu") for a in AB for b in AB]
    t += [(G, a, b, 1.0, T, "cpu") for G in (16, 32) for T in (0.25, 4.0) for a in AB for b in AB]
    t += [(96, a, b, 1.0, 1.0, "gpu-stored") for a in AB for b in AB]
    return t


def run(task):
    G, a, b, v, T, ref = task
    from computational_tools import analysis_tools as ant, numerical_tools as num
    from computational_tools import supplements as sup, mass_analysis_modal as mam
    import contextlib, io
    N_LIST = np.array([0, G // 4, G // 2, 3 * G // 4])
    K = num.compute_K(G, G, T)
    rk = K // MCF
    dT = num.compute_dT(G, G)
    t0 = time.perf_counter()
    if ref == "cpu":
        s = [np.zeros(rk) for _ in range(5)]
        D_LAYER, A_LAYER = sup.initialize_layers(G, G)
        with contextlib.redirect_stdout(io.StringIO()):
            ant.comp_mass_analysis_respect_to_time(G, G, a, b, v, T, N_LIST.copy(), D_LAYER, A_LAYER,
                                                   *s, MCF, rk)
        tm_step = s[4]
        wall_step = time.perf_counter() - t0
    else:
        tm_step = np.fromfile(f"{T1_96}/tm_a{a:g}_b{b:g}.dat", dtype=np.float64)
        assert tm_step.size == rk, (tm_step.size, rk)
        wall_step = float("nan")
    m = [np.zeros(rk) for _ in range(5)]
    info = mam.comp_mass_analysis_modal_expansion(G, G, a, b, v, T, N_LIST.copy(), *m, MCF, rk)
    e = tm_step - m[4]
    j = int(np.argmax(np.abs(e)))
    lam = info["decay_rate"].real
    row = dict(G=G, a=a, b=b, v=v, T=T, reference=ref, K=K, samples=rk, dT=dT,
               L_inf=float(np.abs(e).max()), t_at_L_inf=j * MCF * dT, M_at_L_inf=float(tm_step[j]),
               L_2=float(np.sqrt(MCF * dT * np.sum(e ** 2))),
               L_inf_rel=float(np.max(np.abs(e) / np.abs(tm_step))),
               M_T=float(tm_step[-1]),
               n_state=info["n_state"], n_reduced=info["n_reduced"],
               invariance_residual=info["invariance_residual"], cond_V=info["cond_V"],
               lambda_1=float(lam[0]), lambda_2=float(lam[1]),
               wall_step=wall_step, wall_modal=info["wall_total"], wall_eig=info["wall_eig"])
    with open(LOG, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    done = set()
    if os.path.exists(LOG):
        for line in open(LOG):
            r = json.loads(line)
            done.add((r["G"], r["a"], r["b"], r["v"], r["T"], r["reference"]))
    todo = [t for t in tasks() if t not in done]
    # most expensive first (cost ~ T G^6), so the long runs overlap the short ones
    todo.sort(key=lambda t: -t[4] * t[0] ** 6 * (t[5] == "cpu"))
    print(f"{len(todo)} runs ({len(done)} already done), {workers} workers", flush=True)
    t0 = time.time()
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run, todo), 1):
            print(f"[{i}/{len(todo)} {time.time() - t0:7.0f}s] G={r['G']} a={r['a']:g} b={r['b']:g} "
                  f"v={r['v']:g} T={r['T']:g}: L_inf={r['L_inf']:.2e} L_2={r['L_2']:.2e} "
                  f"cond={r['cond_V']:.1e} step {r['wall_step']:.0f}s modal {r['wall_modal']:.1f}s",
                  flush=True)
