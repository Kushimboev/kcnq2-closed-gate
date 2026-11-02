#!/usr/bin/env python3
"""Are the trajectories statistically independent? (reviewer comment 1, 2026-11-02) -> out2/replica_memory.json

For each system: (1) integrated autocorrelation time of R_min, Ser314 diagonal, N_g and N_c, and the number of statistically independent samples;
(2) block-averaged standard error of R_min per trajectory; (3) variance between trajectories against the error expected from within-trajectory sampling;
(4) divergence from the seed: pore-domain and S6 C-alpha RMSD from the frame of trajectory a that seeded each replica, as a function of time, and the pairwise
RMSD between replicas of one system at equal time.
Usage: replica_memory.py
"""
import os, sys, json
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
from trajectories import FILES, NAME, ORDER, SEED, LENGTH
import MDAnalysis as mda
from MDAnalysis.analysis import align, rms

R = json.load(open(H + "/out2/summary2.json"))


def tau(x, dt):
    x = np.asarray(x, float); x = x - x.mean(); n = len(x)
    if x.std() == 0: return 0.0
    f = np.fft.rfft(x, 2 * n); c = np.fft.irfft(f * np.conj(f))[:n]; c /= c[0]
    s = 0.5; M = 1
    for k in range(1, n):
        s += c[k]; M = k
        if k >= 5 * s: break      # Sokal window
    return float(s * dt)


def block_sem(x, nb=5):
    b = np.array_split(np.asarray(x, float), nb); m = np.array([q.mean() for q in b]); return float(m.std(ddof=1) / np.sqrt(nb))


out = {"tau": {}}
for l in ORDER:
    r = R[l]; t = np.array(r["t_series"]); dt = float(t[1] - t[0]); T = t[-1] - t[0]; q = {}
    for k, nm in (("rmin_series", "Rmin"), ("diag314_series", "D314"), ("nw_series", "Ng"), ("w_cav_series", "Nc")):
        ta = tau(r[k], dt); q[nm] = dict(tau_ns=ta, n_eff=float(T / max(2 * ta, dt)))
    q["Rmin_block_sem"] = block_sem(r["rmin_series"]); q["Rmin_mean"] = float(np.mean(r["rmin_series"]))
    out["tau"][NAME[l]] = q
    print(NAME[l], {k: (round(v["tau_ns"], 2), round(v["n_eff"], 1)) for k, v in q.items() if isinstance(v, dict)}, "SEM", round(q["Rmin_block_sem"], 4), flush=True)
m = np.array([out["tau"][NAME[l]]["Rmin_mean"] for l in ORDER]); e = np.array([out["tau"][NAME[l]]["Rmin_block_sem"] for l in ORDER])
out["Rmin_between_sd"] = float(m.std(ddof=1)); out["Rmin_rms_block_sem"] = float(np.sqrt((e ** 2).mean()))
print("between SD", out["Rmin_between_sd"], "rms SEM", out["Rmin_rms_block_sem"])

# divergence from the seed and between replicas (frames every 2 ns)
SEL = "name CA and resid 233-325"; STEP = 20
out["divergence"] = {}
sets = {}
for sysn in ("S0", "S3"):
    par = [l for l in ORDER if NAME[l] == sysn + "-a"][0]; up = mda.Universe(*FILES[par]); cap = up.select_atoms(SEL)
    tp = np.array([ts.time / 1000.0 for ts in up.trajectory])
    reps = [l for l in ORDER if NAME[l].startswith(sysn) and SEED[l] is not None]; X = {}
    for l in reps:
        up.trajectory[int(np.argmin(abs(tp - SEED[l])))]; ref = cap.positions.copy()
        u = mda.Universe(*FILES[l]); ca = u.select_atoms(SEL); assert len(ca) == len(cap)
        C = np.array([ca.positions.copy() for ts in u.trajectory[::STEP]]); tt = np.array([ts.time / 1000.0 for ts in u.trajectory[::STEP]])
        d = np.array([rms.rmsd(c, ref, center=True, superposition=True) for c in C]) / 10
        X[l] = (tt, C); out["divergence"][NAME[l]] = dict(t=tt.tolist(), rmsd_from_seed=d.tolist())
        print(NAME[l], "RMSD from seed: 2 ns %.3f, 20 ns %.3f, 50 ns %.3f, 200 ns %.3f nm" % (d[1], d[10], d[25], d[-1]), flush=True)
    pair = []
    for i, a in enumerate(reps):
        for b in reps[i + 1:]:
            n = min(len(X[a][0]), len(X[b][0]))
            pair.append([rms.rmsd(X[a][1][k], X[b][1][k], center=True, superposition=True) / 10 for k in range(n)])
    pair = np.array(pair); out["divergence"][sysn + "_pair"] = dict(t=X[reps[0]][0][:pair.shape[1]].tolist(), mean=pair.mean(0).tolist())
    print(sysn, "pairwise replica RMSD: 2 ns %.3f, 20 ns %.3f, 50 ns %.3f, 200 ns %.3f nm" % tuple(pair.mean(0)[[1, 10, 25, -1]]), flush=True)
json.dump(out, open(H + "/out2/replica_memory.json", "w"), indent=1)
