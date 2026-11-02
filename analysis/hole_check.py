#!/usr/bin/env python3
"""Check of Eq. (1) against the HOLE program itself (reviewer comment 3, 2026-11-02). -> out2/hole_check.json

For N frames of each trajectory (evenly spaced, t >= 20 ns): the protein is written in the pore frame of pore_axis.py and HOLE 2.3.1 (conda env hole2_env, simple.rad)
is started at the origin with CVECT along the pore axis, SAMPLE 0.5 A. HOLE places its sphere by Monte Carlo annealing, free to leave the axis.
From the .sph output: R(s), centre offset from the axis. Compared in the same frame with out2/<label>.npz: on-axis r_c = 0.3 nm and unconfined r_c = 0.9 nm, HOLE radius set.
Usage: hole_check.py [nframes]
"""
import os, sys, json, subprocess, tempfile, shutil, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
warnings.filterwarnings("ignore")
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H); from trajectories import FILES, NAME, ORDER
HOLE = "hole"; RAD = "simple.rad"
NF = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def one(lab):
    import MDAnalysis as mda
    from gate_geom import minimum_image, gate_center_and_diagonal, select_gate_ca, SF_RESIDS
    d = np.load(os.path.join(H, "out2", f"{lab}.npz"), allow_pickle=True); t = d["t_ns"]; zs = d["zs"]; rz = d["rz"][:, 0]  # HOLE set: [frame, rc, z] in A
    u = mda.Universe(*FILES[lab]); ca = select_gate_ca(u); sfo = u.select_atoms(f"resid {SF_RESIDS} and name O"); prot = u.select_atoms("protein")
    tt = np.array([ts.time / 1000.0 for ts in u.trajectory]); idx = np.where(t >= 20)[0]; pick = idx[np.linspace(0, len(idx) - 1, NF).astype(int)]
    gate = np.abs(zs) <= 6.0; rows = []
    for k in pick:
        ts = u.trajectory[int(np.argmin(abs(tt - t[k])))]; box = ts.dimensions[:3].astype(float)
        c, _, _ = gate_center_and_diagonal(ca.positions.astype(float), box)
        v = minimum_image(sfo.positions.astype(float) - c, box).mean(axis=0); ax = v / np.linalg.norm(v)
        prot.positions = minimum_image(prot.positions.astype(float) - c, box)
        tmp = tempfile.mkdtemp(prefix="hole_", dir=os.environ.get("HOLE_TMP"))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore"); prot.write(os.path.join(tmp, "p.pdb"))
            open(os.path.join(tmp, "in"), "w").write(f"coord p.pdb\nradius {RAD}\nsphpdb h.sph\ncvect {ax[0]:.5f} {ax[1]:.5f} {ax[2]:.5f}\ncpoint 0 0 0\nsample 0.5\nendrad 15\nignore TIP3\n")
            subprocess.run([HOLE], stdin=open(os.path.join(tmp, "in")), stdout=open(os.path.join(tmp, "o"), "w"), stderr=subprocess.DEVNULL, cwd=tmp, timeout=300)
            S = []
            for l in open(os.path.join(tmp, "h.sph")):
                if l.startswith("ATOM"):
                    x, y, zz = float(l[30:38]), float(l[38:46]), float(l[46:54]); r = float(l[60:66]); S.append((x, y, zz, r))
            S = np.array(S)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        s = S[:, :3] @ ax; rho = np.linalg.norm(S[:, :3] - np.outer(s, ax), axis=1); m = np.abs(s) <= 6.0
        if m.sum() == 0: continue
        j = np.where(m)[0][int(np.argmin(S[m, 3]))]
        rows.append(dict(t=float(t[k]), hole=float(S[j, 3] / 10), hole_rho=float(rho[j] / 10), hole_s=float(s[j] / 10), hole_rho_mean=float(rho[m].mean() / 10), hole_rho_max=float(rho[m].max() / 10),
                         axis03=float(rz[k, 1, gate].min() / 10), axis015=float(rz[k, 0, gate].min() / 10), free09=float(rz[k, 3, gate].min() / 10)))
    print(NAME[lab], len(rows), "HOLE %.3f rho %.2f | on-axis %.3f | free %.3f" % tuple(np.mean([[r["hole"], r["hole_rho"], r["axis03"], r["free09"]] for r in rows], axis=0)), flush=True)
    return lab, rows


if __name__ == "__main__":
    with ProcessPoolExecutor(4) as ex:
        res = dict(ex.map(one, ORDER))
    a = [r for v in res.values() for r in v]; f = lambda k: np.array([r[k] for r in a])
    S = dict(n=len(a), hole_mean=float(f("hole").mean()), axis03_mean=float(f("axis03").mean()), free09_mean=float(f("free09").mean()), hole_max=float(f("hole").max()),
             frac_hole_below_water=float((f("hole") < 0.14).mean()), corr_hole_axis=float(np.corrcoef(f("hole"), f("axis03"))[0, 1]), corr_hole_free=float(np.corrcoef(f("hole"), f("free09"))[0, 1]),
             mean_abs_diff_axis=float(abs(f("hole") - f("axis03")).mean()), mean_abs_diff_free=float(abs(f("hole") - f("free09")).mean()), rho_mean=float(f("hole_rho").mean()), rho_median=float(np.median(f("hole_rho"))),
             frac_rho_within_rc=float((f("hole_rho") <= 0.3).mean()), rho_track_max_median=float(np.median(f("hole_rho_max"))))
    S["within_rc_hole"] = float(f("hole")[f("hole_rho") <= 0.3].mean()) if (f("hole_rho") <= 0.3).any() else None
    S["within_rc_axis"] = float(f("axis03")[f("hole_rho") <= 0.3].mean()) if (f("hole_rho") <= 0.3).any() else None
    print(json.dumps(S, indent=1))
    json.dump(dict(summary=S, frames=res), open(os.path.join(H, "out2", "hole_check.json"), "w"), indent=1)
