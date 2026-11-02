#!/usr/bin/env python3
"""Summarise out2/<label>.npz (pore_axis.py) -> out2/SUMMARY2.md + out2/summary2.json.

Production window t >= T0 ns (default 20). Regions along the pore axis, z relative to the S314 CA centre
(ring planes measured on S0-eq100: L318 -5.6, S314 0.0, G310 +5.4, A306 +11.6, V302 +16.1, T277 +22.1, G281 +33.2 A):
  gate    |z| <= 6 A        (L318 ... G310)
  cavity  +6 <= z < +20 A   (G310 ... below T277)
  filter  +20 <= z < +35 A
"""
import json, os, sys, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
sys.path.insert(0, HERE)
from trajectories import ORDER
SYS = {"S0": "no RTG, PIP2", "S2": "RTG, PIP2", "S3": "RTG, no PIP2"}
BULK = 33.4                                         # nm^-3, TIP3P, reproduced by the zero test
RW = 0.14                                           # nm, radius of a water molecule
POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "HSD", "HSE", "HIS", "ASP", "GLU", "LYS", "ARG"}


def sysof(lab):
    return lab.replace("REP-", "")[:2]


def gap_length(row, zc, i0):
    """Length (A) of the contiguous water-free stretch of 1-A slabs that contains slab i0 (0 if occupied)."""
    if row[i0] > 0:
        return 0.0
    a = i0
    while a > 0 and row[a - 1] == 0:
        a -= 1
    b = i0
    while b < len(row) - 1 and row[b + 1] == 0:
        b += 1
    return float(b - a + 1)


res = {}
for lab in ORDER:
    f = os.path.join(HERE, "out2", lab + ".npz")
    if not os.path.exists(f):
        continue
    d = np.load(f, allow_pickle=True)
    t = d["t_ns"]; m = t >= T0; n = int(m.sum()); ar = np.arange(n)
    zs = d["zs"]; rcs = list(d["rcs"]); gz = np.abs(zs) <= 6.0
    RZ = d["rz"][m] / 10.0                          # nm, (frames, radius set, rc, z); set 0 = HOLE simple.rad, 1 = alternative radii
    WHO = d["who"][m]; OFF = d["off"][m] / 10.0
    r = dict(system=sysof(lab), n_frames=n, t_ns=[float(t[m].min()), float(t[m].max())], zs=zs.tolist())
    for iset, pre in ((0, ""), (1, "camp_")):
        for ir, rc in enumerate(rcs):
            rz = RZ[:, iset, ir, :]
            g = np.where(gz[None, :], rz, np.inf)
            iz = g.argmin(axis=1); rmin = rz[ar, iz]
            w = WHO[ar, iset, ir, iz]
            rn = d["wall_resname"][w]; ri = d["wall_resid"][w]; an = d["wall_name"][w]
            cres = collections.Counter(f"{a}{b}" for a, b in zip(rn, ri))
            q = dict(rmin_mean=float(rmin.mean()), rmin_sd=float(rmin.std()),
                     p05=float(np.percentile(rmin, 5)), p95=float(np.percentile(rmin, 95)), max=float(rmin.max()),
                     frac_below_water=float(np.mean(rmin < RW)), z_mean=float(zs[iz].mean()), z_sd=float(zs[iz].std()),
                     off_mean=float(OFF[ar, iset, ir, iz].mean()), residues={k: v / n for k, v in cres.most_common(8)},
                     prof_mean=rz.mean(0).tolist(), prof_sd=rz.std(0).tolist())
            r[f"{pre}rc{rc:g}"] = q
            if iset == 0 and rc == 3.0:
                cat = collections.Counter(f"{a}{b}:{c}" for a, b, c in zip(rn, ri, an))
                grp = collections.Counter()
                for a_, c_ in zip(rn, an):
                    if a_ == "SER": grp["ser_oh" if c_ in ("OG", "HG1") else "ser_other"] += 1
                    elif a_ == "LEU": grp["leu"] += 1
                    else: grp["other"] += 1
                r["groups"] = {k: grp.get(k, 0) / n for k in ("ser_oh", "ser_other", "leu", "other")}
                r["residues"] = q["residues"]; r["atoms"] = {k: v / n for k, v in cat.most_common(8)}
                r["zlim_hist"] = {f"{z:+.1f}": int(c) for z, c in zip(*np.unique(zs[iz], return_counts=True))}
                r["rmin_series"] = rmin.tolist()
                for name, z0 in (("L318", -5.5), ("S314", 0.0), ("G310", 5.5), ("A306", 11.5), ("V302", 16.0)):
                    j = int(np.argmin(np.abs(zs - z0)))
                    r[f"R_at_{name}"] = [float(rz[:, j].mean()), float(rz[:, j].std())]
                sfm = (zs >= 22) & (zs <= 34)
                r["R_sf_min"] = [float(rz[:, sfm].min(1).mean()), float(rz[:, sfm].min(1).std())]
    zb = d["zb"]; zc = 0.5 * (zb[:-1] + zb[1:])
    w3 = d["w3"][m].astype(float); w6 = d["w6"][m].astype(float); k4 = d["k4"][m].astype(float); l4 = d["l4"][m].astype(float)
    v3 = np.pi * 0.3 ** 2 * 0.1; v6 = np.pi * 0.6 ** 2 * 0.1          # nm^3 per 1-A slab
    r["zc"] = zc.tolist()
    r["w3_rel"] = (w3.mean(0) / v3 / BULK).tolist(); r["w6_rel"] = (w6.mean(0) / v6 / BULK).tolist()
    r["k4_mean"] = k4.mean(0).tolist(); r["l4_mean"] = l4.mean(0).tolist()
    i0 = int(np.argmin(np.abs(zc - 0.0)))
    gaps = np.array([gap_length(row, zc, i0) for row in w3])
    r["gap_mean"] = float(gaps.mean()); r["gap_sd"] = float(gaps.std()); r["gap_min"] = float(gaps.min())
    r["frac_gap_ge3"] = float(np.mean(gaps >= 3)); r["frac_no_gap"] = float(np.mean(gaps == 0))
    gate = (zc > -6) & (zc < 6); core = (zc > -2) & (zc < 2); cav = (zc > 6) & (zc < 20); sf = (zc > 20) & (zc < 35)
    for nm, mk, src in (("w_core_r3", core, w3), ("w_gate_r6", gate, w6), ("w_cav_r6", cav, w6), ("k_cav", cav, k4), ("k_sf", sf, k4), ("lip_gatecav", gate | cav, l4)):
        x = src[:, mk].sum(1)
        r[nm] = [float(x.mean()), float(x.std()), float(np.mean(x == 0))]
    cavw = w6[:, cav].sum(1); tt = t[m]
    r["w_cav_blocks20"] = [float(cavw[(tt >= a) & (tt < a + 20)].mean()) for a in np.arange(tt.min(), tt.max() - 19.99, 20.0)]
    r["w_cav_series"] = cavw.tolist(); r["t_series"] = tt.tolist()
    nw = d["nw"][m].astype(float)
    r["nw_cyl"] = [float(nw.mean()), float(nw.std()), float(np.mean(nw == 0))]; r["nw_series"] = nw.tolist()
    r["nw_blocks20"] = [float(nw[(tt >= a) & (tt < a + 20)].mean()) for a in np.arange(tt.min(), tt.max() - 19.99, 20.0)]
    diag = d["diag"][m]
    r["diag_mean"] = {int(k): float(v) for k, v in zip(d["rings"], diag.mean(0))}
    r["diag_sd"] = {int(k): float(v) for k, v in zip(d["rings"], diag.std(0))}
    r["diag314_series"] = diag[:, list(d["rings"]).index(314)].tolist()
    r["diag_all_series_t"] = d["t_ns"].tolist(); r["diag314_all"] = d["diag"][:, list(d["rings"]).index(314)].tolist()
    r["sf_z"] = [float(d["sf_z"][m].mean()), float(d["sf_z"][m].std())]; r["tilt_deg"] = float(d["tilt_deg"][m].mean())
    res[lab] = r

json.dump(res, open(os.path.join(HERE, "out2", "summary2.json"), "w"))
P = lambda x: f"{100 * x:.1f} %"
L = [f"# On-axis closed-state summary (unbiased; t >= {T0:g} ns)\n", "Generated by `summarize2.py` from `out2/*.npz` (pore_axis.py). Radii in nm, HOLE simple.rad set unless stated.\n",
     "## 1. Ring diagonals (Calpha, A) and on-axis minimum radius in the gate region (|z| <= 6 A, r_c = 3 A)\n",
     "| trajectory | system | frames | L318 | S314 | G310 | A306 | R_min | 5–95 % | max | P(R_min < 0.14) | z(R_min) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for lab, r in res.items():
    dm, ds, q = r["diag_mean"], r["diag_sd"], r["rc3"]
    L.append(f"| {lab} | {r['system']} ({SYS[r['system']]}) | {r['n_frames']} | {dm[318]:.2f} ± {ds[318]:.2f} | {dm[314]:.2f} ± {ds[314]:.2f} | {dm[310]:.2f} ± {ds[310]:.2f} | {dm[306]:.2f} ± {ds[306]:.2f} | "
             f"{q['rmin_mean']:.3f} ± {q['rmin_sd']:.3f} | {q['p05']:.3f}–{q['p95']:.3f} | {q['max']:.3f} | {q['frac_below_water']:.3f} | {q['z_mean']:+.1f} ± {q['z_sd']:.1f} |")
L += ["\n## 2. Sensitivity of R_min to the search-disk radius r_c and to the radius set\n",
      "| trajectory | r_c = 1.5 A | 3.0 A | 4.5 A | 9.0 A (unconstrained) | offset of the 9-A optimum (nm) | residues limiting the 9-A optimum | alternative radii, r_c = 3 | alternative radii, 9 A |", "|---|---|---|---|---|---|---|---|---|"]
for lab, r in res.items():
    L.append(f"| {lab} | " + " | ".join(f"{r[k]['rmin_mean']:.3f} ± {r[k]['rmin_sd']:.3f}" for k in ("rc1.5", "rc3", "rc4.5", "rc9")) + f" | {r['rc9']['off_mean']:.2f} | " +
             ", ".join(f"{k} {100 * v:.0f}%" for k, v in list(r["rc9"]["residues"].items())[:4]) + f" | {r['camp_rc3']['rmin_mean']:.3f} | {r['camp_rc9']['rmin_mean']:.3f} |")
L += ["\n## 3. On-axis radius at the ring planes (r_c = 3 A)\n", "| trajectory | L318 (−5.5) | S314 (0) | G310 (+5.5) | A306 (+11.5) | V302 (+16) |", "|---|---|---|---|---|---|"]
for lab, r in res.items():
    L.append(f"| {lab} | " + " | ".join(f"{r['R_at_' + k][0]:.3f} ± {r['R_at_' + k][1]:.3f}" for k in ("L318", "S314", "G310", "A306", "V302")) + " |")
L += ["\n## 4. Atom limiting the on-axis constriction (fraction of frames)\n", "| trajectory | residues | atoms | where (z: frames) |", "|---|---|---|---|"]
for lab, r in res.items():
    L.append(f"| {lab} | " + ", ".join(f"{k} {P(v)}" for k, v in list(r["residues"].items())[:6]) + " | " + ", ".join(f"{k} {P(v)}" for k, v in list(r["atoms"].items())[:5]) +
             " | " + ", ".join(f"{k}: {v}" for k, v in sorted(r["zlim_hist"].items(), key=lambda kv: -kv[1])[:5]) + " |")
L += ["\n## 5. Water and ions along the axis\n",
      "| trajectory | water-free gap at gate (A) mean ± SD / min | P(gap >= 3 A) | P(no gap) | waters, core (|z|<2, r<3) | waters, gate (|z|<6, r<6) | waters, cavity (6<z<20, r<6) | P(cavity empty) | K+ cavity | K+ filter | lipid heavy atoms (r<4, gate+cavity) | frozen cylinder N_w / P(0) |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for lab, r in res.items():
    L.append(f"| {lab} | {r['gap_mean']:.1f} ± {r['gap_sd']:.1f} / {r['gap_min']:.0f} | {r['frac_gap_ge3']:.3f} | {r['frac_no_gap']:.3f} | {r['w_core_r3'][0]:.3f} | {r['w_gate_r6'][0]:.2f} ± {r['w_gate_r6'][1]:.2f} | "
             f"{r['w_cav_r6'][0]:.2f} ± {r['w_cav_r6'][1]:.2f} | {r['w_cav_r6'][2]:.2f} | {r['k_cav'][0]:.2f} | {r['k_sf'][0]:.2f} | {r['lip_gatecav'][0]:.2f} | {r['nw_cyl'][0]:.2f} / {r['nw_cyl'][2]:.2f} |")
L += ["\n## 6. 20-ns block means\n", "| trajectory | cavity waters | frozen-cylinder waters |", "|---|---|---|"]
for lab, r in res.items():
    L.append(f"| {lab} | " + " ".join(f"{b:.1f}" for b in r["w_cav_blocks20"]) + " | " + " ".join(f"{b:.2f}" for b in r["nw_blocks20"]) + " |")
L += ["\n## 7. Axis\n", "| trajectory | filter centre above gate centre (A) | mean axis tilt from z (deg) |", "|---|---|---|"]
for lab, r in res.items():
    L.append(f"| {lab} | {r['sf_z'][0]:.1f} ± {r['sf_z'][1]:.1f} | {r['tilt_deg']:.1f} |")
open(os.path.join(HERE, "out2", "SUMMARY2.md"), "w").write("\n".join(L) + "\n")
print("wrote SUMMARY2.md for", len(res), "trajectories")
