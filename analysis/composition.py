#!/usr/bin/env python3
"""Composition of the simulated systems, counted from the run inputs and from the last coordinates of every trajectory (reviewer comment 2). -> out2/composition.json

Charges come from eq100.tpr; K+ is split into the bulk ions (POT) and the two ions placed in the filter (POTF). Leaflets: lipid phosphorus above or below the mean.
Concentrations: mol per kg of water, from the molecule counts. Filter sites of the placed ions: nearest midpoint between carbonyl planes in the patched coordinates.
"""
import os, sys, json, warnings
import numpy as np, MDAnalysis as mda
warnings.filterwarnings("ignore")
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
from trajectories import FILES, NAME, ORDER
S = os.environ.get("KCNQ2_TRAJ", ".") + "/systems/"; B = S
TPR = {"S0": S + "S0_noRTG_PIP2/prod_eq100/eq100.tpr", "S2": S + "S2_RTG_PIP2/prod_eq100/eq100.tpr", "S3": S + "S3_RTG_noPIP2/prod_eq100/eq100.tpr"}
PATCH = {"S2": S + "S2_RTG_PIP2/build/system_Kfilter.gro", "S3": S + "S3_RTG_noPIP2/build/system_Kfilter.gro"}
out = {}
for s, t in TPR.items():
    u = mda.Universe(t); q = lambda sel: float(u.select_atoms(sel).charges.sum()); n = lambda sel: len(u.select_atoms(sel).residues)
    w = n("resname TIP3"); k = n("resname POT POTF"); c = n("resname CLA")
    out[s] = dict(total=round(float(u.atoms.charges.sum()), 3), protein=round(q("protein"), 3), pip2=round(q("resname SAPI24"), 3), pip2_n=n("resname SAPI24"), rtg_charge=round(q("resname FBX"), 3), rtg_n=n("resname FBX"),
                  k_bulk=n("resname POT"), k_filter=n("resname POTF"), cl=c, water=w, popc=n("resname POPC"), pope=n("resname POPE"), atoms=len(u.atoms), m_k=k / (w * 0.018015), m_cl=c / (w * 0.018015))
    out[s]["pairs"] = min(k, c); out[s]["extra_k"] = k - min(k, c); out[s]["extra_cl"] = c - min(k, c)
for s, g in PATCH.items():
    u = mda.Universe(g); z = {r: u.select_atoms(f"protein and resid {r} and name O").positions[:, 2].mean() for r in range(277, 282)}
    site = {"S4": (z[277] + z[278]) / 2, "S3": (z[278] + z[279]) / 2, "S2": (z[279] + z[280]) / 2, "S1": (z[280] + z[281]) / 2}
    K = u.select_atoms("resname POTF").positions[:, 2]; out[s]["filter_sites"] = sorted(min(site, key=lambda q: abs(site[q] - x)) for x in K)
    P = u.select_atoms("resname POPC POPE SAPI2 SAPI24 and name P"); zc = P.positions[:, 2].mean()
    up = P[P.positions[:, 2] > zc]; lo = P[P.positions[:, 2] < zc]; cnt = lambda A: {k: int((A.resnames == k).sum()) for k in set(A.resnames)}
    out[s]["upper"] = cnt(up); out[s]["lower"] = cnt(lo)
for l in ORDER:
    u = mda.Universe(FILES[l][0]); s = NAME[l][:2]
    assert len(u.select_atoms("resname POT POTF").residues) == out[s]["k_bulk"] + out[s]["k_filter"] and len(u.select_atoms("resname CLA").residues) == out[s]["cl"], l
    assert len(u.atoms) == out[s]["atoms"], l
    assert abs(out[s]["total"]) < 1e-3
json.dump(out, open(H + "/out2/composition.json", "w"), indent=1); print(json.dumps(out)[:1800])
