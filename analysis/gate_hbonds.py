#!/usr/bin/env python3
"""Hydrogen bonds of the water molecules inside the gate (reviewer comment 4, 2026-11-02). -> out2/gate_hbonds.json

Every (frame, molecule) pair with a water oxygen in the regions of gate_water.py: S (0 <= z <= 2 A, r < 3 A, Ser314 ring) or
P (-5 < z < 0 A, r < 3 A, between the rings). All pairs are analysed, not a sample.
Hydrogen bond (geometric, Luzar & Chandler 1996): O...O <= 3.5 A and H-O...O angle <= 30 deg, donor H from either side.
Partners: Ser314 OG (water as acceptor from HG1, water as donor to OG), any other protein O or N, other water oxygens.
Clearance = min_i(|r_i - r_O| - sigma_i) over protein atoms with HOLE radii: the radius of the largest sphere centred on the oxygen;
the limiting atom is recorded; R is the on-axis radius of Eq. 1 at the height of the oxygen. Control: Ser314 OG-OG hydrogen bonds in frames with and without a molecule in P.
Usage: gate_hbonds.py [label ...]
"""
import os, sys, json, pickle
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H); from trajectories import FILES, NAME, ORDER
from gate_water import region
import MDAnalysis as mda
from gate_geom import minimum_image
from scipy.spatial import cKDTree

RO, ANG = 3.5, 30.0
VDW = dict(H=1.00, C=1.85, N=1.75, O=1.65, S=2.00)


def angle(don, hyd, acc, box):
    a = minimum_image(hyd - don, box); b = minimum_image(acc - don, box)
    c = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def run(lab):
    D = pickle.load(open(os.path.join(H, "out2", f"wat_{lab}.pkl"), "rb"))
    u = mda.Universe(*FILES[lab])
    prot = u.select_atoms("protein"); el = np.array([n[0] for n in prot.names]); assert set(el) <= set("HCNOS"), set(el)
    sig = np.array([VDW[e] for e in el])
    og = u.select_atoms("resid 314 and name OG"); hg = u.select_atoms("resid 314 and name HG1"); assert len(og) == 4 and len(hg) == 4
    tt = np.array([ts.time / 1000.0 for ts in u.trajectory])
    npz = np.load(os.path.join(H, "out2", f"{lab}.npz"), allow_pickle=True); nt = npz["t_ns"]; zs = npz["zs"]; rz = npz["rz"][:, 0, 1, :] / 10   # R(z), HOLE radii, r_c = 0.3 nm, nm
    frames = {}
    for i, (ids, z, r, *_) in enumerate(D["f"]):
        hit = [(int(a), float(zz)) for a, zz, rr in zip(ids, z, r) if region(float(zz), float(rr)) in "SP"]
        p_in = any(region(float(zz), float(rr)) == "P" for zz, rr in zip(z, r))
        frames[i] = (hit, p_in)
    rid2idx = {int(a.resid): a.residue.atoms.indices for a in u.select_atoms("resname TIP3 and name OH2")}
    rows = []; ser_with, ser_without = [], []
    for i, (hit, p_in) in frames.items():
        ts = u.trajectory[int(np.argmin(abs(tt - D["t"][i])))]; box = ts.dimensions[:3].astype(float)
        OG, HG = og.positions.astype(float), hg.positions.astype(float)
        dg = np.linalg.norm(minimum_image(OG[:, None] - OG[None], box), axis=2)
        n = 0
        for a in range(4):
            for b in range(4):
                if a != b and dg[a, b] <= RO and angle(OG[a], HG[a], OG[b], box) <= ANG: n += 1
        (ser_with if p_in else ser_without).append(n)
        if not hit:
            continue
        P = prot.positions.astype(float); O_all = u.select_atoms("resname TIP3 and name OH2")
        for mol, z_ in hit:
            at = u.atoms[rid2idx[mol]]; O = at.select_atoms("name OH2").positions[0].astype(float); W = at.select_atoms("name H1 H2").positions.astype(float)
            d = np.linalg.norm(minimum_image(P - O, box), axis=1)
            j = int(np.argmin(d - sig)); clear = float(d[j] - sig[j])
            # Ser314 hydroxyl
            d_og = np.linalg.norm(minimum_image(OG - O, box), axis=1)
            acc = sum(1 for k in range(4) if d_og[k] <= RO and angle(OG[k], HG[k], O, box) <= ANG)      # water accepts from Ser OH
            don = sum(1 for k in range(4) if d_og[k] <= RO and min(angle(O, W[h], OG[k], box) for h in (0, 1)) <= ANG)   # water donates to Ser OG
            # other protein polar atoms within RO (not Ser314 OG): water as donor to O/N, or their H to water
            near = np.nonzero(d <= RO)[0]; other = []
            for q in near:
                at_q = prot[q]
                if at_q.name[0] not in "ON" or (at_q.resid == 314 and at_q.name == "OG"): continue
                ok = min(angle(O, W[h], P[q], box) for h in (0, 1)) <= ANG
                if not ok:      # the protein atom as donor: its hydrogens are the H atoms of the residue within 1.25 A
                    hb = [x for x in at_q.residue.atoms.select_atoms("name H*") if np.linalg.norm(minimum_image(x.position.astype(float) - P[q], box)) < 1.25]
                    ok = any(angle(P[q], x.position.astype(float), O, box) <= ANG for x in hb)
                if ok: other.append(f"{at_q.resname}{at_q.resid}:{at_q.name}")
            # other waters
            OO = O_all.positions.astype(float); dw = np.linalg.norm(minimum_image(OO - O, box), axis=1); nb = np.nonzero((dw <= RO) & (dw > 0.01))[0]
            nww = 0
            for q in nb:
                wq = O_all[q].residue.atoms.select_atoms("name H1 H2").positions.astype(float)
                if min(angle(O, W[h], OO[q], box) for h in (0, 1)) <= ANG or min(angle(OO[q], wq[h], O, box) for h in (0, 1)) <= ANG: nww += 1
            rows.append(dict(i=i, mol=mol, z=z_, reg=region(z_, 0.0), d_og=float(d_og.min()), n33=int((d_og < 3.3).sum()), acc=acc, don=don, other=other, nww=nww,
                             clear=clear, lim=f"{prot[j].resname}{prot[j].resid}:{prot[j].name}",
                             R=float(rz[int(np.argmin(abs(nt - D["t"][i]))), int(np.argmin(abs(zs - z_)))])))
    return rows, ser_with, ser_without


def summ(rows, sw, swo):
    if not rows:
        return dict(n=0, ser_hb_with=None, ser_hb_without=float(np.mean(swo)) if swo else None)
    f = lambda c, rs: float(np.mean([bool(c(r)) for r in rs])) if rs else None
    out = {}
    for tag, rs in (("all", rows), ("P", [r for r in rows if r["reg"] == "P"]), ("S", [r for r in rows if r["reg"] == "S"])):
        out[tag] = dict(n=len(rs), any_ser=f(lambda r: r["acc"] + r["don"] > 0, rs), mean_ser=float(np.mean([r["acc"] + r["don"] for r in rs])) if rs else None,
                        acc=float(np.mean([r["acc"] for r in rs])) if rs else None, don=float(np.mean([r["don"] for r in rs])) if rs else None,
                        any_other_prot=f(lambda r: r["other"], rs), any_water=f(lambda r: r["nww"], rs), none=f(lambda r: r["acc"] + r["don"] + len(r["other"]) + r["nww"] == 0, rs),
                        d_og=float(np.median([r["d_og"] for r in rs])) if rs else None,
                        clear_med=float(np.median([r["clear"] for r in rs])) if rs else None, clear_below=f(lambda r: r["clear"] < 1.4, rs),
                        lim_ser_oh=f(lambda r: r["lim"].startswith("SER314:") and r["lim"].split(":")[1] in ("OG", "HG1"), rs),
                        R_below=f(lambda r: r["R"] < 0.14, rs), n33=float(np.mean([r["n33"] for r in rs])) if rs else None)
    from collections import Counter
    c = Counter(x for r in rows for x in set(r["other"])); out["other_partners"] = c.most_common(8)
    c2 = Counter(r["lim"] for r in rows); out["limiting"] = c2.most_common(6)
    out["ser_hb_with"] = float(np.mean(sw)) if sw else None; out["ser_hb_without"] = float(np.mean(swo)) if swo else None
    out["nfr_with"] = len(sw); out["nfr_without"] = len(swo); out["n"] = len(rows)
    return out


if __name__ == "__main__":
    labs = sys.argv[1:] or ORDER
    f = os.path.join(H, "out2", "gate_hbonds.json"); RES = json.load(open(f)) if os.path.exists(f) else {}
    for lab in labs:
        rows, sw, swo = run(lab); RES[lab] = summ(rows, sw, swo)
        q = RES[lab]; print(NAME[lab], q["n"], json.dumps(q.get("all"))[:260], "SerHB", q.get("ser_hb_with"), q.get("ser_hb_without"), flush=True)
        json.dump({k: RES[k] for k in ORDER if k in RES}, open(f, "w"), indent=1)
