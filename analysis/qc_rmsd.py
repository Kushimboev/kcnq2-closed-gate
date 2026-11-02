#!/usr/bin/env python3
"""QC for the unbiased trajectories: Calpha RMSD (whole construct, pore domain 233-325) and retigabine retention.

The tetramer is unwrapped by minimum image about the S314 CA centre; the unwrapping is verified in every frame
by the largest consecutive Calpha-Calpha distance inside each chain (must stay < 4.3 A).
Reference = first frame of each trajectory. Output: out2/qc_<label>.json
"""
import sys, os, json
import numpy as np, MDAnalysis as mda
from MDAnalysis.analysis.align import rotation_matrix
HERE = os.path.dirname(os.path.abspath(__file__))
from gate_geom import minimum_image, gate_center_and_diagonal, select_gate_ca

def unwrap(pos, c, box):
    """Chain-wise sequential unwrapping anchored at each chain's S314 CA (which lies next to the axis).
    A plain minimum image about the gate centre fails for peripheral VSD atoms > half a box away."""
    out = np.empty_like(pos)
    for k in range(4):
        sl = slice(k * 261, (k + 1) * 261); p = pos[sl]; q = np.empty_like(p); a = 314 - 70
        q[a] = minimum_image(p[a] - c, box)
        for i in range(a + 1, 261):
            q[i] = q[i - 1] + minimum_image(p[i] - p[i - 1], box)
        for i in range(a - 1, -1, -1):
            q[i] = q[i + 1] + minimum_image(p[i] - p[i + 1], box)
        out[sl] = q
    return out


lab, gro, xtc = sys.argv[1:4]
u = mda.Universe(gro, xtc)
ca = u.select_atoms("protein and name CA"); assert len(ca) == 1044
g = select_gate_ca(u)
pd = np.isin(ca.resids, np.arange(233, 326))
chain_break = np.nonzero(np.diff(ca.resids) != 1)[0]                 # index before each chain start
fbx = u.select_atoms("resname FBX and not name H*")
w236 = u.select_atoms("resid 236 and not name H* and not name N CA C O")
nlig = len(fbx.residues)
T, R_all, R_pd, MAXB, LIGC, LIGD = [], [], [], [], [], []
ref = None
for ts in u.trajectory[::2]:
    box = ts.dimensions[:3].astype(float)
    c, _, _ = gate_center_and_diagonal(g.positions.astype(float), box)
    x = unwrap(ca.positions.astype(float), c, box)
    b = np.linalg.norm(np.diff(x, axis=0), axis=1); b[chain_break] = 0.0
    MAXB.append(float(b.max()))
    if ref is None:
        ref = x.copy(); refc = ref - ref.mean(0)
        if nlig:
            lig0 = None
    xc = x - x.mean(0)
    Rm, _ = rotation_matrix(xc[pd] - xc[pd].mean(0), refc[pd] - refc[pd].mean(0))
    xf = (xc - xc[pd].mean(0)) @ Rm.T + refc[pd].mean(0)
    R_pd.append(float(np.sqrt(((xf[pd] - refc[pd]) ** 2).sum(1).mean())))
    Rm2, rall = rotation_matrix(xc, refc); R_all.append(float(rall))
    if nlig:
        # contact with Trp236 side chain and displacement of each ligand centre in the pore-domain frame
        lp = minimum_image(fbx.positions.astype(float) - c, box)
        wp = minimum_image(w236.positions.astype(float) - c, box)
        cont, cen = [], []
        for res in fbx.residues:
            sel = fbx.resindices == res.resindex
            dmin = np.min(np.linalg.norm(lp[sel][:, None, :] - wp[None, :, :], axis=-1))
            cont.append(bool(dmin < 4.5))
            cen.append(((lp[sel].mean(0) - x.mean(0)) - xc[pd].mean(0)) @ Rm.T)
        cen = np.array(cen)
        if lig0 is None:
            lig0 = cen.copy()
        LIGC.append(cont); LIGD.append(np.linalg.norm(cen - lig0, axis=1).tolist())
    T.append(ts.time / 1000.0)
T = np.array(T); m = T >= 20
out = dict(label=lab, n=len(T), max_ca_ca=float(np.max(MAXB)),
           rmsd_all_mean=float(np.mean(np.array(R_all)[m])), rmsd_all_last=float(np.mean(np.array(R_all)[-50:])),
           rmsd_pd_mean=float(np.mean(np.array(R_pd)[m])), rmsd_pd_last=float(np.mean(np.array(R_pd)[-50:])),
           rmsd_pd_max=float(np.max(R_pd)), t=T.tolist(), rmsd_all=R_all, rmsd_pd=R_pd)
if nlig:
    C = np.array(LIGC); Dd = np.array(LIGD)
    out.update(n_lig=nlig, lig_w236_contact_frac=C[m].mean(0).tolist(), lig_disp_mean=Dd[m].mean(0).tolist(), lig_disp_max=Dd.max(0).tolist())
json.dump(out, open(os.path.join(HERE, "out2", f"qc_{lab}.json"), "w"))
print(lab, "max CA-CA", round(out["max_ca_ca"], 2), "RMSD all/PD (t>20)", round(out["rmsd_all_mean"], 2), round(out["rmsd_pd_mean"], 2),
      ("lig contact " + str(np.round(out["lig_w236_contact_frac"], 2)) + " disp " + str(np.round(out["lig_disp_mean"], 1))) if nlig else "")
