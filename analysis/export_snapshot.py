#!/usr/bin/env python3
"""Export one unbiased frame as a PDB centred on the S314 CA centre with the pore axis along z.

Protein is made whole (chain-wise sequential unwrapping anchored at S314 CA, side chains relative to their CA);
waters within 9 A of the axis and K+ within 6 A are kept. Chains are labelled A-D in topology order.
Usage: export_snapshot.py <gro> <xtc> <time_ns> <out.pdb>
"""
import sys, os
import numpy as np, MDAnalysis as mda
HERE = os.path.dirname(os.path.abspath(__file__))
from gate_geom import minimum_image, gate_center_and_diagonal, select_gate_ca

gro, xtc, tns, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
u = mda.Universe(gro, xtc)
fr = int(np.argmin([abs(ts.time / 1000.0 - tns) for ts in u.trajectory])); u.trajectory[fr]
box = u.dimensions[:3].astype(float)
g = select_gate_ca(u)
c, diag, _ = gate_center_and_diagonal(g.positions.astype(float), box)
prot = u.select_atoms("protein")
ca = prot.select_atoms("name CA"); assert len(ca) == 1044
pos = np.empty((len(prot), 3)); chain = np.empty(len(prot), dtype="U1")
ca_pos = ca.positions.astype(float); ca_unw = np.empty_like(ca_pos)
for k in range(4):
    sl = slice(k * 261, (k + 1) * 261); p = ca_pos[sl]; q = np.empty_like(p); a = 314 - 70
    q[a] = minimum_image(p[a] - c, box)
    for i in range(a + 1, 261): q[i] = q[i - 1] + minimum_image(p[i] - p[i - 1], box)
    for i in range(a - 1, -1, -1): q[i] = q[i + 1] + minimum_image(p[i] - p[i + 1], box)
    ca_unw[sl] = q
resix = prot.resindices - prot.resindices.min()            # 0..1043 in topology order
pos = ca_unw[resix] + minimum_image(prot.positions.astype(float) - ca_pos[resix], box)
chain = np.array(list("ABCD"))[resix // 261]
ow = u.select_atoms("resname TIP3 and name OH2")
d = minimum_image(ow.positions.astype(float) - c, box)
keep = (np.hypot(d[:, 0], d[:, 1]) < 9.0) & (d[:, 2] > -16) & (d[:, 2] < 40)
wat = ow[keep].residues.atoms
k = u.select_atoms("resname POT POTF"); dk = minimum_image(k.positions.astype(float) - c, box)
kk = (np.hypot(dk[:, 0], dk[:, 1]) < 6.0) & (dk[:, 2] > -16) & (dk[:, 2] < 40)
L = []; n = 0
def line(rec, name, resn, ch, resi, xyz, el):
    global n; n += 1
    nm = name if len(name) == 4 else " " + name
    return f"{rec:<6}{n % 100000:>5} {nm:<4} {resn[:3]:>3} {ch}{resi % 10000:>4}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {el:>2}"
for a, p, ch in zip(prot, pos, chain):
    el = "C" if a.name[0].isdigit() else a.name[0]
    L.append(line("ATOM", a.name, a.resname, ch, a.resid, p, el))
for i, r in enumerate(wat.residues):
    o = minimum_image(r.atoms.positions[0].astype(float) - c, box)
    for a in r.atoms:
        p = o + minimum_image(a.position.astype(float) - r.atoms.positions[0].astype(float), box)
        L.append(line("HETATM", "O" if a.name == "OH2" else a.name, "HOH", "W", i + 1, p, "O" if a.name == "OH2" else "H"))
for i, p in enumerate(dk[kk]):
    L.append(line("HETATM", "K", "K", "K", i + 1, p, "K"))
open(out, "w").write("\n".join(L) + "\nEND\n")
print(f"frame {fr} t={u.trajectory.time/1000:.1f} ns  S314 diag {diag:.2f} A  waters kept {len(wat.residues)}  K+ kept {int(kk.sum())} -> {out}")
