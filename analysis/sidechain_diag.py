#!/usr/bin/env python3
"""Side-chain cross-pore distances at the two gate rings, for comparison with the cryo-EM 'shortest diagonal
atom-to-atom distance' (Li 2021; Ma 2023) and with d4/d5 of Roscioni 2026 (S314 OG-OG, L318 CD-CD).
-> out2/sidechain.json"""
import sys, os, json, collections
import numpy as np, MDAnalysis as mda
HERE = os.path.dirname(os.path.abspath(__file__))
from gate_geom import minimum_image
sys.path.insert(0, HERE)
from trajectories import FILES, ORDER
TR = {l: FILES[l] for l in ORDER}
OUTF = os.path.join(HERE, "out2", "sidechain.json")
out = json.load(open(OUTF)) if os.path.exists(OUTF) else {}
for lab, (g, x) in TR.items():
    if lab in out: continue
    u = mda.Universe(g, x)
    og = u.select_atoms("resid 314 and name OG"); assert len(og) == 4
    cd = [u.select_atoms(f"resid 318 and name CD1 CD2")[2 * k:2 * k + 2] for k in range(4)]
    OG, CD = [], []
    for ts in u.trajectory:
        if ts.time < 20000: continue
        box = ts.dimensions[:3].astype(float)
        p = og.positions.astype(float)
        d = np.linalg.norm(minimum_image(p[:, None] - p[None], box), axis=-1)
        pair = max((((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))), key=lambda q: d[q[0]] + d[q[1]])
        OG.append(min(d[pair[0]], d[pair[1]]))                       # shortest of the two OG-OG diagonals
        # L318: shortest CD-CD distance across each diagonal (same pairing)
        best = []
        for (i, j) in pair:
            dd = np.linalg.norm(minimum_image(cd[i].positions[:, None].astype(float) - cd[j].positions[None].astype(float), box), axis=-1)
            best.append(dd.min())
        CD.append(min(best))
    out[lab] = dict(og_diag=[float(np.mean(OG)), float(np.std(OG))], cd_diag=[float(np.mean(CD)), float(np.std(CD))])
    print(lab, "S314 OG-OG shortest diagonal %.2f ± %.2f A | L318 CD-CD shortest %.2f ± %.2f A" % (*out[lab]["og_diag"], *out[lab]["cd_diag"]), flush=True)
json.dump(out, open(OUTF, "w"), indent=1)
