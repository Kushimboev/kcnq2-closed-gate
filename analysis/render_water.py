#!/usr/bin/env python3
"""PyMOL axial close-up of the Ser314 ring with hydrogen bonds (geometric: O...O <= 3.5 A, H-O...O <= 30 deg) from an exported frame.
Waters with the oxygen within 3 A of the axis and -5 < z < 2 A are drawn. Run: pymol -cq render_water.py -- frame.pdb out.png
"""
import sys, math
from pymol import cmd, stored
pdb, png = sys.argv[-2], sys.argv[-1]
cmd.load(pdb, "m"); cmd.hide("everything"); cmd.bg_color("white")
cmd.set("ray_opaque_background", 1); cmd.set("antialias", 2); cmd.set("ray_shadows", 0); cmd.set("orthoscopic", 1); cmd.set("depth_cue", 0)
cmd.set("specular", 0.15); cmd.set("ambient", 0.45); cmd.set("stick_radius", 0.17); cmd.set("sphere_scale", 0.26); cmd.set("dash_gap", 0.22); cmd.set("dash_radius", 0.075); cmd.set("dash_color", "grey20"); cmd.set("label_size", 0)
stored.a = []
cmd.iterate_state(1, "(resi 314 and name OG+HG1) or (resn HOH)", "stored.a.append((model, index, chain, resi, name, x, y, z))")
ser = {}; wat = {}
for m, i, ch, ri, nm, x, y, z in stored.a:
    d = ser if nm in ("OG", "HG1") else wat
    d.setdefault((ch, ri), {})[nm] = (i, (x, y, z))
keep = {k: v for k, v in wat.items() if "O" in v and math.hypot(v["O"][1][0], v["O"][1][1]) < 3 and -5 < v["O"][1][2] < 2}
print("waters drawn:", len(keep))
def ang(D, H, A):
    a = [H[i] - D[i] for i in range(3)]; b = [A[i] - D[i] for i in range(3)]
    n = math.sqrt(sum(q * q for q in a)) * math.sqrt(sum(q * q for q in b))
    return math.degrees(math.acos(max(-1, min(1, sum(a[i] * b[i] for i in range(3)) / n))))
dist = lambda p, q: math.dist(p, q)
n = 0; nss = 0; nsw = 0
def bond(i, j):
    global n
    n += 1; cmd.distance(f"hb{n}", f"m and index {i}", f"m and index {j}", mode=0); cmd.hide("labels", f"hb{n}")
S = list(ser.items())
for ka, a in S:
    for kb, b in S:
        if ka != kb and dist(a["OG"][1], b["OG"][1]) <= 3.5 and ang(a["OG"][1], a["HG1"][1], b["OG"][1]) <= 30: bond(a["OG"][0], b["OG"][0]); nss += 1
for kw, w in keep.items():
    for ka, a in S:
        if dist(w["O"][1], a["OG"][1]) <= 3.5 and (ang(a["OG"][1], a["HG1"][1], w["O"][1]) <= 30 or min(ang(w["O"][1], w[h][1], a["OG"][1]) for h in ("H1", "H2")) <= 30): bond(w["O"][0], a["OG"][0]); nsw += 1
print("Ser-Ser", nss, "Ser-water", nsw)
cmd.show("cartoon", "m and resi 306-322"); cmd.set("cartoon_transparency", 0.80); cmd.set("cartoon_color", "grey80")
cmd.show("sticks", "m and resi 314 and not name N+C+O+H+HA+HB*")
cmd.show("spheres", "m and resi 314 and name OG+HG1")
cmd.color("gold", "m and resi 314 and elem C"); cmd.color("red", "m and resi 314 and name OG"); cmd.color("white", "m and resi 314 and name HG1")
if keep:
    sel = "m and index " + "+".join(str(v[n_][0]) for v in keep.values() for n_ in ("O", "H1", "H2"))
    cmd.select("wg", sel); cmd.show("sticks", "wg"); cmd.show("spheres", "wg"); cmd.color("marine", "wg and elem O"); cmd.color("white", "wg and elem H")
cmd.reset(); cmd.zoom("m and resi 314 and name CB+OG+HG1", 1.5)
cmd.png(png, width=900, height=900, dpi=300, ray=1)
