#!/usr/bin/env python3
"""PyMOL renders of the closed gate from an exported unbiased frame (export_snapshot.py).

  <tag>_side.png   two opposite subunits (A, C), pore domain, gate rings as sticks, water and K+ within 6 A of the axis
  <tag>_S314.png   Ser314 ring, van der Waals spheres, viewed along the pore axis from the cavity side
  <tag>_L318.png   Leu318 ring, van der Waals spheres, viewed along the pore axis from the cytoplasmic side
Run: pymol -cq render_gate.py -- snap/S0_eq100_100ns.pdb ../manuscript/figures/render_S0
"""
import sys, math
from pymol import cmd
pdb, tag = sys.argv[-2], sys.argv[-1]
cmd.load(pdb, "m")
cmd.hide("everything")
cmd.bg_color("white")
cmd.set("ray_opaque_background", 1); cmd.set("antialias", 2); cmd.set("ray_shadows", 0)
cmd.set("orthoscopic", 1); cmd.set("depth_cue", 0); cmd.set("specular", 0.15); cmd.set("ambient", 0.45)
cmd.set("cartoon_transparency", 0.0); cmd.set("stick_radius", 0.22); cmd.set("label_size", 20)
# put the A-C diagonal along x
from pymol import stored
stored.p = []
cmd.iterate_state(1, "m and chain A+C and resi 314 and name CA", "stored.p.append((chain, x, y, z))")
assert len(stored.p) == 2, stored.p
a = [q[1:] for q in stored.p if q[0] == "A"][0]; c = [q[1:] for q in stored.p if q[0] == "C"][0]
ang = math.degrees(math.atan2(c[1] - a[1], c[0] - a[0]))
cmd.rotate("z", -ang, "m", camera=0, origin=[0, 0, 0])
cmd.color("grey80", "m and elem C"); cmd.color("red", "elem O"); cmd.color("blue", "elem N"); cmd.color("white", "elem H")
cmd.color("purple", "resn K")

# ---- side view
cmd.reset(); cmd.turn("x", -90)
pd = "m and chain A+C and resi 233-325"
cmd.show("cartoon", pd); cmd.set("cartoon_color", "grey80")
cmd.show("sticks", "m and chain A+C and resi 318+314+310+306 and not name N+C+O and not elem H")
cmd.show("sticks", "m and chain A+C and resi 277-281 and name N+CA+C+O+OG1+CB")
cmd.color("gold", "m and resi 314 and elem C"); cmd.color("grey30", "m and resi 318 and elem C")
cmd.color("grey50", "m and resi 310+306 and elem C")
cmd.show("spheres", "m and chain A+C and resi 310 and name CA"); cmd.set("sphere_scale", 0.35, "m and resi 310 and name CA")
cmd.set("label_color", "black"); cmd.set("label_font_id", 7); cmd.set("label_size", 15)
stored.z = {}
cmd.iterate_state(1, "m and chain A and name CA and resi 318+314+310+306+277+281", "stored.z[resi] = z")
for nm, ri in (("Leu318", "318"), ("Ser314", "314"), ("Gly310", "310"), ("Ala306", "306"), ("Thr277", "277"), ("Gly281", "281")):
    cmd.pseudoatom("lab_" + ri, pos=[21.0, 0.0, stored.z[ri]], label=nm)
stored.keep = []
cmd.iterate_state(1, "resn HOH and name O", "stored.keep.append((ID, x, y, z))")
ids = [str(i) for i, x, y, z in stored.keep if x * x + y * y < 36.0 and -14 < z < 38]
if ids:
    cmd.select("lumw", "resn HOH and name O and id " + "+".join(ids))
    cmd.show("spheres", "lumw"); cmd.set("sphere_scale", 0.45, "lumw")
cmd.show("spheres", "resn K"); cmd.set("sphere_scale", 0.55, "resn K")
cmd.zoom(pd + " or lab_*", 1)
cmd.png(tag + "_side.png", width=1100, height=1500, dpi=300, ray=1)

# ---- axial views of the two rings
cmd.hide("everything"); cmd.delete("lab_*")
cmd.set("sphere_scale", 1.0)
for name, resi, flip in (("S314", "314", False), ("L318", "318", True)):
    cmd.hide("everything")
    cmd.show("cartoon", "m and resi 300-325"); cmd.set("cartoon_transparency", 0.75)
    cmd.show("spheres", f"m and resi {resi}")
    cmd.color("gold" if resi == "314" else "grey30", f"m and resi {resi} and elem C")
    cmd.reset()
    if flip:
        cmd.turn("y", 180)
    cmd.zoom(f"m and resi {resi}", 6)
    cmd.png(f"{tag}_{name}.png", width=1100, height=1100, dpi=300, ray=1)
