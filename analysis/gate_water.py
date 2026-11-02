#!/usr/bin/env python3
"""Single water molecules inside the gate: occupancy of the space between the two rings, where the molecules come from,
and whether any molecule crosses the gate.  -> out2/gate_water.json (read by make_tables.py)

Pore frame as in pore_axis.py (origin = Ser314 CA centroid, axis to the filter carbonyls, z in A, positive towards the filter).
Regions of a water oxygen (A):  C cavity z > 2, r < 6 | S Ser314 ring 0 <= z <= 2, r < 3 | P between the rings -5 < z < 0, r < 3
| L Leu318 ring -7 <= z <= -5, r < 3 | Y cytoplasm z < -8, r < 9 | O elsewhere in the tracked box | X outside it.
A water wire = oxygens within r < 4 A linked by O-O <= 3.5 A from z > 2 to z < -8 (to z < -1.5 for the cavity-to-pocket wire).
A visit = a maximal run of frames in S/P/L that contains P.  Its origin (exit) is the last C or Y label before (first after)
the visit, provided that the molecule is never outside the tracked box (X) in between.  A crossing = a visit whose origin
and exit are on opposite sides.  The per-frame positions are cached in out2/wat_<label>.pkl (made on the first run).
Usage: gate_water.py            (all trajectories in trajectories.ORDER)
"""
import os, sys, json, pickle
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H); from trajectories import FILES, NAME, ORDER

T0 = 20.0


def track(lab):
    import MDAnalysis as mda
    from gate_geom import minimum_image, gate_center_and_diagonal, select_gate_ca
    import pore_axis as PA
    gro, xtc = FILES[lab]; u = mda.Universe(gro, xtc)
    ca314 = select_gate_ca(u); sfo = u.select_atoms(f"resid {PA.SF_RESIDS} and name O"); ow = u.select_atoms("resname TIP3 and name OH2")
    assert len(ca314) == 4 and len(sfo) == 20 and len(ow) > 1000
    T, F = [], []
    for ts in u.trajectory:
        if ts.time / 1000.0 < T0 - 1e-6:
            continue
        box = ts.dimensions[:3].astype(float)
        c, _, _ = gate_center_and_diagonal(ca314.positions.astype(float), box)
        v = minimum_image(sfo.positions.astype(float) - c, box).mean(axis=0); tilt = v[:2] / v[2]
        d = minimum_image(ow.positions.astype(float) - c, box); z = d[:, 2]; r = np.hypot(d[:, 0] - tilt[0] * z, d[:, 1] - tilt[1] * z)
        k = (r < 12) & (z > -20) & (z < 12); x = d[:, 0] - tilt[0] * z; y = d[:, 1] - tilt[1] * z
        T.append(ts.time / 1000.0); F.append((ow.resids[k].astype(np.int32), z[k].astype(np.float32), r[k].astype(np.float32),
                                              x[k].astype(np.float32), y[k].astype(np.float32)))
    return dict(t=np.array(T), f=F)


def region(z, r):
    if r < 3 and 0 <= z <= 2: return "S"
    if r < 3 and -5 < z < 0: return "P"
    if r < 3 and -7 <= z <= -5: return "L"
    if z > 2 and r < 6: return "C"
    if z < -8 and r < 9: return "Y"
    return "O"


def analyse(D):
    t = D["t"]; n = len(t); seq = {}
    for i, (ids, z, r, *_) in enumerate(D["f"]):
        for a, zz, rr in zip(ids, z, r):
            seq.setdefault(int(a), {})[i] = region(float(zz), float(rr))
    occ = np.zeros(n, int); inside = set(); visits = []
    for a, s in seq.items():
        if not any(v in "SPL" for v in s.values()):
            continue
        L = [s.get(i, "X") for i in range(n)]
        for i, v in enumerate(L):
            if v == "P": occ[i] += 1
        i = 0
        while i < n:
            if L[i] in "SPL":
                j = i
                while j + 1 < n and L[j + 1] in "SPL": j += 1
                if "P" in L[i:j + 1]:
                    inside.add(a)
                    org = next((L[k] for k in range(i - 1, -1, -1) if L[k] in "CYX"), "-"); ext = next((L[k] for k in range(j + 1, n) if L[k] in "CYX"), "-")
                    visits.append(dict(mol=a, t_in=float(t[i]), t_out=float(t[j]), origin=org, exit=ext))
                i = j + 1
            else:
                i += 1
    cross = [v for v in visits if {v["origin"], v["exit"]} == {"C", "Y"}]
    wire = np.zeros(n, bool); wire_in = np.zeros(n, bool)   # cavity-to-cytoplasm / cavity-to-pocket chains of H-bond distance
    for i, (ids, z, r, x, y) in enumerate(D["f"]):
        k = (r < 4) & (z > -12) & (z < 6)
        if k.sum() < 2:
            continue
        P = np.c_[x[k], y[k], z[k]].astype(float); zz = P[:, 2]
        A = np.linalg.norm(P[:, None] - P[None], axis=2) <= 3.5
        reach = zz > 2; prev = None
        while prev is None or reach.sum() > prev:
            prev = reach.sum(); reach = reach | A[reach].any(0)
        wire[i] = bool((reach & (zz < -8)).any()); wire_in[i] = bool((reach & (zz < -1.5)).any())
    return dict(frames=n, frac_occupied=float(np.mean(occ > 0)), max_molecules=int(occ.max()), n_molecules=len(inside), n_visits=len(visits),
                from_cavity=sum(v["origin"] == "C" for v in visits), from_cytoplasm=sum(v["origin"] == "Y" for v in visits),
                to_cavity=sum(v["exit"] == "C" for v in visits), to_cytoplasm=sum(v["exit"] == "Y" for v in visits),
                n_crossings=len(cross), n_crossing_molecules=len({v["mol"] for v in cross}), crossings=cross,
                frac_wire=float(wire.mean()), frac_wire_to_pocket=float(wire_in.mean()), t_wire=[float(x) for x in t[wire]], visits=visits)


def contacts(lab, D, nmax=200):
    """For up to nmax sampled (frame, molecule) pairs with a water oxygen in S or P: on-axis radius at its height (HOLE set,
    r_c = 0.3 nm), distance to the nearest Ser314 OG and number of Ser314 OG within 3.3 A."""
    import MDAnalysis as mda
    from gate_geom import minimum_image
    hits = [(i, int(a), float(z_)) for i, (ids, z, r, *_) in enumerate(D["f"]) for a, z_, r_ in zip(ids, z, r) if region(float(z_), float(r_)) in "SP"]
    if not hits:
        return None
    hits = hits[::max(1, len(hits) // nmax)]
    d = np.load(os.path.join(H, "out2", f"{lab}.npz"), allow_pickle=True); m = d["t_ns"] >= T0; zs = d["zs"]; rz = d["rz"][m][:, 0, 1, :] / 10
    u = mda.Universe(*FILES[lab]); og = u.select_atoms("resid 314 and name OG"); assert len(og) == 4
    tt = np.array([ts.time / 1000.0 for ts in u.trajectory]); R, dog, nog = [], [], []
    for i, a, z_ in hits:
        u.trajectory[int(np.argmin(abs(tt - D["t"][i])))]; box = u.dimensions[:3].astype(float)
        w = u.select_atoms(f"resname TIP3 and name OH2 and resid {a}").positions[0].astype(float)
        x = np.linalg.norm(minimum_image(og.positions.astype(float) - w, box), axis=1)
        R.append(float(rz[i, int(np.argmin(abs(zs - z_)))])); dog.append(float(x.min())); nog.append(int((x < 3.3).sum()))
    return dict(n=len(hits), R_median=float(np.median(R)), frac_R_below_water=float(np.mean(np.array(R) < 0.14)),
                d_og_median=float(np.median(dog)), frac_hbond=float(np.mean(np.array(dog) < 3.3)), n_og_mean=float(np.mean(nog)))


if __name__ == "__main__":
    out = {}
    for lab in ORDER:
        f = os.path.join(H, "out2", f"wat_{lab}.pkl")
        if not os.path.exists(f):
            pickle.dump(track(lab), open(f, "wb"))
        D = pickle.load(open(f, "rb")); out[lab] = analyse(D); out[lab]["contacts"] = contacts(lab, D)
        q = out[lab]
        print(f"{NAME[lab]:5s} frames {q['frames']} | between rings {100 * q['frac_occupied']:5.1f}% (max {q['max_molecules']}) | molecules {q['n_molecules']:3d} "
              f"| visits {q['n_visits']:3d}: from cavity {q['from_cavity']}, from cytoplasm {q['from_cytoplasm']}; to cavity {q['to_cavity']}, to cytoplasm {q['to_cytoplasm']} "
              f"| crossing molecules {q['n_crossing_molecules']} | wire cav-cyt {100 * q['frac_wire']:.2f}% ({len(q['t_wire'])} fr) | wire cav-pocket {100 * q['frac_wire_to_pocket']:.1f}%"
              + ("" if q["contacts"] is None else f" | OW-OG {q['contacts']['d_og_median']:.2f} A, n OG {q['contacts']['n_og_mean']:.1f}, R<0.14 {100 * q['contacts']['frac_R_below_water']:.0f}%"))
    json.dump(out, open(os.path.join(H, "out2", "gate_water.json"), "w"), indent=1)
