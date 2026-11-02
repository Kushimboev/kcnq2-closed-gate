#!/usr/bin/env python3
"""On-axis pore radius and axial water/ion density for UNBIASED trajectories (v3, 2026-10-02).

The probe sphere centre is confined to a disk of radius r_c about the pore axis, the line through the Ser314 CA centre (z = 0) and the centre of the
TIGYG carbonyl oxygens. r_c = 1.5, 3.0, 4.5 and 9.0 A are evaluated in the same pass, for two radius sets: 0 = HOLE simple.rad (primary), 1 = O 1.40, N 1.65 A (test).

Per frame: R(z), limiting atom and radial offset for every (radius set, r_c), z in [-14, +36] A;
water-oxygen counts per 1-A slab within 3 A and 6 A of the axis; K+ within 4 A; lipid/ligand heavy atoms
within 4 A; ring diagonals; frozen-cylinder water count.

Usage: pore_axis.py <label> <gro> <xtc> [--start i --stop j] -> out2/chunks/<label>_<start>.npz
"""
import argparse, os, sys
import numpy as np
import MDAnalysis as mda
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
from gate_geom import (minimum_image, gate_center_and_diagonal, select_gate_ca,
                       count_gate_waters, SF_RESIDS)
from gate_geom import elements, KNN

VDW_SETS = (dict(H=1.00, C=1.85, N=1.75, O=1.65, S=2.00, P=2.10),      # HOLE simple.rad
            dict(H=1.00, C=1.85, N=1.65, O=1.40, S=2.00, P=2.10))      # alternative set
RINGS = (306, 310, 314, 318)
RCS = (1.5, 3.0, 4.5, 9.0)
ZS = np.arange(-14.0, 36.0 + 1e-9, 0.5)


def _grid(step, rmin, rmax):
    g = np.arange(-rmax, rmax + 1e-9, step)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    b = np.column_stack([gx.ravel(), gy.ravel()]); r = np.hypot(b[:, 0], b[:, 1])
    m = (r <= rmax) & (r > rmin)
    return b[m]


BASE = np.vstack([_grid(0.3, -1.0, 4.5), _grid(0.4, 4.5, 9.0)])       # fine inside 4.5 A, 0.4 A outside 
BR = np.hypot(BASE[:, 0], BASE[:, 1])
FINE = BR <= 4.5
ZB = np.arange(-16.0, 38.0 + 1e-9, 1.0)


def profile(pos, rads, center, tilt, box):
    d = minimum_image(pos - center, box)
    near = (d[:, 2] > ZS[0] - 8.0) & (d[:, 2] < ZS[-1] + 8.0) & (d[:, 0] ** 2 + d[:, 1] ** 2 < (RCS[-1] + 14.0) ** 2)
    if near.sum() < 50:
        raise SystemExit(f"ERROR: only {near.sum()} wall atoms near the pore")
    inear = np.nonzero(near)[0]
    P = d[near]; Rs = [r[near] for r in rads]
    tree = cKDTree(P); k = min(KNN, len(P))
    ns, nr, nz = len(rads), len(RCS), len(ZS)
    out = np.empty((ns, nr, nz), dtype=np.float32); who = np.empty((ns, nr, nz), dtype=np.int32); off = np.empty((ns, nr, nz), dtype=np.float32)
    for iz, z in enumerate(ZS):
        ax = tilt * z
        dist, idx = tree.query(np.column_stack([BASE + ax, np.full(len(BASE), z)]), k=k)
        for s in range(ns):
            c = dist - Rs[s][idx]; j = c.argmin(axis=1); rows = np.arange(len(BASE))
            c0 = c[rows, j]; w0 = idx[rows, j]
            for ir, rc in enumerate(RCS):
                b = int(np.argmax(np.where(BR <= rc, c0, -np.inf)))
                best, bw, bxy = c0[b], w0[b], BASE[b].copy()
                step = 0.3 if FINE[b] else 0.4
                for _ in range(3):
                    step /= 4.0
                    o = np.array([-2, -1, 0, 1, 2]) * step
                    ox, oy = np.meshgrid(o, o, indexing="ij")
                    cand = np.column_stack([bxy[0] + ox.ravel(), bxy[1] + oy.ravel()])
                    cand = cand[np.hypot(cand[:, 0], cand[:, 1]) <= rc]
                    dd, ii = tree.query(np.column_stack([cand + ax, np.full(len(cand), z)]), k=k)
                    cc = dd - Rs[s][ii]; jj = cc.argmin(axis=1); rr = np.arange(len(cand))
                    m = int(np.argmax(cc[rr, jj]))
                    if cc[m, jj[m]] > best:
                        best, bw, bxy = cc[m, jj[m]], ii[m, jj[m]], cand[m].copy()
                out[s, ir, iz] = best; who[s, ir, iz] = inear[bw]; off[s, ir, iz] = np.hypot(bxy[0], bxy[1])
    return out, who, off


def slab_counts(pos, center, tilt, box, rmax):
    d = minimum_image(pos - center, box)
    r = np.hypot(d[:, 0] - tilt[0] * d[:, 2], d[:, 1] - tilt[1] * d[:, 2])
    return np.histogram(d[r < rmax, 2], bins=ZB)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label"); ap.add_argument("gro"); ap.add_argument("xtc")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--stop", type=int, default=10 ** 9)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    outdir = os.path.join(HERE, "out2", "chunks"); os.makedirs(outdir, exist_ok=True)

    u = mda.Universe(a.gro, a.xtc)
    ca314 = select_gate_ca(u)
    rings = {r: u.select_atoms(f"resid {r} and name CA") for r in RINGS}
    assert all(len(s) == 4 for s in rings.values())
    wall = u.select_atoms("protein")
    el = elements(wall)
    assert set(el) <= set("HCNOS"), set(el)
    rads = [np.array([v[e] for e in el]) for v in VDW_SETS]
    ow = u.select_atoms("resname TIP3 and name OH2")
    kions = u.select_atoms("resname POT POTF")
    sfo = u.select_atoms(f"resid {SF_RESIDS} and name O")
    other = u.select_atoms("(resname POPC POPE SAPI2 SAPI24 FBX CLA) and not name H*")
    assert len(sfo) == 20 and len(ow) > 1000 and len(kions) > 10

    stop = min(a.stop, u.trajectory.n_frames)
    T, D, NW, RZ, WHO, OFF, W3, W6, K4, L4, SFZ, TILT = ([] for _ in range(12))
    for ts in u.trajectory[a.start:stop:a.stride]:
        box = ts.dimensions[:3].astype(float)
        center, _, _ = gate_center_and_diagonal(ca314.positions.astype(float), box)
        dd = [gate_center_and_diagonal(rings[r].positions.astype(float), box, check=False)[1] for r in RINGS]
        v = minimum_image(sfo.positions.astype(float) - center, box).mean(axis=0)
        tilt = v[:2] / v[2]
        rz, who, off = profile(wall.positions.astype(float), rads, center, tilt, box)
        O = ow.positions.astype(float)
        T.append(ts.time / 1000.0); D.append(dd); NW.append(count_gate_waters(O, center, box))
        RZ.append(rz); WHO.append(who); OFF.append(off)
        W3.append(slab_counts(O, center, tilt, box, 3.0)); W6.append(slab_counts(O, center, tilt, box, 6.0))
        K4.append(slab_counts(kions.positions.astype(float), center, tilt, box, 4.0))
        L4.append(slab_counts(other.positions.astype(float), center, tilt, box, 4.0))
        SFZ.append(v[2]); TILT.append(np.degrees(np.arctan(np.hypot(*tilt))))
    np.savez_compressed(os.path.join(outdir, f"{a.label}_{a.start:05d}.npz"),
                        t_ns=np.array(T), diag=np.array(D), rings=np.array(RINGS), nw=np.array(NW),
                        zs=ZS, rcs=np.array(RCS), rz=np.array(RZ), who=np.array(WHO), off=np.array(OFF),
                        zb=ZB, w3=np.array(W3, dtype=np.int16), w6=np.array(W6, dtype=np.int16),
                        k4=np.array(K4, dtype=np.int16), l4=np.array(L4, dtype=np.int16),
                        sf_z=np.array(SFZ), tilt_deg=np.array(TILT),
                        wall_resid=wall.resids, wall_resname=np.array(wall.resnames), wall_name=np.array(wall.names))
    print(a.label, a.start, "DONE", len(T), "frames", flush=True)


if __name__ == "__main__":
    main()
