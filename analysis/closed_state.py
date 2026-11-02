#!/usr/bin/env python3
"""Closed-state characterisation of the KCNQ2 S6 gate from UNBIASED trajectories.


Per frame:
  - Calpha cross-subunit diagonals of the four S6 rings A306 / G310 / S314 / L318
  - number of water oxygens in the frozen gate cylinder (|z| < 5 A, r < 6 A on S314 CA)
  - HOLE-type pore radius profile R(z), z in [-12, +12] A about the S314 CA centre,
    with the limiting atom and the (x, y) offset of the optimal sphere at every z
  - K+ count in the selectivity filter (r < 5 A, |z| < 9 A about the TIGYG carbonyl centre)

All geometric definitions are imported from gate_geom.py
(gate_geom.py, pore_radius.py) and are NOT redefined here.

Usage: closed_state.py <label> <gro> <xtc> [--stride 1]
Output: out/<label>.npz, out/<label>.zero.txt
"""
import argparse, os, sys, io, contextlib
import numpy as np
import MDAnalysis as mda
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
from gate_geom import (minimum_image, gate_center_and_diagonal, select_gate_ca,
                       count_gate_waters, SF_RESIDS)
from pore_radius import (Z_LO, Z_HI, Z_STEP, XY_HALF, XY_STEP, XY_REFINE, KNN,
                         VDW, VDW_DEFAULT, elements)
import pore_radius, hydration

RINGS = (306, 310, 314, 318)
_g = np.arange(-XY_HALF, XY_HALF + 1e-9, XY_STEP)
_gx, _gy = np.meshgrid(_g, _g, indexing="ij")
BASE = np.column_stack([_gx.ravel(), _gy.ravel()])
BASE = BASE[BASE[:, 0] ** 2 + BASE[:, 1] ** 2 <= XY_HALF ** 2]
ZS = np.arange(Z_LO, Z_HI + 1e-9, Z_STEP)


def profile_with_atoms(pos, rad, center, box):
    """R(z), index of the limiting atom, and radial offset of the optimal sphere."""
    d = minimum_image(pos - center, box)
    near = (np.abs(d[:, 2]) < abs(Z_LO) + 8.0) & \
           (d[:, 0] ** 2 + d[:, 1] ** 2 < (XY_HALF + 12.0) ** 2)
    if near.sum() < 50:
        raise SystemExit(f"ERROR: only {near.sum()} wall atoms near the pore (PBC/selection)")
    inear = np.nonzero(near)[0]
    P, R = d[near], rad[near]
    tree = cKDTree(P)
    k = min(KNN, len(P))
    out = np.empty(len(ZS)); who = np.empty(len(ZS), dtype=np.int64); off = np.empty(len(ZS))
    for iz, z in enumerate(ZS):
        pts = np.column_stack([BASE, np.full(len(BASE), z)])
        step = XY_STEP
        for _ in range(XY_REFINE + 1):
            dist, idx = tree.query(pts, k=k)
            clear = dist - R[idx]
            jmin = clear.argmin(axis=1)
            cval = clear[np.arange(len(pts)), jmin]
            b = int(np.argmax(cval))
            out[iz] = cval[b]; who[iz] = inear[idx[b, jmin[b]]]
            off[iz] = float(np.hypot(pts[b, 0], pts[b, 1]))
            step /= 4.0
            c = pts[b, :2]
            o = np.array([-2, -1, 0, 1, 2]) * step
            ox, oy = np.meshgrid(o, o, indexing="ij")
            pts = np.column_stack([c[0] + ox.ravel(), c[1] + oy.ravel(),
                                   np.full(ox.size, z)])
    return out, who, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label"); ap.add_argument("gro"); ap.add_argument("xtc")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--zero-only", action="store_true", help="write the zero tests and stop")
    a = ap.parse_args()
    outdir = os.path.join(HERE, "out")

    # zero tests (known-answer controls) on this very trajectory
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        hydration.zero_test(a.gro, a.xtc, 20000.0)
        pore_radius.zero_test(a.gro, a.xtc, "protein")
    open(os.path.join(outdir, a.label + ".zero.txt"), "w").write(buf.getvalue())
    if a.zero_only:
        print(a.label, "zero tests written"); return

    u = mda.Universe(a.gro, a.xtc)
    ca314 = select_gate_ca(u)
    rings = {}
    for r in RINGS:
        s = u.select_atoms(f"resid {r} and name CA")
        assert len(s) == 4, f"resid {r} CA -> {len(s)} atoms"
        rings[r] = s
    wall = u.select_atoms("protein")
    rad = np.array([VDW.get(e, VDW_DEFAULT) for e in elements(wall)])
    ow = u.select_atoms("resname TIP3 and name OH2")
    kions = u.select_atoms("resname POT POTF")
    sfo = u.select_atoms(f"resid {SF_RESIDS} and name O")
    assert len(sfo) == 20, f"SF carbonyl O -> {len(sfo)}"
    assert len(ow) > 1000 and len(kions) > 10

    T, D, NW, SFK, RZ, WHO, OFF = [], [], [], [], [], [], []
    for i, ts in enumerate(u.trajectory):
        if i % a.stride:
            continue
        box = ts.dimensions[:3].astype(float)
        center, d314, _ = gate_center_and_diagonal(ca314.positions.astype(float), box)
        dd = [gate_center_and_diagonal(rings[r].positions.astype(float), box, check=False)[1]
              for r in RINGS]
        nw = count_gate_waters(ow.positions.astype(float), center, box)
        ref = sfo.positions[0].astype(float)
        csf = ref + minimum_image(sfo.positions.astype(float) - ref, box).mean(axis=0)
        dk = minimum_image(kions.positions.astype(float) - csf, box)
        nk = int(np.count_nonzero((np.abs(dk[:, 2]) < 9.0) &
                                  (dk[:, 0] ** 2 + dk[:, 1] ** 2 < 25.0)))
        rz, who, off = profile_with_atoms(wall.positions.astype(float), rad, center, box)
        T.append(ts.time / 1000.0); D.append(dd); NW.append(nw); SFK.append(nk)
        RZ.append(rz); WHO.append(who); OFF.append(off)
        if len(T) % 200 == 0:
            print(a.label, len(T), "frames", flush=True)

    np.savez_compressed(os.path.join(outdir, a.label + ".npz"),
                        t_ns=np.array(T), diag=np.array(D), rings=np.array(RINGS),
                        nw=np.array(NW), sfk=np.array(SFK), zs=ZS,
                        rz=np.array(RZ), who=np.array(WHO), off=np.array(OFF),
                        wall_resid=wall.resids, wall_resname=np.array(wall.resnames),
                        wall_name=np.array(wall.names))
    print(a.label, "DONE", len(T), "frames", flush=True)


if __name__ == "__main__":
    main()
