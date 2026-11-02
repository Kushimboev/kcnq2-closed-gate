#!/usr/bin/env python3
"""The single list of unbiased trajectories used by the Turanian paper (all analysis scripts import it).

label  : file label          name : name used in the paper (system + letter)
seed_ns: time of the frame of trajectory "a" from which a replica was started (inputs/replicas_all.csv)
A trajectory enters the paper when out2/<label>.npz exists.
"""
import os
B = os.environ.get("KCNQ2_TRAJ", ".")   # set KCNQ2_TRAJ to the folder that holds the trajectories (not in this repository)
S = B + "/systems/"
HERE = os.path.dirname(os.path.abspath(__file__))


def _rep(r):
    return (f"{B}/replicas/{r}/rep.gro", f"{B}/replicas/{r}/ship.xtc")


ALL = [  # label, name, (gro, xtc), length ns, seed ns
    ("S0-eq100", "S0-a", (S + "S0_noRTG_PIP2/prod_eq100/eq100.gro", S + "S0_noRTG_PIP2/prod_eq100/eq100.xtc"), 100, None),
    ("REP-S0-wet2", "S0-b", _rep("REP-S0-wet2"), 200, 100.0),
    ("REP-S0-dry1", "S0-c", _rep("REP-S0-dry1"), 200, 81.3),
    ("REP-S0-wet1", "S0-d", _rep("REP-S0-wet1"), 200, 85.0),
    ("S2-eq100", "S2-a", (S + "S2_RTG_PIP2/prod_eq100/eq100.gro", S + "S2_RTG_PIP2/prod_eq100/eq100.xtc"), 100, None),
    ("S3-eq100", "S3-a", (S + "S3_RTG_noPIP2/prod_eq100/eq100.gro", S + "S3_RTG_noPIP2/prod_eq100/eq100.xtc"), 100, None),
    ("REP-S3-wet1", "S3-b", _rep("REP-S3-wet1"), 200, 86.8),
    ("REP-S3-wet2", "S3-c", _rep("REP-S3-wet2"), 200, 89.8),
    ("REP-S3-dry2", "S3-d", _rep("REP-S3-dry2"), 200, 100.0),
]
FILES = {l: f for l, n, f, t, s in ALL}
ORDER = [l for l, n, f, t, s in ALL if os.path.exists(os.path.join(HERE, "out2", l + ".npz"))]
NAME = {l: n for l, n, f, t, s in ALL}
LENGTH = {l: t for l, n, f, t, s in ALL}
SEED = {l: s for l, n, f, t, s in ALL}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(*FILES[sys.argv[1]])
    else:
        for l in ORDER: print(l, NAME[l])
