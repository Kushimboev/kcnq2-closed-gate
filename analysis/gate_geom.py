#!/usr/bin/env python3
"""Single definition of the gate geometry, imported by the analysis scripts."""
import numpy as np

GATE_RESID = 314                    # Ser314 CA atoms define the origin (four chains)
SF_RESIDS = "277:281"               # selectivity filter TIGYG
CYL_HALF_Z = 5.0                    # counting cylinder for N_g: |z| < 5.0 A
CYL_R = 6.0                         # counting cylinder for N_g: r < 6.0 A
GATE_MIN_A, GATE_MAX_A = 4.0, 20.0  # physically plausible range of the diagonal; outside it a PBC error is assumed
KNN = 60                            # neighbours examined per grid point


def minimum_image(d, box):
    """d: (..., 3) displacement, box: (3,) orthorhombic edges -> minimum-image displacement."""
    return d - box * np.round(d / box)


def gate_center_and_diagonal(ca_pos, box, check=True):
    """Four Ser314 CA positions -> (centre, mean diagonal, diagonal pair). The diagonal pairing is the longest of the three possible."""
    ca_pos = np.asarray(ca_pos, dtype=float)
    if ca_pos.shape != (4, 3):
        raise SystemExit(f"ERROR: the gate needs 4 CA atoms, got {ca_pos.shape}. Use a .gro or .pdb topology (residue numbering).")
    ref = ca_pos[0]
    unwrapped = ref + minimum_image(ca_pos - ref, box)
    center = unwrapped.mean(axis=0)
    d = np.linalg.norm(unwrapped[:, None, :] - unwrapped[None, :, :], axis=-1)
    pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
    (i, j) = pairings[int(np.argmax([d[a] + d[b] for a, b in pairings]))]
    diag = 0.5 * (d[i] + d[j])
    if check and not (GATE_MIN_A < diag < GATE_MAX_A):
        raise SystemExit(f"ERROR: gate diagonal {diag:.1f} A is outside {GATE_MIN_A}-{GATE_MAX_A} A (periodic boundary problem?).")
    return center, diag, (i, j)


def select_gate_ca(u):
    """Ser314 CA atoms; the count is always checked."""
    ca = u.select_atoms(f"resid {GATE_RESID} and name CA")
    if len(ca) != 4:
        raise SystemExit(f"ERROR: 'resid {GATE_RESID} and name CA' -> {len(ca)} atoms (must be 4). Use a .gro or .pdb topology.")
    return ca


def count_gate_waters(ow_pos, center, box):
    """Water oxygens in the counting cylinder (minimum image)."""
    d = minimum_image(np.asarray(ow_pos, dtype=float) - center, box)
    inz = np.abs(d[:, 2]) < CYL_HALF_Z
    r2 = d[inz, 0] ** 2 + d[inz, 1] ** 2
    return int(np.count_nonzero(r2 < CYL_R ** 2))


def elements(ag):
    """Element from a CHARMM atom name (.gro has no element column)."""
    return np.array(["C" if n.strip()[:1].isdigit() else n.strip()[0].upper() for n in ag.names])
