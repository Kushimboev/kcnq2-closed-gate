# The closed activation gate of KCNQ2: inputs, analysis code and per-trajectory values

Companion to the manuscript *The closed activation gate of the KCNQ2 potassium channel: on-axis pore radius and hydration from unbiased molecular dynamics*, S. Qoshimboyev and J. Razzokov, Turanian Journal (under review, manuscript UR030102).

Unbiased all-atom molecular dynamics of closed human KCNQ2 (PDB 7CR2, residues 70-330): nine trajectories, 1.5 us, three compositions.

| system | composition |
|---|---|
| S0 | PIP2, no retigabine (derived from S2 by deleting the ligand) |
| S2 | PIP2 and retigabine |
| S3 | retigabine, no PIP2 (separate build) |

Software: GROMACS 2024.3, CHARMM36m / CHARMM36 / CHARMM TIP3P, CGenFF for retigabine, CHARMM-GUI, Python with MDAnalysis, HOLE 2.3.1, PyMOL.

## Contents

- `inputs/<system>/equilibration/`: parameter files of the equilibration stages.
- `inputs/<system>/production/`: parameter file and start coordinates of trajectory a.
- `inputs/<system>/replicas/<id>/`: parameter file and start coordinates of each replica (trajectories b-d).
- `inputs/<system>/topol.top`, `index.ndx`, `inputs/toppar/`: topology (point the include lines at `inputs/toppar/`).
- `inputs/replicas_all.csv`: start frame, water count in the start frame and velocity seed of each replica.
- `analysis/`: the scripts used for the paper; the entry points are listed below.
- `data/`: per-trajectory values and `numbers.tex`, which holds every number quoted in the paper. The scripts read and write `analysis/out2/`; `data/` is a copy of those results without the per-frame series.

## Trajectory names

| paper | file label | paper | file label |
|---|---|---|---|
| S0-a | S0-eq100 | S3-a | S3-eq100 |
| S0-b | REP-S0-wet2 | S3-b | REP-S3-wet1 |
| S0-c | REP-S0-dry1 | S3-c | REP-S3-wet2 |
| S0-d | REP-S0-wet1 | S3-d | REP-S3-dry2 |
| S2-a | S2-eq100 | | |

The labels wet and dry record whether the start frame had water molecules in the counting cylinder around the gate (`inputs/replicas_all.csv`).

## Analysis scripts

| script | purpose |
|---|---|
| `pore_axis.py` | on-axis radius, Eq. (1) of the paper; water and ion counts along the axis |
| `hole_check.py` | comparison with the HOLE program |
| `gate_water.py` | single water molecules through the gate, water chains |
| `gate_hbonds.py` | hydrogen bonds and clearance of those molecules |
| `replica_memory.py` | autocorrelation times; divergence of the replicas from their start frames |
| `composition.py` | ion counts, charges, leaflets, filter sites |
| `qc_rmsd.py`, `sidechain_diag.py`, `closed_state.py` | stability, side-chain distances, bulk zero test |
| `summarize2.py`, `make_tables.py`, `make_figures.py` | aggregates, tables and macros, figures |
| `export_snapshot.py`, `render_gate.py`, `render_water.py` | exported frames and PyMOL renders |

Set `KCNQ2_TRAJ` to the folder that holds the trajectories (`trajectories.py` lists the expected paths).

## Not included

Trajectories and per-frame data; they are available from the corresponding author on reasonable request (qushimboyevshavkatjon@gmail.com).

## License

MIT, see `LICENSE`.
