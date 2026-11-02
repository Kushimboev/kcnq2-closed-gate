#!/usr/bin/env python3
"""Figures for the Turanian manuscript from out2/summary2.json (+ out2/qc_*.json).

fig2_profiles.pdf   on-axis pore radius and axial water density (two stacked panels, shared axis)
fig3_stability.pdf  S314 diagonal and on-axis minimum radius versus time (small multiples by system)
figA1_water.pdf     water counts versus time (appendix)
Colour = system (validated palette, first three categorical slots); one line per trajectory.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "manuscript", "figures")
R = json.load(open(os.path.join(HERE, "out2", "summary2.json")))
COL = {"S0": "#2a78d6", "S2": "#eb6834", "S3": "#1baf7a"}
NAME = {"S0": "S0  (PIP$_2$)", "S2": "S2  (PIP$_2$ + retigabine)", "S3": "S3  (retigabine, no PIP$_2$)"}
SHORT = {"S0": "S0: PIP$_2$", "S2": "S2: PIP$_2$ + RTG", "S3": "S3: RTG, no PIP$_2$"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans", "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300})
RINGS = (("L318", -0.55), ("S314", 0.0), ("G310", 0.54), ("A306", 1.16), ("V302", 1.61), ("T277", 2.21), ("G281", 3.32))
W = 5.9   # in  (15 cm text width)


def smooth(x, n):
    x = np.asarray(x, float)
    if n <= 1: return x
    k = np.ones(n) / n
    return np.convolve(x, k, mode="valid")


def syslegend(ax, **kw):
    from matplotlib.lines import Line2D
    present = [s for s in COL if any(r["system"] == s for r in R.values())]
    h = [Line2D([0], [0], color=COL[s], lw=1.8) for s in present]
    ax.legend(h, [NAME[s] + f"  (n = {sum(r['system'] == s for r in R.values())})" for s in present], **kw)


# ---------------- Fig. 2
fig, (a, b) = plt.subplots(2, 1, figsize=(W, 4.9), sharex=True, gridspec_kw=dict(height_ratios=[1.15, 1], hspace=0.10))
for lab, r in R.items():
    z = np.array(r["zs"]) / 10.0
    a.plot(z, r["rc3"]["prof_mean"], color=COL[r["system"]], lw=1.3, alpha=0.9, solid_capstyle="round")
    b.plot(np.array(r["zc"]) / 10.0, r["w3_rel"], color=COL[r["system"]], lw=1.3, alpha=0.9, solid_capstyle="round")
a.axhline(0.14, color=INK2, lw=0.7); a.text(-1.37, 0.147, "0.14 nm (water)", ha="left", va="bottom", color=INK2, fontsize=7.5)
b.axhline(1.0, color=INK2, lw=0.7); b.text(2.25, 1.03, "bulk water", ha="center", va="bottom", color=INK2, fontsize=7.5)
for ax in (a, b):
    for x0, x1 in ((-0.6, 0.6), (2.0, 3.5)):
        ax.axvspan(x0, x1, color="#c9c8c3", alpha=0.22, lw=0, zorder=0)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
ymax = 0.62
a.set_ylim(0, ymax); a.set_xlim(-1.4, 3.6)
for nm, zz in RINGS:
    a.plot([zz, zz], [ymax * 0.955, ymax], color=INK2, lw=0.7, clip_on=False)
    a.text(zz, ymax * 1.012, nm, ha="center", va="bottom", fontsize=7, color=INK2, rotation=0)
for txt, xx in (("gate", 0.0), ("cavity", 1.3), ("selectivity filter", 2.75), ("cytoplasm", -1.0)):
    a.text(xx, ymax * 0.86, txt, ha="center", va="center", fontsize=8, color=INK)
a.set_ylabel("on-axis pore radius  $R$ (nm)")
b.set_ylabel("water density / bulk"); b.set_xlabel("position along the pore axis  $z$ (nm),  $z$ = 0 at the Ser314 C$\\alpha$ ring")
b.set_ylim(0, None)
syslegend(b, loc="upper left", bbox_to_anchor=(0.46, 1.0), fontsize=7.5, handlelength=1.6, labelspacing=0.3)
a.text(-0.105, 1.03, "(a)", transform=a.transAxes, fontsize=10, fontweight="bold"); b.text(-0.105, 0.97, "(b)", transform=b.transAxes, fontsize=10, fontweight="bold")
fig.savefig(os.path.join(OUT, "fig2_profiles.pdf"), metadata={"CreationDate": None, "ModDate": None}, bbox_inches="tight"); fig.savefig(os.path.join(OUT, "fig2_profiles.png"), bbox_inches="tight"); plt.close(fig)

# ---------------- Fig. 3
systems = [s for s in COL if any(r["system"] == s for r in R.values())]
fig, axs = plt.subplots(2, len(systems), figsize=(W, 4.3), sharex="col", sharey="row", gridspec_kw=dict(hspace=0.12, wspace=0.12))
for j, s in enumerate(systems):
    labs = [l for l, r in R.items() if r["system"] == s]
    for k, lab in enumerate(labs):
        r = R[lab]; t = np.array(r["t_series"]); n = 10          # 1-ns running mean (frames every 0.1 ns)
        al = 0.95 if k == 0 else 0.55
        axs[0, j].plot(smooth(t, n), smooth(np.array(r["diag314_series"]) / 10, n), color=COL[s], lw=0.9, alpha=al)
        axs[1, j].plot(smooth(t, n), smooth(r["rmin_series"], n), color=COL[s], lw=0.9, alpha=al)
    axs[0, j].set_title(SHORT[s], fontsize=8, color=INK, loc="left")
    axs[1, j].axhline(0.14, color=INK2, lw=0.7)
    axs[1, j].set_xlabel("time (ns)")
    for i in (0, 1):
        axs[i, j].grid(axis="y", color=GRID, lw=0.5); axs[i, j].set_axisbelow(True); axs[i, j].set_xlim(20, max(max(R[l]["t_series"]) for l in labs))
for j, s_ in enumerate(systems):
    tmax = max(max(R[l]["t_series"]) for l in R if R[l]["system"] == s_)
    axs[1, j].set_xticks([50, 100, 150] if tmax > 150 else [40, 60, 80])
axs[1, 0].text(196, 0.143, "0.14 nm (water)", ha="right", va="bottom", color=INK2, fontsize=7)
axs[0, 0].set_yticks([0.70, 0.75, 0.80])
axs[0, 0].set_ylabel("Ser314 C$\\alpha$ diagonal (nm)", fontsize=8); axs[1, 0].set_ylabel("$R_{\\min}$ (nm)")
axs[0, 0].set_ylim(0.70, 0.84); axs[1, 0].set_ylim(0, 0.16)
axs[0, 0].text(0.03, 0.90, "(a)", transform=axs[0, 0].transAxes, fontsize=10, fontweight="bold")
axs[1, 0].text(0.03, 0.06, "(b)", transform=axs[1, 0].transAxes, fontsize=10, fontweight="bold")
fig.savefig(os.path.join(OUT, "fig3_stability.pdf"), metadata={"CreationDate": None, "ModDate": None}, bbox_inches="tight"); fig.savefig(os.path.join(OUT, "fig3_stability.png"), bbox_inches="tight"); plt.close(fig)

# ---------------- Fig. A1 (water counts vs time)
fig, axs = plt.subplots(2, len(systems), figsize=(W, 4.3), sharex="col", sharey="row", gridspec_kw=dict(hspace=0.12, wspace=0.12))
for j, s in enumerate(systems):
    labs = [l for l, r in R.items() if r["system"] == s]
    for k, lab in enumerate(labs):
        r = R[lab]; t = np.array(r["t_series"]); n = 20
        al = 0.95 if k == 0 else 0.55
        axs[0, j].plot(smooth(t, n), smooth(r["w_cav_series"], n), color=COL[s], lw=0.9, alpha=al)
        axs[1, j].plot(smooth(t, n), smooth(r["nw_series"], n), color=COL[s], lw=0.9, alpha=al)
    axs[0, j].set_title(SHORT[s], fontsize=8, color=INK, loc="left"); axs[1, j].set_xlabel("time (ns)")
    for i in (0, 1):
        axs[i, j].grid(axis="y", color=GRID, lw=0.5); axs[i, j].set_axisbelow(True); axs[i, j].set_xlim(20, max(max(R[l]["t_series"]) for l in labs))
axs[0, 0].set_ylabel("waters in the cavity"); axs[1, 0].set_ylabel("waters at the gate")
axs[0, 0].set_ylim(0, None); axs[1, 0].set_ylim(0, None)
fig.savefig(os.path.join(OUT, "figA1_water.pdf"), metadata={"CreationDate": None, "ModDate": None}, bbox_inches="tight"); fig.savefig(os.path.join(OUT, "figA1_water.png"), bbox_inches="tight"); plt.close(fig)
# ---------------- Fig. 1 (composite of PyMOL renders, system S2)
from PIL import Image, ImageChops
def crop(p):
    im = Image.open(p).convert("RGB"); bg = Image.new("RGB", im.size, (255, 255, 255))
    bb = ImageChops.difference(im, bg).getbbox(); return im.crop(bb)
side, s314, l318 = (crop(os.path.join(OUT, f"render_S2_{k}.png")) for k in ("side", "S314", "L318"))
fig = plt.figure(figsize=(W, 4.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], wspace=0.02, hspace=0.06)
ax0 = fig.add_subplot(gs[:, 0]); ax1 = fig.add_subplot(gs[0, 1]); ax2 = fig.add_subplot(gs[1, 1])
for ax, im, lab in ((ax0, side, "(a)"), (ax1, s314, "(b)  Ser314 ring"), (ax2, l318, "(c)  Leu318 ring")):
    ax.imshow(im); ax.set_axis_off(); ax.text(0.0, 1.0, lab, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="top", ha="left", color=INK)
fig.savefig(os.path.join(OUT, "fig1_structure.pdf"), metadata={"CreationDate": None, "ModDate": None}, bbox_inches="tight", dpi=400); fig.savefig(os.path.join(OUT, "fig1_structure.png"), bbox_inches="tight", dpi=300); plt.close(fig)

# ---------------- Fig. def (definition of the on-axis radius; check against HOLE), neutral ink: colour stays with the systems
import warnings, MDAnalysis as mda
from scipy.spatial import cKDTree
warnings.filterwarnings("ignore")
HC = json.load(open(os.path.join(HERE, "out2", "hole_check.json")))
u = mda.Universe(os.path.join(HERE, "snap", "S2_eq100_100ns.pdb")); pr = u.select_atoms("protein"); P_ = pr.positions.astype(float)
VDW = dict(H=1.00, C=1.85, N=1.75, O=1.65, S=2.00); sg = np.array([VDW[n[0]] for n in pr.names]); tree = cKDTree(P_)
def best(z, rc, step=0.1):
    g = np.arange(-rc, rc + 1e-9, step); X, Y = np.meshgrid(g, g); m = np.hypot(X, Y) <= rc; Q = np.c_[X[m], Y[m], np.full(m.sum(), z)]
    d, i = tree.query(Q, k=60); c = (d - sg[i]).min(1); j = int(np.argmax(c)); return c[j], Q[j]
planes = np.arange(-6, 6.01, 0.5); z0 = planes[int(np.argmin([best(z, 3.0)[0] for z in planes]))]
r_ax, c_ax = best(z0, 3.0, 0.03); r_fr, c_fr = best(z0, 9.0, 0.05)
assert r_fr > r_ax and np.hypot(*c_fr[:2]) > 3.8, "the sphere of the exhaustive search lies inside the Ser314 CA ring in this frame: rewrite the caption of Fig. 1"
fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.95), gridspec_kw=dict(width_ratios=[1, 1], wspace=0.34))
from matplotlib.patches import Circle
for p, s_, nm, ri in zip(P_, sg, pr.names, pr.resids):
    dz = p[2] - z0
    if abs(dz) < s_ and np.hypot(p[0], p[1]) < 13:
        gate = ri in (314, 318); a.add_patch(Circle((p[0] / 10, p[1] / 10), np.sqrt(s_ ** 2 - dz ** 2) / 10, fc="#a9a8a3" if gate else "#dddcd7", ec="white", lw=0.3, zorder=1))
for rc_, ls_ in ((0.3, "-"), (0.9, "--")):
    a.add_patch(Circle((0, 0), rc_, fc="none", ec=INK, lw=0.7, ls=ls_, zorder=3))
a.add_patch(Circle((c_ax[0] / 10, c_ax[1] / 10), r_ax / 10, fc=INK, ec=INK, lw=0.5, zorder=4)); a.add_patch(Circle((c_fr[0] / 10, c_fr[1] / 10), r_fr / 10, fc="white", ec=INK, lw=1.0, zorder=4))
box_ = dict(fc="white", ec="none", pad=1.2)
a.annotate("on-axis sphere", (c_ax[0] / 10, c_ax[1] / 10), (-0.62, -1.08), fontsize=7.2, color=INK, ha="center", va="center", bbox=box_, arrowprops=dict(arrowstyle="-", lw=0.6, color=INK), zorder=6)
a.annotate("exhaustive search", (c_fr[0] / 10, c_fr[1] / 10), (0.56, 1.08), fontsize=7.2, color=INK, ha="center", va="center", bbox=box_, arrowprops=dict(arrowstyle="-", lw=0.6, color=INK), zorder=6)
a.set_title("solid: $r_c$ = 0.3 nm;  dashed: $r_c$ = 0.9 nm", fontsize=7.5, color=INK, loc="right", pad=3)
a.set_xlim(-1.2, 1.2); a.set_ylim(-1.2, 1.2); a.set_aspect("equal"); a.set_xlabel("$x$ (nm)"); a.set_ylabel("$y$ (nm)"); a.set_xticks([-1, 0, 1]); a.set_yticks([-1, 0, 1])
Fr = [r for v in HC["frames"].values() for r in v]; hx = np.array([r["hole"] for r in Fr]); ax_ = np.array([r["axis03"] for r in Fr]); fr_ = np.array([r["free09"] for r in Fr])
b.plot([0, .13], [0, .13], color=INK2, lw=0.7, zorder=1)
b.scatter(hx, fr_, s=13, fc="white", ec=INK2, lw=0.6, zorder=2); b.scatter(hx, ax_, s=11, fc=INK, ec="white", lw=0.25, zorder=3)
b.text(0.035, 0.112, "$r_c$ = 0.9 nm (open)", fontsize=7.5, color=INK); b.text(0.062, 0.026, "$r_c$ = 0.3 nm (filled)", fontsize=7.5, color=INK)
b.set_xlim(0, 0.13); b.set_ylim(0, 0.13); b.set_aspect("equal"); b.set_xticks([0, .05, .10]); b.set_yticks([0, .05, .10]); b.grid(color=GRID, lw=0.5); b.set_axisbelow(True)
b.set_xlabel("$R_{\\min}$ from HOLE (nm)"); b.set_ylabel("$R_{\\min}$ from Eq. (1) (nm)")
a.text(-0.2, 1.02, "(a)", transform=a.transAxes, fontsize=10, fontweight="bold"); b.text(-0.2, 1.02, "(b)", transform=b.transAxes, fontsize=10, fontweight="bold")
fig.savefig(os.path.join(OUT, "fig_def.pdf"), metadata={"CreationDate": None, "ModDate": None}, bbox_inches="tight"); fig.savefig(os.path.join(OUT, "fig_def.png"), bbox_inches="tight"); plt.close(fig)
json.dump(dict(z_plane_nm=z0 / 10, R_on_axis=r_ax / 10, R_free=r_fr / 10, off_free=float(np.hypot(*c_fr[:2]) / 10)), open(os.path.join(HERE, "out2", "fig_def_frame.json"), "w"), indent=1)

# ---------------- Fig. water (the Ser314 ring without and with water; the three crossing molecules of S0-d)
import pickle
GW_ = json.load(open(os.path.join(HERE, "out2", "gate_water.json")))["REP-S0-wet1"]; D_ = pickle.load(open(os.path.join(HERE, "out2", "wat_REP-S0-wet1.pkl"), "rb"))
mols = sorted({c["mol"] for c in GW_["crossings"]}, key=lambda m_: min(c["t_in"] for c in GW_["crossings"] if c["mol"] == m_)); assert len(mols) == 3
t_ = D_["t"]; sel = (t_ >= 33) & (t_ <= 46)
Z = {m_: np.full(len(t_), np.nan) for m_ in mols}
for i in np.where(sel)[0]:
    ids, z_ = D_["f"][i][:2]
    for m_ in mols:
        k = np.where(ids == m_)[0]
        if len(k): Z[m_][i] = z_[k[0]] / 10
fig = plt.figure(figsize=(W, 5.1)); gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1], hspace=0.20, wspace=0.03)
ax0 = fig.add_subplot(gs[0, 0]); ax1 = fig.add_subplot(gs[0, 1]); ax2 = fig.add_subplot(gs[1, :])
for ax, f_, lab in ((ax0, "render_water_S3a_100ns.png", "(a)  no water, S3-a"), (ax1, "render_water_S0a_45ns.png", "(b)  one water molecule, S0-a")):
    ax.imshow(crop(os.path.join(OUT, f_))); ax.set_axis_off(); ax.set_title(lab, fontsize=9, fontweight="bold", color=INK, loc="left", pad=3)
for lo_, hi_, tx in ((-0.5, 0.0, "between the rings"),):
    ax2.axhspan(lo_, hi_, color="#c9c8c3", alpha=0.3, lw=0, zorder=0)
for zz, nm in ((-0.55, "Leu318"), (0.0, "Ser314")):
    ax2.axhline(zz, color=INK2, lw=0.7, zorder=1); ax2.text(46.1, zz, nm, fontsize=7.5, color=INK, va="center", ha="left")
for m_, c_, ls_, n_ in zip(mols, (INK, "#6d6c68", "#a2a19c"), ("-", "-", "-"), ("1", "2", "3")):
    ax2.plot(t_, Z[m_], color=c_, lw=1.1, zorder=3, solid_capstyle="round")
    j0 = np.where(~np.isnan(Z[m_]) & (t_ >= min(c["t_in"] for c in GW_["crossings"] if c["mol"] == m_) - 0.6))[0][0]
    ax2.text(t_[j0] - {"1": 0.08, "2": 0.08, "3": 0.55}[n_], {"1": -1.05, "2": -0.85, "3": -1.25}[n_], f"molecule {n_}", fontsize=7, color=INK, ha="right", va="center", zorder=5)
ax2.set_xlim(33, 46); ax2.set_ylim(-1.5, 1.3); ax2.grid(axis="y", color=GRID, lw=0.5); ax2.set_axisbelow(True); ax2.set_xlabel("time in trajectory S0-d (ns)"); ax2.set_ylabel("$z$ of the water oxygen (nm)")
ax2.text(33.2, 1.08, "cavity", fontsize=8, color=INK); ax2.text(33.2, -1.38, "cytoplasm", fontsize=8, color=INK); ax2.text(-0.095, 1.02, "(c)", transform=ax2.transAxes, fontsize=10, fontweight="bold")
fig.savefig(os.path.join(OUT, "fig_gatewater.pdf"), metadata={"CreationDate": None, "ModDate": None}, bbox_inches="tight", dpi=400); fig.savefig(os.path.join(OUT, "fig_gatewater.png"), bbox_inches="tight", dpi=300); plt.close(fig)
print("figures written to", os.path.normpath(OUT))
