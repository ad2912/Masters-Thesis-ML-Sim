"""
angular_s4_s5.py
================
MC-truth photon angular distributions from the 100k SIM files (G4 and CC3).
Feeds slide S4 (sanity check) and slide S5 (parent breakdown).

Physics question
  Where in theta and phi do the photons that CaloClouds could fast-simulate
  actually sit, and which parent particle do they come from?

Selection (identical for every plot)
  MCParticles with PDG == 22, generatorStatus == 1, E >= 10 GeV.
  No parent, theta or phi cut. ISR photons are deliberately kept in.

Read order per event
  1. Open the "events" frame, read the MCParticles collection.
  2. Skip anything that is not a photon, or has E < 10 GeV (getEnergy()).
  3. Skip photons with generatorStatus != 1.
  4. theta, phi from the four-vector built by edm4hep.utils.p4(p)
     (momentum + mass; the mass does not affect the angles).
  5. Parent category from the DIRECT parents (getParents(), one level up):
       pi0   : any parent has |PDG| == 111
       e+-   : any parent has |PDG| == 11   (ISR, and any FSR off final e+-)
       other : everything else (e.g. tau parent, secondaries)
     Priority: pi0 first, then e+-, then other.
     Note: "pi0 daughter" means a daughter of ANY pi0, not only pi0s from tau.

Plots (raw counts, linear y unless --log, no boundary lines)
  S4: theta and phi, G4 (blue solid) and CC3 (red dashed), same selection.
      Both files come from the same generator events, so the curves are
      expected to coincide. The script prints whether they are identical.
  S5: theta and phi coloured by parent category, one figure per simulation.

Binning
  7.5 deg for theta and phi. 7.5 divides the 45 deg module spacing into 6
  bins and puts the crack centres (-157.5 + 45k) and the barrel edges
  (37.5, 142.5 deg) on bin edges.

Output
  ~/thesis-ml-sim/plots/<GROUP>/angular_s4_s5_<tag>/
    s4_theta.png, s4_phi.png,
    s5_g4_theta.png, s5_g4_phi.png, s5_cc3_theta.png, s5_cc3_phi.png,
    photons_g4.npz, photons_cc3.npz  (per-photon theta, phi, category)

Run
  source ~/source.sh
  python3 angular_s4_s5.py -n 1000     # test on 1k events per file
  python3 angular_s4_s5.py             # full 100k
  Options: --log (log y-axis), --group NAME (output group folder)
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio import root_io
import edm4hep

G4_FILE  = "/data/dust/user/alimuham/thesis/sim/tau-pi0-geant4-100kevents-sim.edm4hep.root"
CC3_FILE = "/data/dust/user/alimuham/thesis/sim/tau-pi0-caloclouds-100kevents-sim.edm4hep.root"
PLOTS    = os.path.expanduser("~/thesis-ml-sim/plots")
GROUP    = "initial-theta-phi"

E_MIN = 10.0
BIN_W = 7.5
BINS_PHI   = np.arange(-180.0, 180.0 + BIN_W / 2, BIN_W)
BINS_THETA = np.arange(0.0,    180.0 + BIN_W / 2, BIN_W)

C_G4, C_CC3 = "#2166ac", "#d6604d"
CATS = [
    (r"$\pi^0$ daughter", "#1b7837", "-"),
    (r"$e^\pm$ parent",   "#e08214", "--"),
    ("other",             "#7f7f7f", "-."),
]

plt.rcParams.update({
    "font.family": "serif", "font.size": 15, "axes.labelsize": 17,
    "axes.titlesize": 17, "legend.fontsize": 13, "xtick.labelsize": 14,
    "ytick.labelsize": 14, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150,
})

ap = argparse.ArgumentParser()
ap.add_argument("-n", type=int, default=None, help="max events per file")
ap.add_argument("--log", action="store_true", help="log y-axis")
ap.add_argument("--group", default=GROUP, help="group folder inside plots/")
args = ap.parse_args()

for f in (G4_FILE, CC3_FILE):
    if not os.path.exists(f):
        sys.exit(f"Input file not found: {f}")

tag = "100k" if args.n is None else (f"{args.n // 1000}k" if args.n % 1000 == 0 else str(args.n))
OUT = os.path.join(PLOTS, args.group, f"angular_s4_s5_{tag}")
os.makedirs(OUT, exist_ok=True)
print(f"Output folder: {OUT}")


def load(label, path, nmax):
    print(f"\n{label}: {path}")
    th, ph, cat = [], [], []
    n_ev = n_any_status = 0
    for event in root_io.Reader(path).get("events"):
        if nmax is not None and n_ev >= nmax:
            break
        n_ev += 1
        for p in event.get("MCParticles"):
            if p.getPDG() != 22 or p.getEnergy() < E_MIN:
                continue
            n_any_status += 1
            if p.getGeneratorStatus() != 1:
                continue
            p4 = edm4hep.utils.p4(p)
            th.append(np.degrees(float(p4.theta())))
            ph.append(np.degrees(float(p4.phi())))
            parents = {abs(q.getPDG()) for q in p.getParents()}
            cat.append(0 if 111 in parents else (1 if 11 in parents else 2))
        if n_ev % 20000 == 0:
            print(f"  processed {n_ev} events", flush=True)
    d = {"theta": np.array(th), "phi": np.array(ph), "cat": np.array(cat, dtype=int)}
    N = len(d["theta"])
    print(f"  Events: {n_ev}")
    print(f"  Photons GS=1, E>={E_MIN:g} GeV: {N}  ({N / max(n_ev, 1):.4f} per event)")
    print(f"  Photons E>={E_MIN:g} GeV, any genStatus: {n_any_status}")
    for c, (name, _, _) in enumerate(CATS):
        n = int((d["cat"] == c).sum())
        print(f"    {name:18s}: {n:7d}  ({100 * n / max(N, 1):.1f}%)")
    np.savez(os.path.join(OUT, f"photons_{label.lower()}.npz"), **d)
    return d


def finish(fig, ax, xlabel, title, fname):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Photons / {BIN_W:g}° bin")
    ax.set_title(title)
    if args.log:
        ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


def plot_s4(a_g4, a_cc3, bins, xlabel, title, fname):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(a_g4, bins=bins, histtype="step", linewidth=2.5, color=C_G4,
            label=f"Geant4  (N={len(a_g4):,})")
    ax.hist(a_cc3, bins=bins, histtype="step", linewidth=2.5, color=C_CC3,
            linestyle="--", label=f"CaloClouds3  (N={len(a_cc3):,})")
    finish(fig, ax, xlabel, title, fname)


def plot_s5(d, key, bins, xlabel, title, fname):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for c, (name, col, ls) in enumerate(CATS):
        m = d["cat"] == c
        if m.sum() == 0:
            continue
        ax.hist(d[key][m], bins=bins, histtype="step", linewidth=2.5, color=col,
                linestyle=ls, label=f"{name}  (N={int(m.sum()):,})")
    finish(fig, ax, xlabel, title, fname)


g4  = load("G4",  G4_FILE,  args.n)
cc3 = load("CC3", CC3_FILE, args.n)

print("\nG4 vs CC3 histograms identical?")
for key, bins in (("theta", BINS_THETA), ("phi", BINS_PHI)):
    h1, _ = np.histogram(g4[key], bins=bins)
    h2, _ = np.histogram(cc3[key], bins=bins)
    print(f"  {key}: identical = {np.array_equal(h1, h2)}  (max |diff| per bin = {np.abs(h1 - h2).max()})")

print("\nPlotting...")
TH, PH = r"Photon polar angle $\theta$ [deg]", r"Photon azimuth $\phi$ [deg]"
plot_s4(g4["theta"], cc3["theta"], BINS_THETA, TH, "Photon θ distribution", "s4_theta.png")
plot_s4(g4["phi"],   cc3["phi"],   BINS_PHI,   PH, "Photon φ distribution", "s4_phi.png")
for name, d, f in (("Geant4", g4, "g4"), ("CaloClouds3", cc3, "cc3")):
    plot_s5(d, "theta", BINS_THETA, TH, f"Photon θ by parent: {name}", f"s5_{f}_theta.png")
    plot_s5(d, "phi",   BINS_PHI,   PH, f"Photon φ by parent: {name}", f"s5_{f}_phi.png")

print(f"\nSaved to {OUT}")
