"""
angular_fastsim_breakdown.py
============================
Selected MC photons in theta and phi, split by how they were simulated:
  Geant4       : photons in the Geant4 SIM file (full-sim reference, all photons)
  CC3 fast-sim : photons in the CC3 SIM file with MCParticle.isHandledByFastSim() == True
  CC3 fallback : photons in the CC3 SIM file with the flag == False
                 (CaloClouds did not take them; Geant4 simulated them inside the CC3 run)

Physics question
  Where does CaloClouds actually take over from Geant4 in theta and phi,
  and where does the ISR peak sit relative to that region?

Selection (identical to angular_s4_s5.py, no parent cut on the plots)
  MCParticles with PDG == 22, generatorStatus == 1, E >= 10 GeV.

Read order per event
  1. Open the "events" frame, read MCParticles.
  2. Keep photons: PDG 22, genStatus 1, E >= 10 GeV (getEnergy()).
  3. theta, phi from edm4hep.utils.p4(p).
  4. Flag: p.isHandledByFastSim() (Key4hep nightlies only).
  5. Parent: True if any DIRECT parent has |PDG| == 11 (used only for the printed checks).

Plots (raw counts, linear y unless --log, 7.5 deg bins, no boundary lines)
  theta and phi, three curves:
    Geant4 (blue), CC3 fast-sim (red), CC3 fallback to Geant4 (grey dashed).
  The legend sits in extra headroom above the data (upper right), so it never
  covers a peak.
  Expected: both files hold the same generator photons, so
  (CC3 fast-sim + CC3 fallback) == Geant4 in every bin. Printed as a check.

Printed checks (no effect on plots)
  - flag count in the Geant4 file (expected 0)
  - fast-sim fraction for e+- parent vs non-e+- photons, overall and inside an
    approximate CC3 acceptance region (theta: barrel 37.5-142.5, endcap 10-25 and
    155-170 deg; phi-crack veto: dead zone [C, C+8] deg for C = -157.5 + 45k).
    Assumption: the phi veto applies to barrel and endcap alike.
    The sign convention of the phi veto is taken from setup.md, not re-derived here.
  - purity of the in-acceptance sample with and without the e+- parent cut
  - the first few fast-simmed e+- parent photons, for inspection

Output
  ~/thesis-ml-sim/plots/<GROUP>/fastsim_breakdown_<tag>/
    fastsim_theta.png, fastsim_phi.png, photons_g4.npz, photons_cc3.npz

Run
  source ~/source.sh        # nightlies needed for isHandledByFastSim()
  python3 angular_fastsim_breakdown.py -n 1000
  python3 angular_fastsim_breakdown.py
  Options: --log, --group NAME
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
GROUP    = "s4-s5-angular"

E_MIN = 10.0
BIN_W = 7.5
BINS_PHI   = np.arange(-180.0, 180.0 + BIN_W / 2, BIN_W)
BINS_THETA = np.arange(0.0,    180.0 + BIN_W / 2, BIN_W)

C_G4, C_FAST, C_FALL = "#2166ac", "#d6604d", "#7f7f7f"

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
OUT = os.path.join(PLOTS, args.group, f"fastsim_breakdown_{tag}")
os.makedirs(OUT, exist_ok=True)
print(f"Output folder: {OUT}")


def in_theta_acc(t):
    return ((t >= 37.5) & (t <= 142.5)) | ((t >= 10) & (t <= 25)) | ((t >= 155) & (t <= 170))


def in_phi_crack(ph):
    dead = np.zeros(len(ph), dtype=bool)
    for k in range(8):
        c = -157.5 + 45.0 * k
        dead |= ((ph - c) % 360.0) <= 8.0
    return dead


def load(label, path, nmax):
    print(f"\n{label}: {path}")
    th, ph, en, fast, epar = [], [], [], [], []
    n_ev = 0
    for event in root_io.Reader(path).get("events"):
        if nmax is not None and n_ev >= nmax:
            break
        n_ev += 1
        for p in event.get("MCParticles"):
            if p.getPDG() != 22 or p.getGeneratorStatus() != 1 or p.getEnergy() < E_MIN:
                continue
            try:
                flag = bool(p.isHandledByFastSim())
            except AttributeError:
                sys.exit("isHandledByFastSim() not found. Run 'source ~/source.sh' (Key4hep nightlies).")
            p4 = edm4hep.utils.p4(p)
            th.append(np.degrees(float(p4.theta())))
            ph.append(np.degrees(float(p4.phi())))
            en.append(p.getEnergy())
            fast.append(flag)
            epar.append(any(abs(q.getPDG()) == 11 for q in p.getParents()))
        if n_ev % 20000 == 0:
            print(f"  processed {n_ev} events", flush=True)
    d = {"theta": np.array(th), "phi": np.array(ph), "energy": np.array(en),
         "fast": np.array(fast, dtype=bool), "epar": np.array(epar, dtype=bool)}
    N = len(d["theta"])
    print(f"  Events: {n_ev}")
    print(f"  Photons GS=1, E>={E_MIN:g} GeV: {N}")
    print(f"  Flagged fast-sim: {int(d['fast'].sum())}  |  fallback (flag False): {int((~d['fast']).sum())}")
    np.savez(os.path.join(OUT, f"photons_{label.lower()}.npz"), **d)
    return d


def report(d):
    ta = in_theta_acc(d["theta"])
    fa = ta & ~in_phi_crack(d["phi"])
    regions = [("all angles", np.ones(len(ta), dtype=bool)),
               ("theta acceptance only", ta),
               ("theta acc + phi-crack veto", fa)]
    for name, region in regions:
        for lab, sel in (("e+- parent", d["epar"]), ("not e+- parent", ~d["epar"])):
            s = region & sel
            n, nf = int(s.sum()), int((s & d["fast"]).sum())
            print(f"  {name:28s} {lab:15s} N={n:7d}  fast-simmed={nf:7d}  ({100 * nf / max(n, 1):.2f}%)")
    print("\n  Purity of the in-acceptance sample (theta acc + phi-crack veto):")
    for lab, sel in (("with e+- parent cut (not e+- only)", ~d["epar"]), ("without the cut (all photons)", np.ones(len(fa), dtype=bool))):
        s = fa & sel
        n, nf = int(s.sum()), int((s & d["fast"]).sum())
        pur = nf / max(n, 1)
        err = np.sqrt(pur * (1 - pur) / max(n, 1))
        print(f"    {lab:36s} N={n:7d}  fast-simmed={nf:7d}  purity={100 * pur:.2f} +- {100 * err:.2f}%")
    n_all, n_e = int(fa.sum()), int((fa & d["epar"]).sum())
    print(f"    e+- parent photons removed by the cut: {n_e} of {n_all} ({100 * n_e / max(n_all, 1):.1f}%)")


def plot3(key, bins, xlabel, title, fname, g4, cc3):
    fast, fall = cc3["fast"], ~cc3["fast"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(g4[key], bins=bins, histtype="step", linewidth=2.5, color=C_G4,
            label=f"Geant4  (N={len(g4[key]):,})")
    ax.hist(cc3[key][fast], bins=bins, histtype="step", linewidth=2.5, color=C_FAST,
            label=f"CC3 fast-sim  (N={int(fast.sum()):,})")
    ax.hist(cc3[key][fall], bins=bins, histtype="step", linewidth=2.5, color=C_FALL,
            linestyle="--", label=f"CC3 fallback to G4  (N={int(fall.sum()):,})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Photons / {BIN_W:g}° bin")
    ax.set_title(title)
    if args.log:
        ax.set_yscale("log")
    y0, y1 = ax.get_ylim()
    if args.log:
        ax.set_ylim(y0, y1 * (y1 / y0) ** 0.45)
    else:
        ax.set_ylim(0, y1 * 1.55)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


g4  = load("G4",  G4_FILE,  args.n)
cc3 = load("CC3", CC3_FILE, args.n)

print("\nCheck 1: Geant4 file flagged fast-sim (expected 0): ", int(g4["fast"].sum()))

print("\nCheck 2: CC3 fast-sim + CC3 fallback == Geant4, per bin?")
for key, bins in (("theta", BINS_THETA), ("phi", BINS_PHI)):
    h1, _ = np.histogram(g4[key], bins=bins)
    h2, _ = np.histogram(cc3[key], bins=bins)
    print(f"  {key}: identical = {np.array_equal(h1, h2)}  (max |diff| per bin = {np.abs(h1 - h2).max()})")

print("\nCheck 3: fast-sim fraction, CC3 file")
report(cc3)

ex = np.where(cc3["fast"] & cc3["epar"])[0][:10]
print(f"\nFast-simmed photons with e+- parent: {int((cc3['fast'] & cc3['epar']).sum())}")
for i in ex:
    print(f"  theta={cc3['theta'][i]:7.2f}  phi={cc3['phi'][i]:8.2f}  E={cc3['energy'][i]:7.2f}")

print("\nPlotting...")
TH, PH = r"Photon polar angle $\theta$ [deg]", r"Photon azimuth $\phi$ [deg]"
plot3("theta", BINS_THETA, TH, "Photon θ by simulation mode", "fastsim_theta.png", g4, cc3)
plot3("phi",   BINS_PHI,   PH, "Photon φ by simulation mode", "fastsim_phi.png",   g4, cc3)
print(f"\nSaved to {OUT}")
