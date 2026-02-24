"""
particle_histos.py
==================
Simple histograms comparing PFO counts and energies between
Geant4 and CaloClouds3 for different particle types.

Particles we look at:
  PDG 22  = photon
  PDG 11  = electron
  PDG -11 = positron
  PDG 13  = muon
  PDG 211 = charged pion
  PDG 2112= neutron (neutral, no track)

Run from thesis-ml-sim/:
  python3 analysis/particle_histos.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from podio import root_io
except ImportError:
    print("Source your Key4hep environment first: source ~/source.sh")
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECO_DIR = os.path.join(BASE, "results", "reco")
PLOT_DIR = os.path.join(BASE, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

G4_FILE  = os.path.join(RECO_DIR, "tau_geant4_REC.edm4hep.root")
CC3_FILE = os.path.join(RECO_DIR, "tau_caloclouds_REC.edm4hep.root")

# ── Particle types to study ────────────────────────────────────────────────────
# Each entry: (PDG code, label for plots)
PARTICLES = [
    (22,   "Photon (γ)"),
    (11,   "Electron (e-)"),
    (-11,  "Positron (e+)"),
    (13,   "Muon (μ-)"),
    (-13,  "Muon (μ+)"),
    (211,  "Charged pion (π+)"),
    (-211, "Charged pion (π-)"),
]

# ── Data extraction ────────────────────────────────────────────────────────────

def extract_all_pfos(filepath):
    """
    For each event, collect energy and PDG of every PFO.
    Returns dict: pdg_code -> list of energies
    Also returns per-event multiplicity for each PDG.
    """
    reader = root_io.Reader(filepath)

    # Store energies per PDG
    energies = {}   # pdg -> [energy, energy, ...]
    counts   = {}   # pdg -> [n_per_event, ...]

    n_events = 0
    for event in reader.get("events"):
        n_events += 1
        pfos = event.get("PandoraPFOs")

        # count per event for each PDG we care about
        event_counts = {pdg: 0 for pdg, _ in PARTICLES}

        for pfo in pfos:
            pdg = pfo.getPDG()
            e   = pfo.getEnergy()

            if pdg not in energies:
                energies[pdg] = []
                counts[pdg]   = []

            energies[pdg].append(e)
            if pdg in event_counts:
                event_counts[pdg] += 1

        for pdg, _ in PARTICLES:
            if pdg not in counts:
                counts[pdg] = []
            counts[pdg].append(event_counts.get(pdg, 0))

    print(f"  {n_events} events from {os.path.basename(filepath)}")
    return energies, counts, n_events


print("── Geant4 ──")
g4_energies, g4_counts, g4_n = extract_all_pfos(G4_FILE)

print("── CaloClouds3 ──")
cc3_energies, cc3_counts, cc3_n = extract_all_pfos(CC3_FILE)

# ── Print summary table ────────────────────────────────────────────────────────
print(f"\n{'Particle':<22} {'G4 total':>10} {'G4/evt':>8} {'CC3 total':>10} {'CC3/evt':>8}")
print("─" * 62)
for pdg, label in PARTICLES:
    g4_e  = g4_energies.get(pdg,  [])
    cc3_e = cc3_energies.get(pdg, [])
    g4_c  = g4_counts.get(pdg,  [0])
    cc3_c = cc3_counts.get(pdg, [0])
    print(f"{label:<22} {len(g4_e):>10} {np.mean(g4_c):>8.3f} "
          f"{len(cc3_e):>10} {np.mean(cc3_c):>8.3f}")
print("─" * 62)
print(f"{'NOTE':<22} {'G4: '+str(g4_n)+' events':>19} {'CC3: '+str(cc3_n)+' events':>19}")

# ── Plotting ───────────────────────────────────────────────────────────────────
# One figure per particle type: left = energy distribution, right = multiplicity

for pdg, label in PARTICLES:
    g4_e  = np.array(g4_energies.get(pdg,  []))
    cc3_e = np.array(cc3_energies.get(pdg, []))
    g4_c  = np.array(g4_counts.get(pdg,  [0] * g4_n))
    cc3_c = np.array(cc3_counts.get(pdg, [0] * cc3_n))

    total_g4  = len(g4_e)
    total_cc3 = len(cc3_e)

    # skip if both samples have basically nothing
    if total_g4 < 5 and total_cc3 < 5:
        print(f"Skipping {label}: too few entries (G4={total_g4}, CC3={total_cc3})")
        continue

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"{label}  |  G4: {total_g4} PFOs ({g4_n} evt)  "
                 f"CC3: {total_cc3} PFOs ({cc3_n} evt)", fontsize=11)

    # Left: energy distribution
    emax = max(g4_e.max() if len(g4_e) else 1,
               cc3_e.max() if len(cc3_e) else 1)
    emax = min(emax, 100)  # cap at 100 GeV for display

    kw = dict(bins=40, range=(0, emax), histtype="step",
              linewidth=2, density=True)
    if len(g4_e)  > 0: ax1.hist(g4_e,  **kw, color="#2166ac", label=f"Geant4  μ={g4_e.mean():.2f} GeV")
    if len(cc3_e) > 0: ax1.hist(cc3_e, **kw, color="#d6604d", label=f"CC3     μ={cc3_e.mean():.2f} GeV")
    ax1.set_xlabel("Energy [GeV]")
    ax1.set_ylabel("Normalised entries")
    ax1.set_title("Energy distribution")
    ax1.legend(fontsize=9)

    # Right: multiplicity per event
    max_mult = int(max(g4_c.max(), cc3_c.max())) + 2
    bins_m = np.arange(0, max_mult) - 0.5
    kw_m = dict(bins=bins_m, histtype="step", linewidth=2, density=True)
    ax2.hist(g4_c,  **kw_m, color="#2166ac",
             label=f"Geant4  μ={g4_c.mean():.2f}/evt")
    ax2.hist(cc3_c, **kw_m, color="#d6604d",
             label=f"CC3     μ={cc3_c.mean():.2f}/evt")
    ax2.set_xlabel("Number of PFOs per event")
    ax2.set_ylabel("Normalised entries")
    ax2.set_title("Multiplicity per event")
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.tight_layout()
    safe_name = label.split("(")[0].strip().lower().replace(" ", "_")
    out = os.path.join(PLOT_DIR, f"pfo_{safe_name}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved: {out}")

print("\nDone. Copy plots with:")
print("  cp /afs/desy.de/user/a/alimuham/thesis-ml-sim/plots/pfo_*.png ~/Desktop/")
