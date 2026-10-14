"""
esum_per_event.py
==================
Physics question:
  Does CC3 reconstruct more total photon energy per event than G4,
  or does it just fragment the same energy into more PFOs?

Selection:
  All PandoraPFO with PDG == 22, no energy floor, no truth-matching.
  One Esum value per event = sum of all reco photon PFO energies.

Samples: 10k events

Run:
    source ~/source.sh
    python3 esum_per_event.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio import root_io
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
# G4_FILE  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-1000events-test_REC.edm4hep.root"
# CC3_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-1000events-test_REC.edm4hep.root"
G4_FILE  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-10k-events-test_REC.edm4hep.root"
CC3_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root"

PLOT_DIR  = os.path.expanduser("~/thesis-ml-sim/plots/esum_per_event")
os.makedirs(PLOT_DIR, exist_ok=True)

PHOTON_PDG = 22
MAX_EVENTS = 10000

# ── COLLECT ───────────────────────────────────────────────────────────────────
def collect_esum(filepath, label):
    reader = root_io.Reader(filepath)
    esum_vals    = []
    n_pfo_counts = []

    for i_ev, event in enumerate(reader.get("events")):
        if i_ev >= MAX_EVENTS:
            break
        photon_energies = [
            pfo.getEnergy()
            for pfo in event.get("PandoraPFOs")
            if pfo.getPDG() == PHOTON_PDG
        ]
        esum_vals.append(sum(photon_energies))
        n_pfo_counts.append(len(photon_energies))

    esum_vals    = np.array(esum_vals)
    n_pfo_counts = np.array(n_pfo_counts)

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Events processed      : {len(esum_vals)}")
    print(f"  Events with 0 photons : {np.sum(n_pfo_counts == 0)}")
    print(f"  Mean photon PFOs/event: {np.mean(n_pfo_counts):.3f}")
    print(f"  Esum mean [GeV]       : {np.mean(esum_vals):.1f}")

    return esum_vals, n_pfo_counts


esum_g4,  npfo_g4  = collect_esum(G4_FILE,  "G4")
esum_cc3, npfo_cc3 = collect_esum(CC3_FILE, "CC3")

# ── PLOT 1: Esum per event ────────────────────────────────────────────────────
bins = np.arange(0, 205, 5)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(esum_g4,  bins=bins, histtype="step", lw=2,
        color="#2166ac", label="G4")
ax.hist(esum_cc3, bins=bins, histtype="step", lw=2, linestyle="--",
        color="#d6604d", label="CC3")

ax.set_xlabel("$\\sum E_{\\rm reco}^{\\gamma}$ per event [GeV]", fontsize=12)
ax.set_ylabel("Events / 5 GeV  [raw counts]", fontsize=12)
ax.set_title(
    "Total reco photon energy per event — G4 vs CC3\n"
    "All PandoraPFO with PDG=22, no energy floor, no truth-matching\n"
    "10k events, $\\tau\\to\\pi^0\\nu$ at $\\sqrt{s}=250$ GeV",
    fontsize=10
)
ax.legend(fontsize=10)
fig.tight_layout()
outpath = f"{PLOT_DIR}/esum_per_event_10k.png"
fig.savefig(outpath, dpi=150)
plt.close(fig)
print(f"\nSaved: {outpath}")

# ── PLOT 2: N photon PFOs per event ──────────────────────────────────────────
n_max  = max(np.max(npfo_g4), np.max(npfo_cc3))
bins_n = np.arange(0, n_max + 2) - 0.5

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(npfo_g4,  bins=bins_n, histtype="step", lw=2,
        color="#2166ac", label=f"G4  (mean = {np.mean(npfo_g4):.3f})")
ax.hist(npfo_cc3, bins=bins_n, histtype="step", lw=2, linestyle="--",
        color="#d6604d", label=f"CC3 (mean = {np.mean(npfo_cc3):.3f})")

ax.set_xlabel("N photon PFOs per event", fontsize=12)
ax.set_ylabel("Events / bin  [raw counts]", fontsize=12)
ax.set_title(
    "Number of reco photon PFOs per event — G4 vs CC3\n"
    "All PandoraPFO with PDG=22, no energy floor, no truth-matching\n"
    "10k events, $\\tau\\to\\pi^0\\nu$ at $\\sqrt{s}=250$ GeV",
    fontsize=10
)
ax.legend(fontsize=10)
fig.tight_layout()
outpath = f"{PLOT_DIR}/npfo_per_event_10k.png"
fig.savefig(outpath, dpi=150)
plt.close(fig)
print(f"Saved: {outpath}")

print("\nDone.")
