"""
phi_diagnostic.py
=================
Two-part diagnostic:

PART A — Raw phi inspection
  Prints phi values of MC photons event-by-event for the first N events.
  Lets you see with your own eyes what is happening in the raw data
  before drawing any conclusions.

PART B — Skim bias analysis
  Compares the phi distribution of:
    - ALL events in the original generator file (elec_pos_10k.edm4hep.root)
    - Events that PASSED the skim (tau_pi0_10GeV_filtered.edm4hep.root)
  If the skim introduces a phi bias, the two distributions will differ.
  A flat phi in the full file but a peaked phi in the skimmed file
  would confirm that the 10 GeV energy cut is breaking azimuthal symmetry.

Run with:
    source ~/source.sh
    python3 phi_diagnostic.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio import root_io
import os

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
FULL_GEN   = "/afs/desy.de/user/a/alimuham/thesis-ml-sim/steering/elec_pos_10k.edm4hep.root"
SKIMMED    = "/afs/desy.de/user/a/alimuham/thesis-ml-sim/steering/tau_pi0_10GeV_filtered.edm4hep.root"
PLOT_DIR   = os.path.expanduser("~/thesis-ml-sim/plots/phi-diagnostic")
os.makedirs(PLOT_DIR, exist_ok=True)

PHOTON_PDG = 22
TAU_PDG    = 15
PI0_PDG    = 111
N_EVENTS_INSPECT = 10   # how many events to print in Part A

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def to_phi(px, py):
    return np.degrees(np.arctan2(py, px))

def to_theta(px, py, pz):
    p = np.sqrt(px**2 + py**2 + pz**2)
    if p == 0:
        return 0.0
    return np.degrees(np.arccos(np.clip(pz / p, -1, 1)))

def sim_label(ax):
    ax.text(0.01, 1.01, "ILD simulation (preliminary)",
            transform=ax.transAxes, fontsize=9, color="gray", va="bottom")

plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 12,
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 12,
    "legend.fontsize"  : 10,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 150,
})

# ─────────────────────────────────────────────────────────────────────────────
# PART A — RAW PHI INSPECTION (first N events)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"  PART A — Raw phi values, first {N_EVENTS_INSPECT} events")
print(f"  File: tau_pi0_10GeV_filtered.edm4hep.root (skimmed)")
print("="*60)

reader = root_io.Reader(SKIMMED)
event_count = 0

for event in reader.get("events"):
    if event_count >= N_EVENTS_INSPECT:
        break
    event_count += 1

    mc_particles = event.get("MCParticles")
    photons = [(p.getEnergy(),
                to_phi(p.getMomentum().x, p.getMomentum().y),
                to_theta(p.getMomentum().x, p.getMomentum().y, p.getMomentum().z),
                p.getMomentum().x, p.getMomentum().y)
               for p in mc_particles if p.getPDG() == PHOTON_PDG]

    print(f"\n  Event {event_count}:  {len(photons)} MC photons")
    print(f"  {'Energy':>10}  {'phi':>8}  {'theta':>8}  {'px':>10}  {'py':>10}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")
    for e, phi, theta, px, py in sorted(photons, key=lambda x: -x[0])[:15]:
        flag = " ← high E" if e >= 10.0 else ""
        print(f"  {e:>10.3f}  {phi:>8.2f}  {theta:>8.2f}  {px:>10.4f}  {py:>10.4f}{flag}")
    if len(photons) > 15:
        print(f"  ... ({len(photons)-15} more photons not shown)")

# ─────────────────────────────────────────────────────────────────────────────
# PART B — SKIM BIAS ANALYSIS
# Compare phi distributions before and after the skim
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n" + "="*60)
print("  PART B — Skim bias analysis")
print("  Comparing phi BEFORE skim vs AFTER skim")
print("="*60)

def collect_photon_phi(path, label, max_events=None):
    """Collect phi of all MC photons, and separately for photons >= 10 GeV."""
    reader = root_io.Reader(path)
    all_phi = []
    hi_phi  = []
    n_events = 0

    for event in reader.get("events"):
        if max_events and n_events >= max_events:
            break
        n_events += 1
        for p in event.get("MCParticles"):
            if p.getPDG() != PHOTON_PDG:
                continue
            phi = to_phi(p.getMomentum().x, p.getMomentum().y)
            all_phi.append(phi)
            if p.getEnergy() >= 10.0:
                hi_phi.append(phi)

    print(f"  {label}: {n_events} events, "
          f"{len(all_phi)} MC photons, "
          f"{len(hi_phi)} >= 10 GeV")
    return np.array(all_phi), np.array(hi_phi), n_events

# Read full generator file — use first 2112 events (same size as skimmed file
# would have come from)
phi_full_all, phi_full_hi, n_full = collect_photon_phi(
    FULL_GEN, "Full generator (all events)", max_events=None)
phi_skim_all, phi_skim_hi, n_skim = collect_photon_phi(
    SKIMMED,  "Skimmed file   (kept events)")

print(f"\n  Skim kept {n_skim} events out of {n_full} "
      f"({100*n_skim/max(n_full,1):.1f}%)")

bins_ph = np.linspace(-180, 180, 72)  # 5-degree bins

# ── PLOT 1: All photon phi — full vs skimmed
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(phi_full_all, bins=bins_ph, histtype="step", color="#2166ac",
        linewidth=2, density=True, label=f"Full generator (N={len(phi_full_all)})")
ax.hist(phi_skim_all, bins=bins_ph, histtype="step", color="#d6604d",
        linewidth=2, linestyle="--", density=True,
        label=f"After skim  (N={len(phi_skim_all)})")
ax.set_xlabel("MC Photon φ [degrees]")
ax.set_ylabel("Normalised")
ax.set_title("All MC Photons\nFull generator vs after skim")
ax.legend()
sim_label(ax)

ax = axes[1]
ax.hist(phi_full_hi, bins=bins_ph, histtype="step", color="#2166ac",
        linewidth=2, density=True,
        label=f"Full generator E≥10 GeV (N={len(phi_full_hi)})")
ax.hist(phi_skim_hi, bins=bins_ph, histtype="step", color="#d6604d",
        linewidth=2, linestyle="--", density=True,
        label=f"After skim E≥10 GeV  (N={len(phi_skim_hi)})")
ax.set_xlabel("MC Photon φ [degrees]")
ax.set_ylabel("Normalised")
ax.set_title("MC Photons E ≥ 10 GeV only\nFull generator vs after skim")
ax.legend()
sim_label(ax)

plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/phi_skim_bias_comparison.png", bbox_inches="tight")
plt.close()
print(f"\n  ✓  phi_skim_bias_comparison.png")

# ── PLOT 2: Just the skimmed file phi — all energies and >= 10 GeV
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(phi_skim_all, bins=bins_ph, histtype="stepfilled", color="#d6604d",
        alpha=0.5, label=f"All photons (N={len(phi_skim_all)})")
ax.hist(phi_skim_all, bins=bins_ph, histtype="step", color="#d6604d", linewidth=2)
ax.set_xlabel("MC Photon φ [degrees]")
ax.set_ylabel("Photons / bin")
ax.set_title("Skimmed file — all MC photon φ\n(should be flat if no bias)")
ax.legend()
sim_label(ax)

ax = axes[1]
ax.hist(phi_skim_hi, bins=bins_ph, histtype="stepfilled", color="#f4a582",
        alpha=0.5, label=f"E ≥ 10 GeV (N={len(phi_skim_hi)})")
ax.hist(phi_skim_hi, bins=bins_ph, histtype="step", color="#f4a582", linewidth=2)
ax.set_xlabel("MC Photon φ [degrees]")
ax.set_ylabel("Photons / bin")
ax.set_title("Skimmed file — E ≥ 10 GeV photon φ\n(CaloClouds trigger region)")
ax.legend()
sim_label(ax)

plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/phi_skimmed_file_inspection.png", bbox_inches="tight")
plt.close()
print(f"  ✓  phi_skimmed_file_inspection.png")

# ── PLOT 3: Full generator phi — sanity check, should be perfectly flat
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(phi_full_all, bins=bins_ph, histtype="stepfilled", color="#2166ac",
        alpha=0.5, label=f"All MC photons (N={len(phi_full_all)})")
ax.hist(phi_full_all, bins=bins_ph, histtype="step", color="#2166ac", linewidth=2)
ax.set_xlabel("MC Photon φ [degrees]")
ax.set_ylabel("Photons / bin")
ax.set_title("Full generator file — MC photon φ\n"
             "(sanity check: must be flat before any cuts)")
ax.legend()
sim_label(ax)
plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/phi_full_generator_sanity.png", bbox_inches="tight")
plt.close()
print(f"  ✓  phi_full_generator_sanity.png")

print(f"\n  All plots saved to: {PLOT_DIR}")
print(f"\n  WHAT TO LOOK FOR:")
print(f"  1. phi_full_generator_sanity.png — must be flat.")
print(f"     If not flat: problem is in the original generator, not the skim.")
print(f"  2. phi_skim_bias_comparison.png — compare blue (full) vs red (skimmed).")
print(f"     If red has a spike at 0 but blue is flat: skim bias confirmed.")
print(f"  3. phi_skimmed_file_inspection.png — look at the spike closely.")
print(f"     Is it at exactly phi=0? Is it narrow or broad?")
