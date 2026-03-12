"""
photon_parentage.py
===================
Investigates the parentage of MC photons near phi = 0 degrees.

Specifically answers:
  1. What are the parent particles of photons near phi = 0?
  2. Are the spike photons exactly at phi = 0, or spread over a small range?
  3. How does the parentage break down across all phi values?
  4. Is the spike purely from beam particles (PDG = 11 / -11)?

Checks both the SIM file and the skimmed generator file.

Run with:
    source ~/source.sh
    python3 photon_parentage.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio import root_io
from collections import Counter
import os

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
SIM_G4  = "/data/dust/user/alimuham/thesis/sim/tau_pi0_SIM_geant4.edm4hep.root"
SKIMMED = "/afs/desy.de/user/a/alimuham/thesis-ml-sim/steering/tau_pi0_10GeV_filtered.edm4hep.root"

PLOT_DIR = os.path.expanduser("~/thesis-ml-sim/plots/phi-diagnostic")
os.makedirs(PLOT_DIR, exist_ok=True)

PHOTON_PDG  = 22
PHI_WINDOW  = 5.0    # degrees — defines "near phi=0" for the spike region
ENERGY_CUT  = 10.0   # GeV

# PDG name lookup for common particles
PDG_NAMES = {
    22:   "photon (γ)",
    11:   "electron (e-)",
    -11:  "positron (e+)",
    111:  "pi0",
    211:  "pi+",
    -211: "pi-",
    15:   "tau-",
    -15:  "tau+",
    13:   "muon-",
    -13:  "muon+",
    2112: "neutron",
    2212: "proton",
    -2212:"antiproton",
    0:    "unknown/beam",
}

def pdg_name(pdg):
    return PDG_NAMES.get(pdg, f"PDG={pdg}")

plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 11,
    "axes.titlesize"   : 12,
    "axes.labelsize"   : 11,
    "legend.fontsize"  : 9,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 150,
})

def sim_label(ax):
    ax.text(0.01, 1.01, "ILD simulation (preliminary)",
            transform=ax.transAxes, fontsize=9, color="gray", va="bottom")

def to_phi(px, py):
    return np.degrees(np.arctan2(py, px))

def to_theta(px, py, pz):
    p = np.sqrt(px**2 + py**2 + pz**2)
    if p == 0:
        return 0.0
    return np.degrees(np.arccos(np.clip(pz / p, -1, 1)))

# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTION — collect photons with parentage info
# ─────────────────────────────────────────────────────────────────────────────
def collect_photons_with_parents(path, label, energy_cut=None, max_events=None):
    """
    For every MC photon, record:
      - energy, phi, theta
      - parent PDG (or -999 if no parent)
      - parent energy
      - generation depth (how many steps from a beam particle)
    """
    reader = root_io.Reader(path)
    records = []
    n_events = 0

    for event in reader.get("events"):
        if max_events and n_events >= max_events:
            break
        n_events += 1

        for p in event.get("MCParticles"):
            if p.getPDG() != PHOTON_PDG:
                continue

            e = p.getEnergy()
            if energy_cut and e < energy_cut:
                continue

            mom = p.getMomentum()
            phi   = to_phi(mom.x, mom.y)
            theta = to_theta(mom.x, mom.y, mom.z)

            parents = p.getParents()
            if len(parents) > 0:
                parent_pdg = parents[0].getPDG()
                parent_e   = parents[0].getEnergy()
            else:
                parent_pdg = 0
                parent_e   = -1.0

            records.append({
                "energy"    : e,
                "phi"       : phi,
                "theta"     : theta,
                "parent_pdg": parent_pdg,
                "parent_e"  : parent_e,
            })

    print(f"  {label}: {n_events} events, {len(records)} photons collected")
    return records

# ─────────────────────────────────────────────────────────────────────────────
# PART A — SIM FILE (Geant4), all photons >= 10 GeV
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  PART A — SIM Geant4 file, photons >= 10 GeV")
print("="*60)

records_sim = collect_photons_with_parents(SIM_G4, "SIM Geant4", energy_cut=ENERGY_CUT)

# Split into spike (near phi=0) and flat (rest)
spike = [r for r in records_sim if abs(r["phi"]) <= PHI_WINDOW]
flat  = [r for r in records_sim if abs(r["phi"]) >  PHI_WINDOW]

print(f"\n  Phi window for spike: |φ| ≤ {PHI_WINDOW}°")
print(f"  Photons in spike region : {len(spike)}")
print(f"  Photons outside spike   : {len(flat)}")

# Parent breakdown for spike photons
print(f"\n  ── Parent PDG breakdown for SPIKE photons (|φ| ≤ {PHI_WINDOW}°) ──")
spike_parents = Counter(r["parent_pdg"] for r in spike)
for pdg, count in spike_parents.most_common():
    pct = 100 * count / max(len(spike), 1)
    print(f"    {pdg_name(pdg):<25}  {count:>5}  ({pct:.1f}%)")

# Parent breakdown for flat photons
print(f"\n  ── Parent PDG breakdown for FLAT photons (|φ| > {PHI_WINDOW}°) ──")
flat_parents = Counter(r["parent_pdg"] for r in flat)
for pdg, count in flat_parents.most_common():
    pct = 100 * count / max(len(flat), 1)
    print(f"    {pdg_name(pdg):<25}  {count:>5}  ({pct:.1f}%)")

# Print the first 20 spike photons in detail
print(f"\n  ── First 20 spike photons in detail ──")
print(f"  {'Energy':>8}  {'phi':>8}  {'theta':>8}  {'Parent':>25}  {'Parent E':>10}")
print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*25}  {'-'*10}")
for r in sorted(spike, key=lambda x: -x["energy"])[:20]:
    print(f"  {r['energy']:>8.2f}  {r['phi']:>8.3f}  "
          f"{r['theta']:>8.3f}  {pdg_name(r['parent_pdg']):>25}  "
          f"{r['parent_e']:>10.2f}")

# ── Is the spike truly at exactly phi=0 or spread over a small range?
print(f"\n  ── Phi values of spike photons (sorted) ──")
spike_phis = sorted([r["phi"] for r in spike])
print(f"  Min phi : {min(spike_phis):.4f}°")
print(f"  Max phi : {max(spike_phis):.4f}°")
print(f"  Mean phi: {np.mean(spike_phis):.4f}°")
print(f"  Std phi : {np.std(spike_phis):.4f}°")

# ─────────────────────────────────────────────────────────────────────────────
# PART B — SKIMMED GENERATOR FILE, all photons (no energy cut)
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n" + "="*60)
print("  PART B — Skimmed generator file, ALL photon energies")
print("="*60)

records_gen = collect_photons_with_parents(SKIMMED, "Skimmed generator")

spike_gen = [r for r in records_gen if abs(r["phi"]) <= PHI_WINDOW]
flat_gen  = [r for r in records_gen if abs(r["phi"]) >  PHI_WINDOW]

print(f"\n  Photons in spike region (|φ| ≤ {PHI_WINDOW}°): {len(spike_gen)}")
print(f"  Photons outside spike               : {len(flat_gen)}")

print(f"\n  ── Parent PDG breakdown for SPIKE photons ──")
spike_gen_parents = Counter(r["parent_pdg"] for r in spike_gen)
for pdg, count in spike_gen_parents.most_common():
    pct = 100 * count / max(len(spike_gen), 1)
    print(f"    {pdg_name(pdg):<25}  {count:>5}  ({pct:.1f}%)")

print(f"\n  ── Parent PDG breakdown for FLAT photons ──")
flat_gen_parents = Counter(r["parent_pdg"] for r in flat_gen)
for pdg, count in flat_gen_parents.most_common():
    pct = 100 * count / max(len(flat_gen), 1)
    print(f"    {pdg_name(pdg):<25}  {count:>5}  ({pct:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  Saving plots")
print("="*60)

bins_ph   = np.linspace(-180, 180, 72)
bins_ph_zoom = np.linspace(-10, 10, 80)  # zoomed into spike region

# ── PLOT 1: Phi distribution coloured by parent type — SIM file
phi_from_pi0     = [r["phi"] for r in records_sim if r["parent_pdg"] == 111]
phi_from_beam    = [r["phi"] for r in records_sim
                   if r["parent_pdg"] in (11, -11)]
phi_from_other   = [r["phi"] for r in records_sim
                   if r["parent_pdg"] not in (111, 11, -11)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(phi_from_pi0,   bins=bins_ph, histtype="step", color="#2166ac",
        linewidth=2, label=f"Parent = π⁰  (N={len(phi_from_pi0)})")
ax.hist(phi_from_beam,  bins=bins_ph, histtype="step", color="#d6604d",
        linewidth=2, linestyle="--",
        label=f"Parent = e±  (N={len(phi_from_beam)})")
ax.hist(phi_from_other, bins=bins_ph, histtype="step", color="#4dac26",
        linewidth=2, linestyle="-.",
        label=f"Parent = other  (N={len(phi_from_other)})")
ax.set_xlabel("MC Photon φ [degrees]")
ax.set_ylabel("Photons / bin")
ax.set_title(f"SIM Geant4 — photons E ≥ {ENERGY_CUT} GeV\nColoured by parent particle")
ax.legend()
sim_label(ax)

# Zoomed into spike
ax = axes[1]
ax.hist(phi_from_pi0,   bins=bins_ph_zoom, histtype="step", color="#2166ac",
        linewidth=2, label=f"Parent = π⁰  (N={len(phi_from_pi0)})")
ax.hist(phi_from_beam,  bins=bins_ph_zoom, histtype="step", color="#d6604d",
        linewidth=2, linestyle="--",
        label=f"Parent = e±  (N={len(phi_from_beam)})")
ax.hist(phi_from_other, bins=bins_ph_zoom, histtype="step", color="#4dac26",
        linewidth=2, linestyle="-.",
        label=f"Parent = other  (N={len(phi_from_other)})")
ax.set_xlabel("MC Photon φ [degrees]  (zoomed)")
ax.set_ylabel("Photons / bin")
ax.set_title(f"SIM Geant4 — ZOOMED into spike region\n|φ| ≤ 10°")
ax.legend()
sim_label(ax)

plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/parentage_by_phi_sim.png", bbox_inches="tight")
plt.close()
print(f"  ✓  parentage_by_phi_sim.png")


# ── PLOT 2: Energy distribution of spike vs flat photons
fig, ax = plt.subplots(figsize=(9, 5))
bins_e = np.linspace(0, 130, 60)
spike_e = [r["energy"] for r in records_sim]
e_spike_only = [r["energy"] for r in spike]
e_flat_only  = [r["energy"] for r in flat]

ax.hist(e_flat_only,  bins=bins_e, histtype="step", color="#2166ac",
        linewidth=2, label=f"Flat photons |φ|>{PHI_WINDOW}°  (N={len(e_flat_only)})")
ax.hist(e_spike_only, bins=bins_e, histtype="step", color="#d6604d",
        linewidth=2, linestyle="--",
        label=f"Spike photons |φ|≤{PHI_WINDOW}°  (N={len(e_spike_only)})")
ax.set_xlabel("MC Photon Energy [GeV]")
ax.set_ylabel("Photons / bin")
ax.set_title(f"SIM Geant4 — Energy of spike vs flat photons\n"
             f"(both E ≥ {ENERGY_CUT} GeV)")
ax.set_yscale("log")
ax.legend()
sim_label(ax)
plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/spike_vs_flat_energy.png", bbox_inches="tight")
plt.close()
print(f"  ✓  spike_vs_flat_energy.png")


# ── PLOT 3: Theta distribution of spike photons — are they truly forward?
fig, ax = plt.subplots(figsize=(9, 5))
bins_th = np.linspace(0, 180, 50)
theta_spike = [r["theta"] for r in spike]
theta_flat  = [r["theta"] for r in flat]

ax.hist(theta_flat,  bins=bins_th, histtype="step", color="#2166ac",
        linewidth=2, label=f"Flat photons |φ|>{PHI_WINDOW}°  (N={len(theta_flat)})",
        density=True)
ax.hist(theta_spike, bins=bins_th, histtype="step", color="#d6604d",
        linewidth=2, linestyle="--",
        label=f"Spike photons |φ|≤{PHI_WINDOW}°  (N={len(theta_spike)})",
        density=True)
ax.set_xlabel("MC Photon θ [degrees]")
ax.set_ylabel("Normalised")
ax.set_title(f"SIM Geant4 — θ of spike vs flat photons\n"
             f"Are spike photons forward (small θ)?")
ax.legend()
sim_label(ax)
plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/spike_vs_flat_theta.png", bbox_inches="tight")
plt.close()
print(f"  ✓  spike_vs_flat_theta.png")

print(f"\n  All plots saved to: {PLOT_DIR}")
print(f"\n  KEY QUESTIONS THESE PLOTS ANSWER:")
print(f"  1. parentage_by_phi_sim.png")
print(f"     → Does the spike come entirely from e± parents?")
print(f"     → Are pi0-daughter photons flat in phi? (they should be)")
print(f"  2. spike_vs_flat_energy.png")
print(f"     → Do spike photons have a different energy profile?")
print(f"  3. spike_vs_flat_theta.png")
print(f"     → Are spike photons forward (small theta = endcap region)?")
print(f"     → If yes, this confirms beam bremsstrahlung hypothesis")
