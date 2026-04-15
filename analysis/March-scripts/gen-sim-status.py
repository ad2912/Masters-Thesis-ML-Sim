"""
gen_sim_status.py
=================
Investigates generator status and simulator status of MC photons
near phi = 0 degrees (the spike region).

Three cases analysed separately:
  Case 1 — No cuts: all MC photons
  Case 2 — Phi cut only: |phi| <= 5 degrees, no energy cut
            (lets us see where the LOW energy spike photons go)
  Case 3 — Phi + energy cut: |phi| <= 5 degrees AND E >= 10 GeV
            (what we looked at before)

For each case, reports:
  - generatorStatus breakdown (0, 1, 2, 3)
  - isCreatedInSimulation() fraction
  - Simulator status bits
  - Parent PDG breakdown

Run with:
    source ~/source.sh
    python3 gen_sim_status.py
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
SIM_CC  = "/data/dust/user/alimuham/thesis/sim/tau_pi0_SIM_caloclouds.edm4hep.root"

PLOT_DIR = os.path.expanduser("~/thesis-ml-sim/plots/phi-diagnostic")
os.makedirs(PLOT_DIR, exist_ok=True)

PHOTON_PDG = 22
PHI_WINDOW = 5.0   # degrees

# Simulator status bit positions (EDM4hep)
BIT_CREATED_IN_SIM = 30
BIT_BACKSCATTER    = 29
BIT_DECAYED_TRACKER= 27
BIT_DECAYED_CALO   = 26
BIT_LEFT_DETECTOR  = 25
BIT_STOPPED        = 24

def check_bit(status, bit):
    return bool((status >> bit) & 1)

# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────
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

PDG_NAMES = {
    22: "photon", 11: "e-", -11: "e+",
    111: "pi0", 211: "pi+", -211: "pi-",
    15: "tau-", -15: "tau+", 0: "none/beam",
}
def pdg_name(pdg):
    return PDG_NAMES.get(pdg, f"PDG={pdg}")

GEN_STATUS_MEANING = {
    0: "not set / sim-created",
    1: "final state (stable, from generator)",
    2: "decayed by generator",
    3: "documentation only",
}

# ─────────────────────────────────────────────────────────────────────────────
# COLLECT — no energy filter here, we filter later
# ─────────────────────────────────────────────────────────────────────────────
def collect(path, label):
    reader   = root_io.Reader(path)
    records  = []
    n_events = 0

    for event in reader.get("events"):
        n_events += 1
        for p in event.get("MCParticles"):
            if p.getPDG() != PHOTON_PDG:
                continue

            mom   = p.getMomentum()
            phi   = to_phi(mom.x, mom.y)
            theta = to_theta(mom.x, mom.y, mom.z)
            e     = p.getEnergy()
            gs    = p.getGeneratorStatus()
            ss    = p.getSimulatorStatus()

            parents    = p.getParents()
            parent_pdg = parents[0].getPDG()    if len(parents) > 0 else 0
            parent_e   = parents[0].getEnergy() if len(parents) > 0 else -1.0

            records.append({
                "energy"         : e,
                "phi"            : phi,
                "theta"          : theta,
                "gen_status"     : gs,
                "sim_status"     : ss,
                "created_in_sim" : check_bit(ss, BIT_CREATED_IN_SIM),
                "backscatter"    : check_bit(ss, BIT_BACKSCATTER),
                "decayed_calo"   : check_bit(ss, BIT_DECAYED_CALO),
                "left_detector"  : check_bit(ss, BIT_LEFT_DETECTOR),
                "stopped"        : check_bit(ss, BIT_STOPPED),
                "parent_pdg"     : parent_pdg,
                "parent_e"       : parent_e,
            })

    print(f"  {label}: {n_events} events, {len(records)} total MC photons")
    return records

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSE ONE CASE
# ─────────────────────────────────────────────────────────────────────────────
def analyse_case(records, case_label, energy_cut=None):
    """
    Apply filters and print breakdown.
    energy_cut: if None, no energy filter applied.
    """
    # Apply energy cut if requested
    if energy_cut is not None:
        filtered = [r for r in records if r["energy"] >= energy_cut]
    else:
        filtered = records

    spike = [r for r in filtered if abs(r["phi"]) <= PHI_WINDOW]
    flat  = [r for r in filtered if abs(r["phi"]) >  PHI_WINDOW]

    ecut_str = f"E ≥ {energy_cut} GeV" if energy_cut else "no energy cut"
    print(f"\n  ┌─ {case_label} ({ecut_str}) ─────────────────")
    print(f"  │  Photons after filter : {len(filtered)}")
    print(f"  │  Spike (|φ|≤{PHI_WINDOW}°)   : {len(spike)}")
    print(f"  │  Flat  (|φ|>{PHI_WINDOW}°)   : {len(flat)}")

    # Generator status
    print(f"  │")
    print(f"  │  Generator Status breakdown:")
    all_gs = sorted(set(r["gen_status"] for r in filtered))
    for gs in all_gs:
        n_s = sum(1 for r in spike if r["gen_status"] == gs)
        n_f = sum(1 for r in flat  if r["gen_status"] == gs)
        ps  = 100 * n_s / max(len(spike), 1)
        pf  = 100 * n_f / max(len(flat),  1)
        meaning = GEN_STATUS_MEANING.get(gs, f"status {gs}")
        print(f"  │    genStatus={gs}: spike={n_s}({ps:.0f}%)  "
              f"flat={n_f}({pf:.0f}%)  → {meaning}")

    # isCreatedInSimulation
    n_ss = sum(1 for r in spike if r["created_in_sim"])
    n_sg = sum(1 for r in spike if not r["created_in_sim"])
    n_fs = sum(1 for r in flat  if r["created_in_sim"])
    n_fg = sum(1 for r in flat  if not r["created_in_sim"])
    print(f"  │")
    print(f"  │  isCreatedInSimulation():")
    print(f"  │    Spike: sim-created={n_ss}({100*n_ss/max(len(spike),1):.0f}%)  "
          f"from-generator={n_sg}({100*n_sg/max(len(spike),1):.0f}%)")
    print(f"  │    Flat : sim-created={n_fs}({100*n_fs/max(len(flat),1):.0f}%)  "
          f"from-generator={n_fg}({100*n_fg/max(len(flat),1):.0f}%)")

    # Parent PDG for spike
    print(f"  │")
    print(f"  │  Parent PDG (spike photons):")
    spike_parents = Counter(r["parent_pdg"] for r in spike)
    for pdg, count in spike_parents.most_common(5):
        pct = 100 * count / max(len(spike), 1)
        print(f"  │    {pdg_name(pdg):<12} {count:>5} ({pct:.1f}%)")
    print(f"  └{'─'*55}")

    return spike, flat, filtered

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  Loading SIM files (all photons, no cuts yet)")
print("="*60)

rec_g4 = collect(SIM_G4, "SIM Geant4    ")
rec_cc = collect(SIM_CC, "SIM CaloClouds")

# ─────────────────────────────────────────────────────────────────────────────
# RUN THREE CASES FOR EACH FILE
# ─────────────────────────────────────────────────────────────────────────────
for rec, sim_lbl in [(rec_g4, "SIM GEANT4"), (rec_cc, "SIM CALOCLOUDS")]:
    print(f"\n{'='*60}")
    print(f"  {sim_lbl}")
    print(f"{'='*60}")

    spike1, flat1, filt1 = analyse_case(
        rec, "Case 1 — No cuts",         energy_cut=None)
    spike2, flat2, filt2 = analyse_case(
        rec, "Case 2 — Phi cut only",    energy_cut=None)
    spike3, flat3, filt3 = analyse_case(
        rec, "Case 3 — Phi + E≥10 GeV", energy_cut=10.0)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  Saving plots")
print("="*60)

bins_e    = np.linspace(0,   130, 60)
bins_ph   = np.linspace(-180, 180, 72)
bins_zoom = np.linspace(-10,  10,  80)

# ── PLOT 1: Phi coloured by origin — 3 rows (cases) x 2 cols (G4, CC)
fig, axes = plt.subplots(3, 2, figsize=(14, 15))
case_configs = [
    ("Case 1 — all photons",          None),
    ("Case 2 — phi cut only (no E)",  None),
    ("Case 3 — phi cut + E ≥ 10 GeV", 10.0),
]

for row, (case_lbl, ecut) in enumerate(case_configs):
    for col, (rec, sim_lbl) in enumerate([(rec_g4, "Geant4"),
                                           (rec_cc, "CaloClouds")]):
        filtered = [r for r in rec if ecut is None or r["energy"] >= ecut]
        sim_c = [r for r in filtered if r["created_in_sim"]]
        gen_c = [r for r in filtered if not r["created_in_sim"]]

        ax = axes[row, col]
        ax.hist([r["phi"] for r in gen_c], bins=bins_ph,
                histtype="step", color="#2166ac", linewidth=2,
                label=f"From generator (N={len(gen_c)})")
        ax.hist([r["phi"] for r in sim_c], bins=bins_ph,
                histtype="step", color="#d6604d", linewidth=2, linestyle="--",
                label=f"Sim-created    (N={len(sim_c)})")
        ax.set_xlabel("MC Photon φ [degrees]")
        ax.set_ylabel("Photons / bin")
        ax.set_title(f"{sim_lbl} — {case_lbl}")
        ax.legend(fontsize=8)
        sim_label(ax)

plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/gen_sim_status_3cases.png", bbox_inches="tight")
plt.close()
print(f"  ✓  gen_sim_status_3cases.png")


# ── PLOT 2: Energy distribution — spike vs flat, all photons (no cut)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (rec, lbl) in zip(axes, [(rec_g4, "SIM Geant4"),
                                   (rec_cc, "SIM CaloClouds")]):
    spike_all = [r for r in rec if abs(r["phi"]) <= PHI_WINDOW]
    flat_all  = [r for r in rec if abs(r["phi"]) >  PHI_WINDOW]

    ax.hist([r["energy"] for r in flat_all],  bins=bins_e,
            histtype="step", color="#2166ac", linewidth=2,
            label=f"Flat |φ|>{PHI_WINDOW}°  (N={len(flat_all)})")
    ax.hist([r["energy"] for r in spike_all], bins=bins_e,
            histtype="step", color="#d6604d", linewidth=2, linestyle="--",
            label=f"Spike |φ|≤{PHI_WINDOW}°  (N={len(spike_all)})")
    ax.axvline(10.0, color="black", linestyle=":", linewidth=1.5,
               label="E = 10 GeV")
    ax.set_xlabel("Photon Energy [GeV]")
    ax.set_ylabel("Photons / bin")
    ax.set_title(f"{lbl}\nEnergy of spike vs flat — ALL photons")
    ax.set_yscale("log")
    ax.legend()
    sim_label(ax)

plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/spike_energy_all_photons.png", bbox_inches="tight")
plt.close()
print(f"  ✓  spike_energy_all_photons.png")


# ── PLOT 3: Zoomed phi — generator-created only, all 3 cases
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (case_lbl, ecut) in zip(axes, case_configs):
    g4_gen = [r for r in rec_g4
              if not r["created_in_sim"] and
              (ecut is None or r["energy"] >= ecut)]
    cc_gen = [r for r in rec_cc
              if not r["created_in_sim"] and
              (ecut is None or r["energy"] >= ecut)]

    ax.hist([r["phi"] for r in g4_gen], bins=bins_zoom,
            histtype="step", color="#2166ac", linewidth=2,
            label=f"Geant4  (N={len(g4_gen)})")
    ax.hist([r["phi"] for r in cc_gen], bins=bins_zoom,
            histtype="step", color="#d6604d", linewidth=2, linestyle="--",
            label=f"CaloClouds  (N={len(cc_gen)})")
    ax.set_xlabel("MC Photon φ [degrees] (zoomed)")
    ax.set_ylabel("Photons / bin")
    ax.set_title(f"Generator-created photons only\n{case_lbl}")
    ax.legend()
    sim_label(ax)

plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/generator_photons_zoomed_3cases.png", bbox_inches="tight")
plt.close()
print(f"  ✓  generator_photons_zoomed_3cases.png")

print(f"\n  All plots saved to: {PLOT_DIR}")
print(f"\n  WHAT TO LOOK FOR:")
print(f"  gen_sim_status_3cases.png")
print(f"    Row 1 (no cuts): do generator-created photons have a spike?")
print(f"    Row 2 (phi only): are low-energy spike photons sim-created or gen?")
print(f"    Row 3 (phi+10GeV): high-energy spike — what fraction is sim-created?")
print(f"  spike_energy_all_photons.png")
print(f"    Does the spike extend below 10 GeV? How does the energy")
print(f"    distribution of spike photons compare to flat photons?")
print(f"  generator_photons_zoomed_3cases.png")
print(f"    Thomas's crossing angle hypothesis: if gen-created photons")
print(f"    show a spike at phi=0, that confirms the boost effect.")
print(f"    If spike is only in sim-created: it's pure bremsstrahlung.")
