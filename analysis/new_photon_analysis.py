"""
photon_analysis.py
==================
Complete reco photon analysis: Geant4 vs CaloClouds.

What this script does, in order:
  1. Load reco photons from all three files (G4 part1 + part2 merged, CC).
  2. For every reco photon in the CC file: classify as ML or Geant4 using
     the step-length logic via EcalBarrelRelationsSimRec + EcalEndcapsRelationsSimRec.
  3. Compute energy, theta, phi for every photon in every category.
  4. Make 5 publication-ready plots.

ML-ID logic (from your existing script, unchanged):
  For each reco photon -> walk its clusters -> walk reco hits -> look up
  sim hit via SimRec relation -> walk contributions -> if getStepLength()==0,
  increment ml_hit_count. If ml_hit_count > 10 across the whole photon: ML.

Files used:
  REC G4  : tau_geant4_REC.edm4hep.root      (621 events)
           + tau_geant4_part2_REC.edm4hep.root (378 events) → 999 total
  REC CC  : tau_caloclouds_REC.edm4hep.root  (1000 events)

Run:
  source ~/source.sh
  python3 photon_analysis.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from podio import root_io
import os

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
REC_G4_P1 = "/data/dust/user/alimuham/thesis/reco/tau_geant4_REC.edm4hep.root"
REC_G4_P2 = "/data/dust/user/alimuham/thesis/reco/tau_geant4_part2_REC.edm4hep.root"
REC_CC    = "/data/dust/user/alimuham/thesis/reco/tau_caloclouds_REC.edm4hep.root"
PLOT_DIR  = "/data/dust/user/alimuham/thesis/plots/photon_analysis"

os.makedirs(PLOT_DIR, exist_ok=True)

PHOTON_PDG    = 22
ML_THRESHOLD  = 10   # number of zero-steplength contributions to call it ML
ML_ENERGY_GEV = 10.0 # CaloClouds trigger energy

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR SCHEME — consistent across all plots
# ─────────────────────────────────────────────────────────────────────────────
C_G4_FILE  = "#2166ac"   # blue  — full Geant4 reco file
C_CC_ALL   = "#d6604d"   # red   — all CC reco photons
C_ML       = "#f4a582"   # orange — ML-classified photons inside CC file
C_CC_G4    = "#4dac26"   # green  — G4-classified photons inside CC file

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def header(msg):
    print(f"\n{'='*65}\n  {msg}\n{'='*65}")

def momentum_to_angles(px, py, pz):
    """
    Convert 3-momentum to theta (polar) and phi (azimuthal).
    theta: angle from +z axis, in degrees [0, 180]
    phi  : azimuthal angle, in degrees [-180, 180]
    """
    p  = np.sqrt(px**2 + py**2 + pz**2)
    theta = np.degrees(np.arccos(np.clip(pz / p, -1, 1))) if p > 0 else 0.0
    phi   = np.degrees(np.arctan2(py, px))
    return theta, phi


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: collect reco photons from a single file, no ML classification
# ─────────────────────────────────────────────────────────────────────────────
def collect_g4_photons(path, label):
    """
    Read all reco photons (PDG==22) from a file.
    Returns dict with lists: energy, theta, phi.
    """
    reader = root_io.Reader(path)
    energies, thetas, phis = [], [], []

    for event in reader.get("events"):
        pfos = event.get("PandoraPFOs")
        for pfo in pfos:
            if pfo.getPDG() != PHOTON_PDG:
                continue
            energies.append(pfo.getEnergy())
            mom = pfo.getMomentum()
            th, ph = momentum_to_angles(mom.x, mom.y, mom.z)
            thetas.append(th)
            phis.append(ph)

    print(f"  {label}: {len(energies)} reco photons")
    return {"energy": np.array(energies),
            "theta":  np.array(thetas),
            "phi":    np.array(phis)}


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: collect and classify reco photons from CaloClouds file
# ─────────────────────────────────────────────────────────────────────────────
def collect_cc_photons(path, label):
    """
    Read all reco photons from the CaloClouds reco file.
    For each photon, apply the ML-ID logic:
      - Walk clusters → reco hits → SimRec relation → sim hit → contributions
      - Count contributions with getStepLength() == 0
      - If count > ML_THRESHOLD → ML photon, else G4 photon

    Returns two dicts (ml, g4), each with energy/theta/phi arrays.
    Also returns combined dict for all CC photons.
    """
    reader = root_io.Reader(path)

    ml_e,  ml_th,  ml_ph  = [], [], []
    g4_e,  g4_th,  g4_ph  = [], [], []

    n_no_relation = 0   # hits we couldn't trace back to a sim hit

    for event in reader.get("events"):
        pfos          = event.get("PandoraPFOs")
        barrel_rels   = event.get("EcalBarrelRelationsSimRec")
        endcap_rels   = event.get("EcalEndcapsRelationsSimRec")

        # Build lookup: reco hit index → sim hit
        # The relation stores: getFrom() = reco hit, getTo() = sim hit
        hit_lookup = {}
        for rel in barrel_rels:
            hit_lookup[rel.getFrom().getObjectID().index] = rel.getTo()
        for rel in endcap_rels:
            hit_lookup[rel.getFrom().getObjectID().index] = rel.getTo()

        for pfo in pfos:
            if pfo.getPDG() != PHOTON_PDG:
                continue

            ml_hit_count = 0

            for cluster in pfo.getClusters():
                for rec_hit in cluster.getHits():
                    sim_hit = hit_lookup.get(rec_hit.getObjectID().index)
                    if sim_hit is None:
                        n_no_relation += 1
                        continue
                    for contrib in sim_hit.getContributions():
                        if contrib.getStepLength() == 0:
                            ml_hit_count += 1
                            break   # one zero-step contrib is enough per sim hit

            mom = pfo.getMomentum()
            th, ph = momentum_to_angles(mom.x, mom.y, mom.z)
            e = pfo.getEnergy()

            if ml_hit_count > ML_THRESHOLD:
                ml_e.append(e);  ml_th.append(th);  ml_ph.append(ph)
            else:
                g4_e.append(e);  g4_th.append(th);  g4_ph.append(ph)

    total = len(ml_e) + len(g4_e)
    print(f"  {label}: {total} reco photons total")
    print(f"    → ML-classified  : {len(ml_e)}  ({100*len(ml_e)/max(total,1):.1f}%)")
    print(f"    → G4-classified  : {len(g4_e)}  ({100*len(g4_e)/max(total,1):.1f}%)")
    if n_no_relation > 0:
        print(f"    ⚠ Hits with no SimRec relation: {n_no_relation} (not counted)")

    ml = {"energy": np.array(ml_e), "theta": np.array(ml_th), "phi": np.array(ml_ph)}
    g4 = {"energy": np.array(g4_e), "theta": np.array(g4_th), "phi": np.array(g4_ph)}
    all_cc = {k: np.concatenate([ml[k], g4[k]]) for k in ml}
    return ml, g4, all_cc


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
header("Loading Geant4 reco (part 1 + part 2)")
g4_p1 = collect_g4_photons(REC_G4_P1, "G4 part1 (events 1–621)  ")
g4_p2 = collect_g4_photons(REC_G4_P2, "G4 part2 (events 623–999)")

# Merge the two G4 reco files
g4 = {k: np.concatenate([g4_p1[k], g4_p2[k]]) for k in g4_p1}
print(f"\n  G4 MERGED TOTAL: {len(g4['energy'])} reco photons across 999 events")

header("Loading and classifying CaloClouds reco")
cc_ml, cc_g4, cc_all = collect_cc_photons(REC_CC, "CaloClouds (1000 events)")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
header("Summary")
print(f"  {'Category':<40}  {'Count':>6}  {'Mean E [GeV]':>12}")
print(f"  {'-'*40}  {'-'*6}  {'-'*12}")
print(f"  {'Geant4 reco (all, 999 events)':<40}  "
      f"{len(g4['energy']):>6}  {np.mean(g4['energy']):>12.2f}")
print(f"  {'CaloClouds reco (all, 1000 events)':<40}  "
      f"{len(cc_all['energy']):>6}  {np.mean(cc_all['energy']):>12.2f}")
print(f"  {'  └─ ML-classified photons':<40}  "
      f"{len(cc_ml['energy']):>6}  {np.mean(cc_ml['energy']) if len(cc_ml['energy'])>0 else 0:>12.2f}")
print(f"  {'  └─ G4-classified photons (in CC)':<40}  "
      f"{len(cc_g4['energy']):>6}  {np.mean(cc_g4['energy']) if len(cc_g4['energy'])>0 else 0:>12.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT STYLE
# ─────────────────────────────────────────────────────────────────────────────
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

def add_lumi_label(ax, text="ILD simulation (preliminary)"):
    ax.text(0.01, 1.01, text, transform=ax.transAxes,
            fontsize=9, color="gray", va="bottom")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1: Energy spectrum — full range, log y
# Question: how does the overall reco photon energy compare G4 vs CC?
# ─────────────────────────────────────────────────────────────────────────────
header("Saving plots")

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0, 130, 80)

ax.hist(g4["energy"],     bins=bins, histtype="step", color=C_G4_FILE,
        linewidth=2, label=f"Full Geant4  (N={len(g4['energy'])})")
ax.hist(cc_all["energy"], bins=bins, histtype="step", color=C_CC_ALL,
        linewidth=2, linestyle="--", label=f"CaloClouds all  (N={len(cc_all['energy'])})")

ax.axvline(ML_ENERGY_GEV, color="black", linestyle=":", linewidth=1.5,
           label=f"CaloClouds trigger = {ML_ENERGY_GEV} GeV")
ax.set_xlabel("Reconstructed Photon Energy [GeV]")
ax.set_ylabel("Photons / bin")
ax.set_title("Reco Photon Energy Spectrum — Full Range")
ax.set_yscale("log")
ax.legend()
add_lumi_label(ax)
plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/plot1_energy_full.png")
plt.close()
print(f"  ✓  plot1_energy_full.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2: Energy spectrum — zoomed above 10 GeV, with ML/G4 breakdown in CC
# Question: where the ML model operates, how do the photons look?
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
bins_zoom = np.linspace(10, 130, 60)

ax.hist(g4["energy"][g4["energy"] >= ML_ENERGY_GEV],
        bins=bins_zoom, histtype="step", color=C_G4_FILE,
        linewidth=2, label=f"Full Geant4  (N={np.sum(g4['energy']>=ML_ENERGY_GEV)})")

ax.hist(cc_ml["energy"][cc_ml["energy"] >= ML_ENERGY_GEV] if len(cc_ml["energy"])>0 else [],
        bins=bins_zoom, histtype="stepfilled", color=C_ML,
        alpha=0.5, label=f"CC — ML photons  (N={np.sum(cc_ml['energy']>=ML_ENERGY_GEV)})")

ax.hist(cc_ml["energy"][cc_ml["energy"] >= ML_ENERGY_GEV] if len(cc_ml["energy"])>0 else [],
        bins=bins_zoom, histtype="step", color=C_ML, linewidth=1.5)

ax.set_xlabel("Reconstructed Photon Energy [GeV]")
ax.set_ylabel("Photons / bin")
ax.set_title(f"Reco Photon Energy — Above {ML_ENERGY_GEV} GeV\n"
             f"(CaloClouds trigger region)")
ax.legend()
add_lumi_label(ax)
plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/plot2_energy_above10GeV.png")
plt.close()
print(f"  ✓  plot2_energy_above10GeV.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3: Full breakdown — G4 file vs CC components
# Question: where does each category sit in energy? Is the CC G4-component
#           consistent with the G4 file below 10 GeV?
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(0, 130, 80)

ax.hist(g4["energy"],     bins=bins, histtype="step", color=C_G4_FILE,
        linewidth=2.5, label=f"Full Geant4 (N={len(g4['energy'])})")
ax.hist(cc_g4["energy"],  bins=bins, histtype="step", color=C_CC_G4,
        linewidth=2, linestyle="--",
        label=f"CC — G4-type photons (N={len(cc_g4['energy'])})")
ax.hist(cc_ml["energy"],  bins=bins, histtype="step", color=C_ML,
        linewidth=2, linestyle="-.",
        label=f"CC — ML photons (N={len(cc_ml['energy'])})")

ax.axvline(ML_ENERGY_GEV, color="black", linestyle=":", linewidth=1.5,
           label=f"ML trigger = {ML_ENERGY_GEV} GeV")
ax.set_xlabel("Reconstructed Photon Energy [GeV]")
ax.set_ylabel("Photons / bin")
ax.set_title("Reco Photon Energy — Full Breakdown\n"
             "G4 file vs CaloClouds components")
ax.set_yscale("log")
ax.legend()
add_lumi_label(ax)
plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/plot3_energy_breakdown.png")
plt.close()
print(f"  ✓  plot3_energy_breakdown.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4: Theta (polar angle) distributions
# Question: are the photons going in the same directions in both simulations?
# Physics: ECAL endcap sits at low theta (forward). Barrel at mid-theta.
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
bins_th = np.linspace(0, 180, 60)

# Left: G4 file vs CC all
ax = axes[0]
ax.hist(g4["theta"],     bins=bins_th, histtype="step", color=C_G4_FILE,
        linewidth=2, density=True, label=f"Full Geant4")
ax.hist(cc_all["theta"], bins=bins_th, histtype="step", color=C_CC_ALL,
        linewidth=2, density=True, linestyle="--", label=f"CaloClouds all")
ax.set_xlabel("Polar angle θ [degrees]")
ax.set_ylabel("Normalised")
ax.set_title("Theta Distribution\nGeant4 vs CaloClouds (normalised)")
ax.legend()
add_lumi_label(ax)

# Right: CC breakdown
ax = axes[1]
ax.hist(cc_g4["theta"], bins=bins_th, histtype="step", color=C_CC_G4,
        linewidth=2, density=True, linestyle="--", label="CC — G4-type")
ax.hist(cc_ml["theta"], bins=bins_th, histtype="step", color=C_ML,
        linewidth=2, density=True, linestyle="-.", label="CC — ML")
ax.hist(g4["theta"],    bins=bins_th, histtype="step", color=C_G4_FILE,
        linewidth=2, density=True, label="Full Geant4", alpha=0.7)
ax.set_xlabel("Polar angle θ [degrees]")
ax.set_ylabel("Normalised")
ax.set_title("Theta Distribution — CC Breakdown\nvs Full Geant4 (normalised)")
ax.legend()
add_lumi_label(ax)

plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/plot4_theta.png")
plt.close()
print(f"  ✓  plot4_theta.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5: Phi (azimuthal angle) distributions
# Question: is there azimuthal symmetry? Any detector artefacts?
# Physics: phi should be flat for a symmetric detector. Any peaks
#          would indicate detector acceptance effects.
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
bins_ph = np.linspace(-180, 180, 60)

# Left: G4 vs CC all
ax = axes[0]
ax.hist(g4["phi"],     bins=bins_ph, histtype="step", color=C_G4_FILE,
        linewidth=2, density=True, label="Full Geant4")
ax.hist(cc_all["phi"], bins=bins_ph, histtype="step", color=C_CC_ALL,
        linewidth=2, density=True, linestyle="--", label="CaloClouds all")
ax.set_xlabel("Azimuthal angle φ [degrees]")
ax.set_ylabel("Normalised")
ax.set_title("Phi Distribution\nGeant4 vs CaloClouds (normalised)")
ax.legend()
add_lumi_label(ax)

# Right: CC breakdown
ax = axes[1]
ax.hist(cc_g4["phi"], bins=bins_ph, histtype="step", color=C_CC_G4,
        linewidth=2, density=True, linestyle="--", label="CC — G4-type")
ax.hist(cc_ml["phi"], bins=bins_ph, histtype="step", color=C_ML,
        linewidth=2, density=True, linestyle="-.", label="CC — ML")
ax.hist(g4["phi"],    bins=bins_ph, histtype="step", color=C_G4_FILE,
        linewidth=2, density=True, label="Full Geant4", alpha=0.7)
ax.set_xlabel("Azimuthal angle φ [degrees]")
ax.set_ylabel("Normalised")
ax.set_title("Phi Distribution — CC Breakdown\nvs Full Geant4 (normalised)")
ax.legend()
add_lumi_label(ax)

plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/plot5_phi.png")
plt.close()
print(f"  ✓  plot5_phi.png")


header("Done")
print(f"  All plots saved to: {PLOT_DIR}")
print()
print("  WHAT TO CHECK IN EACH PLOT:")
print("  Plot 1 — Do G4 and CC total counts look similar? Large difference = investigate.")
print("  Plot 2 — Do ML photons fill the high-energy region as expected above 10 GeV?")
print("  Plot 3 — Does CC G4-type curve match the G4 file below 10 GeV? It should.")
print("  Plot 4 — Do theta distributions match? ML photons concentrated in endcap?")
print("  Plot 5 — Is phi flat for both? Any asymmetry = detector or generator artefact.")
