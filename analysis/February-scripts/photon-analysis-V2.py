#!/usr/bin/env python3
"""
photon_validation.py
====================

Full photon validation for Geant4 vs CaloClouds3.

Produces:
  - photon_energy_log.png
  - photon_multiplicity.png
  - photon_response.png

Prints:
  - total events
  - total true photons
  - total reco photons
  - total matched photons
  - reconstruction efficiency
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from podio import root_io


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECO_DIR = os.path.join(BASE, "results", "reco")
PLOT_DIR = os.path.join(BASE, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

G4_FILES = [
    os.path.join(RECO_DIR, "tau_geant4_REC.edm4hep.root"),
    os.path.join(RECO_DIR, "tau_geant4_part2_REC.edm4hep.root"),
]

CC3_FILES = [
    os.path.join(RECO_DIR, "tau_caloclouds_REC.edm4hep.root"),
]


# ----------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------

def extract_photons(file_list):

    photon_energies = []
    photon_mult = []
    true_photon_energies = []
    matched_reco = []
    matched_true = []

    total_events = 0
    total_true = 0
    total_reco = 0
    total_matched = 0

    for filepath in file_list:

        reader = root_io.Reader(filepath)

        for event in reader.get("events"):
            total_events += 1

            # --- reco photons ---
            pfos = event.get("PandoraPFOs")
            event_count = 0

            for pfo in pfos:
                if pfo.getPDG() == 22:
                    e = pfo.getEnergy()
                    if e > 1e-6:
                        photon_energies.append(e)
                        event_count += 1
                        total_reco += 1

            photon_mult.append(event_count)

            # --- true photons ---
            mc_particles = event.get("MCParticles")
            for mc in mc_particles:
                if mc.getPDG() == 22:
                    true_photon_energies.append(mc.getEnergy())
                    total_true += 1

            # --- matching ---
            try:
                links = event.get("RecoMCTruthLink")
                for link in links:
                    reco = link.getFrom()
                    mc = link.getTo()
                    if reco.getPDG() == 22 and mc.getPDG() == 22:
                        if reco.getEnergy() > 1e-6:
                            matched_reco.append(reco.getEnergy())
                            matched_true.append(mc.getEnergy())
                            total_matched += 1
            except:
                pass

    print("\n==============================")
    print(f"Events:            {total_events}")
    print(f"True photons:      {total_true}")
    print(f"Reco photons:      {total_reco}")
    print(f"Matched photons:   {total_matched}")
    if total_true > 0:
        print(f"Reco efficiency:   {total_matched/total_true:.4f}")
    print("==============================\n")

    return {
        "photon_energies": np.array(photon_energies),
        "photon_mult": np.array(photon_mult),
        "matched_reco": np.array(matched_reco),
        "matched_true": np.array(matched_true),
    }


# ----------------------------------------------------------------------
# Run extraction
# ----------------------------------------------------------------------

print("\n--- Geant4 sample ---")
g4 = extract_photons(G4_FILES)

print("\n--- CaloClouds3 sample ---")
cc3 = extract_photons(CC3_FILES)


# ----------------------------------------------------------------------
# Plot 1: Photon energy (log scale)
# ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7,5))

ax.hist(g4["photon_energies"],
        bins=60, range=(0,60),
        histtype="step", linewidth=2,
        label="Geant4", density=True)

ax.hist(cc3["photon_energies"],
        bins=60, range=(0,60),
        histtype="step", linewidth=2,
        label="CaloClouds3", density=True)

ax.set_yscale("log")
ax.set_ylim(1e-4, None)
ax.set_xlabel("Photon energy [GeV]")
ax.set_ylabel("Normalised entries")
ax.set_title("Photon energy distribution (log scale)")
ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "photon_energy_log.png"))
plt.close(fig)


# ----------------------------------------------------------------------
# Plot 2: Photon multiplicity
# ----------------------------------------------------------------------

max_mult = max(g4["photon_mult"].max(), cc3["photon_mult"].max()) + 1
bins = np.arange(0, max_mult+1) - 0.5

fig, ax = plt.subplots(figsize=(7,5))

ax.hist(g4["photon_mult"], bins=bins,
        histtype="step", linewidth=2,
        label="Geant4", density=True)

ax.hist(cc3["photon_mult"], bins=bins,
        histtype="step", linewidth=2,
        label="CaloClouds3", density=True)

ax.set_xlabel("Photons per event")
ax.set_ylabel("Normalised entries")
ax.set_title("Photon multiplicity")
ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "photon_multiplicity_validation.png"))
plt.close(fig)


# ----------------------------------------------------------------------
# Plot 3: Photon response
# ----------------------------------------------------------------------

if len(g4["matched_true"]) > 0:

    g4_resp = g4["matched_reco"] / g4["matched_true"]
    cc3_resp = cc3["matched_reco"] / cc3["matched_true"]

    fig, ax = plt.subplots(figsize=(7,5))

    ax.hist(g4_resp, bins=50, range=(0,2),
            histtype="step", linewidth=2,
            label="Geant4", density=True)

    ax.hist(cc3_resp, bins=50, range=(0,2),
            histtype="step", linewidth=2,
            label="CaloClouds3", density=True)

    ax.set_xlabel("E_reco / E_true")
    ax.set_ylabel("Normalised entries")
    ax.set_title("Photon energy response")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "photon_response.png"))
    plt.close(fig)

print("Photon validation complete.\n")
