"""
mc_angle_analysis.py
====================
MC truth angular distributions of photons from the SIM files.

Reads MCParticles directly from the two SIM files — no reco needed.
Produces 4 clean plots:
  1. Theta — all MC photons
  2. Theta — MC photons >= 10 GeV (CaloClouds trigger region)
  3. Phi   — all MC photons
  4. Phi   — MC photons >= 10 GeV

Run with:
    source ~/source.sh
    python3 mc_angle_analysis.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio import root_io
import os
import edm4hep

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
SIM_G4   = "/data/dust/user/alimuham/thesis/sim/tau_pi0_SIM_geant4.edm4hep.root"
SIM_CC   = "/data/dust/user/alimuham/thesis/sim/tau_pi0_SIM_caloclouds.edm4hep.root"
PLOT_DIR = os.path.expanduser("~/thesis-ml-sim/plots/mc_angles")
os.makedirs(PLOT_DIR, exist_ok=True)

PHOTON_PDG    = 22
ML_ENERGY_GEV = 10.0

C_G4 = "#2166ac"   # blue
C_CC = "#d6604d"   # red

# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 12,
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 12,
    "legend.fontsize"  : 11,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 150,
})

def sim_label(ax):
    ax.text(0.01, 1.01, "ILD simulation (preliminary)",
            transform=ax.transAxes, fontsize=9, color="gray", va="bottom")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MC PHOTON ANGLES
# ─────────────────────────────────────────────────────────────────────────────
def load_mc_angles(path, label):
    reader = root_io.Reader(path)
    all_theta, all_phi = [], []
    hi_theta,  hi_phi  = [], []

    for event in reader.get("events"):
        for p in event.get("MCParticles"):
            if p.getPDG() != PHOTON_PDG:
                continue
            p4=edm4hep.utils.p4(p)
            theta= np.degrees(p4.theta())
            phi= np.degrees(p4.phi())
           
            
            mom = p.getMomentum()
#            px, py, pz = mom.x, mom.y, mom.z
#            pmag = np.sqrt(px**2 + py**2 + pz**2)
#            if pmag == 0:
#                continue
#           theta = np.degrees(np.arccos(np.clip(pz / pmag, -1, 1)))
#            phi   = np.degrees(np.arctan2(py, px))
            all_theta.append(theta)
            all_phi.append(phi)
            if p.getEnergy() >= ML_ENERGY_GEV:
                hi_theta.append(theta)
                hi_phi.append(phi)

    print(f"  {label}: {len(all_theta)} MC photons  "
          f"({len(hi_theta)} above {ML_ENERGY_GEV} GeV)")
    return (np.array(all_theta), np.array(all_phi),
            np.array(hi_theta),  np.array(hi_phi))

print("Loading SIM files...")
g4_th, g4_ph, g4_th_hi, g4_ph_hi = load_mc_angles(SIM_G4, "SIM Geant4    ")
cc_th, cc_ph, cc_th_hi, cc_ph_hi = load_mc_angles(SIM_CC, "SIM CaloClouds")

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
bins_th = np.linspace(0,    180, 50)
bins_ph = np.linspace(-180, 180, 50)

def make_angle_plot(g4_data, cc_data, bins, xlabel, title, filename, n_g4, n_cc):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(g4_data, bins=bins, histtype="step", color=C_G4,
            linewidth=2, label=f"SIM Geant4  (N={n_g4})")
    ax.hist(cc_data, bins=bins, histtype="step", color=C_CC,
            linewidth=2, linestyle="--", label=f"SIM CaloClouds  (N={n_cc})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Photons / bin")
    ax.set_title(title)
    ax.legend()
    sim_label(ax)
    plt.tight_layout()
    fig.savefig(f"{PLOT_DIR}/{filename}")
    plt.close()
    print(f"  ✓  {filename}")

print("\nSaving plots...")

make_angle_plot(
    g4_th, cc_th, bins_th,
    xlabel   = "MC Photon Polar Angle θ [degrees]",
    title    = "MC Truth Photon θ — All Energies",
    filename = "theta_all.png",
    n_g4=len(g4_th), n_cc=len(cc_th)
)

make_angle_plot(
    g4_th_hi, cc_th_hi, bins_th,
    xlabel   = "MC Photon Polar Angle θ [degrees]",
    title    = f"MC Truth Photon θ — E ≥ {ML_ENERGY_GEV} GeV  (CaloClouds trigger region)",
    filename = "theta_above10GeV.png",
    n_g4=len(g4_th_hi), n_cc=len(cc_th_hi)
)

make_angle_plot(
    g4_ph, cc_ph, bins_ph,
    xlabel   = "MC Photon Azimuthal Angle φ [degrees]",
    title    = "MC Truth Photon φ — All Energies",
    filename = "phi_all.png",
    n_g4=len(g4_ph), n_cc=len(cc_ph)
)

make_angle_plot(
    g4_ph_hi, cc_ph_hi, bins_ph,
    xlabel   = "MC Photon Azimuthal Angle φ [degrees]",
    title    = f"MC Truth Photon φ — E ≥ {ML_ENERGY_GEV} GeV  (CaloClouds trigger region)",
    filename = "phi_above10GeV.png",
    n_g4=len(g4_ph_hi), n_cc=len(cc_ph_hi)
)

print(f"\n  All plots saved to: {PLOT_DIR}")
