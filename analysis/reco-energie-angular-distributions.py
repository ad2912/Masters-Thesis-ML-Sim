"""
reco_photon_distributions.py
=============================
Reco-level photon distributions for two signal definitions.
Compares G4 and CC3 reconstructed photons using PandoraPFOs + RecoMCTruthLink.

Signal definitions:
  A. pi0 daughters  : genStatus==1, E_mc>=10 GeV, parent PDG==111
  B. not-ISR        : genStatus==1, E_mc>=10 GeV, parent PDG!=11

For CC3: split into fast-sim handled vs G4-handled using isHandledByFastSim().

Plots produced (6 total):
  signal_A_energy.png   — reco energy, pi0 daughters
  signal_A_theta.png    — reco theta, pi0 daughters
  signal_A_phi.png      — reco phi,   pi0 daughters
  signal_B_energy.png   — reco energy, not-ISR
  signal_B_theta.png    — reco theta,  not-ISR
  signal_B_phi.png      — reco phi,    not-ISR

All plots: raw counts, NOT normalized.
CC3 shown as: total (dashed) + fast-sim component (filled orange) + G4-handled (green).

Run:
    source ~/source.sh
    python3 reco_photon_distributions.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import edm4hep
from podio import root_io
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
G4_FILE  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-1000events-test_REC.edm4hep.root"
CC3_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-1000events-test_REC.edm4hep.root"

PLOT_DIR = os.path.expanduser("~/thesis-ml-sim/plots/reco_distributions_V2")
os.makedirs(PLOT_DIR, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
PHOTON_PDG   = 22
PI0_PDG      = 111
ELECTRON_PDG = 11
E_THRESH     = 10.0
MAX_EVENTS   = 1000

# ─── Colors — consistent with sim-level scripts ───────────────────────────────
C_G4     = "#2166ac"   # blue       — Geant4
C_CC3    = "#d6604d"   # red        — CC3 total
C_CC_FS  = "#f4a582"   # orange     — CC3 fast-sim handled
C_CC_G4  = "#4dac26"   # green      — CC3 G4-handled

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 12,
    "axes.titlesize"   : 11,
    "axes.labelsize"   : 12,
    "legend.fontsize"  : 10,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 150,
})

# ─── Binning ──────────────────────────────────────────────────────────────────
# Energy: 2 GeV bins 0-130 GeV (same as sim scripts)
# Theta:  5 deg bins 0-180 deg
# Phi:    5 deg bins -180-180 deg
BINS_E     = np.arange(0, 132, 2)
BINS_THETA = np.linspace(0,   180, 37)
BINS_PHI   = np.linspace(-180, 180, 73)

# Detector boundary markers (same as sim scripts)
THETA_VLINES = [10.0, 35.0, 40.0, 140.0, 145.0, 170.0]
PHI_VLINES   = [-180 + 45 * i for i in range(9)]


# ─── Loader ───────────────────────────────────────────────────────────────────
def load_file(label, path, is_cc3=False):
    """
    Single pass. For each PandoraPFO with PDG==22:
      - follow RecoMCTruthLink to get matched MCParticle
      - apply signal cuts
      - store reco energy, reco theta, reco phi
      - store MC energy for reference
      - store fast-sim flag (CC3 only)

    Two signal masks applied after loading:
      A: pi0 daughter
      B: not ISR
    Both branch from the same base: genStatus==1, E_mc>=10 GeV.
    """
    print(f"  Loading {label} ...")

    if not os.path.exists(path):
        print(f"  ERROR: not found: {path}")
        return None

    reader = root_io.Reader(path)

    reco_e_arr     = []
    reco_theta_arr = []
    reco_phi_arr   = []
    mc_e_arr       = []
    is_pi0_arr     = []
    is_isr_arr     = []
    is_fs_arr      = []

    n_events = 0
    for event in reader.get("events"):
        if n_events >= MAX_EVENTS:
            break
        n_events += 1

        # Build pfo_index -> mc map from RecoMCTruthLink
        reco_to_mc = {}
        try:
            for link in event.get("RecoMCTruthLink"):
                reco = link.getFrom()
                mc   = link.getTo()
                reco_to_mc[reco.getObjectID().index] = mc
        except Exception:
            continue

        try:
            pfos = list(event.get("PandoraPFOs"))
        except Exception:
            continue

        for pfo in pfos:
            if pfo.getPDG() != PHOTON_PDG:
                continue

            mc = reco_to_mc.get(pfo.getObjectID().index, None)
            if mc is None:
                continue

            # Base cuts: genStatus==1, MC PDG==22, E_mc>=10 GeV
            if mc.getGeneratorStatus() != 1:
                continue
            if mc.getPDG() != PHOTON_PDG:
                continue
            if mc.getEnergy() < E_THRESH:
                continue

            # Reco kinematics from PFO momentum
            p4    = edm4hep.utils.p4(pfo)
            theta = np.degrees(float(p4.theta()))
            phi   = np.degrees(float(p4.phi()))

            parent_pdgs = [abs(par.getPDG()) for par in mc.getParents()]

            reco_e_arr.append(pfo.getEnergy())
            reco_theta_arr.append(theta)
            reco_phi_arr.append(phi)
            mc_e_arr.append(mc.getEnergy())
            is_pi0_arr.append(PI0_PDG in parent_pdgs)
            is_isr_arr.append(ELECTRON_PDG in parent_pdgs)
            is_fs_arr.append(bool(mc.isHandledByFastSim()) if is_cc3 else False)

    d = {
        "reco_e"    : np.array(reco_e_arr),
        "reco_theta": np.array(reco_theta_arr),
        "reco_phi"  : np.array(reco_phi_arr),
        "mc_e"      : np.array(mc_e_arr),
        "is_pi0"    : np.array(is_pi0_arr,  dtype=bool),
        "is_isr"    : np.array(is_isr_arr,  dtype=bool),
        "is_fs"     : np.array(is_fs_arr,   dtype=bool),
        "n_events"  : n_events,
    }

    n = len(d["reco_e"])
    print(f"    {n_events} events | {n} base photons (genStatus==1, E_mc>=10 GeV)")
    print(f"    pi0 daughters: {d['is_pi0'].sum()}  |  not-ISR: {(~d['is_isr']).sum()}"
          + (f"  |  fast-sim: {d['is_fs'].sum()}" if is_cc3 else ""))
    return d


# ─── Plot function ─────────────────────────────────────────────────────────────
def make_plot(observable, bins, title, filename,
              g4_vals, cc3_total_vals, cc3_fs_vals, cc3_g4_vals,
              n_g4, n_cc3, xlabel):
    """
    Single-panel histogram.
    G4: solid blue step.
    CC3 total: dashed red step.
    CC3 fast-sim: filled orange + step.
    CC3 G4-handled: dash-dot green step.
    Raw counts, not normalized.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(g4_vals, bins=bins, histtype="step",
            color=C_G4, linewidth=2,
            label=f"G4 (N={len(g4_vals)}, {n_g4} ev)")

    ax.hist(cc3_total_vals, bins=bins, histtype="step",
            color=C_CC3, linewidth=2, linestyle="--",
            label=f"CC3 total (N={len(cc3_total_vals)}, {n_cc3} ev)")

    ax.hist(cc3_fs_vals, bins=bins, histtype="stepfilled",
            color=C_CC_FS, alpha=0.5,
            label=f"CC3 fast-sim (N={len(cc3_fs_vals)})")
    ax.hist(cc3_fs_vals, bins=bins, histtype="step",
            color=C_CC_FS, linewidth=1.5)

    ax.hist(cc3_g4_vals, bins=bins, histtype="step",
            color=C_CC_G4, linewidth=2, linestyle="-.",
            label=f"CC3 G4-handled (N={len(cc3_g4_vals)})")

    # Detector boundaries
    if observable == "theta":
        for xv in THETA_VLINES:
            ax.axvline(xv, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
        ax.axvline(THETA_VLINES[0], color="gray", linewidth=0.8,
                   linestyle=":", alpha=0.7, label="detector boundaries")
    elif observable == "phi":
        for xv in PHI_VLINES:
            ax.axvline(xv, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
        ax.axvline(PHI_VLINES[0], color="gray", linewidth=0.8,
                   linestyle=":", alpha=0.7, label="ECal octagon boundaries (45°)")
    elif observable == "energy":
        ax.axvline(E_THRESH, color="black", linewidth=1.2, linestyle=":",
                   label=f"CC3 trigger threshold ({E_THRESH:.0f} GeV)")
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Photons / bin  [raw counts]")
    ax.set_title(title)
    ax.legend(framealpha=0.5, fontsize=9)
    ax.text(0.01, 1.01,
            "ILD reco — 1000 events  |  raw counts, not normalized  |  reco kinematics, MC truth signal cuts",
            transform=ax.transAxes, fontsize=8, color="gray", va="bottom")

    plt.tight_layout()
    out = f"{PLOT_DIR}/{filename}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {filename}")


# ─── Main ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Reco photon distributions — 1000 events")
print("=" * 60 + "\n")

g4  = load_file("G4  (full sim)", G4_FILE,  is_cc3=False)
cc3 = load_file("CC3 (fast sim)", CC3_FILE, is_cc3=True)

if not g4 or not cc3:
    raise SystemExit("ERROR: could not load one or both files.")

print()

# ─── Signal masks ─────────────────────────────────────────────────────────────
# Signal A: pi0 daughters
g4_A  = g4["is_pi0"]
cc3_A = cc3["is_pi0"]
cc3_A_fs  = cc3_A &  cc3["is_fs"]
cc3_A_g4h = cc3_A & ~cc3["is_fs"]

# Signal B: not ISR
g4_B  = ~g4["is_isr"]
cc3_B = ~cc3["is_isr"]
cc3_B_fs  = cc3_B &  cc3["is_fs"]
cc3_B_g4h = cc3_B & ~cc3["is_fs"]

n_g4  = g4["n_events"]
n_cc3 = cc3["n_events"]

# ─── Signal A plots ───────────────────────────────────────────────────────────
print("Signal A (pi0 daughters):")

make_plot(
    "energy", BINS_E,
    title    = "Reco Energy — genStatus==1, E_mc≥10 GeV, π⁰ daughter | G4 vs CC3",
    filename = "signal_A_energy.png",
    g4_vals       = g4["reco_e"][g4_A],
    cc3_total_vals = cc3["reco_e"][cc3_A],
    cc3_fs_vals    = cc3["reco_e"][cc3_A_fs],
    cc3_g4_vals    = cc3["reco_e"][cc3_A_g4h],
    n_g4=n_g4, n_cc3=n_cc3,
    xlabel = "Reco Photon Energy [GeV]",
)

make_plot(
    "theta", BINS_THETA,
    title    = "Reco θ — genStatus==1, E_mc≥10 GeV, π⁰ daughter | G4 vs CC3",
    filename = "signal_A_theta.png",
    g4_vals       = g4["reco_theta"][g4_A],
    cc3_total_vals = cc3["reco_theta"][cc3_A],
    cc3_fs_vals    = cc3["reco_theta"][cc3_A_fs],
    cc3_g4_vals    = cc3["reco_theta"][cc3_A_g4h],
    n_g4=n_g4, n_cc3=n_cc3,
    xlabel = "Reco Photon θ [deg]",
)

make_plot(
    "phi", BINS_PHI,
    title    = "Reco φ — genStatus==1, E_mc≥10 GeV, π⁰ daughter | G4 vs CC3",
    filename = "signal_A_phi.png",
    g4_vals       = g4["reco_phi"][g4_A],
    cc3_total_vals = cc3["reco_phi"][cc3_A],
    cc3_fs_vals    = cc3["reco_phi"][cc3_A_fs],
    cc3_g4_vals    = cc3["reco_phi"][cc3_A_g4h],
    n_g4=n_g4, n_cc3=n_cc3,
    xlabel = "Reco Photon φ [deg]",
)

# ─── Signal B plots ───────────────────────────────────────────────────────────
print("Signal B (not ISR):")

make_plot(
    "energy", BINS_E,
    title    = "Reco Energy — genStatus==1, E_mc≥10 GeV, not ISR | G4 vs CC3",
    filename = "signal_B_energy.png",
    g4_vals       = g4["reco_e"][g4_B],
    cc3_total_vals = cc3["reco_e"][cc3_B],
    cc3_fs_vals    = cc3["reco_e"][cc3_B_fs],
    cc3_g4_vals    = cc3["reco_e"][cc3_B_g4h],
    n_g4=n_g4, n_cc3=n_cc3,
    xlabel = "Reco Photon Energy [GeV]",
)

make_plot(
    "theta", BINS_THETA,
    title    = "Reco θ — genStatus==1, E_mc≥10 GeV, not ISR | G4 vs CC3",
    filename = "signal_B_theta.png",
    g4_vals       = g4["reco_theta"][g4_B],
    cc3_total_vals = cc3["reco_theta"][cc3_B],
    cc3_fs_vals    = cc3["reco_theta"][cc3_B_fs],
    cc3_g4_vals    = cc3["reco_theta"][cc3_B_g4h],
    n_g4=n_g4, n_cc3=n_cc3,
    xlabel = "Reco Photon θ [deg]",
)

make_plot(
    "phi", BINS_PHI,
    title    = "Reco φ — genStatus==1, E_mc≥10 GeV, not ISR | G4 vs CC3",
    filename = "signal_B_phi.png",
    g4_vals       = g4["reco_phi"][g4_B],
    cc3_total_vals = cc3["reco_phi"][cc3_B],
    cc3_fs_vals    = cc3["reco_phi"][cc3_B_fs],
    cc3_g4_vals    = cc3["reco_phi"][cc3_B_g4h],
    n_g4=n_g4, n_cc3=n_cc3,
    xlabel = "Reco Photon φ [deg]",
)

print(f"\n  All 6 plots saved to: {PLOT_DIR}\n")
