"""
pi0_mass_reco.py

Physics question:
    For each truth pi0 -> gamma gamma from a tau decay (any depth in decay chain),
    find both daughter photons as reconstructed PandoraPFOs via MCTruthRecoLink
    (highest-weight match), compute the two-photon invariant mass, and compare
    the mass spectrum between G4 and CC3.

    Within CC3, stratify by fast-sim handling of the two photons:
        - both handled by CC3
        - one CC3, one G4-fallback
        - both G4-fallback

    Expected: peak at ~135 MeV. Shift or broadening in CC3 = energy/angular bias.

Cuts applied:
    - pi0 must descend from a tau (PDG=15) anywhere up the mother chain
    - Both daughter photons must have a matched PFO (highest-weight link)
    - No explicit energy cut on photons here — we want to see what reco gives us,
      including sub-10 GeV photons that Geant4 handles in CC3

Output:
    plots/pi0_mass/pi0_mass_spectrum.png
    plots/pi0_mass/pi0_mass_cc3_stratified.png
    plots/pi0_mass/diagnostic_link_weights.png  (first 5 events, for validation)
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import podio
import podio.root_io

# ── paths ────────────────────────────────────────────────────────────────────
RECO_DIR = Path("/data/dust/user/alimuham/thesis/reco")
G4_FILE  = RECO_DIR / "tau-pi0-g4-taureco-1000events-test_REC.edm4hep.root"
CC3_FILE = RECO_DIR / "tau-pi0-cc3-taureco-1000events-test_REC.edm4hep.root"

PLOT_DIR = Path("/afs/desy.de/user/a/alimuham/thesis-ml-sim/plots/pi0_mass")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
PDG_TAU    = 15
PDG_PI0    = 111
PDG_PHOTON = 22

# Mass axis: 0 to 300 MeV, 60 bins -> 5 MeV per bin
# Motivated by: pi0 mass = 135 MeV, ILD ECAL resolution ~17%/sqrt(E) ~ few MeV
# at E~60 GeV -> ~2 MeV resolution, but reco mass folded over both photons,
# expect peak width O(10 MeV). 5 MeV bins are fine for 1000 events.
MASS_BINS = np.linspace(0, 0.300, 61)   # GeV


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def has_tau_ancestor(mc_particle):
    """
    Walk up the mother chain from mc_particle.
    Return True if any ancestor has PDG abs = 15 (tau).
    Stops at generator-level particles (no mothers) or if tau found.
    """
    visited = set()
    queue = list(mc_particle.getParents())
    while queue:
        parent = queue.pop()
        pid = id(parent)
        if pid in visited:
            continue
        visited.add(pid)
        if abs(parent.getPDG()) == PDG_TAU:
            return True
        queue.extend(list(parent.getParents()))
    return False


def build_mc_to_pfo_map(link_collection):
    """
    Build a dict: mc_particle_object_id -> (pfo, weight)
    keeping only the highest-weight link per MCParticle.

    Link type: Link<ReconstructedParticle, MCParticle>
    so link.getFrom() = PFO, link.getTo() = MCParticle
    """
    mc_to_pfo = {}   # mc_id -> (pfo, weight)
    for link in link_collection:
        pfo    = link.getFrom()
        mc     = link.getTo()
        weight = link.getWeight()
        mc_id  = mc.id()
        if mc_id not in mc_to_pfo or weight > mc_to_pfo[mc_id][1]:
            mc_to_pfo[mc_id] = (pfo, weight)
    return mc_to_pfo


def invariant_mass(pfo_a, pfo_b):
    """
    Compute invariant mass of two PFOs from their 4-momenta.
    PFO momentum is a 3-vector; energy = pfo.getEnergy().
    Returns mass in GeV, or None if unphysical.
    """
    p_a = pfo_a.getMomentum()
    p_b = pfo_b.getMomentum()
    e_a = pfo_a.getEnergy()
    e_b = pfo_b.getEnergy()

    e_sum  = e_a + e_b
    px_sum = p_a.x + p_b.x
    py_sum = p_a.y + p_b.y
    pz_sum = p_a.z + p_b.z

    m2 = e_sum**2 - px_sum**2 - py_sum**2 - pz_sum**2
    if m2 < 0:
        return None
    return np.sqrt(m2)


def fast_sim_label(photon_a, photon_b):
    """
    Given two MCParticle photons, return a string describing
    which were handled by CC3.
    """
    a = photon_a.isHandledByFastSim()
    b = photon_b.isHandledByFastSim()
    if a and b:
        return "both_cc3"
    elif a or b:
        return "mixed"
    else:
        return "both_g4"


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC: print link weights for first N events
# ═══════════════════════════════════════════════════════════════════════════════

def print_link_weight_diagnostic(filepath, n_events=5):
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC: MCTruthRecoLink weights — {filepath.name}")
    print(f"{'='*60}")
    reader = podio.root_io.Reader(str(filepath))
    for i, event in enumerate(reader.get("events")):
        if i >= n_events:
            break
        links = event.get("MCTruthRecoLink")
        print(f"\n  Event {i}: {len(links)} links")
        for j, link in enumerate(links):
            pfo = link.getFrom()
            mc  = link.getTo()
            w   = link.getWeight()
            print(f"    link {j:2d}: PFO E={pfo.getEnergy():.2f} GeV  "
                  f"MC PDG={mc.getPDG():5d}  weight={w:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def process_file(filepath, label):
    """
    Returns:
        masses_all   : list of float (GeV) — all matched pi0s
        masses_strat : dict {"both_cc3": [...], "mixed": [...], "both_g4": [...]}
        counters     : dict with diagnostic counts
    """
    reader = podio.root_io.Reader(str(filepath))

    masses_all   = []
    masses_strat = {"both_cc3": [], "mixed": [], "both_g4": []}

    counters = {
        "events":              0,
        "pi0s_from_tau":       0,
        "pi0s_both_matched":   0,
        "pi0s_one_matched":    0,
        "pi0s_none_matched":   0,
        "pi0s_unphysical_m2":  0,
    }

    for event in reader.get("events"):
        counters["events"] += 1

        mc_particles = event.get("MCParticles")
        link_col     = event.get("MCTruthRecoLink")

        # Build reverse lookup: mc_id -> (pfo, weight)
        mc_to_pfo = build_mc_to_pfo_map(link_col)

        # Find all pi0s that descend from a tau
        for mc in mc_particles:
            if mc.getPDG() != PDG_PI0:
                continue
            if not has_tau_ancestor(mc):
                continue

            counters["pi0s_from_tau"] += 1

            # Get daughter photons
            daughters = list(mc.getDaughters())
            photon_daughters = [d for d in daughters if d.getPDG() == PDG_PHOTON]

            if len(photon_daughters) != 2:
                # Not a clean pi0->gg decay (e.g. Dalitz), skip
                continue

            ph_a, ph_b = photon_daughters

            # Look up PFOs
            pfo_a_result = mc_to_pfo.get(ph_a.id())
            pfo_b_result = mc_to_pfo.get(ph_b.id())

            if pfo_a_result is None and pfo_b_result is None:
                counters["pi0s_none_matched"] += 1
                continue
            elif pfo_a_result is None or pfo_b_result is None:
                counters["pi0s_one_matched"] += 1
                continue

            counters["pi0s_both_matched"] += 1

            pfo_a, w_a = pfo_a_result
            pfo_b, w_b = pfo_b_result

            mass = invariant_mass(pfo_a, pfo_b)
            if mass is None:
                counters["pi0s_unphysical_m2"] += 1
                continue

            masses_all.append(mass)

            # Stratify by fast-sim flag (only meaningful for CC3 file,
            # but we call it for both — for G4 file isHandledByFastSim()
            # should always return False)
            strat = fast_sim_label(ph_a, ph_b)
            masses_strat[strat].append(mass)

    print(f"\n[{label}] Processing complete:")
    for k, v in counters.items():
        print(f"  {k:<25s}: {v}")
    print(f"  masses collected      : {len(masses_all)}")

    return masses_all, masses_strat


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_overlay(masses_g4, masses_cc3):
    """Main comparison: G4 vs CC3 total, raw counts."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(masses_g4,  bins=MASS_BINS, histtype="step", linewidth=2,
            color="blue",  label=f"G4  (N={len(masses_g4)})")
    ax.hist(masses_cc3, bins=MASS_BINS, histtype="step", linewidth=2,
            color="red",   label=f"CC3 (N={len(masses_cc3)})")

    ax.axvline(0.135, color="gray", linestyle="--", linewidth=1, label="m(π⁰)=135 MeV")

    ax.set_xlabel("Invariant mass of matched photon PFO pair [GeV]", fontsize=12)
    ax.set_ylabel("π⁰ candidates (raw counts)", fontsize=12)
    ax.set_title(
        "π⁰ → γγ invariant mass from truth-matched PandoraPFOs\n"
        "V2 reco, 1000 events | π⁰ from τ decay chain | highest-weight MCTruthRecoLink match",
        fontsize=10
    )
    ax.legend(fontsize=11)
    ax.set_xlim(0, 0.300)

    out = PLOT_DIR / "pi0_mass_spectrum.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


def plot_cc3_stratified(masses_strat):
    """CC3 only, broken down by fast-sim handling of the two photons."""
    colors = {
        "both_cc3": "orange",
        "mixed":    "purple",
        "both_g4":  "green",
    }
    labels = {
        "both_cc3": "Both photons CC3-handled",
        "mixed":    "One CC3, one G4-fallback",
        "both_g4":  "Both G4-fallback",
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for key in ["both_cc3", "mixed", "both_g4"]:
        m = masses_strat[key]
        if len(m) == 0:
            print(f"  WARNING: no entries for stratum '{key}'")
            continue
        ax.hist(m, bins=MASS_BINS, histtype="step", linewidth=2,
                color=colors[key], label=f"{labels[key]} (N={len(m)})")

    ax.axvline(0.135, color="gray", linestyle="--", linewidth=1, label="m(π⁰)=135 MeV")

    ax.set_xlabel("Invariant mass of matched photon PFO pair [GeV]", fontsize=12)
    ax.set_ylabel("π⁰ candidates (raw counts)", fontsize=12)
    ax.set_title(
        "CC3: π⁰ → γγ invariant mass stratified by fast-sim handling\n"
        "V2 reco, 1000 events | π⁰ from τ decay chain | highest-weight MCTruthRecoLink match",
        fontsize=10
    )
    ax.legend(fontsize=10)
    ax.set_xlim(0, 0.300)

    out = PLOT_DIR / "pi0_mass_cc3_stratified.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Step 0: diagnostic — always run this first to validate link weights
    print_link_weight_diagnostic(CC3_FILE, n_events=5)

    # Step 1: process both files
    print("\nProcessing G4...")
    masses_g4,  strat_g4  = process_file(G4_FILE,  "G4")

    print("\nProcessing CC3...")
    masses_cc3, strat_cc3 = process_file(CC3_FILE, "CC3")

    # Step 2: sanity check before plotting
    print(f"\nSanity check:")
    print(f"  G4  masses: {len(masses_g4)}")
    print(f"  CC3 masses: {len(masses_cc3)}")
    print(f"  CC3 strat — both_cc3: {len(strat_cc3['both_cc3'])}, "
          f"mixed: {len(strat_cc3['mixed'])}, "
          f"both_g4: {len(strat_cc3['both_g4'])}")

    if len(masses_g4) == 0 or len(masses_cc3) == 0:
        print("\nERROR: no masses collected for one or both files.")
        print("Check the diagnostic output above and the counter printout.")
        sys.exit(1)

    # Step 3: plots
    plot_overlay(masses_g4, masses_cc3)
    plot_cc3_stratified(strat_cc3)

    print("\nDone.")
