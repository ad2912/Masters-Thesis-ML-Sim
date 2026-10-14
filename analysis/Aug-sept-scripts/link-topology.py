# link_topology_raw_study.py
#
# Purpose: build ONE shared per-event table of raw (truth, reco, weight)
# links -- no threshold applied -- and derive BOTH directions' counts from
# it, so n_split (MC side) and n_contribs (reco side) are guaranteed to be
# internally consistent (same underlying links, same weight decode).
#
# MC side: for true-CC3 barrel photons (CC3 file) and phase-space-matched
#   barrel photons (G4 file), how many distinct reco PFOs does each link to
#   at all (weight > 0, no threshold)?
# Reco side: for photon-PDG PFOs that are linked to at least one of our
#   selected photons, how many distinct truth particles link into that PFO
#   AT ALL (not restricted to our selected sample -- a merge partner could
#   be anything: another photon, a charged-pion leak, noise). PFO fast-sim
#   status label (CC3 file only) = flag of its single dominant contributor.
#
# NOT YET RUN.

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio import root_io
import edm4hep
import os
from collections import defaultdict

CC3_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root"
G4_FILE  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-10k-events-test_REC.edm4hep.root"
MAX_EVENTS = 10000
PLOT_DIR = os.path.expanduser("~/thesis-ml-sim/plots/link_topology_raw_study")
os.makedirs(PLOT_DIR, exist_ok=True)

PHOTON_PDG = 22
ELECTRON_PDG = 11
E_MC_MIN = 10.0
THETA_SAFE_LO_DEG = 37.5
THETA_SAFE_HI_DEG = 142.5
PHI_CRACK_CENTERS_DEG = [-157.5 + 45 * k for k in range(8)]
PHI_CRACK_REJECT_WIDTH_DEG = 8.0

C_G4  = "#2166ac"
C_CC3 = "#d6604d"

plt.rcParams.update({
    "font.family": "serif", "font.size": 12, "axes.titlesize": 11,
    "axes.labelsize": 12, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def decode_cluster_weight(raw_weight):
    return (int(round(raw_weight)) // 10000) / 1000.0


def get_parent_pdg(mc_part):
    parents = mc_part.getParents()
    return abs(parents[0].getPDG()) if len(parents) > 0 else 0


def passes_layer_a(mc_part):
    if mc_part.getGeneratorStatus() != 1:
        return False
    if mc_part.getPDG() != PHOTON_PDG:
        return False
    if mc_part.getEnergy() < E_MC_MIN:
        return False
    if get_parent_pdg(mc_part) == ELECTRON_PDG:
        return False
    return True


def in_phi_safe_region(phi_deg):
    for center in PHI_CRACK_CENTERS_DEG:
        raw = ((phi_deg - center + 180) % 360) - 180
        if 0.0 <= raw <= PHI_CRACK_REJECT_WIDTH_DEG:
            return False
    return True


def passes_layer_b(mc_part):
    p4 = edm4hep.utils.p4(mc_part)
    theta_deg = np.degrees(float(p4.theta()))
    phi_deg = np.degrees(float(p4.phi()))
    if not (THETA_SAFE_LO_DEG <= theta_deg <= THETA_SAFE_HI_DEG):
        return False
    if not in_phi_safe_region(phi_deg):
        return False
    return True


def process_file(filepath, label, tag_fastsim):
    """
    One pass. Builds, per event, the full raw link table restricted to
    photon-PDG PandoraPFOs on the reco side (no restriction on the truth
    side of the raw table -- we need ALL contributors for the reco-side
    count).

    Returns:
      n_split_true_cc3, n_split_true_g4fb (or n_split_all if tag_fastsim=False):
        lists of raw link counts, one per selected truth photon.
      n_contribs_by_pfo_fastsim_status: dict "true_cc3"/"true_g4fb"/"g4file" ->
        list of raw contributor counts, one per relevant PFO.
    """
    print(f"\n{'='*60}\n  Processing: {label}\n{'='*60}")
    reader = root_io.Reader(filepath)

    n_split_true_cc3 = []
    n_split_true_g4fb = []
    n_split_all = []  # used when tag_fastsim=False (G4 file)

    n_contribs_true_cc3_pfo = []
    n_contribs_true_g4fb_pfo = []
    n_contribs_g4file_pfo = []

    n_events = 0
    for event in reader.get("events"):
        if n_events >= MAX_EVENTS:
            break
        n_events += 1

        selected_photons = {}  # mc local id -> is_fastsim (or None)
        for mc in event.get("MCParticles"):
            if not passes_layer_a(mc):
                continue
            if not passes_layer_b(mc):
                continue
            mc_id = mc.getObjectID().index
            is_fastsim = bool(mc.isHandledByFastSim()) if tag_fastsim else None
            selected_photons[mc_id] = is_fastsim

        photon_pfo_ids = set()
        for pfo in event.get("PandoraPFOs"):
            if pfo.getPDG() == PHOTON_PDG:
                photon_pfo_ids.add(pfo.getObjectID().index)

        # Raw links, restricted to reco side being a photon-PDG PFO.
        # mc side is UNRESTRICTED here (need full contributor picture).
        links_by_mc = defaultdict(set)     # mc_id -> set of reco_id (weight>0)
        links_by_reco = defaultdict(dict)  # reco_id -> {mc_id: weight}

        for link in event.get("RecoMCTruthLink"):
            reco = link.getFrom()
            reco_id = reco.getObjectID().index
            if reco_id not in photon_pfo_ids:
                continue
            mc = link.getTo()
            mc_id = mc.getObjectID().index
            w = decode_cluster_weight(link.getWeight())
            if w <= 0:
                continue
            links_by_mc[mc_id].add(reco_id)
            links_by_reco[reco_id][mc_id] = w

        # --- MC side: n_split for selected photons ---
        for mc_id, is_fastsim in selected_photons.items():
            n = len(links_by_mc.get(mc_id, set()))
            if tag_fastsim:
                if is_fastsim:
                    n_split_true_cc3.append(n)
                else:
                    n_split_true_g4fb.append(n)
            else:
                n_split_all.append(n)

        # --- Reco side: n_contribs for PFOs linked to >=1 selected photon ---
        relevant_pfo_ids = set()
        for mc_id in selected_photons:
            relevant_pfo_ids |= links_by_mc.get(mc_id, set())

        for reco_id in relevant_pfo_ids:
            contributors = links_by_reco.get(reco_id, {})
            n_contribs = len(contributors)
            if tag_fastsim:
                # Label this PFO by its dominant contributor's fastsim status
                if not contributors:
                    continue
                dominant_mc_id = max(contributors, key=contributors.get)
                # Need to re-check flag for the dominant contributor -- it may
                # not be in selected_photons if it's a merge partner outside
                # our Layer A+B selection. Re-derive from the event's MCParticles.
                dominant_is_fastsim = selected_photons.get(dominant_mc_id, None)
                if dominant_is_fastsim is None:
                    # Dominant contributor wasn't in our selected sample --
                    # look it up directly.
                    for mc in event.get("MCParticles"):
                        if mc.getObjectID().index == dominant_mc_id:
                            dominant_is_fastsim = bool(mc.isHandledByFastSim())
                            break
                if dominant_is_fastsim:
                    n_contribs_true_cc3_pfo.append(n_contribs)
                else:
                    n_contribs_true_g4fb_pfo.append(n_contribs)
            else:
                n_contribs_g4file_pfo.append(n_contribs)

    print(f"  Events processed: {n_events}")
    if tag_fastsim:
        print(f"  true-CC3 photons (n_split entries)      : {len(n_split_true_cc3)}")
        print(f"  true-G4-fallback photons (n_split entries): {len(n_split_true_g4fb)}")
        print(f"  PFOs dominated by true-CC3               : {len(n_contribs_true_cc3_pfo)}")
        print(f"  PFOs dominated by true-G4-fallback        : {len(n_contribs_true_g4fb_pfo)}")
    else:
        print(f"  G4-file photons (n_split entries)  : {len(n_split_all)}")
        print(f"  G4-file PFOs (n_contribs entries)  : {len(n_contribs_g4file_pfo)}")

    return dict(
        n_split_true_cc3=n_split_true_cc3, n_split_true_g4fb=n_split_true_g4fb,
        n_split_all=n_split_all,
        n_contribs_true_cc3_pfo=n_contribs_true_cc3_pfo,
        n_contribs_true_g4fb_pfo=n_contribs_true_g4fb_pfo,
        n_contribs_g4file_pfo=n_contribs_g4file_pfo,
    )


def print_dist(name, arr):
    arr = np.array(arr)
    if len(arr) == 0:
        print(f"  {name}: no entries")
        return
    vals, counts = np.unique(arr, return_counts=True)
    print(f"  {name} (N={len(arr)}, mean={arr.mean():.3f}):")
    for v, c in zip(vals, counts):
        print(f"    {int(v)}: {c}  ({100*c/len(arr):.1f}%)")


cc3_results = process_file(CC3_FILE, "CC3 file", tag_fastsim=True)
g4_results  = process_file(G4_FILE,  "G4 file",  tag_fastsim=False)

print("\n" + "="*60)
print("  MC SIDE: raw n_split distributions (no threshold)")
print("="*60)
print_dist("true-CC3 (CC3 file)", cc3_results["n_split_true_cc3"])
print_dist("true-G4-fallback (CC3 file)", cc3_results["n_split_true_g4fb"])
print_dist("phase-space-matched (G4 file)", g4_results["n_split_all"])

print("\n" + "="*60)
print("  RECO SIDE: raw n_contribs distributions (no threshold)")
print("="*60)
print_dist("PFO dominated by true-CC3", cc3_results["n_contribs_true_cc3_pfo"])
print_dist("PFO dominated by true-G4-fallback", cc3_results["n_contribs_true_g4fb_pfo"])
print_dist("PFO (G4 file, phase-space-matched)", g4_results["n_contribs_g4file_pfo"])

# --- Plots ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

max_split = 6
bins = np.arange(0.5, max_split + 1.5, 1)
axes[0].hist(cc3_results["n_split_true_cc3"], bins=bins, histtype="step", color=C_CC3,
             linewidth=2, density=True, label=f"true-CC3 (N={len(cc3_results['n_split_true_cc3'])})")
axes[0].hist(g4_results["n_split_all"], bins=bins, histtype="step", color=C_G4,
             linewidth=2, density=True, label=f"G4, phase-space-matched (N={len(g4_results['n_split_all'])})")
axes[0].set_xlabel("n_split (raw, weight>0, no threshold)")
axes[0].set_ylabel("Normalized density")
axes[0].set_title("MC side: how many PFOs does each\ntruth photon link to?", fontsize=10)
axes[0].legend(fontsize=8)

max_contrib = 6
axes[1].hist(cc3_results["n_contribs_true_cc3_pfo"], bins=bins, histtype="step", color=C_CC3,
             linewidth=2, density=True, label=f"CC3-dominated PFO (N={len(cc3_results['n_contribs_true_cc3_pfo'])})")
axes[1].hist(g4_results["n_contribs_g4file_pfo"], bins=bins, histtype="step", color=C_G4,
             linewidth=2, density=True, label=f"G4 PFO (N={len(g4_results['n_contribs_g4file_pfo'])})")
axes[1].set_xlabel("n_contribs (raw, weight>0, no threshold)")
axes[1].set_title("Reco side: how many distinct truth\nparticles feed into each PFO?", fontsize=10)
axes[1].legend(fontsize=8)

plt.tight_layout()
out = f"{PLOT_DIR}/raw_link_topology.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out}")
