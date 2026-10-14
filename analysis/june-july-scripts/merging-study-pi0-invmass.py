"""
diagnostic_1reco_pi0.py
=======================
For each π⁰ that lands in the "1 reco photon" category,
find the UNMATCHED MC photon daughter and ask:
  - Does it have ANY reco PFO linked to it at all (any weight)?
  - If yes → what is the best weight? Did it fail our weight>0.5 cut?
  - Is the best-match PFO already claimed by a different MC particle
    (i.e., the PFO's highest-weight match is NOT our daughter)?
  - Is the unmatched daughter outside CC3/detector acceptance (E<1 GeV, |cos θ|>0.99)?

Categories:
  TYPE_A1: Daughter has a reco PFO linked to it, but weight ≤ 0.5 → weight cut failure
  TYPE_A2: Daughter has a reco PFO linked but it's "stolen" — that PFO's best match
           is a different MC particle (our daughter is not the highest-weight match)
  TYPE_B:  Daughter has a reco PFO linked with weight > 0.5, but that PFO is
           already matched to the FIRST daughter (same reco PFO objectID) → merging
  TYPE_C:  Daughter has zero reco PFO links at all → never reconstructed
  TYPE_C2: Daughter has zero links AND E_mc < 1 GeV or outside barrel → acceptance loss
"""

import numpy as np
from podio import root_io
from collections import defaultdict
import math

G4_FILE  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-10k-events-test_REC.edm4hep.root"
CC3_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root"
MAX_EVENTS = 10000
WEIGHT_CUT = 0.5
PHOTON_PDG = 22
PI0_PDG    = 111

def costheta(mc):
    mom = mc.getMomentum()
    p = math.sqrt(mom.x**2 + mom.y**2 + mom.z**2)
    return abs(mom.z / p) if p > 0 else 0.0

def run_diagnostic(filepath, label):
    print(f"\n{'='*60}")
    print(f"  1-reco-photon diagnostic: {label}")
    print(f"{'='*60}")

    reader   = root_io.Reader(filepath)
    n_events = 0

    counts = {
        "TYPE_A1_weight_cut_failure"     : 0,
        "TYPE_A2_stolen_by_other_mc"     : 0,
        "TYPE_B_merged_same_pfo"         : 0,
        "TYPE_C_no_links_at_all"         : 0,
        "TYPE_C2_no_links_out_of_acc"    : 0,
    }

    for event in reader.get("events"):
        if n_events >= MAX_EVENTS:
            break
        n_events += 1

        # ── Build ALL reco→MC links (not just best weight) ──────────────────
        # For each reco PFO: list of (mc_obj_id, weight)
        reco_all_links = defaultdict(list)
        # For each MC particle: list of (reco_obj_id, weight)
        mc_all_links   = defaultdict(list)

        for link in event.get("RecoMCTruthLink"):
            reco   = link.getFrom()
            mc     = link.getTo()
            weight = link.getWeight()
            rid    = reco.getObjectID().index
            mid    = mc.getObjectID().index
            reco_all_links[rid].append((mid, weight))
            mc_all_links[mid].append((rid, weight))

        # Strategy A: best-weight match per reco PFO
        reco_best_mc = {}
        reco_best_wgt = {}
        for link in event.get("RecoMCTruthLink"):
            reco   = link.getFrom()
            mc     = link.getTo()
            weight = link.getWeight()
            rid    = reco.getObjectID().index
            if rid not in reco_best_wgt or weight > reco_best_wgt[rid]:
                reco_best_mc[rid]  = mc.getObjectID().index
                reco_best_wgt[rid] = weight

        # ── Build π⁰ → MC daughters map ─────────────────────────────────────
        pi0_mc_daughters = defaultdict(list)  # pi0_obj_id → [mc_obj_id, ...]
        mc_obj_by_id = {}

        for mc in event.get("MCParticles"):
            mc_obj_by_id[mc.getObjectID().index] = mc
            if mc.getGeneratorStatus() != 1:
                continue
            if mc.getPDG() != PHOTON_PDG:
                continue
            parents = mc.getParents()
            if len(parents) == 0:
                continue
            parent = parents[0]
            if parent.getPDG() != PI0_PDG:
                continue
            pi0_obj_id = parent.getObjectID().index
            pi0_mc_daughters[pi0_obj_id].append(mc.getObjectID().index)

        # ── Build π⁰ → matched reco PFOs map (same logic as main script) ────
        pi0_reco_photons = defaultdict(list)  # pi0_obj_id → [reco_obj_id, ...]

        for pfo in event.get("PandoraPFOs"):
            if pfo.getPDG() != PHOTON_PDG:
                continue
            rid = pfo.getObjectID().index
            if rid not in reco_best_mc:
                continue
            if reco_best_wgt[rid] <= WEIGHT_CUT:
                continue
            mc_id = reco_best_mc[rid]
            if mc_id not in mc_obj_by_id:
                continue
            mc = mc_obj_by_id[mc_id]
            if mc.getGeneratorStatus() != 1 or mc.getPDG() != PHOTON_PDG:
                continue
            parents = mc.getParents()
            if len(parents) == 0:
                continue
            parent = parents[0]
            if parent.getPDG() != PI0_PDG:
                continue
            pi0_reco_photons[parent.getObjectID().index].append((rid, mc_id))

        # ── Find 1-reco-photon π⁰s and classify the unmatched daughter ──────
        for pi0_id, reco_list in pi0_reco_photons.items():
            if len(reco_list) != 1:
                continue

            # All MC daughters of this π⁰
            all_daughters = pi0_mc_daughters.get(pi0_id, [])
            if len(all_daughters) != 2:
                continue  # skip unusual cases (should be rare)

            matched_mc_id   = reco_list[0][1]   # MC id of the matched daughter
            matched_reco_id = reco_list[0][0]   # reco PFO id of the matched daughter

            # Find the UNMATCHED daughter
            unmatched_mc_id = None
            for d_id in all_daughters:
                if d_id != matched_mc_id:
                    unmatched_mc_id = d_id
                    break
            if unmatched_mc_id is None:
                # Both daughters have same objectID? Shouldn't happen, skip
                continue

            unmatched_mc = mc_obj_by_id.get(unmatched_mc_id)
            if unmatched_mc is None:
                continue

            # Does the unmatched daughter have ANY reco links?
            links_for_unmatched = mc_all_links.get(unmatched_mc_id, [])

            if len(links_for_unmatched) == 0:
                # TYPE C: never linked to any reco PFO
                ct = costheta(unmatched_mc)
                e  = unmatched_mc.getEnergy()
                if e < 1.0 or ct > 0.99:
                    counts["TYPE_C2_no_links_out_of_acc"] += 1
                else:
                    counts["TYPE_C_no_links_at_all"] += 1

            else:
                # Has at least one reco link — find best
                best_rid, best_w = max(links_for_unmatched, key=lambda x: x[1])

                # TYPE B: the best-linked reco PFO is the SAME as the matched daughter's PFO
                # → both daughters point to the same reco PFO = merging
                if best_rid == matched_reco_id:
                    counts["TYPE_B_merged_same_pfo"] += 1

                # Has a different reco PFO linked
                elif best_w <= WEIGHT_CUT:
                    # TYPE A1: linked but weight too low → weight cut failure
                    counts["TYPE_A1_weight_cut_failure"] += 1

                else:
                    # Weight > 0.5 but didn't appear in our selected photons.
                    # Check if that reco PFO's best MC match is NOT our daughter
                    # (i.e., "stolen" by a different MC particle)
                    pfo_best_mc = reco_best_mc.get(best_rid)
                    if pfo_best_mc != unmatched_mc_id:
                        counts["TYPE_A2_stolen_by_other_mc"] += 1
                    else:
                        # weight > 0.5, PFO's best match IS our daughter,
                        # but it didn't appear in reco list — shouldn't happen
                        # if main script logic is correct. Flag it.
                        counts["TYPE_A2_stolen_by_other_mc"] += 1  # flag as anomaly

    total_1reco = sum(counts.values())
    print(f"\n  Total 1-reco π⁰s classified: {total_1reco}")
    print(f"  {'─'*50}")
    for name, val in counts.items():
        pct = 100*val/max(total_1reco, 1)
        print(f"  {name:<40}: {val:5d}  ({pct:.1f}%)")
    print(f"\n  KEY: If TYPE_B dominates → real cluster merging")
    print(f"       If TYPE_A1/A2 dominates → link/selection artifact")
    print(f"       If TYPE_C dominates → acceptance/threshold loss")

run_diagnostic(G4_FILE,  "Geant4")
run_diagnostic(CC3_FILE, "CC3")
print("\nDone.\n")
