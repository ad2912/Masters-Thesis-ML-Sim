# link_topology_floor_scan.py
#
# Extends link_topology_raw_study.py: scans several weight floors instead of
# using "any nonzero weight", to separate real splitting/merging from
# hit-sharing crosstalk noise. Also adds three cheap extras from the same
# single pass:
#   1. Weight-sum sanity check: do a PFO's incoming cluster_weights sum to
#      ~1.0? (verifies our understanding of what cluster_weight represents)
#   2. Completeness: for each selected truth photon, sum (cluster_weight *
#      PFO energy) across ALL its linked PFOs, divide by E_mc -- how much of
#      the real energy was recovered in total, not just via the best link.
#   3. Second-largest contributor weight, for PFOs with >=2 contributors
#      above the floor -- measures HOW confused a merge is, not just whether
#      one exists.
#
# NOT YET RUN. CC3 100k filename is a PLACEHOLDER -- confirm before running
# on 100k; 10k paths are filled in and safe to run now.

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio import root_io
import edm4hep
import os
from collections import defaultdict

RUN_SIZE = "100k"   # switch to "100k" once CC3 100k filename is confirmed

RUN_TABLE = {
    "10k": dict(
        cc3="/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root",
        g4="/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-10k-events-test_REC.edm4hep.root",
        max_events=10000, label="10k events",
    ),
    "100k": dict(
        cc3="/data/dust/user/alimuham/thesis/reco/tau-pi0-caloclouds-100kevents-reco-test_REC.edm4hep.root",
        g4="/data/dust/user/alimuham/thesis/reco/tau-pi0-geant4-100kevents-reco-test_REC.edm4hep.root",
        max_events=100000, label="100k events",
    ),
}
_cfg = RUN_TABLE[RUN_SIZE]
CC3_FILE, G4_FILE, MAX_EVENTS, RUN_LABEL = _cfg["cc3"], _cfg["g4"], _cfg["max_events"], _cfg["label"]

PLOT_DIR = os.path.expanduser(f"~/thesis-ml-sim/plots/link_topology_floor_scan_{RUN_SIZE}")
os.makedirs(PLOT_DIR, exist_ok=True)

PHOTON_PDG = 22
ELECTRON_PDG = 11
E_MC_MIN = 10.0
THETA_SAFE_LO_DEG = 37.5
THETA_SAFE_HI_DEG = 142.5
PHI_CRACK_CENTERS_DEG = [-157.5 + 45 * k for k in range(8)]
PHI_CRACK_REJECT_WIDTH_DEG = 8.0
WEIGHT_FLOORS = [0.0, 0.02, 0.05, 0.1, 0.2]

C_G4, C_CC3 = "#2166ac", "#d6604d"
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
    if mc_part.getGeneratorStatus() != 1: return False
    if mc_part.getPDG() != PHOTON_PDG: return False
    if mc_part.getEnergy() < E_MC_MIN: return False
    if get_parent_pdg(mc_part) == ELECTRON_PDG: return False
    return True


def in_phi_safe_region(phi_deg):
    for center in PHI_CRACK_CENTERS_DEG:
        raw = ((phi_deg - center + 180) % 360) - 180
        if 0.0 <= raw <= PHI_CRACK_REJECT_WIDTH_DEG:
            return False
    return True


def passes_layer_b(mc_part):
    p4 = edm4hep.utils.p4(mc_part)
    theta_deg, phi_deg = np.degrees(float(p4.theta())), np.degrees(float(p4.phi()))
    if not (THETA_SAFE_LO_DEG <= theta_deg <= THETA_SAFE_HI_DEG): return False
    if not in_phi_safe_region(phi_deg): return False
    return True


def process_file(filepath, label, tag_fastsim):
    print(f"\n{'='*60}\n  Processing: {label}\n{'='*60}")
    reader = root_io.Reader(filepath)

    n_split_by_floor = {f: {"true_cc3": [], "true_g4fb": [], "all": []} for f in WEIGHT_FLOORS}
    n_contribs_by_floor = {f: {"true_cc3_pfo": [], "true_g4fb_pfo": [], "all_pfo": []} for f in WEIGHT_FLOORS}
    second_contrib_weight = {"true_cc3_pfo": [], "true_g4fb_pfo": [], "all_pfo": []}
    completeness_vals = {"true_cc3": [], "true_g4fb": [], "all": []}
    weight_sums = []  # sanity check: sum of cluster_weight into each PFO

    n_events = 0
    for event in reader.get("events"):
        if n_events >= MAX_EVENTS:
            break
        n_events += 1

        selected_photons = {}
        e_mc_by_id = {}
        for mc in event.get("MCParticles"):
            if not passes_layer_a(mc) or not passes_layer_b(mc):
                continue
            mc_id = mc.getObjectID().index
            selected_photons[mc_id] = bool(mc.isHandledByFastSim()) if tag_fastsim else None
            e_mc_by_id[mc_id] = mc.getEnergy()

        photon_pfo_ids = set()
        pfo_energy = {}
        for pfo in event.get("PandoraPFOs"):
            if pfo.getPDG() == PHOTON_PDG:
                pid = pfo.getObjectID().index
                photon_pfo_ids.add(pid)
                pfo_energy[pid] = pfo.getEnergy()

        links_by_mc_all = defaultdict(dict)   # mc_id -> {reco_id: weight}, unfiltered
        links_by_reco_all = defaultdict(dict) # reco_id -> {mc_id: weight}, unfiltered

        for link in event.get("RecoMCTruthLink"):
            reco_id = link.getFrom().getObjectID().index
            if reco_id not in photon_pfo_ids:
                continue
            mc_id = link.getTo().getObjectID().index
            w = decode_cluster_weight(link.getWeight())
            if w <= 0:
                continue
            links_by_mc_all[mc_id][reco_id] = w
            links_by_reco_all[reco_id][mc_id] = w

        # sanity check: sum of weights into each relevant PFO
        relevant_pfos = set()
        for mc_id in selected_photons:
            relevant_pfos |= set(links_by_mc_all.get(mc_id, {}).keys())
        for reco_id in relevant_pfos:
            weight_sums.append(sum(links_by_reco_all[reco_id].values()))

        for f in WEIGHT_FLOORS:
            for mc_id, is_fastsim in selected_photons.items():
                links = links_by_mc_all.get(mc_id, {})
                n = sum(1 for w in links.values() if w >= f)
                key = "true_cc3" if (tag_fastsim and is_fastsim) else ("true_g4fb" if tag_fastsim else "all")
                n_split_by_floor[f][key].append(n)

            for reco_id in relevant_pfos:
                contribs = links_by_reco_all[reco_id]
                surviving = {mc_id: w for mc_id, w in contribs.items() if w >= f}
                n_c = len(surviving)
                if tag_fastsim:
                    if not surviving:
                        continue
                    dominant_mc = max(surviving, key=surviving.get)
                    dom_flag = selected_photons.get(dominant_mc)
                    if dom_flag is None:
                        for mc in event.get("MCParticles"):
                            if mc.getObjectID().index == dominant_mc:
                                dom_flag = bool(mc.isHandledByFastSim())
                                break
                    key = "true_cc3_pfo" if dom_flag else "true_g4fb_pfo"
                else:
                    key = "all_pfo"
                n_contribs_by_floor[f][key].append(n_c)

                if f == 0.05 and n_c >= 2:
                    sorted_w = sorted(surviving.values(), reverse=True)
                    second_contrib_weight[key].append(sorted_w[1])

        # completeness at floor=0 (use ALL links, since we want TOTAL energy recovered)
        for mc_id, is_fastsim in selected_photons.items():
            links = links_by_mc_all.get(mc_id, {})
            e_mc = e_mc_by_id[mc_id]
            if e_mc <= 0:
                continue
            recovered = sum(w * pfo_energy.get(rid, 0.0) for rid, w in links.items())
            comp = recovered / e_mc
            key = "true_cc3" if (tag_fastsim and is_fastsim) else ("true_g4fb" if tag_fastsim else "all")
            completeness_vals[key].append(comp)

    print(f"  Events processed: {n_events}")
    ws = np.array(weight_sums)
    if len(ws) > 0:
        print(f"\n  Sanity check -- sum of cluster_weight into each relevant PFO (N={len(ws)}):")
        print(f"    mean={ws.mean():.3f}  median={np.median(ws):.3f}  "
              f"p5={np.percentile(ws,5):.3f}  p95={np.percentile(ws,95):.3f}")
        print("    (expect close to 1.0 if cluster_weight fully accounts for a PFO's energy)")

    return n_split_by_floor, n_contribs_by_floor, second_contrib_weight, completeness_vals


cc3_split, cc3_contribs, cc3_second, cc3_comp = process_file(CC3_FILE, "CC3 file", tag_fastsim=True)
g4_split, g4_contribs, g4_second, g4_comp = process_file(G4_FILE, "G4 file", tag_fastsim=False)

print("\n" + "="*70)
print("  n_split MEAN vs floor")
print("="*70)
print(f"  {'floor':>6} {'true-CC3':>10} {'true-G4fb':>10} {'G4-file':>10}")
for f in WEIGHT_FLOORS:
    a = np.mean(cc3_split[f]["true_cc3"]) if cc3_split[f]["true_cc3"] else float("nan")
    b = np.mean(cc3_split[f]["true_g4fb"]) if cc3_split[f]["true_g4fb"] else float("nan")
    c = np.mean(g4_split[f]["all"]) if g4_split[f]["all"] else float("nan")
    print(f"  {f:>6.2f} {a:>10.3f} {b:>10.3f} {c:>10.3f}")

print("\n" + "="*70)
print("  n_contribs MEAN vs floor")
print("="*70)
print(f"  {'floor':>6} {'CC3-dom PFO':>12} {'G4fb-dom PFO':>13} {'G4-file PFO':>12}")
for f in WEIGHT_FLOORS:
    a = np.mean(cc3_contribs[f]["true_cc3_pfo"]) if cc3_contribs[f]["true_cc3_pfo"] else float("nan")
    b = np.mean(cc3_contribs[f]["true_g4fb_pfo"]) if cc3_contribs[f]["true_g4fb_pfo"] else float("nan")
    c = np.mean(g4_contribs[f]["all_pfo"]) if g4_contribs[f]["all_pfo"] else float("nan")
    print(f"  {f:>6.2f} {a:>12.3f} {b:>13.3f} {c:>12.3f}")

print("\n" + "="*70)
print("  Second-largest contributor weight (at floor=0.05, PFOs with n_contribs>=2)")
print("="*70)
for key in ["true_cc3_pfo", "true_g4fb_pfo"]:
    arr = np.array(cc3_second[key])
    if len(arr) > 0:
        print(f"  {key}: N={len(arr)}, mean={arr.mean():.3f}, median={np.median(arr):.3f}")
arr = np.array(g4_second["all_pfo"])
if len(arr) > 0:
    print(f"  G4-file: N={len(arr)}, mean={arr.mean():.3f}, median={np.median(arr):.3f}")

print("\n" + "="*70)
print("  Completeness (energy recovered / E_mc), mean")
print("="*70)
for key, results in [("true-CC3", cc3_comp["true_cc3"]), ("true-G4fb", cc3_comp["true_g4fb"]),
                      ("G4-file", g4_comp["all"])]:
    arr = np.array(results)
    if len(arr) > 0:
        print(f"  {key}: N={len(arr)}, mean={arr.mean():.3f}, median={np.median(arr):.3f}")

# --- plot: n_contribs mean vs floor, the key diagnostic ---
fig, ax = plt.subplots(figsize=(8, 5.5))
cc3_means = [np.mean(cc3_contribs[f]["true_cc3_pfo"]) if cc3_contribs[f]["true_cc3_pfo"] else np.nan for f in WEIGHT_FLOORS]
g4fb_means = [np.mean(cc3_contribs[f]["true_g4fb_pfo"]) if cc3_contribs[f]["true_g4fb_pfo"] else np.nan for f in WEIGHT_FLOORS]
g4_means = [np.mean(g4_contribs[f]["all_pfo"]) if g4_contribs[f]["all_pfo"] else np.nan for f in WEIGHT_FLOORS]
ax.plot(WEIGHT_FLOORS, cc3_means, "s--", color=C_CC3, label="CC3-dominated PFO")
ax.plot(WEIGHT_FLOORS, g4fb_means, "^:", color="gray", label="G4-fallback-dominated PFO")
ax.plot(WEIGHT_FLOORS, g4_means, "o-", color=C_G4, label="G4-file PFO")
ax.set_xlabel("Weight floor")
ax.set_ylabel("Mean n_contribs")
ax.set_title("Does the CC3/G4 gap open up once noise is floored out?", fontsize=10)
ax.legend(fontsize=9)
plt.tight_layout()
out = f"{PLOT_DIR}/n_contribs_vs_floor.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out}")
