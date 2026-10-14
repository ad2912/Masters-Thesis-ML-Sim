"""
fastsim_confidence_study_v1.py
================================
Study: confident reco-level fast-sim (CC3) photon tagging, and a phase-space-
matched CC3-vs-G4 comparison of match efficiency and cluster splitting as a
function of the RecoMCTruthLink weight threshold.

REVISION (2026-09-16, this session): geometry cuts corrected based on
empirical measurement + confirmation from Thomas:
  - theta window: barrel-only, [37.5, 142.5] deg (own empirical scan,
    confirmed as the correct barrel range -- NOT extended to endcap, which
    CC3 also covers, per explicit scope decision to stay barrel-only for now).
  - phi crack cut: ASYMMETRIC, one-sided per crack. Confirmed empirically
    (phi_crack_signed_asymmetry_check.py, 10k events): fraction is ~0.85-0.97
    approaching a crack center from one rotational direction (no penalty at
    all), and EXACTLY 0.000 for a full 8 degrees on the other side of the
    center, recovering sharply past +8 deg. This replaces the earlier
    (WRONG) symmetric half-width model entirely -- there is no "half-width",
    there is a one-sided 8-degree dead window starting AT the crack center.
    Sign convention: raw = ((phi - center + 180) % 360) - 180; dead zone is
    0 <= raw <= 8. Confirmed same direction for all 8 cracks (pooled result
    was clean, not smeared, which would not happen if directions varied).

CONCEPT (read before trusting output):
  Step 0: quantify the effect of Layer B (geometry) cuts on top of Layer A
    (physics only). Reports overall retention, retention separately for
    true-CC3 vs true-G4-fallback, and purity before/after.
    Falsifiable prediction: with the corrected geometry, true-CC3 retention
    should be MUCH closer to 100% than the earlier (wrong-geometry) 49.4%.
    Some residual gap is still expected -- see the unexplained-baseline-
    fallback item in the running list -- but it should be small, not
    dominant.

  Step 1 (truth level, CC3 file only): apply Layer A + corrected Layer B to
    MCParticles. Split survivors by isHandledByFastSim() into true-CC3 /
    true-G4-fallback. Validates the geometry cuts.

  Step 2 (truth level, G4 file): apply the SAME Layer A+B cuts to the G4
    file's MCParticles -- the phase-space-matched G4 comparison population.

  Step 3 (reco level, both files): weight-threshold scan (efficiency +
    n_split) comparing true-CC3 (CC3 file) against phase-space-matched G4
    (G4 file). cluster_weight is itself a purity fraction, so this one scan
    traces efficiency and purity together.

  The true-G4-fallback population (within the CC3 file) is NOT part of the
  main reco-level comparison -- reported for Step 0 bookkeeping only.

NOT YET RUN. Output below (if any) is a prediction of behavior, not a
confirmed result.

Run on NAF:
    source ~/source.sh
    python3 fastsim_confidence_study_v1.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio import root_io
import edm4hep
import os
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================
RUN_SIZE = "10k"   # "10k" first per plan, then "100k"

RUN_TABLE = {
    "10k": dict(
        g4  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-10k-events-test_REC.edm4hep.root",
        cc3 = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root",
        max_events = 10000,
        label = "10k events",
    ),
    "100k": dict(
        g4  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-geant4-100kevents-reco-test_REC.edm4hep.root",
        cc3 = "/data/dust/user/alimuham/thesis/reco/PLACEHOLDER_CONFIRM_CC3_100k_FILENAME_REC.edm4hep.root",
        max_events = 100000,
        label = "100k events",
    ),
}

_cfg = RUN_TABLE[RUN_SIZE]
G4_FILE    = _cfg["g4"]
CC3_FILE   = _cfg["cc3"]
MAX_EVENTS = _cfg["max_events"]
RUN_LABEL  = _cfg["label"]

PLOT_DIR = os.path.expanduser(f"~/thesis-ml-sim/plots/fastsim_confidence_study_{RUN_SIZE}_v1")
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Layer A: physics cuts (Signal B, same as existing scripts) ---
PHOTON_PDG   = 22
ELECTRON_PDG = 11
E_MC_MIN     = 10.0

# --- Layer B: geometry cuts, CORRECTED this revision ---
# theta window: barrel-only, empirically confirmed (37.5/142.5 = EC-barrel
# and barrel-EC crack boundaries from cc3_acceptance_full_v3.py's own scan).
# Deliberately NOT extended to endcap -- barrel-only scope for now.
THETA_SAFE_LO_DEG = 37.5
THETA_SAFE_HI_DEG = 142.5

# phi crack cut: ASYMMETRIC, one-sided, 8 deg wide, starting AT the crack
# center. Confirmed empirically this session -- see revision note above.
PHI_CRACK_CENTERS_DEG = [-157.5 + 45 * k for k in range(8)]
PHI_CRACK_REJECT_WIDTH_DEG = 8.0


def in_phi_safe_region(phi_deg):
    """One-sided rejection per crack: dead zone is [center, center+8] in the
    increasing-phi direction. No restriction on the other side at all."""
    for center in PHI_CRACK_CENTERS_DEG:
        raw = ((phi_deg - center + 180) % 360) - 180  # signed, in [-180, 180)
        if 0.0 <= raw <= PHI_CRACK_REJECT_WIDTH_DEG:
            return False
    return True


# --- Reco-level weight thresholds to scan ---
WEIGHT_THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

C_G4  = "#2166ac"
C_CC3 = "#d6604d"

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


# ============================================================
# HELPERS
# ============================================================
def decode_cluster_weight(raw_weight):
    """Confirmed correct via data ceiling check (2026-09-16 session):
    remainder max = 1000."""
    raw_int = int(round(raw_weight))
    cluster_weight = (raw_int // 10000) / 1000.0
    return cluster_weight


def get_parent_pdg(mc_part):
    parents = mc_part.getParents()
    return abs(parents[0].getPDG()) if len(parents) > 0 else 0


def passes_layer_a(mc_part):
    """Physics-only cuts (Signal B), no geometry."""
    if mc_part.getGeneratorStatus() != 1:
        return False
    if mc_part.getPDG() != PHOTON_PDG:
        return False
    if mc_part.getEnergy() < E_MC_MIN:
        return False
    if get_parent_pdg(mc_part) == ELECTRON_PDG:
        return False
    return True


def passes_layer_b_geometry(mc_part):
    """Geometry-only cut: theta-safe barrel window AND phi outside the
    asymmetric one-sided crack dead zones."""
    p4 = edm4hep.utils.p4(mc_part)
    theta_deg = np.degrees(float(p4.theta()))
    phi_deg   = np.degrees(float(p4.phi()))

    if not (THETA_SAFE_LO_DEG <= theta_deg <= THETA_SAFE_HI_DEG):
        return False
    if not in_phi_safe_region(phi_deg):
        return False
    return True


# ============================================================
# STEP 0 + 1 + 2: truth-level population building
# ============================================================
def build_truth_populations(filepath, label, tag_fastsim):
    print(f"\n{'='*60}")
    print(f"  Building truth population: {label}  ({RUN_LABEL})")
    print(f"{'='*60}")

    reader = root_io.Reader(filepath)

    n_layerA_total     = 0
    n_layerA_true_cc3  = 0
    n_layerA_true_g4fb = 0
    n_layerAB_total     = 0
    n_layerAB_true_cc3  = 0
    n_layerAB_true_g4fb = 0

    truth_photons = {}
    links_by_photon = defaultdict(list)

    n_events = 0
    for event in reader.get("events"):
        if n_events >= MAX_EVENTS:
            break

        event_photon_keys = {}

        for mc in event.get("MCParticles"):
            if not passes_layer_a(mc):
                continue
            n_layerA_total += 1

            is_fastsim = None
            if tag_fastsim:
                is_fastsim = bool(mc.isHandledByFastSim())
                if is_fastsim:
                    n_layerA_true_cc3 += 1
                else:
                    n_layerA_true_g4fb += 1

            if not passes_layer_b_geometry(mc):
                continue
            n_layerAB_total += 1
            if tag_fastsim:
                if is_fastsim:
                    n_layerAB_true_cc3 += 1
                else:
                    n_layerAB_true_g4fb += 1

            mc_local_id = mc.getObjectID().index
            key = (n_events, mc_local_id)
            truth_photons[key] = is_fastsim
            event_photon_keys[mc_local_id] = key

        for link in event.get("RecoMCTruthLink"):
            mc = link.getTo()
            mc_local_id = mc.getObjectID().index
            if mc_local_id not in event_photon_keys:
                continue
            key = event_photon_keys[mc_local_id]
            reco_id = link.getFrom().getObjectID().index
            cw = decode_cluster_weight(link.getWeight())
            links_by_photon[key].append((reco_id, cw))

        n_events += 1

    print(f"  Events processed          : {n_events}")
    print(f"\n  --- Step 0: geometry-cut effect ---")
    print(f"  Layer A only (physics)     : {n_layerA_total}")
    print(f"  Layer A+B (physics+geom)   : {n_layerAB_total}")
    overall_retention = n_layerAB_total / max(n_layerA_total, 1)
    print(f"  Overall retention after geometry cut : {100*overall_retention:.1f}%  "
          f"({n_layerA_total - n_layerAB_total} photons lost to geometry)")

    geom_stats = dict(
        n_layerA_total=n_layerA_total, n_layerAB_total=n_layerAB_total,
        overall_retention=overall_retention,
    )

    if tag_fastsim:
        purity_before = n_layerA_true_cc3 / max(n_layerA_total, 1)
        purity_after  = n_layerAB_true_cc3 / max(n_layerAB_total, 1)
        retention_cc3   = n_layerAB_true_cc3  / max(n_layerA_true_cc3,  1)
        retention_g4fb  = n_layerAB_true_g4fb / max(n_layerA_true_g4fb, 1)

        print(f"\n  Purity (fraction true-CC3) BEFORE geometry cut : {100*purity_before:.1f}%  "
              f"(N_true-CC3={n_layerA_true_cc3}, N_true-G4fb={n_layerA_true_g4fb})")
        print(f"  Purity (fraction true-CC3) AFTER  geometry cut : {100*purity_after:.1f}%  "
              f"(N_true-CC3={n_layerAB_true_cc3}, N_true-G4fb={n_layerAB_true_g4fb})")
        print(f"\n  Retention among true-CC3 photons      : {100*retention_cc3:.1f}%  "
              f"({'as expected, near 100%' if retention_cc3 >= 0.90 else 'STILL LOWER THAN EXPECTED -- investigate further'})")
        print(f"  Retention among true-G4-fallback photons : {100*retention_g4fb:.1f}%  "
              f"(expected to be low -- this is where the geometry cut is supposed to bite)")

        geom_stats.update(dict(
            purity_before=purity_before, purity_after=purity_after,
            retention_cc3=retention_cc3, retention_g4fb=retention_g4fb,
            n_layerA_true_cc3=n_layerA_true_cc3, n_layerA_true_g4fb=n_layerA_true_g4fb,
            n_layerAB_true_cc3=n_layerAB_true_cc3, n_layerAB_true_g4fb=n_layerAB_true_g4fb,
        ))

        total_ab = max(n_layerAB_true_cc3 + n_layerAB_true_g4fb, 1)
        print(f"\n  --- Step 1: 90/10 validation (post-geometry-cut) ---")
        print(f"  true-CC3   : {n_layerAB_true_cc3}  ({100*n_layerAB_true_cc3/total_ab:.1f}%)")
        print(f"  true-G4-fb : {n_layerAB_true_g4fb}  ({100*n_layerAB_true_g4fb/total_ab:.1f}%)")
        pct = 100*n_layerAB_true_cc3/total_ab
        print(f"  Expectation check: ~90/10 split -- {'MATCHES' if 85 <= pct <= 97 else 'DOES NOT MATCH, investigate'}")

    return truth_photons, links_by_photon, geom_stats


# ============================================================
# STEP 3: weight-threshold scan
# ============================================================
def threshold_scan(truth_photons, links_by_photon, population_filter=None):
    if population_filter is None:
        keys = list(truth_photons.keys())
    else:
        keys = [k for k, v in truth_photons.items() if v == population_filter]

    n_total = len(keys)
    results = {}
    for thr in WEIGHT_THRESHOLDS:
        n_matched, n_clean, n_multisplit, n_lost = 0, 0, 0, 0
        for key in keys:
            links = links_by_photon.get(key, [])
            n_clearing = sum(1 for (_, w) in links if w >= thr)
            if n_clearing == 0:
                n_lost += 1
            elif n_clearing == 1:
                n_clean += 1
                n_matched += 1
            else:
                n_multisplit += 1
                n_matched += 1

        results[thr] = dict(
            efficiency = n_matched / n_total if n_total > 0 else float("nan"),
            clean_frac = n_clean / n_total if n_total > 0 else float("nan"),
            split_frac = n_multisplit / n_total if n_total > 0 else float("nan"),
            lost_frac  = n_lost / n_total if n_total > 0 else float("nan"),
            n_total    = n_total,
        )
    return results


# ============================================================
# PLOTTING
# ============================================================
def plot_geometry_effect(geom_stats_cc3):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["Before\n(Layer A only)", "After\n(Layer A+B)"]
    purities = [geom_stats_cc3["purity_before"], geom_stats_cc3["purity_after"]]
    ax1.bar(labels, purities, color=[C_G4, C_CC3], alpha=0.8, width=0.5)
    for i, p in enumerate(purities):
        ax1.text(i, p + 0.02, f"{100*p:.1f}%", ha="center", fontsize=11)
    ax1.set_ylabel("Purity: fraction true-CC3\namong all signal photons")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Effect of corrected geometry cut on purity\n(CC3 file, barrel-only)", fontsize=10)

    cats = ["true-CC3", "true-G4-fallback"]
    retentions = [geom_stats_cc3["retention_cc3"], geom_stats_cc3["retention_g4fb"]]
    ax2.bar(cats, retentions, color=["#4daf4a", "#999999"], alpha=0.85, width=0.5)
    for i, r in enumerate(retentions):
        ax2.text(i, r + 0.02, f"{100*r:.1f}%", ha="center", fontsize=11)
    ax2.axhline(1.0, color="black", lw=1, ls=":", alpha=0.6)
    ax2.set_ylabel("Fraction retained after geometry cut")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Geometry-cut retention by true fast-sim status\n"
                   "(corrected: barrel theta [37.5,142.5], asymmetric 8deg phi crack)", fontsize=10)

    plt.tight_layout()
    out = f"{PLOT_DIR}/fig0_geometry_effect.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig0_geometry_effect.png")


def plot_efficiency_vs_threshold(cc3_results, g4_results):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    thr = WEIGHT_THRESHOLDS
    cc3_eff = [cc3_results[t]["efficiency"] for t in thr]
    g4_eff  = [g4_results[t]["efficiency"] for t in thr]

    ax.plot(thr, cc3_eff, "s--", color=C_CC3, linewidth=2, markersize=7,
            label=f"true-CC3 (CC3 file, N={cc3_results[thr[0]]['n_total']})")
    ax.plot(thr, g4_eff, "o-", color=C_G4, linewidth=2, markersize=7,
            label=f"phase-space-matched G4 (G4 file, N={g4_results[thr[0]]['n_total']})")

    ax.set_xlabel("RecoMCTruthLink cluster_weight threshold\n(higher = stricter purity requirement)")
    ax.set_ylabel("Match efficiency\n(fraction with >=1 reco PFO above threshold)")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Reco-level match efficiency vs weight threshold\n"
        "Layer A+B truth photons (barrel-only), phase-space matched between files",
        fontsize=10)
    ax.legend(framealpha=0.6)
    plt.tight_layout()
    out = f"{PLOT_DIR}/fig1_efficiency_vs_threshold.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig1_efficiency_vs_threshold.png")


def plot_splitting_vs_threshold(cc3_results, g4_results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    thr = WEIGHT_THRESHOLDS

    for ax, results, name in [
        (axes[0], cc3_results, "true-CC3 (CC3 file)"),
        (axes[1], g4_results,  "phase-space-matched G4 (G4 file)"),
    ]:
        clean = [results[t]["clean_frac"] for t in thr]
        split = [results[t]["split_frac"] for t in thr]
        lost  = [results[t]["lost_frac"]  for t in thr]

        ax.stackplot(thr, clean, split, lost,
                     labels=["clean (n=1)", "split (n>=2)", "lost (n=0)"],
                     colors=["#4daf4a", "#ff7f00", "#999999"], alpha=0.85)
        ax.set_xlabel("cluster_weight threshold")
        ax.set_title(name, fontsize=10)
        ax.set_ylim(0, 1)
        if ax is axes[0]:
            ax.set_ylabel("Fraction of truth photons")
        ax.legend(fontsize=8, loc="center left")

    fig.suptitle("Clean / split / lost fractions vs weight threshold (barrel-only)", fontsize=11)
    plt.tight_layout()
    out = f"{PLOT_DIR}/fig2_splitting_vs_threshold.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig2_splitting_vs_threshold.png")


# ============================================================
# MAIN
# ============================================================
print("\n" + "="*60)
print("  fastsim_confidence_study_v1.py  (geometry-corrected revision)")
print(f"  Dataset: {RUN_SIZE} ({RUN_LABEL})")
print(f"  Layer B geometry (empirically confirmed this session):")
print(f"    theta-safe window (barrel-only) : [{THETA_SAFE_LO_DEG}, {THETA_SAFE_HI_DEG}] deg")
print(f"    phi crack centers               : {PHI_CRACK_CENTERS_DEG}")
print(f"    phi crack dead zone             : one-sided, [center, center+{PHI_CRACK_REJECT_WIDTH_DEG}] deg")
print("="*60)

cc3_truth, cc3_links, cc3_geom = build_truth_populations(CC3_FILE, "CC3", tag_fastsim=True)
g4_truth,  g4_links,  g4_geom  = build_truth_populations(G4_FILE,  "G4",  tag_fastsim=False)

print("\n-- Running weight-threshold scan --")
cc3_scan = threshold_scan(cc3_truth, cc3_links, population_filter=True)
g4_scan  = threshold_scan(g4_truth,  g4_links,  population_filter=None)

print("\nEfficiency by threshold:")
print(f"  {'threshold':>10} {'CC3 eff':>10} {'G4 eff':>10}")
for t in WEIGHT_THRESHOLDS:
    print(f"  {t:>10.2f} {cc3_scan[t]['efficiency']:>10.3f} {g4_scan[t]['efficiency']:>10.3f}")

print("\nGenerating plots...")
plot_geometry_effect(cc3_geom)
plot_efficiency_vs_threshold(cc3_scan, g4_scan)
plot_splitting_vs_threshold(cc3_scan, g4_scan)

print(f"\nAll plots saved to: {PLOT_DIR}")
print("Done.\n")
