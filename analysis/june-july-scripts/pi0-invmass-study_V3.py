"""
pi0_invariant_mass_v3.py
========================
pi0 -> gamma gamma invariant mass analysis: MC truth + reco (Tier 1 + Tier 2)
+ cluster-splitting diagnostic.

Changes from v2:
  - FIXED: RecoMCTruthLink.weight decoding. The raw weight field packs two
    values into one float: raw = 10000*clusterWeight + trackWeight, with
    BOTH clusterWeight and trackWeight expressed in permill (i.e. weight*1000).
    v2 applied the weight cut directly to this raw packed value, which is not
    a meaningful physical quantity on its own. v3 decodes clusterWeight and
    trackWeight separately and cuts on clusterWeight (since photons are purely
    calorimetric, cluster weight is the physically relevant one).

    NOTE (2026-07-29): an earlier version of this v3 fix divided the cluster
    term by 10000.0 instead of 1000.0, which shrank every decoded cluster
    weight by 10x (e.g. true 0.844 became 0.0844) and made essentially every
    link fail the ">0.5" cut -- this produced the "0 reco photons, empty
    plots" result. Verified against the actual RecoMCTruthLinker.cc source
    comment ("weight = 10000*calo weight + track weight (weights in
    permill)") and numerically against the dataset: with the corrected
    /1000.0 scaling, decoded clusterWeight is properly bounded in [0, 1]
    (max exactly 1.0 in this dataset) and about 30% of links pass a >0.5
    cut -- a sane, non-degenerate selection. This file has the corrected
    scaling.

  - ADDED: cluster-splitting diagnostic (Section 5). For pi0s with 3+ matched
    reco photons (the "split" bucket), this reports:
      - energy sharing among the split fragments (leading fragment energy
        fraction of total matched energy)
      - how many extra fragments beyond 2 are typically present
      - a comparison histogram, G4 vs CC3
    This turns the previously silent n_pi0_3plus counter into an actual
    diagnostic you can look at, not just a number.

  - Everything else (pi0-constrained pairing, Tier 1 / Tier 2 definitions,
    Gaussian+linear fit, truth sanity check) is unchanged from v2.

Run on NAF:
    source ~/source.sh
    python3 pi0_invariant_mass_v3.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from podio import root_io
from collections import defaultdict
import os, math

# CONFIG
USE_1K = False

if USE_1K:
    G4_FILE  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-1000events-test_REC.edm4hep.root"
    CC3_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-1000events-test_REC.edm4hep.root"
    MAX_EVENTS = 1000
    RUN_LABEL  = "1k events [dev]"
else:
    G4_FILE  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-10k-events-test_REC.edm4hep.root"
    CC3_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root"
    MAX_EVENTS = 10000
    RUN_LABEL  = "10k events"

PLOT_DIR = os.path.expanduser(
    f"~/thesis-ml-sim/plots/pi0_invariant_mass_{'1k' if USE_1K else '10k'}_v3"
)
os.makedirs(PLOT_DIR, exist_ok=True)

# PHYSICS
PHOTON_PDG   = 22
PI0_PDG      = 111
E_MC_TIER2   = 10.0
WEIGHT_CUT   = 0.5
PI0_MASS_MEV = 134.97

# HISTOGRAM SETTINGS
M_BINS        = np.arange(0.080, 0.201, 0.002)
M_CENTERS     = 0.5 * (M_BINS[:-1] + M_BINS[1:])
M_BINS_TRUTH  = np.arange(0.120, 0.155, 0.001)
M_CENTERS_TRUTH = 0.5 * (M_BINS_TRUTH[:-1] + M_BINS_TRUTH[1:])

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


def m_gg(e1, px1, py1, pz1, e2, px2, py2, pz2):
    m2 = 2.0 * (e1*e2 - px1*px2 - py1*py2 - pz1*pz2)
    return math.sqrt(max(m2, 0.0))


def decode_weight(raw_weight):
    """Decode the packed RecoMCTruthLink weight.

    ILD convention (RecoMCTruthLinker.cc source comment):
        raw = 10000 * clusterWeight + trackWeight
    with BOTH clusterWeight and trackWeight expressed in permill, i.e.
    each is (true fractional weight) * 1000, not *10000.

    Returns (clusterWeight, trackWeight), both in [0, 1] (track_weight can
    occasionally exceed 1 slightly due to how hit-count fractions are
    formed upstream -- that's expected and harmless, we don't cut on it
    here).
    """
    raw_int = int(round(raw_weight))
    cluster_weight = (raw_int // 10000) / 1000.0
    track_weight   = (raw_int % 10000) / 1000.0
    return cluster_weight, track_weight


def gauss_plus_linear(x, N, mu, sigma, a0, a1):
    return N * np.exp(-0.5 * ((x - mu) / sigma)**2) + a0 + a1 * x


def run_truth(filepath, label):
    print(f"\n{'='*60}")
    print(f"  Analysis A (truth): {label}  [{RUN_LABEL}]")
    print(f"{'='*60}")

    reader   = root_io.Reader(filepath)
    masses   = []
    n_events = 0
    n_pi0    = 0

    for event in reader.get("events"):
        if n_events >= MAX_EVENTS:
            break
        n_events += 1
        pi0_photons = defaultdict(list)

        for mc in event.get("MCParticles"):
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
            mom = mc.getMomentum()
            pi0_photons[parent.getObjectID().index].append(
                (mc.getEnergy(), mom.x, mom.y, mom.z)
            )

        for pi0_id, photons in pi0_photons.items():
            if len(photons) != 2:
                continue
            n_pi0 += 1
            e1, px1, py1, pz1 = photons[0]
            e2, px2, py2, pz2 = photons[1]
            masses.append(m_gg(e1, px1, py1, pz1, e2, px2, py2, pz2))

    masses = np.array(masses)
    print(f"  Events processed : {n_events}")
    print(f"  pi0 pairs found  : {n_pi0}")
    print(f"  M_gg computed    : {len(masses)}")
    if len(masses) > 0:
        print(f"  Mean M_gg [MeV]  : {1000*np.mean(masses):.3f}")
        print(f"  Std  M_gg [MeV]  : {1000*np.std(masses):.3f}")
    return masses


def run_reco(filepath, label):
    print(f"\n{'='*60}")
    print(f"  Reco analysis: {label}  [{RUN_LABEL}]")
    print(f"{'='*60}")

    reader = root_io.Reader(filepath)

    tier1_masses = []
    tier2_masses = []
    n_events     = 0

    n_pi0_mc_total  = 0
    n_pi0_0reco     = 0
    n_pi0_1reco     = 0
    n_pi0_exactly2  = 0
    n_pi0_3plus     = 0

    split_leading_frac = []
    split_n_fragments   = []

    for event in reader.get("events"):
        if n_events >= MAX_EVENTS:
            break
        n_events += 1

        mc_pi0_ids_with_photon_daughters = set()
        for mc in event.get("MCParticles"):
            if mc.getGeneratorStatus() != 1:
                continue
            if mc.getPDG() != PHOTON_PDG:
                continue
            parents = mc.getParents()
            if len(parents) == 0:
                continue
            parent = parents[0]
            if parent.getPDG() == PI0_PDG:
                mc_pi0_ids_with_photon_daughters.add(parent.getObjectID().index)
        n_pi0_mc_total += len(mc_pi0_ids_with_photon_daughters)

        # Strategy A -- best-match MC per reco PFO, using DECODED cluster weight
        reco_to_mc  = {}
        reco_to_wgt = {}
        for link in event.get("RecoMCTruthLink"):
            reco   = link.getFrom()
            mc     = link.getTo()
            raw_w  = link.getWeight()
            cluster_w, track_w = decode_weight(raw_w)
            rid    = reco.getObjectID().index
            if rid not in reco_to_wgt or cluster_w > reco_to_wgt[rid]:
                reco_to_mc[rid]  = mc
                reco_to_wgt[rid] = cluster_w

        pi0_reco_photons = defaultdict(list)

        for pfo in event.get("PandoraPFOs"):
            if pfo.getPDG() != PHOTON_PDG:
                continue
            rid = pfo.getObjectID().index
            if rid not in reco_to_mc:
                continue
            if reco_to_wgt[rid] <= WEIGHT_CUT:
                continue
            mc = reco_to_mc[rid]
            if mc.getGeneratorStatus() != 1:
                continue
            if mc.getPDG() != PHOTON_PDG:
                continue

            mc_parents = mc.getParents()
            if len(mc_parents) == 0:
                continue
            mc_parent = mc_parents[0]
            if mc_parent.getPDG() != PI0_PDG:
                continue

            pi0_obj_id = mc_parent.getObjectID().index
            mom = pfo.getMomentum()
            pi0_reco_photons[pi0_obj_id].append(
                (pfo.getEnergy(), mom.x, mom.y, mom.z, mc.getEnergy())
            )

        seen_pi0_reco_ids = set()

        for pi0_id, photons in pi0_reco_photons.items():
            seen_pi0_reco_ids.add(pi0_id)
            n = len(photons)

            if n == 1:
                n_pi0_1reco += 1
            elif n == 2:
                n_pi0_exactly2 += 1
                e1, px1, py1, pz1, emc1 = photons[0]
                e2, px2, py2, pz2, emc2 = photons[1]
                mass = m_gg(e1, px1, py1, pz1, e2, px2, py2, pz2)
                tier1_masses.append(mass)
                if emc1 > E_MC_TIER2 and emc2 > E_MC_TIER2:
                    tier2_masses.append(mass)
            elif n >= 3:
                n_pi0_3plus += 1
                energies = sorted([p[0] for p in photons], reverse=True)
                total_e = sum(energies)
                if total_e > 0:
                    split_leading_frac.append(energies[0] / total_e)
                    split_n_fragments.append(n)

        n_pi0_0reco += len(mc_pi0_ids_with_photon_daughters - seen_pi0_reco_ids)

    print(f"\n  pi0 RECONSTRUCTION EFFICIENCY [{label}]")
    print(f"  {'-'*45}")
    print(f"  MC pi0s (denominator)         : {n_pi0_mc_total}")
    print(f"  0 reco photons (lost entirely): {n_pi0_0reco:6d}  "
          f"({100*n_pi0_0reco/max(n_pi0_mc_total,1):.1f}%)")
    print(f"  1 reco photon  (merged/lost 1): {n_pi0_1reco:6d}  "
          f"({100*n_pi0_1reco/max(n_pi0_mc_total,1):.1f}%)")
    print(f"  2 reco photons (clean -> mass) : {n_pi0_exactly2:6d}  "
          f"({100*n_pi0_exactly2/max(n_pi0_mc_total,1):.1f}%)  <- USED FOR MASS")
    print(f"  3+ reco photons (split)       : {n_pi0_3plus:6d}  "
          f"({100*n_pi0_3plus/max(n_pi0_mc_total,1):.1f}%)")
    print(f"  Clean reco efficiency         : "
          f"{100*n_pi0_exactly2/max(n_pi0_mc_total,1):.1f}%")
    print(f"\n  Tier 1 masses (no energy gate): {len(tier1_masses)}")
    print(f"  Tier 2 masses (E_mc>10 GeV)   : {len(tier2_masses)}")

    eff = {
        "n_mc_total"  : n_pi0_mc_total,
        "n_0reco"     : n_pi0_0reco,
        "n_1reco"     : n_pi0_1reco,
        "n_exactly2"  : n_pi0_exactly2,
        "n_3plus"     : n_pi0_3plus,
        "eff_pct"     : 100*n_pi0_exactly2/max(n_pi0_mc_total, 1),
    }
    split_diag = {
        "leading_frac": np.array(split_leading_frac),
        "n_fragments" : np.array(split_n_fragments),
    }
    return np.array(tier1_masses), np.array(tier2_masses), eff, split_diag


def fit_gaussian(centers, counts, label):
    fit_mask = (centers > 0.110) & (centers < 0.170)
    x = centers[fit_mask]
    y = counts[fit_mask].astype(float)

    if len(x) < 5 or y.max() < 5:
        print(f"  Fit [{label}] SKIPPED: insufficient statistics")
        return None, None

    mu0    = PI0_MASS_MEV / 1000.0
    sigma0 = 0.015
    N0     = y.max() * sigma0 * np.sqrt(2 * np.pi)
    p0     = [N0, mu0, sigma0, 1.0, 0.0]
    lower  = [  0,  0.100,  0.005, -1e6, -1e6]
    upper  = [1e9,  0.200,  0.040,  1e6,  1e6]

    try:
        popt, pcov = curve_fit(gauss_plus_linear, x, y,
                               p0=p0, bounds=(lower, upper), maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        mu, sigma = popt[1], popt[2]
        dmu, dsigma = perr[1], perr[2]
        bias_mev = 1000 * (mu - PI0_MASS_MEV/1000.0)
        print(f"  Fit [{label:3s}]: mu={1000*mu:.2f}+/-{1000*dmu:.2f} MeV | "
              f"sigma={1000*sigma:.2f}+/-{1000*dsigma:.2f} MeV | bias={bias_mev:+.2f} MeV")
        return popt, perr
    except RuntimeError as e:
        print(f"  Fit [{label}] FAILED: {e}")
        return None, None


def plot_truth(g4_masses, cc3_masses):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(1000*g4_masses,  bins=1000*M_BINS_TRUTH, histtype="step",
            color=C_G4,  linewidth=2, label=f"Geant4 (N={len(g4_masses)})")
    ax.hist(1000*cc3_masses, bins=1000*M_BINS_TRUTH, histtype="step",
            color=C_CC3, linewidth=2, linestyle="--", label=f"CC3 (N={len(cc3_masses)})")
    ax.axvline(PI0_MASS_MEV, color="black", lw=1.0, ls=":", alpha=0.7,
               label=f"PDG pi0 mass ({PI0_MASS_MEV} MeV)")
    ax.set_xlabel("M(gamma gamma) [MeV]")
    ax.set_ylabel("pi0 candidates / 1 MeV")
    ax.set_title("Analysis A -- MC Truth M(gamma gamma): G4 vs CC3", fontsize=10)
    ax.legend(framealpha=0.5)
    plt.tight_layout()
    out = f"{PLOT_DIR}/analysisA_truth_mgg.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: analysisA_truth_mgg.png")


def plot_reco(g4_t1, cc3_t1, g4_t2, cc3_t2, g4_eff, cc3_eff):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    tiers = [
        (axes[0], g4_t1, cc3_t1, g4_eff, cc3_eff, "Tier 1", "no energy gate"),
        (axes[1], g4_t2, cc3_t2, g4_eff, cc3_eff, "Tier 2", "E_mc>10 GeV both daughters"),
    ]
    for ax, g4_m, cc3_m, g4e, cc3e, tier, subtitle in tiers:
        g4_counts,  _ = np.histogram(g4_m,  bins=M_BINS)
        cc3_counts, _ = np.histogram(cc3_m, bins=M_BINS)
        ax.hist(g4_m,  bins=M_BINS, histtype="step", color=C_G4, linewidth=2,
                label=f"Geant4 (N={len(g4_m)}, eff={g4e['eff_pct']:.1f}%)")
        ax.hist(cc3_m, bins=M_BINS, histtype="step", color=C_CC3, linewidth=2,
                linestyle="--", label=f"CC3 (N={len(cc3_m)}, eff={cc3e['eff_pct']:.1f}%)")
        ax.axvline(PI0_MASS_MEV/1000, color="gray", lw=1.0, ls=":",
                   label=f"PDG: {PI0_MASS_MEV:.0f} MeV")

        fit_texts = []
        for counts, flabel, color in [(g4_counts, "G4", C_G4), (cc3_counts, "CC3", C_CC3)]:
            popt, perr = fit_gaussian(M_CENTERS, counts, flabel)
            if popt is not None:
                x_fit = np.linspace(0.100, 0.180, 800)
                ax.plot(x_fit, gauss_plus_linear(x_fit, *popt), color=color, lw=1.8, alpha=0.75)
                mu_mev, sig_mev = 1000*popt[1], 1000*popt[2]
                bias_mev = mu_mev - PI0_MASS_MEV
                fit_texts.append(f"{flabel}: mu={mu_mev:.1f} MeV, sigma={sig_mev:.1f} MeV, bias={bias_mev:+.1f} MeV")

        if fit_texts:
            ax.text(0.97, 0.97, "\n".join(fit_texts), transform=ax.transAxes,
                    fontsize=7.5, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

        ax.set_xlabel("M(gamma gamma) [GeV]")
        ax.set_ylabel("pi0 candidates / 2 MeV")
        ax.set_title(f"Reco M(gamma gamma) -- {tier}\n{subtitle}", fontsize=9.5)
        ax.legend(framealpha=0.5, fontsize=9)

    plt.tight_layout()
    out = f"{PLOT_DIR}/reco_mgg_tier1_tier2.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: reco_mgg_tier1_tier2.png")


def plot_cluster_splitting(g4_split, cc3_split, g4_eff, cc3_eff):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    max_frag = 8
    bins = np.arange(2.5, max_frag + 1.5, 1)
    if len(g4_split["n_fragments"]) > 0:
        ax.hist(g4_split["n_fragments"], bins=bins, histtype="step", color=C_G4,
                linewidth=2, label=f"Geant4 (N={g4_eff['n_3plus']})")
    if len(cc3_split["n_fragments"]) > 0:
        ax.hist(cc3_split["n_fragments"], bins=bins, histtype="step", color=C_CC3,
                linewidth=2, linestyle="--", label=f"CC3 (N={cc3_eff['n_3plus']})")
    ax.set_xlabel("Number of matched reco fragments per pi0")
    ax.set_ylabel("pi0 count")
    ax.set_title("Cluster splitting: fragment multiplicity\n(pi0s with 3+ reco photon daughters)",
                 fontsize=9.5)
    if len(g4_split["n_fragments"]) > 0 or len(cc3_split["n_fragments"]) > 0:
        ax.legend(framealpha=0.5)

    ax = axes[1]
    bins2 = np.linspace(0.2, 1.0, 25)
    if len(g4_split["leading_frac"]) > 0:
        ax.hist(g4_split["leading_frac"], bins=bins2, histtype="step", color=C_G4,
                linewidth=2, density=True, label="Geant4")
    if len(cc3_split["leading_frac"]) > 0:
        ax.hist(cc3_split["leading_frac"], bins=bins2, histtype="step", color=C_CC3,
                linewidth=2, linestyle="--", density=True, label="CC3")
    ax.set_xlabel("Leading fragment energy / total matched energy")
    ax.set_ylabel("Normalized density")
    ax.set_title("Cluster splitting: energy sharing\n(1.0 = one dominant fragment, lower = more even split)",
                 fontsize=9.5)
    if len(g4_split["leading_frac"]) > 0 or len(cc3_split["leading_frac"]) > 0:
        ax.legend(framealpha=0.5)

    plt.tight_layout()
    out = f"{PLOT_DIR}/cluster_splitting_diagnostic.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: cluster_splitting_diagnostic.png")

    if len(g4_split["n_fragments"]) > 0:
        print(f"\n  Cluster-splitting summary [G4]:")
        print(f"    Mean fragments per split pi0 : {g4_split['n_fragments'].mean():.2f}")
        print(f"    Mean leading fragment frac    : {g4_split['leading_frac'].mean():.3f}")
    if len(cc3_split["n_fragments"]) > 0:
        print(f"\n  Cluster-splitting summary [CC3]:")
        print(f"    Mean fragments per split pi0 : {cc3_split['n_fragments'].mean():.2f}")
        print(f"    Mean leading fragment frac    : {cc3_split['leading_frac'].mean():.3f}")


def print_efficiency_comparison(g4_eff, cc3_eff):
    print(f"\n{'='*60}")
    print("  pi0 RECONSTRUCTION EFFICIENCY COMPARISON")
    print(f"  {'-'*58}")
    print(f"  {'Category':<32}  {'G4':>10}  {'CC3':>10}")
    print(f"  {'-'*58}")
    rows = [
        ("MC pi0s (denominator)",    "n_mc_total", False),
        ("0 reco photons (lost)",    "n_0reco",    True),
        ("1 reco photon (merged)",   "n_1reco",    True),
        ("2 reco photons (-> mass)", "n_exactly2", True),
        ("3+ reco photons (split)",  "n_3plus",    True),
    ]
    for name, key, show_pct in rows:
        g4v, cc3v = g4_eff[key], cc3_eff[key]
        if show_pct:
            g4_pct  = 100 * g4v  / max(g4_eff["n_mc_total"],  1)
            cc3_pct = 100 * cc3v / max(cc3_eff["n_mc_total"], 1)
            print(f"  {name:<32}  {g4v:5d} ({g4_pct:4.1f}%)  {cc3v:5d} ({cc3_pct:4.1f}%)")
        else:
            print(f"  {name:<32}  {g4v:>10}  {cc3v:>10}")
    print(f"  {'-'*58}")
    print(f"  {'Clean reco efficiency':<32}  {g4_eff['eff_pct']:9.1f}%  {cc3_eff['eff_pct']:9.1f}%")
    delta = cc3_eff['eff_pct'] - g4_eff['eff_pct']
    print(f"  {'Delta efficiency (CC3-G4)':<32}  {delta:+.1f} pp")
    print(f"{'='*60}")


print("\n" + "="*60)
print("  pi0_invariant_mass_v3.py")
print("  Analysis A (truth) + Tier 1 + Tier 2 reco M_gg + splitting diagnostic")
print("  FIX: decoded RecoMCTruthLink weight (cluster/track split, /1000.0 scaling)")
print("="*60)

print("\n-- Analysis A: MC Truth --")
g4_truth  = run_truth(G4_FILE,  "Geant4")
cc3_truth = run_truth(CC3_FILE, "CC3")

print("\n-- Reco: G4 --")
g4_t1, g4_t2, g4_eff, g4_split = run_reco(G4_FILE, "G4")

print("\n-- Reco: CC3 --")
cc3_t1, cc3_t2, cc3_eff, cc3_split = run_reco(CC3_FILE, "CC3")

print_efficiency_comparison(g4_eff, cc3_eff)

print("\n-- Generating plots --")
plot_truth(g4_truth, cc3_truth)
plot_reco(g4_t1, cc3_t1, g4_t2, cc3_t2, g4_eff, cc3_eff)
plot_cluster_splitting(g4_split, cc3_split, g4_eff, cc3_eff)

print(f"\n  All plots saved to: {PLOT_DIR}")
print(f"  Done.\n")
