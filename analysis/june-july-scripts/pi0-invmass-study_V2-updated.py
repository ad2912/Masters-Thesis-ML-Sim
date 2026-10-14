"""
pi0_invariant_mass_v2.py updated
========================
pi0 -> gamma gamma invariant mass analysis: MC truth + reco (Tier 1 + Tier 2)
+ cluster-splitting diagnostic.

Same core logic as your original v2: pi0-constrained pairing, Strategy A
weight matching (raw weight, no decoding), Tier 1/Tier 2 definitions,
Gaussian+linear fit, truth sanity check.

Addition: cluster-splitting diagnostic (Section 5). For pi0s with 3+ matched
reco photons (the "split" bucket), this reports:
  - energy sharing among the split fragments (leading fragment energy
    fraction of total matched energy)
  - how many extra fragments beyond 2 are typically present
  - a comparison histogram, G4 vs CC3

Run on NAF:
    source ~/source.sh
    python3 pi0_invariant_mass_v2.py
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
    f"~/thesis-ml-sim/plots/pi0_invariant_mass_{'1k' if USE_1K else '10k'}_v2"
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

        # Strategy A -- best-weight MC match per reco PFO, RAW weight, no decoding
        reco_to_mc  = {}
        reco_to_wgt = {}
        for link in event.get("RecoMCTruthLink"):
            reco   = link.getFrom()
            mc     = link.getTo()
            weight = link.getWeight()
            rid    = reco.getObjectID().index
            if rid not in reco_to_wgt or weight > reco_to_wgt[rid]:
                reco_to_mc[rid]  = mc
                reco_to_wgt[rid] = weight

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
print("  pi0_invariant_mass_v2.py")
print("  Analysis A (truth) + Tier 1 + Tier 2 reco M_gg + splitting diagnostic")
print("  Weight cut: raw RecoMCTruthLink.weight, no decoding")
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



"""
pi0_invariant_mass_v2.py
========================
π⁰ → γγ invariant mass analysis: MC truth + reco (Tier 1 + Tier 2).

Changes from v1:
  - Reco pairing is now π⁰-CONSTRAINED: reco photons are grouped by their
    MC π⁰ parent object ID. Only pairs from the SAME π⁰ are formed.
    This eliminates combinatorial background entirely.
  - π⁰ reconstruction efficiency is tracked:
      n_pi0_mc_total, n_pi0_0reco, n_pi0_1reco, n_pi0_exactly2reco, n_pi0_3plus_reco
  - Skip count (≠2 reco daughters) reported separately for G4 and CC3.
  - Fit model: Gaussian + LINEAR background (not quadratic).
    Quadratic was compensating for combinatorial background that no longer exists.
  - Fit window tightened to 110–170 MeV (combinatorics removed, signal is clean).
  - Split-off annotation removed (was physically wrong — spike is in single-γ
    energy spectrum, not di-photon mass spectrum).
  - Three-tier narrative:
      Analysis A: MC truth (sanity check, δ-function at 134.97 MeV)
      Tier 1:     π⁰-constrained reco, no energy gate (full acceptance)
      Tier 2:     π⁰-constrained reco, E_mc > 10 GeV on both daughters
                  (CC3-active phase space; also selects high-boost π⁰s —
                   stated explicitly on plot title)
  - Fit results (μ, σ, bias) displayed as text box on plot face.

Physics questions:
  Analysis A: Sharp peak at 134.97 MeV? G4 == CC3? Any difference = code bug.
  Tier 1:     Does CC3 reconstruct π⁰ mass peak position and width correctly
              across full acceptance?
  Tier 2:     In CC3-active phase space (high-boost π⁰, both daughters E>10 GeV),
              does CC3 reproduce G4 peak position and resolution?
              Yield difference here = genuine efficiency difference, not artifact.

Link strategy: Strategy A — highest-weight MC match per reco PFO.
Signal (reco): PandoraPFO PDG==22, matched MC genStatus==1 PDG==22, weight>0.5.
π⁰ constraint: matched MC photon must have parent PDG==111.
               Pairs formed only within same π⁰ parent object ID.
Tier 2 gate  : E_mc > 10 GeV on BOTH matched MC photons.

Run:
    source ~/source.sh
    python3 pi0_invariant_mass_v2.py
"""
############################################################

'''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from podio import root_io
from collections import defaultdict
import os, math

# ── CONFIG ────────────────────────────────────────────────────────────────────
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
    f"~/thesis-ml-sim/plots/pi0_invariant_mass_{'1k' if USE_1K else '10k'}_v2"
)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── PHYSICS ───────────────────────────────────────────────────────────────────
PHOTON_PDG   = 22
PI0_PDG      = 111
E_MC_TIER2   = 10.0   # GeV — CC3 active phase space gate (applied to E_mc, not reco)
WEIGHT_CUT   = 0.5
PI0_MASS_MEV = 134.97 # MeV — PDG value

# ── HISTOGRAM SETTINGS ────────────────────────────────────────────────────────
# Reco: 2 MeV bins, 80–200 MeV.
# Motivated by ILD ECAL stochastic ~15%/sqrt(E): at 15 GeV, sigma_m ~ 10-20 MeV
# → need ~5-10 bins across peak. 2 MeV gives ~10 bins across a 20 MeV sigma.
M_BINS        = np.arange(0.080, 0.201, 0.002)   # GeV
M_CENTERS     = 0.5 * (M_BINS[:-1] + M_BINS[1:])

# Truth: 1 MeV bins, 120–155 MeV. No detector smearing → width < 1 MeV expected.
M_BINS_TRUTH  = np.arange(0.120, 0.155, 0.001)
M_CENTERS_TRUTH = 0.5 * (M_BINS_TRUTH[:-1] + M_BINS_TRUTH[1:])

# ── COLORS / STYLE ────────────────────────────────────────────────────────────
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

# ── INVARIANT MASS ────────────────────────────────────────────────────────────
def m_gg(e1, px1, py1, pz1, e2, px2, py2, pz2):
    """Full 4-vector invariant mass. Guard against sqrt(negative) from
    floating-point precision on nearly collinear photons."""
    m2 = 2.0 * (e1*e2 - px1*px2 - py1*py2 - pz1*pz2)
    return math.sqrt(max(m2, 0.0))

# ── FIT MODEL ─────────────────────────────────────────────────────────────────
def gauss_plus_linear(x, N, mu, sigma, a0, a1):
    """Gaussian + linear background.
    Quadratic dropped: combinatorial background is gone by construction
    (π⁰-constrained pairing). Quadratic was compensating for non-existent
    background and destabilized fits on clean signal."""
    return N * np.exp(-0.5 * ((x - mu) / sigma)**2) + a0 + a1 * x

# ── ANALYSIS A: MC TRUTH ──────────────────────────────────────────────────────
def run_truth(filepath, label):
    """
    Strategy: reverse walk — for each genStatus==1 photon, check if parent
    is a π⁰ (PDG==111). Group photons by parent object ID. For each π⁰ with
    exactly 2 photon daughters, compute M_γγ.

    π⁰ generatorStatus == 2 in this sample (Pythia intermediate decay product).
    We filter on the photon's genStatus==1, not the π⁰'s — correct.

    Expected: delta-function at 134.97 MeV, width < 1 MeV.
    G4 and CC3 should be identical (same Pythia generator).
    Any difference = bug in the analysis code.
    """
    print(f"\n{'='*60}")
    print(f"  Analysis A (truth): {label}  [{RUN_LABEL}]")
    print(f"{'='*60}")

    reader   = root_io.Reader(filepath)
    masses   = []
    n_events = 0
    n_pi0    = 0
    probed   = False

    for event in reader.get("events"):
        if n_events >= MAX_EVENTS:
            break
        n_events += 1

        pi0_photons = defaultdict(list)  # pi0_obj_id -> [(e, px, py, pz)]

        for mc in event.get("MCParticles"):
            if mc.getGeneratorStatus() != 1:
                continue
            if mc.getPDG() != PHOTON_PDG:
                continue
            parents = mc.getParents()
            if len(parents) == 0:
                continue
            parent = parents[0]

            if not probed and parent.getPDG() == PI0_PDG:
                print(f"  [probe] First π⁰ parent: PDG={parent.getPDG()}, "
                      f"genStatus={parent.getGeneratorStatus()}, "
                      f"E={parent.getEnergy():.2f} GeV")
                probed = True

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
    print(f"  π⁰ pairs found   : {n_pi0}")
    print(f"  M_γγ computed    : {len(masses)}")
    if len(masses) > 0:
        print(f"  Mean M_γγ [MeV]  : {1000*np.mean(masses):.3f}")
        print(f"  Std  M_γγ [MeV]  : {1000*np.std(masses):.3f}  "
              f"← should be < 1 MeV (generator level, no smearing)")
    return masses


# ── RECO-LEVEL ANALYSIS ───────────────────────────────────────────────────────
def run_reco(filepath, label):
    """
    π⁰-CONSTRAINED pairing (key change from v1).

    For each reco photon PFO:
      1. Find best-weight MC match (Strategy A: highest weight wins per PFO).
      2. Require weight > 0.5, genStatus==1, PDG==22.
      3. Require MC parent PDG==111 (π⁰). This rejects:
           - ISR photons (parent PDG==11)
           - Photons from other sources (FSR, secondaries)
      4. Group the reco PFO by the π⁰ parent's object ID.

    Then for each π⁰ group:
      - Exactly 2 reco photons: form the mass pair (clean reconstruction).
      - ≠ 2 reco photons: SKIP (counted in efficiency table).
        * 0 reco photons: π⁰ completely lost
        * 1 reco photon: cluster merged or one photon lost
        * 3+ reco photons: cluster splitting — one true photon split into
                           multiple PFOs. This is the same effect studied
                           in Chapter 4. Counted here, not included in mass.

    Efficiency table (reported per label):
      n_pi0_mc_total    — all MC π⁰s in events (to compute true denominator)
      n_pi0_0reco       — π⁰s with 0 matched reco photon daughters
      n_pi0_1reco       — π⁰s with 1 matched reco photon daughter
      n_pi0_exactly2    — π⁰s with exactly 2 → used for mass
      n_pi0_3plus       — π⁰s with 3+ (splitting)

    Tier definitions (applied AFTER π⁰-constrained pairing):
      Tier 1: all π⁰-constrained pairs (no energy gate)
              → measures mass reconstruction across full acceptance
      Tier 2: both matched MC photons have E_mc > 10 GeV
              → CC3-active phase space; also selects high-boost π⁰s
                (symmetric decays with high-E daughters)
              → peak position unbiased (M_γγ is Lorentz-invariant)
              → peak width NARROWER than Tier 1 (higher-E photons →
                better stochastic resolution, ~1/sqrt(E))
              → yield difference G4 vs CC3 = genuine efficiency difference
    """
    print(f"\n{'='*60}")
    print(f"  Reco analysis: {label}  [{RUN_LABEL}]")
    print(f"{'='*60}")

    reader = root_io.Reader(filepath)

    tier1_masses = []
    tier2_masses = []
    n_events     = 0

    # Efficiency counters
    n_pi0_mc_total  = 0
    n_pi0_0reco     = 0
    n_pi0_1reco     = 0
    n_pi0_exactly2  = 0
    n_pi0_3plus     = 0

    for event in reader.get("events"):
        if n_events >= MAX_EVENTS:
            break
        n_events += 1

        # ── Step 1: Count all MC π⁰s in this event (denominator) ──────────
        # π⁰s that have at least one genStatus==1 photon daughter in MCParticles.
        # We count distinct π⁰ object IDs that appear as parent of a genStatus==1 photon.
        # This is the correct denominator: π⁰s that actually decayed to photons
        # within the generator's truth record.
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

        # ── Step 2: Strategy A — best-weight MC match per reco PFO ─────────
        reco_to_mc  = {}
        reco_to_wgt = {}
        for link in event.get("RecoMCTruthLink"):
            reco   = link.getFrom()
            mc     = link.getTo()
            weight = link.getWeight()
            rid    = reco.getObjectID().index
            if rid not in reco_to_wgt or weight > reco_to_wgt[rid]:
                reco_to_mc[rid]  = mc
                reco_to_wgt[rid] = weight

        # ── Step 3: Group selected reco photons by MC π⁰ parent ────────────
        # Each entry: (e_reco, px, py, pz, e_mc)
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

            # Require π⁰ parent — key constraint that removes combinatorics
            mc_parents = mc.getParents()
            if len(mc_parents) == 0:
                continue
            mc_parent = mc_parents[0]
            if mc_parent.getPDG() != PI0_PDG:
                continue  # reject ISR photons, FSR, secondaries

            pi0_obj_id = mc_parent.getObjectID().index
            mom = pfo.getMomentum()
            pi0_reco_photons[pi0_obj_id].append(
                (pfo.getEnergy(), mom.x, mom.y, mom.z, mc.getEnergy())
            )

        # ── Step 4: Form masses from exactly-2 groups; count efficiency ────
        # For π⁰s in MC truth that have zero reco photons matched, they won't
        # appear in pi0_reco_photons at all → captured in n_pi0_0reco below.
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

                # Tier 2: E_mc > 10 GeV on both daughters.
                # Cut is on e_mc (true energy), not reco energy.
                # Same cut on both G4 and CC3 → yield difference is pure efficiency.
                # Kinematic note: selects high-boost π⁰s with roughly symmetric
                # decays (both daughters energetic). Peak position unbiased
                # (Lorentz-invariant); peak width narrower (better resolution at
                # higher energy: sigma_E/E ~ 15%/sqrt(E)).
                if emc1 > E_MC_TIER2 and emc2 > E_MC_TIER2:
                    tier2_masses.append(mass)
            elif n >= 3:
                n_pi0_3plus += 1

        # π⁰s in MC truth that produced zero matched reco photons
        n_pi0_0reco += len(mc_pi0_ids_with_photon_daughters - seen_pi0_reco_ids)

    # ── Efficiency report ──────────────────────────────────────────────────
    print(f"\n  π⁰ RECONSTRUCTION EFFICIENCY [{label}]")
    print(f"  {'─'*45}")
    print(f"  MC π⁰s (denominator)         : {n_pi0_mc_total}")
    print(f"  0 reco photons (lost entirely): {n_pi0_0reco:6d}  "
          f"({100*n_pi0_0reco/max(n_pi0_mc_total,1):.1f}%)")
    print(f"  1 reco photon  (merged/lost 1): {n_pi0_1reco:6d}  "
          f"({100*n_pi0_1reco/max(n_pi0_mc_total,1):.1f}%)")
    print(f"  2 reco photons (clean → mass) : {n_pi0_exactly2:6d}  "
          f"({100*n_pi0_exactly2/max(n_pi0_mc_total,1):.1f}%)  ← USED FOR MASS")
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
    return np.array(tier1_masses), np.array(tier2_masses), eff


# ── FIT HELPER ────────────────────────────────────────────────────────────────
def fit_gaussian(centers, counts, label):
    """
    Gaussian + linear background fit.
    Fit window: 110–170 MeV (tightened from v1's 100–200 MeV).
    Combinatorial background gone → no need for wide window or quadratic.
    Starting guess: mu = 135 MeV (PDG prior, not 140 MeV as in v1).
    """
    fit_mask = (centers > 0.110) & (centers < 0.170)
    x = centers[fit_mask]
    y = counts[fit_mask].astype(float)

    if len(x) < 5 or y.max() < 5:
        print(f"  Fit [{label}] SKIPPED: insufficient statistics in fit window")
        return None, None

    mu0    = PI0_MASS_MEV / 1000.0   # 0.13497 GeV — PDG prior
    sigma0 = 0.015
    N0     = y.max() * sigma0 * np.sqrt(2 * np.pi)
    p0     = [N0, mu0, sigma0, 1.0, 0.0]

    lower  = [  0,  0.100,  0.005, -1e6, -1e6]
    upper  = [1e9,  0.200,  0.040,  1e6,  1e6]

    try:
        popt, pcov = curve_fit(gauss_plus_linear, x, y,
                               p0=p0, bounds=(lower, upper),
                               maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        N, mu, sigma = popt[0], popt[1], popt[2]
        dN, dmu, dsigma = perr[0], perr[1], perr[2]
        bias_mev = 1000 * (mu - PI0_MASS_MEV/1000.0)
        print(f"  Fit [{label:3s}]: "
              f"μ = {1000*mu:.2f} ± {1000*dmu:.2f} MeV | "
              f"σ = {1000*sigma:.2f} ± {1000*dsigma:.2f} MeV | "
              f"bias = {bias_mev:+.2f} MeV | "
              f"N = {abs(N):.0f}")
        return popt, perr
    except RuntimeError as e:
        print(f"  Fit [{label}] FAILED: {e}")
        return None, None


# ── PLOT: TRUTH M_γγ ──────────────────────────────────────────────────────────
def plot_truth(g4_masses, cc3_masses):
    """
    Physics question: do G4 and CC3 give identical truth M_γγ?
    Expected: sharp δ-function peak at 134.97 MeV, width < 1 MeV.
    Any difference = bug in analysis code (same Pythia generator used for both).
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(1000*g4_masses,  bins=1000*M_BINS_TRUTH, histtype="step",
            color=C_G4,  linewidth=2, label=f"Geant4  (N={len(g4_masses)})")
    ax.hist(1000*cc3_masses, bins=1000*M_BINS_TRUTH, histtype="step",
            color=C_CC3, linewidth=2, linestyle="--",
            label=f"CC3  (N={len(cc3_masses)})")
    ax.axvline(PI0_MASS_MEV, color="black", lw=1.0, ls=":", alpha=0.7,
               label=f"PDG π⁰ mass ({PI0_MASS_MEV} MeV)")

    ax.set_xlabel("M$_{\\gamma\\gamma}$ [MeV]")
    ax.set_ylabel("π⁰ candidates / 1 MeV")
    ax.set_title(
        "Analysis A — MC Truth M$_{\\gamma\\gamma}$: G4 vs CC3\n"
        "genStatus==1 photons, π⁰ parent (PDG==111), exactly 2 daughters per π⁰\n"
        "No detector effects. Width > 1 MeV = bug in pairing logic.",
        fontsize=10
    )
    ax.legend(framealpha=0.5)
    ax.text(0.01, 1.01, f"ILD sim — {RUN_LABEL} — raw counts",
            transform=ax.transAxes, fontsize=8, color="gray")

    plt.tight_layout()
    out = f"{PLOT_DIR}/analysisA_truth_mgg.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓  Saved: analysisA_truth_mgg.png")


# ── PLOT: RECO M_γγ (TIER 1 + TIER 2 SIDE BY SIDE) ──────────────────────────
def plot_reco(g4_t1, cc3_t1, g4_t2, cc3_t2,
              g4_eff, cc3_eff):
    """
    Physics questions:
      Tier 1: Does CC3 reproduce the π⁰ mass peak position and width
              across full acceptance?
      Tier 2: In CC3-active phase space (E_mc > 10 GeV on both daughters,
              selects high-boost π⁰s with symmetric decays), does CC3
              reproduce G4 peak position and resolution?
              Tier 2 width is EXPECTED to be narrower than Tier 1
              (higher-energy photons → better stochastic resolution ~1/sqrt(E)).
              Yield difference G4 vs CC3 here = genuine efficiency difference.

    Normalization: raw counts shown.
    Event counts equal (same N events per file).
    Pair counts may differ due to different π⁰ reconstruction efficiency
    between G4 and CC3 — stated explicitly on plot.

    Fit: Gaussian + linear background, window 110–170 MeV.
    Fit results (μ, σ, bias) shown as text box on each panel.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    tiers = [
        (axes[0], g4_t1, cc3_t1, g4_eff, cc3_eff, "Tier 1",
         "π⁰-constrained reco pairs, no energy gate\n"
         "(full acceptance, both daughters any E_mc)"),
        (axes[1], g4_t2, cc3_t2, g4_eff, cc3_eff, "Tier 2",
         "π⁰-constrained reco pairs | E$_{mc}$ > 10 GeV on both daughters\n"
         "CC3-active phase space; selects high-boost π⁰, symmetric decays\n"
         "Yield difference G4 vs CC3 = genuine efficiency difference"),
    ]

    for ax, g4_m, cc3_m, g4e, cc3e, tier, subtitle in tiers:
        g4_counts,  _ = np.histogram(g4_m,  bins=M_BINS)
        cc3_counts, _ = np.histogram(cc3_m, bins=M_BINS)

        ax.hist(g4_m,  bins=M_BINS, histtype="step",
                color=C_G4,  linewidth=2,
                label=f"Geant4  (N={len(g4_m)}, "
                      f"eff={g4e['eff_pct']:.1f}%)")
        ax.hist(cc3_m, bins=M_BINS, histtype="step",
                color=C_CC3, linewidth=2, linestyle="--",
                label=f"CC3  (N={len(cc3_m)}, "
                      f"eff={cc3e['eff_pct']:.1f}%)")
        ax.axvline(PI0_MASS_MEV/1000, color="gray", lw=1.0, ls=":",
                   alpha=0.7, label=f"PDG: {PI0_MASS_MEV:.0f} MeV")

        # Fit and plot curves; collect results for text box
        fit_texts = []
        print(f"\n  Fitting {tier}:")
        for counts, flabel, color in [
            (g4_counts,  "G4",  C_G4),
            (cc3_counts, "CC3", C_CC3),
        ]:
            popt, perr = fit_gaussian(M_CENTERS, counts, flabel)
            if popt is not None:
                x_fit = np.linspace(0.100, 0.180, 800)
                ax.plot(x_fit, gauss_plus_linear(x_fit, *popt),
                        color=color, lw=1.8, ls="-", alpha=0.75)
                mu_mev    = 1000 * popt[1]
                sig_mev   = 1000 * popt[2]
                dmu_mev   = 1000 * perr[1]
                dsig_mev  = 1000 * perr[2]
                bias_mev  = mu_mev - PI0_MASS_MEV
                fit_texts.append(
                    f"{flabel}: μ = {mu_mev:.1f}±{dmu_mev:.1f} MeV, "
                    f"σ = {sig_mev:.1f}±{dsig_mev:.1f} MeV, "
                    f"bias = {bias_mev:+.1f} MeV"
                )

        # Text box with fit results on plot face
        if fit_texts:
            box_text = f"PDG: {PI0_MASS_MEV} MeV\n" + "\n".join(fit_texts)
            ax.text(0.97, 0.97, box_text,
                    transform=ax.transAxes,
                    fontsize=7.5,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor="white", alpha=0.85,
                              edgecolor="lightgray"))

        ax.set_xlabel("M$_{\\gamma\\gamma}$ [GeV]")
        ax.set_ylabel("π⁰ candidates / 2 MeV")
        ax.set_title(f"Reco M$_{{\\gamma\\gamma}}$ — {tier}\n{subtitle}",
                     fontsize=9.5)
        ax.legend(framealpha=0.5, fontsize=9)

    axes[0].text(0.01, 1.01,
                 f"ILD reco — {RUN_LABEL} — π⁰-constrained pairs — "
                 f"genStatus==1, weight>0.5 — raw counts",
                 transform=axes[0].transAxes, fontsize=8, color="gray")

    plt.tight_layout()
    out = f"{PLOT_DIR}/reco_mgg_tier1_tier2.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓  Saved: reco_mgg_tier1_tier2.png")


# ── EFFICIENCY SUMMARY TABLE ──────────────────────────────────────────────────
def print_efficiency_comparison(g4_eff, cc3_eff):
    """Print a side-by-side efficiency table for G4 vs CC3."""
    print(f"\n{'='*60}")
    print("  π⁰ RECONSTRUCTION EFFICIENCY COMPARISON")
    print(f"  {'─'*58}")
    print(f"  {'Category':<32}  {'G4':>10}  {'CC3':>10}")
    print(f"  {'─'*58}")
    rows = [
        ("MC π⁰s (denominator)",    "n_mc_total", False),
        ("0 reco photons (lost)",    "n_0reco",    True),
        ("1 reco photon (merged)",   "n_1reco",    True),
        ("2 reco photons (→ mass)",  "n_exactly2", True),
        ("3+ reco photons (split)",  "n_3plus",    True),
    ]
    for name, key, show_pct in rows:
        g4v  = g4_eff[key]
        cc3v = cc3_eff[key]
        if show_pct:
            g4_pct  = 100 * g4v  / max(g4_eff["n_mc_total"],  1)
            cc3_pct = 100 * cc3v / max(cc3_eff["n_mc_total"], 1)
            print(f"  {name:<32}  {g4v:5d} ({g4_pct:4.1f}%)  "
                  f"{cc3v:5d} ({cc3_pct:4.1f}%)")
        else:
            print(f"  {name:<32}  {g4v:>10}  {cc3v:>10}")
    print(f"  {'─'*58}")
    print(f"  {'Clean reco efficiency':<32}  "
          f"{g4_eff['eff_pct']:9.1f}%  {cc3_eff['eff_pct']:9.1f}%")
    delta = cc3_eff['eff_pct'] - g4_eff['eff_pct']
    print(f"  {'Δ efficiency (CC3 − G4)':<32}  {delta:+.1f} pp")
    if abs(delta) > 5.0:
        print(f"  *** Δ > 5 pp — headline result for thesis. ***")
    print(f"{'='*60}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  pi0_invariant_mass_v2.py")
print("  Analysis A (truth) + Tier 1 + Tier 2 reco M_γγ")
print("  π⁰-constrained pairing | efficiency tracking | linear fit")
print("="*60)

print("\n── Analysis A: MC Truth ──")
g4_truth  = run_truth(G4_FILE,  "Geant4")
cc3_truth = run_truth(CC3_FILE, "CC3")

print("\n── Reco: G4 ──")
g4_t1, g4_t2, g4_eff = run_reco(G4_FILE, "G4")

print("\n── Reco: CC3 ──")
cc3_t1, cc3_t2, cc3_eff = run_reco(CC3_FILE, "CC3")

print_efficiency_comparison(g4_eff, cc3_eff)

print("\n── Generating plots ──")
plot_truth(g4_truth, cc3_truth)
plot_reco(g4_t1, cc3_t1, g4_t2, cc3_t2, g4_eff, cc3_eff)

print(f"\n  All plots saved to: {PLOT_DIR}")
print(f"  Done.\n")'''
