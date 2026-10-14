"""
pi0_response_diagnostics.py
============================
Standalone script -- three diagnostic plots for the Chapter 5 mass-bias
hypothesis, extracted from pi0_invariant_mass_v4.py logic (no splitting
diagnostic, no truth-level closure check, no Tier1/Tier2 mass histograms --
just the three plots below):

  1. Energy response: E_reco/E_mc per matched photon (clean n==2 pi0s)
  2. Opening angle bias: dAlpha_reco/dAlpha_mc per clean n==2 pi0 pair
     - FIX (2026-08-03): near-collinear high-E pi0s have dAlpha_mc -> 0,
       which blows up the ratio numerically (not a physics effect -- pure
       division-by-small-number artifact). Guarded with DALPHA_MC_MIN cut,
       and median used instead of mean for the summary stat (robust against
       any remaining tail). Confirmed against a toy reproduction of the
       mechanism: a handful of near-zero-denominator pairs is enough to drag
       the mean to O(1e3-1e4) while the bulk distribution (and median) stay
       physically sane near 1.0.
  3. Mass peak (mu) vs E_mc(pi0) bins: same E_mc bin convention as the rest
     of the analysis ([10, 15, 20, 30, 50, 80, 127] GeV, per setup.md), fit
     per bin with the same Gaussian+linear model used elsewhere in the
     pi0 mass analysis. Tests whether the 8-14 MeV bias is flat across
     energy or grows/shrinks with E_mc -- directly informs whether the
     energy-response and angle-bias effects above are energy-dependent.

Run on NAF:
    source ~/source.sh
    python3 pi0_response_diagnostics.py
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
    f"~/thesis-ml-sim/plots/pi0_response_diagnostics_{'1k' if USE_1K else '10k'}"
)
os.makedirs(PLOT_DIR, exist_ok=True)

# PHYSICS
PHOTON_PDG   = 22
PI0_PDG      = 111
WEIGHT_CUT   = 0.5
PI0_MASS_MEV = 134.97
DALPHA_MC_MIN = 0.001  # rad -- below this, reco/mc angle ratio is numerically unstable

# E_mc(pi0) bins -- same convention as rest of the analysis (setup.md)
E_MC_PI0_BINS = np.array([10, 15, 20, 30, 50, 80, 127])

# HISTOGRAM SETTINGS
M_BINS  = np.arange(0.080, 0.201, 0.002)
M_CENTERS = 0.5 * (M_BINS[:-1] + M_BINS[1:])
E_RESPONSE_BINS = np.arange(0.80, 1.21, 0.01)
ANGLE_RATIO_BINS = np.arange(0.70, 1.31, 0.01)

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
    """Verified 2026-08-03 against real raw weight distributions in both G4
    and CC3 REC files: hard ceiling at 10,001,000-10,001,004, consistent
    with raw = 10000*clusterWeight_permille + trackWeight_permille. /1000.0
    scaling gives cluster_weight cleanly bounded in [0,1]."""
    raw_int = int(round(raw_weight))
    cluster_weight = (raw_int // 10000) / 1000.0
    track_weight   = (raw_int % 10000) / 1000.0
    return cluster_weight, track_weight


def gauss_plus_linear(x, N, mu, sigma, a0, a1):
    return N * np.exp(-0.5 * ((x - mu) / sigma)**2) + a0 + a1 * x


def fit_gaussian(centers, counts, label):
    fit_mask = (centers > 0.110) & (centers < 0.170)
    x = centers[fit_mask]
    y = counts[fit_mask].astype(float)

    if len(x) < 5 or y.max() < 5:
        print(f"    Fit [{label}] SKIPPED: insufficient statistics")
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
        return popt, perr
    except RuntimeError as e:
        print(f"    Fit [{label}] FAILED: {e}")
        return None, None


def run_reco(filepath, label):
    print(f"\n{'='*60}")
    print(f"  Reco pass: {label}  [{RUN_LABEL}]")
    print(f"{'='*60}")

    reader = root_io.Reader(filepath)
    n_events = 0

    energy_response    = []
    opening_angle_ratio = []
    n_angle_skipped_collinear = 0

    # per-pi0 record for the mass-vs-E_mc binning: (mass, E_mc_pi0_estimate)
    mass_records = []

    for event in reader.get("events"):
        if n_events >= MAX_EVENTS:
            break
        n_events += 1

        reco_to_mc  = {}
        reco_to_wgt = {}
        for link in event.get("RecoMCTruthLink"):
            reco  = link.getFrom()
            mc    = link.getTo()
            raw_w = link.getWeight()
            cluster_w, _ = decode_weight(raw_w)
            rid = reco.getObjectID().index
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
            mc_mom = mc.getMomentum()
            pi0_reco_photons[pi0_obj_id].append(
                (pfo.getEnergy(), mom.x, mom.y, mom.z,
                 mc.getEnergy(), mc_mom.x, mc_mom.y, mc_mom.z)
            )

        for pi0_id, photons in pi0_reco_photons.items():
            if len(photons) != 2:
                continue
            e1, px1, py1, pz1, emc1, mcpx1, mcpy1, mcpz1 = photons[0]
            e2, px2, py2, pz2, emc2, mcpx2, mcpy2, mcpz2 = photons[1]

            mass = m_gg(e1, px1, py1, pz1, e2, px2, py2, pz2)
            e_mc_pi0 = emc1 + emc2  # proxy for pi0 energy: sum of matched daughter MC energies
            mass_records.append((mass, e_mc_pi0))

            if emc1 > 0:
                energy_response.append(e1 / emc1)
            if emc2 > 0:
                energy_response.append(e2 / emc2)

            p1_reco = math.sqrt(px1**2 + py1**2 + pz1**2)
            p2_reco = math.sqrt(px2**2 + py2**2 + pz2**2)
            p1_mc   = math.sqrt(mcpx1**2 + mcpy1**2 + mcpz1**2)
            p2_mc   = math.sqrt(mcpx2**2 + mcpy2**2 + mcpz2**2)
            if p1_reco > 0 and p2_reco > 0 and p1_mc > 0 and p2_mc > 0:
                cos_reco = (px1*px2 + py1*py2 + pz1*pz2) / (p1_reco * p2_reco)
                cos_mc   = (mcpx1*mcpx2 + mcpy1*mcpy2 + mcpz1*mcpz2) / (p1_mc * p2_mc)
                dalpha_reco = math.acos(np.clip(cos_reco, -1.0, 1.0))
                dalpha_mc   = math.acos(np.clip(cos_mc, -1.0, 1.0))
                if dalpha_mc > DALPHA_MC_MIN:
                    opening_angle_ratio.append(dalpha_reco / dalpha_mc)
                else:
                    n_angle_skipped_collinear += 1

    energy_response    = np.array(energy_response)
    opening_angle_ratio = np.array(opening_angle_ratio)
    mass_records = np.array(mass_records)  # shape (N, 2): [mass, e_mc_pi0]

    print(f"  Events processed              : {n_events}")
    print(f"  Energy response entries       : {len(energy_response)}")
    print(f"  Opening angle ratio entries   : {len(opening_angle_ratio)}  "
          f"({n_angle_skipped_collinear} skipped, dAlpha_mc < {DALPHA_MC_MIN} rad)")
    if len(energy_response) > 0:
        print(f"  Mean   E_reco/E_mc             : {np.mean(energy_response):.4f}")
    if len(opening_angle_ratio) > 0:
        print(f"  Median dAlpha_reco/dAlpha_mc    : {np.median(opening_angle_ratio):.4f}")
        print(f"  Mean   dAlpha_reco/dAlpha_mc    : {np.mean(opening_angle_ratio):.4f}  "
              f"(sensitive to tail -- see median instead)")

    return energy_response, opening_angle_ratio, mass_records


def plot_energy_response(g4_resp, cc3_resp):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(g4_resp,  bins=E_RESPONSE_BINS, histtype="step", color=C_G4,
            linewidth=2, density=True,
            label=f"Geant4 (N={len(g4_resp)}, mean={np.mean(g4_resp):.3f})")
    ax.hist(cc3_resp, bins=E_RESPONSE_BINS, histtype="step", color=C_CC3,
            linewidth=2, linestyle="--", density=True,
            label=f"CC3 (N={len(cc3_resp)}, mean={np.mean(cc3_resp):.3f})")
    ax.axvline(1.0, color="black", lw=1.0, ls=":", alpha=0.7, label="Perfect response (1.0)")
    ax.set_xlabel("E_reco / E_mc  (per matched photon, clean n==2 pi0s only)")
    ax.set_ylabel("Normalized density")
    ax.set_title("Photon energy response: G4 vs CC3\n"
                 "Signal: matched daughters of clean (n_reco==2) pi0 candidates",
                 fontsize=9.5)
    ax.legend(framealpha=0.5, fontsize=9)
    plt.tight_layout()
    out = f"{PLOT_DIR}/energy_response.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: energy_response.png")


def plot_opening_angle_bias(g4_ratio, cc3_ratio):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(g4_ratio,  bins=ANGLE_RATIO_BINS, histtype="step", color=C_G4,
            linewidth=2, density=True,
            label=f"Geant4 (N={len(g4_ratio)}, median={np.median(g4_ratio):.3f})")
    ax.hist(cc3_ratio, bins=ANGLE_RATIO_BINS, histtype="step", color=C_CC3,
            linewidth=2, linestyle="--", density=True,
            label=f"CC3 (N={len(cc3_ratio)}, median={np.median(cc3_ratio):.3f})")
    ax.axvline(1.0, color="black", lw=1.0, ls=":", alpha=0.7, label="Perfect angle (1.0)")
    ax.set_xlabel("dAlpha_reco / dAlpha_mc  (per clean n==2 pi0 pair)")
    ax.set_ylabel("Normalized density")
    ax.set_title("Opening angle bias: G4 vs CC3\n"
                 f"Signal: clean (n_reco==2) pi0 pairs, dAlpha_mc > {DALPHA_MC_MIN} rad",
                 fontsize=9.5)
    ax.legend(framealpha=0.5, fontsize=9)
    plt.tight_layout()
    out = f"{PLOT_DIR}/opening_angle_bias.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: opening_angle_bias.png")


def plot_mass_vs_emc(g4_records, cc3_records):
    """Fits the pi0 mass peak in each E_mc(pi0) bin, G4 vs CC3, and plots
    mu (with fit uncertainty) as a function of bin center."""
    bin_centers = 0.5 * (E_MC_PI0_BINS[:-1] + E_MC_PI0_BINS[1:])

    results = {"G4": {"mu": [], "dmu": []}, "CC3": {"mu": [], "dmu": []}}

    for records, tag in [(g4_records, "G4"), (cc3_records, "CC3")]:
        if len(records) == 0:
            results[tag]["mu"] = [np.nan] * len(bin_centers)
            results[tag]["dmu"] = [np.nan] * len(bin_centers)
            continue
        masses  = records[:, 0]
        e_mc    = records[:, 1]
        for lo, hi in zip(E_MC_PI0_BINS[:-1], E_MC_PI0_BINS[1:]):
            mask = (e_mc >= lo) & (e_mc < hi)
            m_sel = masses[mask]
            counts, _ = np.histogram(m_sel, bins=M_BINS)
            popt, perr = fit_gaussian(M_CENTERS, counts, f"{tag} [{lo:.0f}-{hi:.0f} GeV]")
            if popt is not None:
                results[tag]["mu"].append(1000 * popt[1])
                results[tag]["dmu"].append(1000 * perr[1])
            else:
                results[tag]["mu"].append(np.nan)
                results[tag]["dmu"].append(np.nan)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(bin_centers, results["G4"]["mu"], yerr=results["G4"]["dmu"],
                marker="o", color=C_G4, linewidth=1.5, capsize=3, label="Geant4")
    ax.errorbar(bin_centers, results["CC3"]["mu"], yerr=results["CC3"]["dmu"],
                marker="s", color=C_CC3, linewidth=1.5, linestyle="--", capsize=3, label="CC3")
    ax.axhline(PI0_MASS_MEV, color="black", lw=1.0, ls=":", alpha=0.7,
               label=f"PDG pi0 mass ({PI0_MASS_MEV} MeV)")
    ax.set_xlabel("E_mc(pi0) bin center [GeV]  (sum of matched daughter MC energies)")
    ax.set_ylabel("Fitted mass peak mu [MeV]")
    ax.set_title("pi0 mass peak vs pi0 energy: G4 vs CC3\n"
                 f"E_mc(pi0) bins: {list(E_MC_PI0_BINS)} GeV",
                 fontsize=9.5)
    ax.legend(framealpha=0.5, fontsize=9)
    plt.tight_layout()
    out = f"{PLOT_DIR}/mass_peak_vs_emc.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: mass_peak_vs_emc.png")

    print("\n  Mass peak vs E_mc(pi0) bin summary:")
    for i, (lo, hi) in enumerate(zip(E_MC_PI0_BINS[:-1], E_MC_PI0_BINS[1:])):
        g4_mu, cc3_mu = results["G4"]["mu"][i], results["CC3"]["mu"][i]
        print(f"    [{lo:>4.0f}-{hi:<4.0f} GeV]  G4 mu={g4_mu:.2f} MeV   CC3 mu={cc3_mu:.2f} MeV")


print("\n" + "="*60)
print("  pi0_response_diagnostics.py")
print("  Energy response + opening angle bias + mass peak vs E_mc(pi0)")
print("="*60)

g4_resp, g4_angle, g4_mass_records   = run_reco(G4_FILE,  "G4")
cc3_resp, cc3_angle, cc3_mass_records = run_reco(CC3_FILE, "CC3")

print("\n-- Generating plots --")
plot_energy_response(g4_resp, cc3_resp)
plot_opening_angle_bias(g4_angle, cc3_angle)
plot_mass_vs_emc(g4_mass_records, cc3_mass_records)

print(f"\n  All plots saved to: {PLOT_DIR}")
print(f"  Done.\n")
