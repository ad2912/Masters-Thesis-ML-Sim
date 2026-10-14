"""
pi0_invariant_mass_v5.py
========================
pi0 -> gamma gamma invariant mass analysis: MC truth (zoomed only) + reco
(Tier 1 + Tier 2) + cluster-splitting diagnostic + energy response + opening
angle bias + E_mc>=10 GeV "fast-sim region" diagnostic set.

Changes from v4:
  - ADDED: RUN_SIZE config ("1k" / "10k" / "100k") replacing the old USE_1K
    boolean. Selects G4/CC3 file paths, MAX_EVENTS, RUN_LABEL, and PLOT_DIR
    from one table -- no other code needs to know which size is active.
  - CHANGED: invariant-mass fit is now a pure Gaussian (gauss_plus_linear
    removed as the primary fit). A separate Crystal-Ball fit is done in
    parallel and BOTH are saved as separate figures per tier, so the two
    models can be compared side by side instead of overlaid.
  - ADDED: "fast-sim region" (Emc>=10 GeV on BOTH matched daughters) parallel
    diagnostic set -- efficiency table, splitting diagnostic, energy
    response, and opening-angle bias are all regenerated under this gate,
    using the same underlying per-event collections as the ungated set (no
    second file read; the gate is applied as a boolean mask at fill time).
  - ADDED: ratio panel (CC3/G4, sqrt(N) propagated errors) under the opening-
    angle histogram, for both the ungated and Emc>=10 GeV variants.
  - DEFERRED (per discussion): barrel-minus-phi-crack angular cut. Left as a
    TODO hook (see FASTSIM_GATE_LABEL) -- not implemented in this version.

Changes from v5 (this revision, 2026-08-31 cleanup):
  - REMOVED: unzoomed truth mass plot (plot_truth / analysisA_truth_mgg.png).
    Only the zoomed truth plot is produced now.
  - CHANGED: truth zoom window is now a FIXED range, 134.5-135.5 MeV, with
    fixed 0.1 MeV bins (10 bins total) -- overriding the old data-derived
    mu +/- 3*sigma window. NOTE: this is a manual override; if the truth-level
    peak mu (printed to console) is not close to 135 MeV, this fixed window
    will clip the peak. Check the printed mu before trusting this plot.

Run on NAF:
    source ~/source.sh
    python3 pi0_invariant_mass_v5.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from scipy.special import erf
from podio import root_io
from collections import defaultdict
import os, math

# ============================================================
# CONFIG
# ============================================================
RUN_SIZE = "100k"   # "1k" | "10k" | "100k"

RUN_TABLE = {
    "1k": dict(
        g4  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-1000events-test_REC.edm4hep.root",
        cc3 = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-1000events-test_REC.edm4hep.root",
        max_events = 1000,
        label = "1k events [dev]",
    ),
    "10k": dict(
        g4  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-10k-events-test_REC.edm4hep.root",
        cc3 = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root",
        max_events = 10000,
        label = "10k events",
    ),
    "100k": dict(
        g4  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-geant4-100kevents-reco-test_REC.edm4hep.root",
        cc3 = "/data/dust/user/alimuham/thesis/reco/tau-pi0-caloclouds-100kevents-reco-test_REC.edm4hep.root",
        max_events = 100000,
        label = "100k events",
    ),
}

_cfg = RUN_TABLE[RUN_SIZE]
G4_FILE    = _cfg["g4"]
CC3_FILE   = _cfg["cc3"]
MAX_EVENTS = _cfg["max_events"]
RUN_LABEL  = _cfg["label"]

PLOT_DIR = os.path.expanduser(
    f"~/thesis-ml-sim/plots/pi0_invariant_mass_{RUN_SIZE}_v5"
)
os.makedirs(PLOT_DIR, exist_ok=True)

# PHYSICS
PHOTON_PDG   = 22
PI0_PDG      = 111
E_MC_TIER2   = 10.0      # GeV, Tier 2 gate (both daughters), same value used
                          # for the "fast-sim region" parallel diagnostic set
WEIGHT_CUT   = 0.5
PI0_MASS_MEV = 134.97

# Fast-sim region gate label, shown on all "region B" plots.
# TODO: extend with barrel-minus-phi-crack angular cut once geometry is
# finalized (deferred per 2026-08-31 discussion). Hook left here so the
# label/config point is centralized when that lands.
FASTSIM_GATE_LABEL = f"E_mc >= {E_MC_TIER2:.0f} GeV (both daughters)"

# Crystal Ball tail side. Flagged explicitly (not silently hard-coded) --
# v4 histograms suggested a HIGH-mass tail, which is atypical for calo pi0
# mass (energy leakage usually pulls mass LOW). Verify against the actual
# Tier 1 histogram shape before trusting fit output; flip to "low" if the
# excess is really on the low-mass side.
CB_TAIL_SIDE = "high"   # "high" | "low"

# Truth-level zoom window: FIXED per 2026-08-31 request (was auto-derived
# from a Gaussian fit to the truth mass sample in the previous revision).
# If the printed truth-level mu is far from 135 MeV, this window will clip
# the peak -- check console output before trusting the plot.
TRUTH_ZOOM_LO_MEV  = 134.5
TRUTH_ZOOM_HI_MEV  = 135.5
TRUTH_ZOOM_BIN_MEV = 0.1

# HISTOGRAM SETTINGS
M_BINS        = np.arange(0.080, 0.201, 0.002)
M_CENTERS     = 0.5 * (M_BINS[:-1] + M_BINS[1:])

E_RESPONSE_BINS  = np.arange(0.80, 1.21, 0.01)
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


# ============================================================
# CORE PHYSICS HELPERS
# ============================================================
def m_gg(e1, px1, py1, pz1, e2, px2, py2, pz2):
    m2 = 2.0 * (e1*e2 - px1*px2 - py1*py2 - pz1*pz2)
    return math.sqrt(max(m2, 0.0))


def decode_weight(raw_weight):
    """Decode the packed RecoMCTruthLink weight.

    Verified 2026-08-03 against real raw weight distributions in both G4 and
    CC3 10k REC files: hard ceiling at 10,001,000-10,001,004, consistent with
    raw = 10000*clusterWeight_permille + trackWeight_permille. Decoded
    cluster_weight is cleanly bounded in [0,1] with /1000.0 scaling (max
    exactly 1.0).
    """
    raw_int = int(round(raw_weight))
    cluster_weight = (raw_int // 10000) / 1000.0
    track_weight   = (raw_int % 10000) / 1000.0
    return cluster_weight, track_weight


# ============================================================
# FIT MODELS
# ============================================================
def gauss_pure(x, N, mu, sigma):
    return N * np.exp(-0.5 * ((x - mu) / sigma)**2)


def crystal_ball(x, N, mu, sigma, alpha, n):
    """Single-sided Crystal Ball. alpha, n > 0 required.

    tail_side controls which side of the peak the power-law tail sits on:
    for "high" tail, we mirror x -> (2*mu - x) before evaluating the
    standard (left-tail) CB form, i.e. the tail points to x > mu.
    """
    z = (x - mu) / sigma
    if CB_TAIL_SIDE == "high":
        z = -z
    abs_alpha = np.abs(alpha)
    A = (n / abs_alpha) ** n * np.exp(-0.5 * abs_alpha**2)
    B = n / abs_alpha - abs_alpha
    out = np.where(
        z > -abs_alpha,
        np.exp(-0.5 * z**2),
        A * (B - z) ** (-n)
    )
    return N * out


# ============================================================
# TRUTH-LEVEL ANALYSIS
# ============================================================
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


# ============================================================
# RECO-LEVEL ANALYSIS
# ============================================================
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

    fs_n_pi0_mc_total  = 0
    fs_n_pi0_0reco     = 0
    fs_n_pi0_1reco     = 0
    fs_n_pi0_exactly2  = 0
    fs_n_pi0_3plus     = 0

    split_leading_frac = []
    split_n_fragments  = []
    fs_split_leading_frac = []
    fs_split_n_fragments  = []

    energy_response     = []
    opening_angle_ratio  = []
    fs_energy_response    = []
    fs_opening_angle_ratio = []

    for event in reader.get("events"):
        if n_events >= MAX_EVENTS:
            break
        n_events += 1

        mc_pi0_ids_with_photon_daughters = set()
        mc_pi0_ids_fastsim = set()
        pi0_daughter_energies = defaultdict(list)

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
                pid = parent.getObjectID().index
                mc_pi0_ids_with_photon_daughters.add(pid)
                pi0_daughter_energies[pid].append(mc.getEnergy())

        for pid, elist in pi0_daughter_energies.items():
            if len(elist) == 2 and min(elist) >= E_MC_TIER2:
                mc_pi0_ids_fastsim.add(pid)

        n_pi0_mc_total += len(mc_pi0_ids_with_photon_daughters)
        fs_n_pi0_mc_total += len(mc_pi0_ids_fastsim)

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
            mc_mom = mc.getMomentum()
            pi0_reco_photons[pi0_obj_id].append(
                (pfo.getEnergy(), mom.x, mom.y, mom.z,
                 mc.getEnergy(), mc_mom.x, mc_mom.y, mc_mom.z)
            )

        seen_pi0_reco_ids = set()
        fs_seen_pi0_reco_ids = set()

        for pi0_id, photons in pi0_reco_photons.items():
            seen_pi0_reco_ids.add(pi0_id)
            is_fastsim_pi0 = pi0_id in mc_pi0_ids_fastsim
            if is_fastsim_pi0:
                fs_seen_pi0_reco_ids.add(pi0_id)

            n = len(photons)

            if n == 1:
                n_pi0_1reco += 1
                if is_fastsim_pi0:
                    fs_n_pi0_1reco += 1
            elif n == 2:
                n_pi0_exactly2 += 1
                if is_fastsim_pi0:
                    fs_n_pi0_exactly2 += 1

                e1, px1, py1, pz1, emc1, mcpx1, mcpy1, mcpz1 = photons[0]
                e2, px2, py2, pz2, emc2, mcpx2, mcpy2, mcpz2 = photons[1]
                mass = m_gg(e1, px1, py1, pz1, e2, px2, py2, pz2)
                tier1_masses.append(mass)

                if emc1 > 0:
                    energy_response.append(e1 / emc1)
                    if is_fastsim_pi0:
                        fs_energy_response.append(e1 / emc1)
                if emc2 > 0:
                    energy_response.append(e2 / emc2)
                    if is_fastsim_pi0:
                        fs_energy_response.append(e2 / emc2)

                p1_reco = math.sqrt(px1**2 + py1**2 + pz1**2)
                p2_reco = math.sqrt(px2**2 + py2**2 + pz2**2)
                p1_mc   = math.sqrt(mcpx1**2 + mcpy1**2 + mcpz1**2)
                p2_mc   = math.sqrt(mcpx2**2 + mcpy2**2 + mcpz2**2)
                if p1_reco > 0 and p2_reco > 0 and p1_mc > 0 and p2_mc > 0:
                    cos_reco = (px1*px2 + py1*py2 + pz1*pz2) / (p1_reco * p2_reco)
                    cos_mc   = (mcpx1*mcpx2 + mcpy1*mcpy2 + mcpz1*mcpz2) / (p1_mc * p2_mc)
                    dalpha_reco = math.acos(np.clip(cos_reco, -1.0, 1.0))
                    dalpha_mc   = math.acos(np.clip(cos_mc, -1.0, 1.0))
                    if dalpha_mc > 0:
                        ratio = dalpha_reco / dalpha_mc
                        opening_angle_ratio.append(ratio)
                        if is_fastsim_pi0:
                            fs_opening_angle_ratio.append(ratio)

                if emc1 > E_MC_TIER2 and emc2 > E_MC_TIER2:
                    tier2_masses.append(mass)
            elif n >= 3:
                n_pi0_3plus += 1
                if is_fastsim_pi0:
                    fs_n_pi0_3plus += 1
                energies = sorted([p[0] for p in photons], reverse=True)
                total_e = sum(energies)
                if total_e > 0:
                    split_leading_frac.append(energies[0] / total_e)
                    split_n_fragments.append(n)
                    if is_fastsim_pi0:
                        fs_split_leading_frac.append(energies[0] / total_e)
                        fs_split_n_fragments.append(n)

        n_pi0_0reco += len(mc_pi0_ids_with_photon_daughters - seen_pi0_reco_ids)
        fs_n_pi0_0reco += len(mc_pi0_ids_fastsim - fs_seen_pi0_reco_ids)

    print(f"\n  pi0 RECONSTRUCTION EFFICIENCY [{label}] (ungated)")
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

    print(f"\n  pi0 RECONSTRUCTION EFFICIENCY [{label}] (fast-sim region: {FASTSIM_GATE_LABEL})")
    print(f"  {'-'*45}")
    print(f"  MC pi0s (denominator)         : {fs_n_pi0_mc_total}")
    print(f"  0 reco photons (lost entirely): {fs_n_pi0_0reco:6d}  "
          f"({100*fs_n_pi0_0reco/max(fs_n_pi0_mc_total,1):.1f}%)")
    print(f"  1 reco photon  (merged/lost 1): {fs_n_pi0_1reco:6d}  "
          f"({100*fs_n_pi0_1reco/max(fs_n_pi0_mc_total,1):.1f}%)")
    print(f"  2 reco photons (clean -> mass) : {fs_n_pi0_exactly2:6d}  "
          f"({100*fs_n_pi0_exactly2/max(fs_n_pi0_mc_total,1):.1f}%)  <- USED FOR MASS")
    print(f"  3+ reco photons (split)       : {fs_n_pi0_3plus:6d}  "
          f"({100*fs_n_pi0_3plus/max(fs_n_pi0_mc_total,1):.1f}%)")
    print(f"  Clean reco efficiency         : "
          f"{100*fs_n_pi0_exactly2/max(fs_n_pi0_mc_total,1):.1f}%")

    print(f"\n  Tier 1 masses (no energy gate): {len(tier1_masses)}")
    print(f"  Tier 2 masses (E_mc>10 GeV)   : {len(tier2_masses)}")
    print(f"  Energy response entries (ungated / fast-sim): "
          f"{len(energy_response)} / {len(fs_energy_response)}")
    print(f"  Opening angle entries   (ungated / fast-sim): "
          f"{len(opening_angle_ratio)} / {len(fs_opening_angle_ratio)}")

    eff = {
        "n_mc_total"  : n_pi0_mc_total,
        "n_0reco"     : n_pi0_0reco,
        "n_1reco"     : n_pi0_1reco,
        "n_exactly2"  : n_pi0_exactly2,
        "n_3plus"     : n_pi0_3plus,
        "eff_pct"     : 100*n_pi0_exactly2/max(n_pi0_mc_total, 1),
    }
    fs_eff = {
        "n_mc_total"  : fs_n_pi0_mc_total,
        "n_0reco"     : fs_n_pi0_0reco,
        "n_1reco"     : fs_n_pi0_1reco,
        "n_exactly2"  : fs_n_pi0_exactly2,
        "n_3plus"     : fs_n_pi0_3plus,
        "eff_pct"     : 100*fs_n_pi0_exactly2/max(fs_n_pi0_mc_total, 1),
    }
    split_diag = {
        "leading_frac": np.array(split_leading_frac),
        "n_fragments" : np.array(split_n_fragments),
    }
    fs_split_diag = {
        "leading_frac": np.array(fs_split_leading_frac),
        "n_fragments" : np.array(fs_split_n_fragments),
    }
    response_diag = {
        "energy_response"    : np.array(energy_response),
        "opening_angle_ratio" : np.array(opening_angle_ratio),
    }
    fs_response_diag = {
        "energy_response"    : np.array(fs_energy_response),
        "opening_angle_ratio" : np.array(fs_opening_angle_ratio),
    }
    return (np.array(tier1_masses), np.array(tier2_masses),
            eff, split_diag, response_diag,
            fs_eff, fs_split_diag, fs_response_diag)


# ============================================================
# FITTING
# ============================================================
def fit_gaussian_pure(centers, counts, label):
    fit_mask = (centers > 0.110) & (centers < 0.170)
    x = centers[fit_mask]
    y = counts[fit_mask].astype(float)

    if len(x) < 5 or y.max() < 5:
        print(f"  Gaussian fit [{label}] SKIPPED: insufficient statistics")
        return None, None

    mu0    = PI0_MASS_MEV / 1000.0
    sigma0 = 0.015
    N0     = y.max()
    p0     = [N0, mu0, sigma0]
    lower  = [0,    0.100, 0.003]
    upper  = [1e9,  0.200, 0.040]

    try:
        popt, pcov = curve_fit(gauss_pure, x, y, p0=p0, bounds=(lower, upper), maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        mu, sigma = popt[1], popt[2]
        dmu, dsigma = perr[1], perr[2]
        bias_mev = 1000 * (mu - PI0_MASS_MEV/1000.0)
        print(f"  Gauss fit [{label:3s}]: mu={1000*mu:.2f}+/-{1000*dmu:.2f} MeV | "
              f"sigma={1000*sigma:.2f}+/-{1000*dsigma:.2f} MeV | bias={bias_mev:+.2f} MeV")
        return popt, perr
    except RuntimeError as e:
        print(f"  Gaussian fit [{label}] FAILED: {e}")
        return None, None


def fit_crystal_ball(centers, counts, label):
    fit_mask = (centers > 0.100) & (centers < 0.180)
    x = centers[fit_mask]
    y = counts[fit_mask].astype(float)

    if len(x) < 8 or y.max() < 5:
        print(f"  CB fit [{label}] SKIPPED: insufficient statistics")
        return None, None

    mu0    = PI0_MASS_MEV / 1000.0
    sigma0 = 0.015
    N0     = y.max()
    p0     = [N0, mu0, sigma0, 1.5, 3.0]
    lower  = [0,    0.100, 0.003, 0.3,  1.1]
    upper  = [1e9,  0.200, 0.040, 10.0, 50.0]

    try:
        popt, pcov = curve_fit(crystal_ball, x, y, p0=p0, bounds=(lower, upper), maxfev=30000)
        perr = np.sqrt(np.diag(pcov))
        mu, sigma, alpha, n = popt[1], popt[2], popt[3], popt[4]
        dmu, dsigma = perr[1], perr[2]
        bias_mev = 1000 * (mu - PI0_MASS_MEV/1000.0)
        print(f"  CB fit    [{label:3s}]: mu={1000*mu:.2f}+/-{1000*dmu:.2f} MeV | "
              f"sigma={1000*sigma:.2f}+/-{1000*dsigma:.2f} MeV | bias={bias_mev:+.2f} MeV | "
              f"alpha={alpha:.2f} n={n:.2f} | tail_side={CB_TAIL_SIDE}")
        return popt, perr
    except RuntimeError as e:
        print(f"  CB fit [{label}] FAILED: {e}")
        return None, None


# ============================================================
# PLOTTING: MASS (Gaussian + Crystal Ball, saved separately)
# ============================================================
def _draw_mass_panel(ax, g4_m, cc3_m, g4_counts, cc3_counts, fit_func, fit_fn, tier, subtitle, gate_suffix=""):
    ax.hist(g4_m,  bins=M_BINS, histtype="step", color=C_G4, linewidth=2,
            label=f"Geant4 (N={len(g4_m)})")
    ax.hist(cc3_m, bins=M_BINS, histtype="step", color=C_CC3, linewidth=2,
            linestyle="--", label=f"CC3 (N={len(cc3_m)})")
    ax.axvline(PI0_MASS_MEV/1000, color="gray", lw=1.0, ls=":",
               label=f"PDG: {PI0_MASS_MEV:.0f} MeV")

    fit_texts = []
    for counts, flabel, color in [(g4_counts, "G4", C_G4), (cc3_counts, "CC3", C_CC3)]:
        popt, perr = fit_fn(M_CENTERS, counts, flabel)
        if popt is not None:
            x_fit = np.linspace(0.095, 0.185, 800)
            ax.plot(x_fit, fit_func(x_fit, *popt), color=color, lw=1.8, alpha=0.8)
            mu_mev, sig_mev = 1000*popt[1], 1000*popt[2]
            bias_mev = mu_mev - PI0_MASS_MEV
            fit_texts.append(f"{flabel}: mu={mu_mev:.1f} MeV, sigma={sig_mev:.1f} MeV, bias={bias_mev:+.1f} MeV")

    if fit_texts:
        ax.text(0.97, 0.97, "\n".join(fit_texts), transform=ax.transAxes,
                fontsize=7.5, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    ax.set_xlabel("M(gamma gamma) [GeV]")
    ax.set_ylabel("pi0 candidates / 2 MeV")
    ax.set_title(f"Reco M(gamma gamma) -- {tier}{gate_suffix}\n{subtitle}", fontsize=9.5)
    ax.legend(framealpha=0.5, fontsize=9)


def plot_reco_mass(g4_t1, cc3_t1, g4_t2, cc3_t2, tag=""):
    """Two separate figures: pure-Gaussian fit, and Crystal-Ball fit."""
    tiers = [
        (g4_t1, cc3_t1, "Tier 1", "no energy gate"),
        (g4_t2, cc3_t2, "Tier 2", "E_mc>10 GeV both daughters"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (g4_m, cc3_m, tier, subtitle) in zip(axes, tiers):
        g4_counts,  _ = np.histogram(g4_m,  bins=M_BINS)
        cc3_counts, _ = np.histogram(cc3_m, bins=M_BINS)
        _draw_mass_panel(ax, g4_m, cc3_m, g4_counts, cc3_counts,
                          gauss_pure, fit_gaussian_pure, tier, subtitle)
    plt.tight_layout()
    out = f"{PLOT_DIR}/reco_mgg_tier1_tier2{tag}_gaussian.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: reco_mgg_tier1_tier2{tag}_gaussian.png")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (g4_m, cc3_m, tier, subtitle) in zip(axes, tiers):
        g4_counts,  _ = np.histogram(g4_m,  bins=M_BINS)
        cc3_counts, _ = np.histogram(cc3_m, bins=M_BINS)
        _draw_mass_panel(ax, g4_m, cc3_m, g4_counts, cc3_counts,
                          crystal_ball, fit_crystal_ball, tier, subtitle)
    plt.tight_layout()
    out = f"{PLOT_DIR}/reco_mgg_tier1_tier2{tag}_crystalball.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: reco_mgg_tier1_tier2{tag}_crystalball.png")


# ============================================================
# PLOTTING: TRUTH (zoomed only -- fixed window)
# ============================================================
def plot_truth_zoomed(g4_masses, cc3_masses):
    """Fixed zoom window per 2026-08-31 request: 134.5-135.5 MeV, 0.1 MeV
    bins (10 bins). Overrides the previous auto-derived mu+/-3sigma window.

    Prints the actual truth-level mean/std for both samples so you can check
    this fixed window isn't clipping the real peak (see module docstring).
    """
    lo, hi, bw = TRUTH_ZOOM_LO_MEV, TRUTH_ZOOM_HI_MEV, TRUTH_ZOOM_BIN_MEV
    n_bins = round((hi - lo) / bw)
    bins = np.linspace(lo, hi, n_bins + 1)

    g4_mean_mev, g4_std_mev = 1000*np.mean(g4_masses), 1000*np.std(g4_masses)
    cc3_mean_mev, cc3_std_mev = 1000*np.mean(cc3_masses), 1000*np.std(cc3_masses)
    print(f"\n  Truth zoom window (FIXED): [{lo}, {hi}] MeV, {n_bins} bins ({bw} MeV/bin)")
    print(f"  Truth G4  mean/std [MeV] : {g4_mean_mev:.4f} / {g4_std_mev:.4f}")
    print(f"  Truth CC3 mean/std [MeV] : {cc3_mean_mev:.4f} / {cc3_std_mev:.4f}")
    if not (lo <= g4_mean_mev <= hi) or not (lo <= cc3_mean_mev <= hi):
        print(f"  WARNING: truth mean(s) fall OUTSIDE the fixed zoom window "
              f"[{lo}, {hi}] MeV -- this plot may be clipping the real peak.")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(1000*g4_masses,  bins=bins, histtype="step",
            color=C_G4,  linewidth=2, label=f"Geant4 (N in window)")
    ax.hist(1000*cc3_masses, bins=bins, histtype="step",
            color=C_CC3, linewidth=2, linestyle="--", label=f"CC3 (N in window)")
    ax.axvline(PI0_MASS_MEV, color="black", lw=1.0, ls=":", alpha=0.7,
               label=f"PDG pi0 mass ({PI0_MASS_MEV} MeV)")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("M(gamma gamma) [MeV]")
    ax.set_ylabel(f"pi0 candidates / {bw:.1f} MeV")
    ax.set_title("Analysis A -- MC Truth M(gamma gamma), zoomed on peak\n"
                 f"Fixed window: {lo}-{hi} MeV, {bw} MeV bins",
                 fontsize=9.5)
    ax.legend(framealpha=0.5)
    plt.tight_layout()
    out = f"{PLOT_DIR}/analysisA_truth_mgg_zoomed.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: analysisA_truth_mgg_zoomed.png")


# ============================================================
# PLOTTING: CLUSTER SPLITTING (parametrized by gate)
# ============================================================
def plot_cluster_splitting(g4_split, cc3_split, g4_eff, cc3_eff, tag="", gate_suffix=""):
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
    ax.set_title(f"Cluster splitting: fragment multiplicity{gate_suffix}\n"
                 "(pi0s with 3+ reco photon daughters)", fontsize=9.5)
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
    ax.set_title(f"Cluster splitting: energy sharing{gate_suffix}\n"
                 "(1.0 = one dominant fragment, lower = more even split)", fontsize=9.5)
    if len(g4_split["leading_frac"]) > 0 or len(cc3_split["leading_frac"]) > 0:
        ax.legend(framealpha=0.5)

    plt.tight_layout()
    out = f"{PLOT_DIR}/cluster_splitting_diagnostic{tag}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: cluster_splitting_diagnostic{tag}.png")

    if len(g4_split["n_fragments"]) > 0:
        print(f"  Cluster-splitting summary [G4{gate_suffix}]:")
        print(f"    Mean fragments per split pi0 : {g4_split['n_fragments'].mean():.2f}")
        print(f"    Mean leading fragment frac    : {g4_split['leading_frac'].mean():.3f}")
    if len(cc3_split["n_fragments"]) > 0:
        print(f"  Cluster-splitting summary [CC3{gate_suffix}]:")
        print(f"    Mean fragments per split pi0 : {cc3_split['n_fragments'].mean():.2f}")
        print(f"    Mean leading fragment frac    : {cc3_split['leading_frac'].mean():.3f}")


# ============================================================
# PLOTTING: ENERGY RESPONSE (parametrized by gate)
# ============================================================
def plot_energy_response(g4_resp, cc3_resp, tag="", gate_suffix=""):
    g4_vals  = g4_resp["energy_response"]
    cc3_vals = cc3_resp["energy_response"]

    if len(g4_vals) == 0 or len(cc3_vals) == 0:
        print(f"\n  Energy response{gate_suffix}: SKIPPED (empty sample, "
              f"G4 N={len(g4_vals)}, CC3 N={len(cc3_vals)})")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(g4_vals,  bins=E_RESPONSE_BINS, histtype="step", color=C_G4,
            linewidth=2, density=True, label=f"Geant4 (N={len(g4_vals)}, mean={np.mean(g4_vals):.3f})")
    ax.hist(cc3_vals, bins=E_RESPONSE_BINS, histtype="step", color=C_CC3,
            linewidth=2, linestyle="--", density=True,
            label=f"CC3 (N={len(cc3_vals)}, mean={np.mean(cc3_vals):.3f})")
    ax.axvline(1.0, color="black", lw=1.0, ls=":", alpha=0.7, label="Perfect response (1.0)")
    ax.set_xlabel("E_reco / E_mc  (per matched photon, clean n==2 pi0s only)")
    ax.set_ylabel("Normalized density")
    ax.set_title(f"Photon energy response: G4 vs CC3{gate_suffix}\n"
                 "Signal: matched daughters of clean (n_reco==2) pi0 candidates",
                 fontsize=9.5)
    ax.legend(framealpha=0.5, fontsize=9)
    plt.tight_layout()
    out = f"{PLOT_DIR}/energy_response{tag}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: energy_response{tag}.png")
    print(f"  G4  mean E_reco/E_mc{gate_suffix}  : {np.mean(g4_vals):.4f}  (std {np.std(g4_vals):.4f})")
    print(f"  CC3 mean E_reco/E_mc{gate_suffix}  : {np.mean(cc3_vals):.4f}  (std {np.std(cc3_vals):.4f})")


# ============================================================
# PLOTTING: OPENING ANGLE BIAS + RATIO PANEL (parametrized by gate)
# ============================================================
def plot_opening_angle_bias(g4_resp, cc3_resp, tag="", gate_suffix=""):
    g4_vals  = g4_resp["opening_angle_ratio"]
    cc3_vals = cc3_resp["opening_angle_ratio"]

    if len(g4_vals) == 0 or len(cc3_vals) == 0:
        print(f"\n  Opening angle bias{gate_suffix}: SKIPPED (empty sample, "
              f"G4 N={len(g4_vals)}, CC3 N={len(cc3_vals)})")
        return

    g4_counts,  _ = np.histogram(g4_vals,  bins=ANGLE_RATIO_BINS)
    cc3_counts, _ = np.histogram(cc3_vals, bins=ANGLE_RATIO_BINS)
    bin_centers = 0.5*(ANGLE_RATIO_BINS[:-1]+ANGLE_RATIO_BINS[1:])

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(g4_counts > 0, cc3_counts / g4_counts, np.nan)
        rel_err_g4  = np.where(g4_counts  > 0, 1.0/np.sqrt(g4_counts),  np.nan)
        rel_err_cc3 = np.where(cc3_counts > 0, 1.0/np.sqrt(cc3_counts), np.nan)
        ratio_err = ratio * np.sqrt(rel_err_g4**2 + rel_err_cc3**2)

    fig = plt.figure(figsize=(9, 7))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax)

    ax.hist(g4_vals,  bins=ANGLE_RATIO_BINS, histtype="step", color=C_G4,
            linewidth=2, density=True, label=f"Geant4 (N={len(g4_vals)}, mean={np.mean(g4_vals):.3f})")
    ax.hist(cc3_vals, bins=ANGLE_RATIO_BINS, histtype="step", color=C_CC3,
            linewidth=2, linestyle="--", density=True,
            label=f"CC3 (N={len(cc3_vals)}, mean={np.mean(cc3_vals):.3f})")
    ax.axvline(1.0, color="black", lw=1.0, ls=":", alpha=0.7, label="Perfect angle (1.0)")
    ax.set_ylabel("Normalized density")
    ax.set_title(f"Opening angle bias: G4 vs CC3{gate_suffix}\n"
                 "Signal: clean (n_reco==2) pi0 candidates, reco vs MC 3-momentum angle",
                 fontsize=9.5)
    ax.legend(framealpha=0.5, fontsize=9)
    ax.tick_params(labelbottom=False)

    ax_ratio.errorbar(bin_centers, ratio, yerr=ratio_err, fmt="o", ms=3,
                       color="black", ecolor="gray", elinewidth=1, capsize=1.5)
    ax_ratio.axhline(1.0, color="gray", lw=1.0, ls=":")
    ax_ratio.set_xlabel("dAlpha_reco / dAlpha_mc  (per clean n==2 pi0 pair)")
    ax_ratio.set_ylabel("CC3 / G4")
    ax_ratio.set_ylim(0.0, 2.0)

    out = f"{PLOT_DIR}/opening_angle_bias{tag}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: opening_angle_bias{tag}.png")
    print(f"  G4  mean dAlpha_reco/dAlpha_mc{gate_suffix}  : {np.mean(g4_vals):.4f}  (std {np.std(g4_vals):.4f})")
    print(f"  CC3 mean dAlpha_reco/dAlpha_mc{gate_suffix}  : {np.mean(cc3_vals):.4f}  (std {np.std(cc3_vals):.4f})")


# ============================================================
# EFFICIENCY TABLE (parametrized by gate)
# ============================================================
def print_efficiency_comparison(g4_eff, cc3_eff, gate_suffix=""):
    print(f"\n{'='*60}")
    print(f"  pi0 RECONSTRUCTION EFFICIENCY COMPARISON{gate_suffix}")
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


# ============================================================
# MAIN
# ============================================================
print("\n" + "="*60)
print("  pi0_invariant_mass_v5.py")
print(f"  Run size: {RUN_SIZE}  ({RUN_LABEL})")
print("  Analysis A (truth, zoomed only) + Tier 1/2 reco M_gg (Gaussian + Crystal Ball)")
print("  + splitting diagnostic + energy response + opening angle bias")
print(f"  + fast-sim region set ({FASTSIM_GATE_LABEL})")
print("="*60)

print("\n-- Analysis A: MC Truth --")
g4_truth  = run_truth(G4_FILE,  "Geant4")
cc3_truth = run_truth(CC3_FILE, "CC3")

print("\n-- Reco: G4 --")
(g4_t1, g4_t2, g4_eff, g4_split, g4_resp,
 g4_fs_eff, g4_fs_split, g4_fs_resp) = run_reco(G4_FILE, "G4")

print("\n-- Reco: CC3 --")
(cc3_t1, cc3_t2, cc3_eff, cc3_split, cc3_resp,
 cc3_fs_eff, cc3_fs_split, cc3_fs_resp) = run_reco(CC3_FILE, "CC3")

print_efficiency_comparison(g4_eff, cc3_eff, gate_suffix=" (ungated)")
print_efficiency_comparison(g4_fs_eff, cc3_fs_eff, gate_suffix=f" ({FASTSIM_GATE_LABEL})")

print("\n-- Generating plots: truth (zoomed only) --")
plot_truth_zoomed(g4_truth, cc3_truth)

print("\n-- Generating plots: reco mass (ungated) --")
plot_reco_mass(g4_t1, cc3_t1, g4_t2, cc3_t2, tag="")

print("\n-- Generating plots: diagnostics (ungated) --")
plot_cluster_splitting(g4_split, cc3_split, g4_eff, cc3_eff, tag="", gate_suffix="")
plot_energy_response(g4_resp, cc3_resp, tag="", gate_suffix="")
plot_opening_angle_bias(g4_resp, cc3_resp, tag="", gate_suffix="")

print(f"\n-- Generating plots: diagnostics (fast-sim region: {FASTSIM_GATE_LABEL}) --")
fs_tag = "_fastsim_Emc10GeV"
fs_gate_suffix = f"\n[{FASTSIM_GATE_LABEL}]"
plot_cluster_splitting(g4_fs_split, cc3_fs_split, g4_fs_eff, cc3_fs_eff,
                        tag=fs_tag, gate_suffix=fs_gate_suffix)
plot_energy_response(g4_fs_resp, cc3_fs_resp, tag=fs_tag, gate_suffix=fs_gate_suffix)
plot_opening_angle_bias(g4_fs_resp, cc3_fs_resp, tag=fs_tag, gate_suffix=fs_gate_suffix)

print(f"\n  All plots saved to: {PLOT_DIR}")
print(f"  Done.\n")
