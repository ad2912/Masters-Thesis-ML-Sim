"""
single_photon_observables_v3.py
=================================
Revised single-photon benchmarking script.

CHANGE LOG vs v2 (this session's revisions):
  - Change 1: decoded-weight quality gate. Raw RecoMCTruthLink.getWeight()
    is now decoded (cluster_weight, track_weight in [0,1]) before ranking,
    and WEIGHT_CUT=0.5 on decoded cluster_weight rejects low-confidence
    winners outright (no fallback to second-best link).
  - Change 2: splitting multiplicity (Plot 3) rebuilt on a weighted MC-side
    view. Instead of inverting the reco->mc partition (which is blind to
    soft/partial splits — confirmed by diagnostic: ~5.4-5.7% of linked
    Signal B photons have a comparable second contributor), we now collect
    ALL links touching each MC photon and count how many survive WEIGHT_CUT.
  - Plot 1 (response/resolution): unchanged definition, Change 1 applied to
    shared partition only.
  - Plot 2a (cluster width, reco): unchanged formula, E-binning now
    toggleable (coarse original vs finer uniform bins).
  - Plot 2c (radial profile, reco, NEW): ring-binned energy density vs
    distance from shower axis, adapted from CaloClouds3 methodology,
    simplified for collision-event + full ILD detector (no particle gun,
    no layer-by-layer breakdown, no JS-divergence). Axis = energy-weighted
    centroid direction (same as 2a — no privileged truth at reco level).
  - Plot 3 (splitting multiplicity): two-panel figure. Panel A = n_split=1
    (clean) grouped bars G4 vs CC3. Panel B = n_split=2 (solid) and n_split
    3+ (hatched) grouped bars, own y-scale. n_split=0 excluded from this
    plot (that is a "lost" story, not a "split" story).
  - Plot 4 (fake rate, REPLACES old N_reco/N_mc efficiency plot): global
    fake rate = N(reco PDG==22 PFOs with no genuine-photon match above
    WEIGHT_CUT) / N(all reco PDG==22 PFOs). No energy cut, no Signal B
    window, no isHandledByFastSim() in the decision.

POST-REVIEW FIXES, ROUND 1 (2026-09-08):
  - FIX A: decode_weight() track_weight divisor corrected /1000.0 -> /10000.0.
    Was allowing track_weight > 1.0. cluster_weight (the only value WEIGHT_CUT
    uses) was already correct -- no previously-plotted number changes.
    track_weight remains unused everywhere here: photons are neutral and
    carry no track, so this is expected, not a gap.
  - FIX B: Plot 2c relabeled "longitudinally integrated" + explicit caveat
    (title + docstring) that a G4/CC3 discrepancy here can reflect a
    transverse-width difference, a longitudinal shower-development
    difference, or both entangled -- cannot be separated from this plot alone.
  - FIX C: Plot 3 suptitle now leads with a plain-language definition of
    n_split (a truth photon's shower reconstructed as >1 PFO = PandoraPFA
    splitting) before the "weighted MC-side view" implementation label.

POST-REVIEW FIXES, ROUND 2 (2026-09-08, after seeing real plots):
  - FIX D: Plot 3 (splitting) now has Poisson-style sqrt(N) error bars on
    every bar in both panels (Panel A: n_split=1 counts; Panel B: n_split=2
    and n_split=3+ counts). Without these, low counts in Panel B (some bins
    had only ~6-40 entries) made G4-vs-CC3 differences unreadable as
    signal-vs-noise.
  - FIX E: RADIAL_MAX_MM reduced from 200mm to 50mm (starting placeholder).
    200mm was ~22 Moliere radii for the ILD SiW-ECAL tungsten absorber
    (Moliere radius ~9mm, looked up 2026-09-08) -- physically, ~95% shower
    containment is ~1 Moliere radius (~9mm) and 99%+ containment is ~2-3
    Moliere radii (~20-27mm), so almost the entire old 200mm range was
    binning noise/tails, not real shower structure. This is why the old
    ratio panels swung wildly (e.g. 0.3-1.9) at large radius -- near-empty
    bins amplifying Poisson noise, not real physics.
    50mm is a safety-padded starting guess (covers >99% containment with
    margin for messy real clusters in a busy tau-decay event, not a clean
    single-photon gun). A ONE-LINE DIAGNOSTIC is added below (see
    print_radial_containment_check()) that computes the actual empirical
    radius containing 99.9% of pooled hit energy from your real data --
    use that printed number to retune RADIAL_MAX_MM, this 50mm is not final.
  - FIX F: Per-ring statistics floor added to the radial profile. Any ring
    with fewer than RADIAL_MIN_HITS (default 20) pooled hits in a given
    E_mc bin is masked to NaN in both the density and ratio panels, instead
    of plotting a ratio computed from a handful of near-random hits. This
    directly targets the low-N noise described in FIX E.
  - FIX G: dataset selection changed from a single USE_1K boolean to a
    three-way DATASET = "1k" | "10k" | "100k" string, adding the 100k reco
    files as a third option (paths below). Cleaner than stacking a second
    boolean toggle alongside USE_1K.

NOT YET IMPLEMENTED (pending separate follow-up):
  - Plot 2b (cluster width, sim level) and Plot 2d (radial profile, sim
    level) are STUBBED. These require attributing SimCalorimeterHit /
    CaloHitContribution energy back to a specific MC truth photon, and
    that attribution path (CaloHitContribution.getParticle()) has not yet
    been verified against real data — see sim_hit_attribution_check.py.
    Do not fill in 2b/2d until that check has run successfully.

Signal: Signal B — genStatus==1, PDG==22, E_mc>=10 GeV, parent PDG!=11
(NOT the same as Signal A, which is pi0-daughter photons used in the
di-photon chapter script — do not conflate the two).

Run:
    source ~/source.sh
    python3 single_photon_observables_v3.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import os
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────
# FIX G: three-way dataset selection, replaces the old USE_1K boolean.
DATASET = "100k"   # one of: "1k", "10k", "100k"

_DATASET_PATHS = {
    "1k": dict(
        G4_FILE  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-1000events-test_REC.edm4hep.root",
        CC3_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-1000events-test_REC.edm4hep.root",
        MAX_EVENTS = 1000,
        RUN_LABEL  = "1k events [dev]",
    ),
    "10k": dict(
        G4_FILE  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-10k-events-test_REC.edm4hep.root",
        CC3_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root",
        MAX_EVENTS = 10000,
        RUN_LABEL  = "10k events",
    ),
    "100k": dict(
        G4_FILE  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-geant4-100kevents-reco-test_REC.edm4hep.root",
        CC3_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-caloclouds-100kevents-reco-test_REC.edm4hep.root",
        MAX_EVENTS = 100000,
        RUN_LABEL  = "100k events",
    ),
}

if DATASET not in _DATASET_PATHS:
    raise ValueError(f"DATASET must be one of {list(_DATASET_PATHS)}, got {DATASET!r}")

_paths     = _DATASET_PATHS[DATASET]
G4_FILE    = _paths["G4_FILE"]
CC3_FILE   = _paths["CC3_FILE"]
MAX_EVENTS = _paths["MAX_EVENTS"]
RUN_LABEL  = _paths["RUN_LABEL"]

# Toggle: coarse original E-binning vs finer uniform binning.
# Set to True once running on the 100k-event file where stats support it.
USE_FINE_E_BINS = False

PLOT_DIR = os.path.expanduser(
    f"~/thesis-ml-sim/plots/single_photon_observables_v3_{DATASET}"
    f"_{'fine' if USE_FINE_E_BINS else 'coarse'}bins"
)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── PHYSICS CONSTANTS ────────────────────────────────────────────────────
PHOTON_PDG   = 22
ELECTRON_PDG = 11
E_MC_MIN     = 10.0
WEIGHT_CUT   = 0.5   # on DECODED cluster_weight, in [0,1] — Change 1

if USE_FINE_E_BINS:
    E_BINS = np.arange(0, 130, 10.0)   # uniform 10 GeV bins
else:
    E_BINS = np.array([0, 10, 20, 30, 40, 60, 80, 100, 125], dtype=float)

E_CENTERS = 0.5 * (E_BINS[:-1] + E_BINS[1:])
N_E_BINS  = len(E_CENTERS)

# Radial profile (Plot 2c) config.
# FIX E: ring width (5mm) matches the real ILD SiW-ECAL cell pitch (5x5mm2
# silicon pads, confirmed against published ILD ECAL design docs 2026-09-08)
# -- this was already a physically sound choice, kept as-is.
# Max radius (50mm) is a STARTING placeholder, reduced from the old 200mm
# (~22 Moliere radii -- almost entirely empty/noise). Retune using the
# printed empirical containment radius from print_radial_containment_check()
# below, once you have real hit data from a run.
RADIAL_BIN_WIDTH_MM = 5.0
RADIAL_MAX_MM       = 50.0     # STARTING VALUE -- retune from empirical check, see FIX E
RADIAL_MIN_HITS     = 20       # FIX F: per-ring stats floor (pooled hits per E_mc bin)
RADIAL_BINS   = np.arange(0, RADIAL_MAX_MM + RADIAL_BIN_WIDTH_MM, RADIAL_BIN_WIDTH_MM)
RADIAL_CENTERS = 0.5 * (RADIAL_BINS[:-1] + RADIAL_BINS[1:])
RADIAL_RING_AREA = np.pi * (RADIAL_BINS[1:]**2 - RADIAL_BINS[:-1]**2)  # mm^2, exact ring area

# ── COLORS / STYLE ───────────────────────────────────────────────────────
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

SIGNAL_SUBTITLE = "Signal B: genStatus==1, E_mc≥10 GeV, parent PDG≠11 (not ISR)"

G4_KW  = dict(color=C_G4,  marker="o", linewidth=2.0, linestyle="-",  markersize=6, label="Geant4")
CC3_KW = dict(color=C_CC3, marker="s", linewidth=2.0, linestyle="--", markersize=6, label="CC3")

# ── HELPERS ───────────────────────────────────────────────────────────────

def get_parent_pdg(mc_part):
    parents = mc_part.getParents()
    return abs(parents[0].getPDG()) if len(parents) > 0 else 0


def signal_b_cuts(mc_part):
    if mc_part.getGeneratorStatus() != 1:
        return False
    if mc_part.getPDG() != PHOTON_PDG:
        return False
    if mc_part.getEnergy() < E_MC_MIN:
        return False
    if get_parent_pdg(mc_part) == ELECTRON_PDG:
        return False
    return True


def decode_weight(raw_weight):
    """
    Decode packed RecoMCTruthLink weight.
    raw = 10000*cluster_weight_permille + track_weight_permille
    Returns (cluster_weight, track_weight), both in [0,1].

    FIX A (2026-09-08): track_weight divisor corrected /1000.0 -> /10000.0
    (was allowing values > 1.0). cluster_weight's divisor was already
    correct and is the only value WEIGHT_CUT compares against, so this fix
    changes no previously-plotted number. track_weight stays unused
    throughout this script: photons carry no track, so it's physically
    inert here, not a gap.

    Reused verbatim (same mechanism as pi0_invariant_mass_v5.py) — do not
    reimplement differently here. cluster_weight ceiling confirmed against
    real data 2026-08-03: hard ceiling at raw ~= 10,001,000-10,001,004.
    """
    raw_int = int(round(raw_weight))
    cluster_weight = (raw_int // 10000) / 1000.0
    track_weight   = (raw_int % 10000) / 10000.0
    return cluster_weight, track_weight


def compute_cluster_width_and_profile(cluster, axis=None):
    """
    Energy-weighted transverse RMS (sigma_perp) AND per-hit (d_perp, E) pairs
    for the radial profile histogram, computed in one pass over the hits.

    axis: optional pre-computed unit vector (ax,ay,az) for the shower axis.
          If None, uses the energy-weighted centroid direction (reco-level
          convention — no privileged truth direction available).

    Returns (sigma_perp, d_perp_array, energy_array) — sigma_perp is NaN if
    cluster has <2 hits or zero total energy; d_perp_array/energy_array are
    empty arrays in that case (safe to concatenate).

    Note: d_perp is the 3D perpendicular distance from the shower AXIS (a
    line through the origin along `axis`), not a distance within one
    transverse plane. Hits at every depth along the shower contribute to
    the same d_perp bin if their perpendicular distance matches -- this
    matters for how Plot 2c's "density" should be read (see FIX B there).
    """
    hits = cluster.getHits()
    if len(hits) < 2:
        return float("nan"), np.array([]), np.array([])

    xs = np.array([h.getPosition().x for h in hits], dtype=float)
    ys = np.array([h.getPosition().y for h in hits], dtype=float)
    zs = np.array([h.getPosition().z for h in hits], dtype=float)
    es = np.array([h.getEnergy()      for h in hits], dtype=float)

    e_tot = es.sum()
    if e_tot == 0:
        return float("nan"), np.array([]), np.array([])

    if axis is None:
        cx = np.dot(es, xs) / e_tot
        cy = np.dot(es, ys) / e_tot
        cz = np.dot(es, zs) / e_tot
        c_mag = math.sqrt(cx**2 + cy**2 + cz**2)
        if c_mag == 0:
            return float("nan"), np.array([]), np.array([])
        ax_, ay_, az_ = cx / c_mag, cy / c_mag, cz / c_mag
    else:
        ax_, ay_, az_ = axis

    r_dot_a = xs * ax_ + ys * ay_ + zs * az_
    r_sq    = xs**2 + ys**2 + zs**2
    d_perp_sq = np.maximum(r_sq - r_dot_a**2, 0.0)
    d_perp    = np.sqrt(d_perp_sq)

    sigma_perp = math.sqrt(np.dot(es, d_perp_sq) / e_tot)

    return sigma_perp, d_perp, es


def bin_index(e_mc, bins, centers):
    idx = np.searchsorted(bins, e_mc, side="right") - 1
    if idx < 0 or idx >= len(centers):
        return -1
    return int(idx)


def print_radial_containment_check(all_d_perp_pooled, all_energy_pooled, label):
    """
    FIX E, one-line diagnostic: given ALL pooled (d_perp, energy) pairs
    across every event/E_mc bin for one sample (G4 or CC3), compute and
    print the empirical radius containing 99.9% of total hit energy.
    Use this printed number -- not the 50mm placeholder -- to set
    RADIAL_MAX_MM once you have real data. Run once per sample; if G4 and
    CC3 disagree noticeably, take the larger of the two so both fit on
    the same axis range.
    """
    if len(all_d_perp_pooled) == 0:
        print(f"  [{label}] no hits pooled -- cannot compute containment radius.")
        return None
    order = np.argsort(all_d_perp_pooled)
    d_sorted = all_d_perp_pooled[order]
    e_sorted = all_energy_pooled[order]
    cum_e = np.cumsum(e_sorted)
    cum_e /= cum_e[-1]
    idx_999 = np.searchsorted(cum_e, 0.999)
    idx_999 = min(idx_999, len(d_sorted) - 1)
    r_999 = d_sorted[idx_999]
    print(f"  [{label}] empirical 99.9% energy-containment radius: {r_999:.1f} mm "
          f"(current RADIAL_MAX_MM={RADIAL_MAX_MM:.0f} mm)")
    return r_999


# ── MAIN LOOP ─────────────────────────────────────────────────────────────

def run(filepath, label):
    """
    Single pass over events. For each Signal B MC truth photon:
      - Always fills: fake-rate denominators (via all PFOs), n_split list
        (weighted MC-side view, Change 2)
      - If reconstructed as photon PFO above WEIGHT_CUT: fills response,
        resolution, cluster width, radial profile hits.

    Change 1: reco_to_mc built from DECODED cluster_weight, with
              WEIGHT_CUT rejecting low-confidence winners outright.
    Change 2: n_split computed from ALL links touching each MC photon
              (not just the reco-side partition), counting how many
              survive WEIGHT_CUT.
    """
    from podio import root_io

    print(f"\n{'='*60}")
    print(f"  Processing: {label}  ({RUN_LABEL})")
    print(f"{'='*60}")

    reader = root_io.Reader(filepath)

    response_ratios = defaultdict(list)
    widths          = defaultdict(list)
    radial_d_perp   = defaultdict(list)   # per E_mc bin: list of d_perp arrays
    radial_energy   = defaultdict(list)   # per E_mc bin: list of energy arrays
    n_split_counts  = defaultdict(list)   # Change 2: per E_mc bin, n_split per photon

    # Fake rate (Plot 4) — global, NOT scoped to Signal B, NOT scoped to energy bin
    n_all_photon_pfos   = 0
    n_fake_photon_pfos  = 0

    n_events_read = 0

    for event in reader.get("events"):
        if n_events_read >= MAX_EVENTS:
            break
        n_events_read += 1

        # ── reco -> mc, best DECODED weight wins, WEIGHT_CUT enforced ──────
        # Change 1: decode before ranking; reject below WEIGHT_CUT outright.
        reco_to_mc = {}
        for link in event.get("RecoMCTruthLink"):
            reco  = link.getFrom()
            mc    = link.getTo()
            raw_w = link.getWeight()
            cw, _ = decode_weight(raw_w)
            rid   = reco.getObjectID().index
            if cw < WEIGHT_CUT:
                continue
            if rid not in reco_to_mc or cw > reco_to_mc[rid][1]:
                reco_to_mc[rid] = (mc, cw)

        # ── mc -> ALL links touching it (decoded), for Change 2 n_split ────
        mc_all_link_weights = defaultdict(list)
        for link in event.get("RecoMCTruthLink"):
            mc    = link.getTo()
            raw_w = link.getWeight()
            cw, _ = decode_weight(raw_w)
            mc_all_link_weights[mc.getObjectID().index].append(cw)

        # ── reco -> mc inverted, gated by PDG==22, for response/width ──────
        mc_to_owned = defaultdict(list)
        for pfo in event.get("PandoraPFOs"):
            if pfo.getPDG() != PHOTON_PDG:
                continue
            rid = pfo.getObjectID().index
            if rid not in reco_to_mc:
                continue
            mc, _ = reco_to_mc[rid]
            mc_to_owned[mc.getObjectID().index].append(pfo)

        # ── Fake rate (Plot 4): loop ALL reco PDG==22 PFOs, unscoped ───────
        for pfo in event.get("PandoraPFOs"):
            if pfo.getPDG() != PHOTON_PDG:
                continue
            n_all_photon_pfos += 1
            rid = pfo.getObjectID().index
            is_fake = True
            if rid in reco_to_mc:
                mc, _ = reco_to_mc[rid]
                if mc.getGeneratorStatus() == 1 and mc.getPDG() == PHOTON_PDG:
                    is_fake = False
            if is_fake:
                n_fake_photon_pfos += 1

        # ── Loop Signal B truth photons ─────────────────────────────────────
        for mc in event.get("MCParticles"):
            if not signal_b_cuts(mc):
                continue

            e_mc  = mc.getEnergy()
            mc_id = mc.getObjectID().index
            b     = bin_index(e_mc, E_BINS, E_CENTERS)
            if b < 0:
                continue

            all_weights = mc_all_link_weights.get(mc_id, [])
            n_split = sum(1 for w in all_weights if w >= WEIGHT_CUT)
            if n_split >= 1:
                n_split_counts[b].append(n_split)

            owned_pfos = mc_to_owned.get(mc_id, [])
            n_owned    = len(owned_pfos)
            if n_owned == 0:
                continue

            best_pfo = max(owned_pfos, key=lambda p: p.getEnergy())
            e_reco   = best_pfo.getEnergy()

            response_ratios[b].append(e_reco / e_mc)

            clusters = best_pfo.getClusters()
            if len(clusters) > 0:
                sigma_perp, d_perp, hit_e = compute_cluster_width_and_profile(clusters[0])
                if not math.isnan(sigma_perp):
                    widths[b].append(sigma_perp)
                if len(d_perp) > 0:
                    radial_d_perp[b].append(d_perp)
                    radial_energy[b].append(hit_e)

    print(f"  Events processed : {n_events_read}")
    print(f"  All reco photon PFOs      : {n_all_photon_pfos}")
    print(f"  Fake photon PFOs          : {n_fake_photon_pfos}")
    if n_all_photon_pfos > 0:
        print(f"  Global fake rate          : {n_fake_photon_pfos/n_all_photon_pfos:.4f}")

    # FIX E: one-line empirical containment-radius diagnostic. Pool every
    # d_perp/energy pair across all E_mc bins for this sample and print the
    # radius containing 99.9% of total hit energy -- use this to retune
    # RADIAL_MAX_MM (currently a 50mm starting placeholder).
    all_d_perp_flat  = ([d for b in range(N_E_BINS) for arr in radial_d_perp[b] for d in arr])
    all_energy_flat  = ([e for b in range(N_E_BINS) for arr in radial_energy[b] for e in arr])
    print_radial_containment_check(np.array(all_d_perp_flat), np.array(all_energy_flat), label)

    # ── Per-bin statistics ───────────────────────────────────────────────
    resp_mean  = np.full(N_E_BINS, np.nan)
    resp_sigma = np.full(N_E_BINS, np.nan)
    width_mean = np.full(N_E_BINS, np.nan)
    width_err  = np.full(N_E_BINS, np.nan)

    for b in range(N_E_BINS):
        r = response_ratios[b]
        if len(r) >= 5:
            resp_mean[b]  = np.mean(r)
            resp_sigma[b] = np.std(r)
        w = widths[b]
        if len(w) >= 5:
            width_mean[b] = np.mean(w)
            width_err[b]  = np.std(w) / math.sqrt(len(w))

    # Radial profile: pool all hits per E_mc bin, histogram d_perp weighted
    # by energy. FIX F: rings with fewer than RADIAL_MIN_HITS pooled hits
    # are masked to NaN -- prevents ratios computed from near-empty rings
    # (the source of the wild swings seen in the old 200mm-range plot).
    radial_profile = np.full((N_E_BINS, len(RADIAL_CENTERS)), np.nan)
    for b in range(N_E_BINS):
        if len(radial_d_perp[b]) == 0:
            continue
        d_all = np.concatenate(radial_d_perp[b])
        e_all = np.concatenate(radial_energy[b])
        e_sum_per_ring, _ = np.histogram(d_all, bins=RADIAL_BINS, weights=e_all)
        hit_count_per_ring, _ = np.histogram(d_all, bins=RADIAL_BINS)
        n_showers = len(radial_d_perp[b])
        ring_density = (e_sum_per_ring / n_showers) / RADIAL_RING_AREA
        ring_density[hit_count_per_ring < RADIAL_MIN_HITS] = np.nan   # FIX F
        radial_profile[b] = ring_density

    # n_split categorical counts per bin: 1, 2, 3+
    n1 = np.zeros(N_E_BINS, dtype=int)
    n2 = np.zeros(N_E_BINS, dtype=int)
    n3plus = np.zeros(N_E_BINS, dtype=int)
    for b in range(N_E_BINS):
        for ns in n_split_counts[b]:
            if ns == 1:
                n1[b] += 1
            elif ns == 2:
                n2[b] += 1
            elif ns >= 3:
                n3plus[b] += 1

    return {
        "resp_mean" : resp_mean,
        "resp_sigma": resp_sigma,
        "width_mean": width_mean,
        "width_err" : width_err,
        "radial_profile": radial_profile,
        "n1": n1, "n2": n2, "n3plus": n3plus,
        "n_all_photon_pfos": n_all_photon_pfos,
        "n_fake_photon_pfos": n_fake_photon_pfos,
        "global_fake_rate": (n_fake_photon_pfos / n_all_photon_pfos
                              if n_all_photon_pfos > 0 else np.nan),
    }


# ── PLOTTING HELPERS ──────────────────────────────────────────────────────

def plot_valid(ax, x, y, **kwargs):
    mask = ~np.isnan(y)
    ax.plot(x[mask], y[mask], **kwargs)


def plot_valid_err(ax, x, y, yerr, **kwargs):
    mask = ~np.isnan(y)
    kw = {k: v for k, v in kwargs.items() if k != "linewidth"}
    ax.errorbar(x[mask], y[mask], yerr=yerr[mask],
                lw=kwargs.get("linewidth", 2.0), capsize=4, **kw)


# ── Figure 1: Energy response and resolution (unchanged definition) ───────
def make_fig_response(g4, cc3):
    fig, (ax_resp, ax_res) = plt.subplots(
        2, 1, figsize=(9, 8),
        gridspec_kw={"height_ratios": [3, 2]},
        sharex=True
    )
    fig.subplots_adjust(hspace=0.06)

    plot_valid(ax_resp, E_CENTERS, g4["resp_mean"],  **G4_KW)
    plot_valid(ax_resp, E_CENTERS, cc3["resp_mean"], **CC3_KW)
    ax_resp.axhline(1.0, color="black", lw=1.0, ls=":", alpha=0.6, label="Perfect response")
    ax_resp.set_ylabel("Mean  $E_\\mathrm{reco}\\,/\\,E_\\mathrm{mc}$")
    ax_resp.set_ylim(0.5, 1.5)
    ax_resp.set_title(
        f"Energy Scale and Resolution — Reco Photons, G4 vs CC3\n{SIGNAL_SUBTITLE}",
        fontsize=10
    )
    ax_resp.legend(framealpha=0.5)
    ax_resp.text(0.01, 1.01,
                 f"ILD ILDConfig reco — {RUN_LABEL} — decoded-weight gate (cut={WEIGHT_CUT})",
                 transform=ax_resp.transAxes, fontsize=8, color="gray")

    plot_valid(ax_res, E_CENTERS, g4["resp_sigma"],  **G4_KW)
    plot_valid(ax_res, E_CENTERS, cc3["resp_sigma"], **CC3_KW)
    ax_res.set_xlabel("MC Truth Photon Energy  $E_\\mathrm{mc}$  [GeV]")
    ax_res.set_ylabel("$\\sigma(E_\\mathrm{reco}\\,/\\,E_\\mathrm{mc})$")
    ax_res.set_title("Energy Resolution per $E_\\mathrm{mc}$ bin", fontsize=10)
    ax_res.set_ylim(bottom=0)
    ax_res.set_xlim(E_BINS[0], E_BINS[-1])

    plt.tight_layout()
    out = f"{PLOT_DIR}/fig1_energy_response_resolution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  \u2713  Saved: fig1_energy_response_resolution.png")


# ── Figure 2a: Cluster width, reco level ──────────────────────────────────
def make_fig_cluster_width(g4, cc3):
    fig, ax = plt.subplots(figsize=(8, 5))

    plot_valid_err(ax, E_CENTERS, g4["width_mean"],  g4["width_err"],  **G4_KW)
    plot_valid_err(ax, E_CENTERS, cc3["width_mean"], cc3["width_err"], **CC3_KW)

    ax.set_xlabel("MC Truth Photon Energy  $E_\\mathrm{mc}$  [GeV]")
    ax.set_ylabel("Mean cluster transverse width  $\\langle\\sigma_\\perp\\rangle$  [mm]")
    ax.set_title(
        "ECAL Cluster Transverse Width vs $E_\\mathrm{mc}$ — G4 vs CC3 (reco level)\n"
        f"{SIGNAL_SUBTITLE}\n"
        "Energy-weighted RMS of hit transverse distances from shower axis",
        fontsize=10
    )
    ax.legend(framealpha=0.5)
    ax.text(0.01, 1.01,
            f"ILD ILDConfig reco — {RUN_LABEL} — decoded-weight gate (cut={WEIGHT_CUT}) — "
            f"{'fine' if USE_FINE_E_BINS else 'coarse'} E-bins",
            transform=ax.transAxes, fontsize=8, color="gray")

    plt.tight_layout()
    out = f"{PLOT_DIR}/fig2a_cluster_width_reco.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  \u2713  Saved: fig2a_cluster_width_reco.png")


# ── Figure 2c: Radial profile, reco level (NEW) ────────────────────────────
def make_fig_radial_profile(g4, cc3):
    """
    One panel per E_mc bin. Each panel: energy density per ring (G4 vs CC3
    overlay) with a ratio panel underneath.
    Method: adapted from CaloClouds3 (ring-binned energy density), simplified
    for collision-event + full ILD geometry (no particle gun, no layer
    breakdown, no JS-divergence).

    FIX B: this profile pools calorimeter hits across the ENTIRE shower
    depth at each transverse distance from the shower axis, then normalizes
    by a flat annulus area -- a longitudinally-integrated quantity, NOT a
    shower-max-plane density.

    CAVEAT: showers narrow near the start and widen near shower max, so
    pooling over depth blends transverse profiles from many depths into one
    composite curve. A G4-vs-CC3 discrepancy here can reflect a transverse
    width difference, a longitudinal shower-development difference, or both
    entangled -- this plot alone cannot separate which.

    FIX E/F: RADIAL_MAX_MM reduced to a physically-motivated range (was
    ~22 Moliere radii, now much tighter -- see FIX E note in CONFIG) and
    rings with fewer than RADIAL_MIN_HITS pooled hits are masked to NaN,
    so the ratio panel no longer shows noise-dominated swings from
    near-empty outer rings.
    """
    n_bins_to_plot = N_E_BINS
    ncols = min(4, n_bins_to_plot)
    nrows = int(np.ceil(n_bins_to_plot / ncols))

    fig = plt.figure(figsize=(4.2 * ncols, 5.2 * nrows))
    for b in range(n_bins_to_plot):
        g4_prof  = g4["radial_profile"][b]
        cc3_prof = cc3["radial_profile"][b]
        if np.all(np.isnan(g4_prof)) and np.all(np.isnan(cc3_prof)):
            continue

        ax_main = fig.add_subplot(nrows * 2, ncols, 2 * (b // ncols) * ncols + (b % ncols) + 1)
        ax_rat  = fig.add_subplot(nrows * 2, ncols,
                                   (2 * (b // ncols) + 1) * ncols + (b % ncols) + 1,
                                   sharex=ax_main)

        ax_main.step(RADIAL_CENTERS, g4_prof,  where="mid", color=C_G4,  label="G4")
        ax_main.step(RADIAL_CENTERS, cc3_prof, where="mid", color=C_CC3, label="CC3", linestyle="--")
        ax_main.set_title(f"[{E_BINS[b]:.0f},{E_BINS[b+1]:.0f}) GeV", fontsize=9)
        ax_main.set_ylabel("E density,\nlong.-integrated\n[GeV/mm$^2$]", fontsize=7)
        ax_main.tick_params(labelbottom=False)
        if b == 0:
            ax_main.legend(fontsize=7)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = cc3_prof / g4_prof
        ax_rat.axhline(1.0, color="black", lw=0.8, ls=":", alpha=0.6)
        ax_rat.step(RADIAL_CENTERS, ratio, where="mid", color="black")
        ax_rat.set_ylim(0, 2)
        ax_rat.set_xlabel("Distance from shower axis [mm]", fontsize=8)
        ax_rat.set_ylabel("CC3/G4", fontsize=7)

    fig.suptitle(
        "Radial Energy Density Profile (longitudinally integrated) — G4 vs CC3 (reco level)\n"
        f"{SIGNAL_SUBTITLE} — pools hits over full shower depth per transverse-distance ring, "
        f"NOT a shower-max-plane slice — ring width={RADIAL_BIN_WIDTH_MM:.0f}mm, "
        f"max radius={RADIAL_MAX_MM:.0f}mm (starting placeholder, retune from printed "
        f"containment check), min {RADIAL_MIN_HITS} hits/ring — decoded-weight gate (cut={WEIGHT_CUT})\n"
        "Caveat: a G4/CC3 difference here may reflect transverse width, longitudinal "
        "shower development, or both entangled — cannot be separated from this plot alone",
        fontsize=8
    )
    plt.tight_layout()
    out = f"{PLOT_DIR}/fig2c_radial_profile_reco.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  \u2713  Saved: fig2c_radial_profile_reco.png")


# ── Figure 3: Splitting multiplicity — two-panel (Change 2 + histogram) ───
def make_fig_splitting(g4, cc3):
    """
    Panel A: n_split=1 (clean) grouped bars, G4 vs CC3, own y-scale.
    Panel B: n_split=2 (solid) + n_split=3+ (hatched, stacked) grouped bars,
             G4 vs CC3, own y-scale.
    n_split=0 (lost) intentionally excluded -- that is a fake-rate/efficiency
    story, not a splitting story.

    FIX C: suptitle leads with the plain-language definition of n_split
    before the "weighted MC-side view" implementation label.

    FIX D: Poisson-style sqrt(N) error bars added on every bar in both
    panels. Panel B counts were low in several bins (single/double digits
    in some) -- without error bars, G4-vs-CC3 differences there were not
    distinguishable from statistical noise.
    """
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    fig.subplots_adjust(hspace=0.15)

    bw = np.diff(E_BINS) * 0.9
    half = bw / 2 * 0.45
    barw = half * 2 * 0.9

    g4_n1_err  = np.sqrt(g4["n1"])
    cc3_n1_err = np.sqrt(cc3["n1"])
    ax_top.bar(E_CENTERS - half, g4["n1"],  width=barw, color=C_G4,  alpha=0.85, label="G4",
               yerr=g4_n1_err, capsize=3, ecolor="black", error_kw=dict(elinewidth=1.0))
    ax_top.bar(E_CENTERS + half, cc3["n1"], width=barw, color=C_CC3, alpha=0.85, label="CC3",
               yerr=cc3_n1_err, capsize=3, ecolor="black", error_kw=dict(elinewidth=1.0))
    ax_top.set_ylabel("N photons (n_split=1, clean)")
    ax_top.set_title("Panel A — Clean (unsplit) population", fontsize=10)
    ax_top.legend(fontsize=9)

    g4_split_total  = g4["n2"] + g4["n3plus"]
    cc3_split_total = cc3["n2"] + cc3["n3plus"]
    g4_split_err  = np.sqrt(g4_split_total)
    cc3_split_err = np.sqrt(cc3_split_total)

    ax_bot.bar(E_CENTERS - half, g4["n2"], width=barw, color=C_G4, alpha=0.85, label="G4: n_split=2")
    ax_bot.bar(E_CENTERS - half, g4["n3plus"], width=barw, bottom=g4["n2"],
               color=C_G4, alpha=0.45, hatch="//", label="G4: n_split=3+",
               yerr=g4_split_err, capsize=3, ecolor="black", error_kw=dict(elinewidth=1.0))
    ax_bot.bar(E_CENTERS + half, cc3["n2"], width=barw, color=C_CC3, alpha=0.85, label="CC3: n_split=2")
    ax_bot.bar(E_CENTERS + half, cc3["n3plus"], width=barw, bottom=cc3["n2"],
               color=C_CC3, alpha=0.45, hatch="//", label="CC3: n_split=3+",
               yerr=cc3_split_err, capsize=3, ecolor="black", error_kw=dict(elinewidth=1.0))
    ax_bot.set_ylabel("N photons (n_split\u22652, split)")
    ax_bot.set_xlabel("MC Truth Photon Energy  $E_\\mathrm{mc}$  [GeV]")
    ax_bot.set_title("Panel B — Split population (own y-scale)", fontsize=10)
    ax_bot.legend(fontsize=8, ncol=2)

    fig.suptitle(
        "Shower Splitting: how many reco PFOs does one truth photon's shower become?\n"
        "n_split = count of confident-weight reco links to a single MC truth photon "
        "(n_split=1: clean; n_split\u22652: PandoraPFA split the shower into multiple PFOs)\n"
        f"{SIGNAL_SUBTITLE} — decoded-weight cut ({WEIGHT_CUT}) — n_split=0 (lost) excluded, "
        "see fake-rate plot instead — error bars: sqrt(N) Poisson counting error",
        fontsize=9
    )
    plt.tight_layout()
    out = f"{PLOT_DIR}/fig3_splitting_multiplicity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  \u2713  Saved: fig3_splitting_multiplicity.png")


# ── Figure 4: Global fake rate (REPLACES old efficiency plot) ─────────────
def make_fig_fake_rate(g4, cc3):
    """
    Single global number per sample (not binned by energy) --
    fake rate = N(reco PDG==22 PFOs with no genuine-photon match above
    WEIGHT_CUT) / N(all reco PDG==22 PFOs).
    No energy cut, no Signal B window, no isHandledByFastSim() in the
    decision -- deliberately simple, per locked spec.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    labels = ["Geant4", "CC3"]
    rates  = [g4["global_fake_rate"], cc3["global_fake_rate"]]
    colors = [C_G4, C_CC3]

    bars = ax.bar(labels, rates, color=colors, alpha=0.85, width=0.5)
    for bar_, rate, d in zip(bars, rates, [g4, cc3]):
        ax.text(bar_.get_x() + bar_.get_width()/2, rate + 0.005,
                f"{rate:.4f}\n(N_fake={d['n_fake_photon_pfos']}, N_all={d['n_all_photon_pfos']})",
                ha="center", fontsize=8)

    ax.set_ylabel("Global fake rate")
    ax.set_ylim(0, max(rates) * 1.4 if max(rates) > 0 else 1.0)
    ax.set_title(
        "Global Photon Fake Rate — G4 vs CC3\n"
        "Fake = reco PDG==22 PFO with no surviving-weight match to a genuine "
        "(genStatus==1, PDG==22) MC particle\n"
        f"No energy cut, no Signal B window — decoded-weight gate (cut={WEIGHT_CUT})",
        fontsize=9
    )
    ax.text(0.01, 1.01, f"ILD ILDConfig reco — {RUN_LABEL}",
            transform=ax.transAxes, fontsize=8, color="gray")

    plt.tight_layout()
    out = f"{PLOT_DIR}/fig4_fake_rate.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  \u2713  Saved: fig4_fake_rate.png")


# ── STUBS: Plot 2b, 2d (sim level) — DO NOT FILL IN YET ───────────────────
def make_fig_cluster_width_sim(g4_sim, cc3_sim):
    """
    TODO: Plot 2b — cluster transverse width, sim level.
    Requires attributing SimCalorimeterHit / CaloHitContribution energy to
    a specific MC truth photon (candidate path: CaloHitContribution.
    getParticle()). NOT YET VERIFIED against real data -- see
    sim_hit_attribution_check.py (had a RelationRange slicing bug, not yet
    rerun successfully). Axis should be the TRUE MC photon direction
    (mc.getMomentum()), not the centroid trick used at reco level.
    Do not implement until attribution path is confirmed.
    """
    raise NotImplementedError(
        "Plot 2b (sim-level cluster width) pending sim-hit attribution check."
    )


def make_fig_radial_profile_sim(g4_sim, cc3_sim):
    """
    TODO: Plot 2d — radial profile, sim level. Same ring-binned method as
    2c, computed on SimCalorimeterHits, axis = true MC photon direction.
    Blocked on the same attribution check as 2b.
    """
    raise NotImplementedError(
        "Plot 2d (sim-level radial profile) pending sim-hit attribution check."
    )


# ── ENTRY POINT ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  single_photon_observables_v3.py")
print(f"  Dataset: {DATASET}  ({RUN_LABEL})")
print("  Plots implemented this pass: 1, 2a, 2c, 3, 4")
print("  Plots 2b, 2d: STUBBED, pending sim-hit attribution check")
print("  Round 1 fixes: A (track_weight divisor), B (radial profile")
print("  labeling+caveat), C (splitting plain-language title)")
print("  Round 2 fixes: D (Poisson error bars, Plot 3), E (radial max")
print("  radius reduced + empirical containment check), F (per-ring")
print("  stats floor, Plot 2c), G (3-way DATASET config)")
print("="*60)

g4_data  = run(G4_FILE,  "Geant4")
cc3_data = run(CC3_FILE, "CC3")

print("\nGenerating figures...")
make_fig_response(g4_data, cc3_data)
make_fig_cluster_width(g4_data, cc3_data)
make_fig_radial_profile(g4_data, cc3_data)
make_fig_splitting(g4_data, cc3_data)
make_fig_fake_rate(g4_data, cc3_data)

print(f"\n  Figures saved to: {PLOT_DIR}")
print(f"  Plots 2b/2d skipped (stubbed, see docstrings).")
print(f"  Reminder: check the printed 99.9% containment radius above and")
print(f"  retune RADIAL_MAX_MM (currently {RADIAL_MAX_MM:.0f}mm) if needed.")
print(f"  Done.\n")
