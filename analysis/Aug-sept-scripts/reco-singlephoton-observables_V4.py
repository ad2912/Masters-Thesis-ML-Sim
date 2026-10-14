"""
single_photon_observables_v4.py
=================================
Revised single-photon benchmarking script.

CHANGE LOG vs v3 (this session's revisions, 2026-09-10, pre-presentation pass):

  - CHANGE H: Fake rate (Plot 4) is now ENERGY-DEPENDENT, not a single global
    scalar. Binned by reco PFO energy (pfo.getEnergy()), reusing the same
    E_BINS edges as the rest of the script for visual consistency across
    plots, even though these bins were originally motivated by MC-truth
    photon energy for Signal B -- fakes have no MC truth to bin against, so
    reco energy is the only available axis. Two scalars (n_all_photon_pfos,
    n_fake_photon_pfos) become two arrays indexed by E_BINS. Poisson error
    bars (binomial-style sqrt(p(1-p)/N)) added since low-N bins would
    otherwise look artificially precise.

  - CHANGE I: Radial profile (Plot 2c) bin width halved, 5mm -> 2.5mm
    (RADIAL_BIN_WIDTH_MM). NOTE: this is now finer than the ILD SiW-ECAL
    5x5mm^2 cell pitch -- flagged per session discussion, kept per explicit
    user instruction. RADIAL_MIN_HITS left at 20; expect more rings masked
    to NaN at fixed statistics since each ring covers less area. Main-panel
    y-axis switched to log scale: at v3's linear scale the G4/CC3 curves
    were visually indistinguishable (dominated by the first ring) even
    though the ratio panel showed a real, smooth 0.9->1.1 trend across
    radius -- log scale is needed to actually see that separation in the
    density curves themselves, not just infer it from the ratio. Ratio
    panel y-range fixed to (0.8, 1.2) per supervisor request (was (0,2)).

  - CHANGE K: Cluster width (Plot 2a) gets a ratio panel underneath
    (CC3/G4 on width_mean, propagated error), mirroring the two-panel
    layout already used in Plot 1. No change to the top panel or to how
    width_mean/width_err are computed.

  - CHANGE J2 (splitting plot, FINAL for this session -- superseded an
    earlier same-session attempt that stacked everything on one shared
    y-axis and looked too messy because n_split=1 dominated the plot):
    Two independent y-scales.
      Panel A = n_split=1 (clean) only, own y-scale.
      Panel B = stacked bars, FIXED categories n_split={2,3,4,"5+"}
      (capped, not fully dynamic anymore -- per explicit instruction),
      own y-scale, distinct hatch per category.
      Ratio panel (under Panel B) = one line per category, with DISTINCT
      COLOR AND MARKER SHAPE per category (not a color gradient -- a
      4-line viridis gradient was hard to read live).
    n_split=0 (lost) still excluded throughout -- see energy-dependent
    fake-rate plot (Change H) instead.

  Everything from v3's Round 1 and Round 2 fixes (decoded-weight gate,
  radial profile relabeling, splitting plain-language title, Poisson error
  bars, empirical containment check, per-ring stats floor, 3-way DATASET
  config) is UNCHANGED and carried forward as-is.

NOT YET IMPLEMENTED (carried over from v3, still pending):
  - Plot 2b (cluster width, sim level) and Plot 2d (radial profile, sim
    level) remain STUBBED -- blocked on sim_hit_attribution_check.py,
    unrelated to this session's changes.

Signal: Signal B — genStatus==1, PDG==22, E_mc>=10 GeV, parent PDG!=11
(NOT the same as Signal A, which is pi0-daughter photons used in the
di-photon chapter script — do not conflate the two).

Run:
    source ~/source.sh
    python3 single_photon_observables_v4.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import os
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────
DATASET = "100k"   # one of: "1k", "10k", "100k"  -- validate on 10k, then switch to 100k

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

USE_FINE_E_BINS = False

PLOT_DIR = os.path.expanduser(
    f"~/thesis-ml-sim/plots/single_photon_observables_v4_{DATASET}"
    f"_{'fine' if USE_FINE_E_BINS else 'coarse'}bins"
)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── PHYSICS CONSTANTS ────────────────────────────────────────────────────
PHOTON_PDG   = 22
ELECTRON_PDG = 11
E_MC_MIN     = 10.0
WEIGHT_CUT   = 0.5   # on DECODED cluster_weight, in [0,1]

if USE_FINE_E_BINS:
    E_BINS = np.arange(0, 130, 10.0)
else:
    E_BINS = np.array([0, 10, 20, 30, 40, 60, 80, 100, 125], dtype=float)

E_CENTERS = 0.5 * (E_BINS[:-1] + E_BINS[1:])
N_E_BINS  = len(E_CENTERS)

# CHANGE H: fake rate now binned by RECO PFO energy, reusing E_BINS edges.
FAKE_E_BINS    = E_BINS
FAKE_E_CENTERS = E_CENTERS
N_FAKE_E_BINS  = N_E_BINS

# Radial profile (Plot 2c) config.
# CHANGE I: bin width halved 5mm -> 2.5mm per supervisor request. This is
# now FINER than the ILD SiW-ECAL 5x5mm^2 cell pitch (flagged explicitly
# this session) -- kept per explicit instruction, not a silent choice.
RADIAL_BIN_WIDTH_MM = 2.5
RADIAL_MAX_MM       = 50.0
RADIAL_MIN_HITS     = 20
RADIAL_BINS   = np.arange(0, RADIAL_MAX_MM + RADIAL_BIN_WIDTH_MM, RADIAL_BIN_WIDTH_MM)
RADIAL_CENTERS = 0.5 * (RADIAL_BINS[:-1] + RADIAL_BINS[1:])
RADIAL_RING_AREA = np.pi * (RADIAL_BINS[1:]**2 - RADIAL_BINS[:-1]**2)

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
    Unchanged from v3 -- confirmed correct against real data (ceiling check,
    2026-08-03) and re-verified by hand earlier this session. Do not touch.
    """
    raw_int = int(round(raw_weight))
    cluster_weight = (raw_int // 10000) / 1000.0
    track_weight   = (raw_int % 10000) / 1000.0
    return cluster_weight, track_weight


def compute_cluster_width_and_profile(cluster, axis=None):
    """
    Energy-weighted transverse RMS (sigma_perp) AND per-hit (d_perp, E) pairs
    for the radial profile histogram, computed in one pass over the hits.
    Unchanged from v3.
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


def bin_index(value, bins, centers):
    idx = np.searchsorted(bins, value, side="right") - 1
    if idx < 0 or idx >= len(centers):
        return -1
    return int(idx)


def print_radial_containment_check(all_d_perp_pooled, all_energy_pooled, label):
    """Unchanged from v3."""
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
    Single pass over events.

    CHANGE H: fake rate is now accumulated into per-reco-energy-bin arrays
    (n_all_photon_pfos_binned, n_fake_photon_pfos_binned), not two scalars.

    n_split remains stored as raw per-photon integers in n_split_counts
    (uncollapsed) -- the bucketing into fixed display categories
    {1, 2, 3, 4, "5+"} happens only at plotting time, in make_fig_splitting.
    """
    from podio import root_io

    print(f"\n{'='*60}")
    print(f"  Processing: {label}  ({RUN_LABEL})")
    print(f"{'='*60}")

    reader = root_io.Reader(filepath)

    response_ratios = defaultdict(list)
    widths          = defaultdict(list)
    radial_d_perp   = defaultdict(list)
    radial_energy   = defaultdict(list)
    n_split_counts  = defaultdict(list)   # raw per-photon n_split, kept uncollapsed

    # CHANGE H: fake rate accumulators, now arrays over FAKE_E_BINS.
    n_all_photon_pfos_binned  = np.zeros(N_FAKE_E_BINS, dtype=int)
    n_fake_photon_pfos_binned = np.zeros(N_FAKE_E_BINS, dtype=int)

    n_events_read = 0

    for event in reader.get("events"):
        if n_events_read >= MAX_EVENTS:
            break
        n_events_read += 1

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

        mc_all_link_weights = defaultdict(list)
        for link in event.get("RecoMCTruthLink"):
            mc    = link.getTo()
            raw_w = link.getWeight()
            cw, _ = decode_weight(raw_w)
            mc_all_link_weights[mc.getObjectID().index].append(cw)

        mc_to_owned = defaultdict(list)
        for pfo in event.get("PandoraPFOs"):
            if pfo.getPDG() != PHOTON_PDG:
                continue
            rid = pfo.getObjectID().index
            if rid not in reco_to_mc:
                continue
            mc, _ = reco_to_mc[rid]
            mc_to_owned[mc.getObjectID().index].append(pfo)

        # ── Fake rate (Plot 4), CHANGE H: binned by reco PFO energy ───────
        for pfo in event.get("PandoraPFOs"):
            if pfo.getPDG() != PHOTON_PDG:
                continue
            fb = bin_index(pfo.getEnergy(), FAKE_E_BINS, FAKE_E_CENTERS)
            if fb < 0:
                continue
            n_all_photon_pfos_binned[fb] += 1
            rid = pfo.getObjectID().index
            is_fake = True
            if rid in reco_to_mc:
                mc, _ = reco_to_mc[rid]
                if mc.getGeneratorStatus() == 1 and mc.getPDG() == PHOTON_PDG:
                    is_fake = False
            if is_fake:
                n_fake_photon_pfos_binned[fb] += 1

        # ── Loop Signal B truth photons ─────────────────────────────────
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
                n_split_counts[b].append(n_split)   # raw value, not collapsed

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
    print(f"  All reco photon PFOs (total)  : {n_all_photon_pfos_binned.sum()}")
    print(f"  Fake photon PFOs (total)      : {n_fake_photon_pfos_binned.sum()}")
    total_all = n_all_photon_pfos_binned.sum()
    if total_all > 0:
        print(f"  Global fake rate (unbinned check): {n_fake_photon_pfos_binned.sum()/total_all:.4f}")

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
        ring_density[hit_count_per_ring < RADIAL_MIN_HITS] = np.nan
        radial_profile[b] = ring_density

    # n_split raw count table: split_table[b, k-1] = count of photons with
    # n_split == k, for k=1..max_n_split. max_n_split is whatever the data
    # actually contains -- bucketing into fixed display categories
    # {1, 2, 3, 4, "5+"} happens later, at plotting time.
    all_observed = [ns for b in range(N_E_BINS) for ns in n_split_counts[b]]
    max_n_split = max(all_observed) if all_observed else 1
    split_table = np.zeros((N_E_BINS, max_n_split), dtype=int)
    for b in range(N_E_BINS):
        for ns in n_split_counts[b]:
            split_table[b, ns - 1] += 1

    return {
        "resp_mean" : resp_mean,
        "resp_sigma": resp_sigma,
        "width_mean": width_mean,
        "width_err" : width_err,
        "radial_profile": radial_profile,
        "split_table": split_table,
        "max_n_split": max_n_split,
        "n_all_photon_pfos_binned": n_all_photon_pfos_binned,
        "n_fake_photon_pfos_binned": n_fake_photon_pfos_binned,
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


def ratio_with_error(num, num_err, den, den_err):
    """Standard error propagation for a ratio of two independent means."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = num / den
        rel_err = np.sqrt((num_err / num) ** 2 + (den_err / den) ** 2)
        ratio_err = np.abs(ratio) * rel_err
    return ratio, ratio_err


# ── Figure 1: Energy response and resolution (unchanged from v3) ─────────
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


# ── Figure 2a: Cluster width, reco level — CHANGE K: + ratio panel ────────
def make_fig_cluster_width(g4, cc3):
    fig, (ax, ax_rat) = plt.subplots(
        2, 1, figsize=(8, 6.5),
        gridspec_kw={"height_ratios": [3, 1.2]},
        sharex=True
    )
    fig.subplots_adjust(hspace=0.06)

    plot_valid_err(ax, E_CENTERS, g4["width_mean"],  g4["width_err"],  **G4_KW)
    plot_valid_err(ax, E_CENTERS, cc3["width_mean"], cc3["width_err"], **CC3_KW)

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

    # CHANGE K: ratio panel, CC3/G4 on width_mean with propagated error.
    ratio, ratio_err = ratio_with_error(cc3["width_mean"], cc3["width_err"],
                                        g4["width_mean"],  g4["width_err"])
    ax_rat.axhline(1.0, color="black", lw=1.0, ls=":", alpha=0.6)
    plot_valid_err(ax_rat, E_CENTERS, ratio, ratio_err,
                   color="black", marker="D", linewidth=1.5, markersize=5, label="CC3/G4")
    ax_rat.set_ylabel("CC3 / G4")
    ax_rat.set_xlabel("MC Truth Photon Energy  $E_\\mathrm{mc}$  [GeV]")
    ax_rat.set_xlim(E_BINS[0], E_BINS[-1])

    plt.tight_layout()
    out = f"{PLOT_DIR}/fig2a_cluster_width_reco.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  \u2713  Saved: fig2a_cluster_width_reco.png")


# ── Figure 2c: Radial profile, reco level — CHANGE I: log-y + tighter ratio
def make_fig_radial_profile(g4, cc3):
    """
    CHANGE I: main-panel y-axis is now log scale. At linear scale (v3), the
    G4/CC3 curves were visually indistinguishable -- dominated by the first
    ring -- even though the ratio panel showed a real, smooth trend from
    ~0.9 to ~1.1 across radius. Log scale surfaces that separation directly
    in the density curves. Zero/negative/NaN densities are masked before
    plotting (can't log a non-positive value).
    Ratio y-range fixed to (0.8, 1.2) per supervisor request (was (0, 2)).
    Bin width halved to 2.5mm (RADIAL_BIN_WIDTH_MM in CONFIG).
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

        g4_prof_pos  = np.where(g4_prof  > 0, g4_prof,  np.nan)
        cc3_prof_pos = np.where(cc3_prof > 0, cc3_prof, np.nan)

        ax_main.step(RADIAL_CENTERS, g4_prof_pos,  where="mid", color=C_G4,  label="G4")
        ax_main.step(RADIAL_CENTERS, cc3_prof_pos, where="mid", color=C_CC3, label="CC3", linestyle="--")
        ax_main.set_yscale("log")
        ax_main.set_title(f"[{E_BINS[b]:.0f},{E_BINS[b+1]:.0f}) GeV", fontsize=9)
        ax_main.set_ylabel("E density,\nlong.-integrated\n[GeV/mm$^2$]", fontsize=7)
        ax_main.tick_params(labelbottom=False)
        if b == 0:
            ax_main.legend(fontsize=7)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = cc3_prof / g4_prof
        ax_rat.axhline(1.0, color="black", lw=0.8, ls=":", alpha=0.6)
        ax_rat.step(RADIAL_CENTERS, ratio, where="mid", color="black")
        ax_rat.set_ylim(0.8, 1.2)   # CHANGE I: was (0, 2)
        ax_rat.set_xlabel("Distance from shower axis [mm]", fontsize=8)
        ax_rat.set_ylabel("CC3/G4", fontsize=7)

    fig.suptitle(
        "Radial Energy Density Profile (longitudinally integrated) — G4 vs CC3 (reco level)\n"
        f"{SIGNAL_SUBTITLE} — pools hits over full shower depth per transverse-distance ring, "
        f"NOT a shower-max-plane slice — ring width={RADIAL_BIN_WIDTH_MM:.1f}mm "
        f"(finer than 5mm ILD SiW-ECAL cell pitch, by request), "
        f"max radius={RADIAL_MAX_MM:.0f}mm, min {RADIAL_MIN_HITS} hits/ring — "
        f"decoded-weight gate (cut={WEIGHT_CUT}) — main panel log-scale\n"
        "Caveat: a G4/CC3 difference here may reflect transverse width, longitudinal "
        "shower development, or both entangled — cannot be separated from this plot alone",
        fontsize=8
    )
    plt.tight_layout()
    out = f"{PLOT_DIR}/fig2c_radial_profile_reco.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  \u2713  Saved: fig2c_radial_profile_reco.png")


# ── Figure 3: Splitting multiplicity — REDESIGN v2 (final, this session):
# two independent y-scales, fixed categories {2, 3, 4, "5+"}, distinct
# color+marker+hatch per category ──────────────────────────────────────
_SPLIT_CATS     = ["n_split=2", "n_split=3", "n_split=4", "n_split=5+"]
_SPLIT_COLORS   = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
_SPLIT_MARKERS  = ["o", "s", "^", "D"]
_SPLIT_HATCHES2 = ["///", "xxx", "...", "***"]


def _bucket_split_table(table, max_k):
    """
    Collapse a (N_E_BINS, max_k) raw per-value count table into fixed
    display buckets: n_split=2, 3, 4, 5+ (columns 0..3 in the returned
    array). Input table columns are indexed by n_split value - 1
    (column 0 = n_split=1, which is handled separately in Panel A).
    """
    out = np.zeros((table.shape[0], 4), dtype=float)
    for b in range(table.shape[0]):
        if max_k >= 2 and table.shape[1] > 1:
            out[b, 0] = table[b, 1]
        if max_k >= 3 and table.shape[1] > 2:
            out[b, 1] = table[b, 2]
        if max_k >= 4 and table.shape[1] > 3:
            out[b, 2] = table[b, 3]
        if max_k >= 5 and table.shape[1] > 4:
            out[b, 3] = table[b, 4:].sum()
    return out


def make_fig_splitting(g4, cc3):
    """
    FINAL redesign this session, after an earlier same-session attempt
    (single shared-axis stacked plot) looked too messy: n_split=1 counts
    are ~5-10x larger than any split category, and completely dominated
    the shared y-axis, crushing Panel B's structure to a sliver.

    Panel A: n_split=1 only, own y-scale (fast "reconstruction works" check).
    Panel B: stacked bars, FIXED categories n_split={2,3,4,"5+"} (capped,
             not fully dynamic -- per explicit instruction), own y-scale,
             distinct hatch per category.
    Ratio panel (under Panel B): one line per category, distinct color AND
    marker shape per category (not a color gradient -- a 4-line viridis
    gradient was hard to read live; swapped for high-contrast discrete
    colors + shapes per explicit feedback).
    n_split=0 (lost) excluded throughout -- see energy-dependent fake-rate
    plot (Change H) instead.
    """
    max_k = max(g4["max_n_split"], cc3["max_n_split"])

    def _pad(table, k):
        if table.shape[1] >= k:
            return table
        pad = np.zeros((table.shape[0], k - table.shape[1]), dtype=int)
        return np.hstack([table, pad])

    g4_table_full  = _pad(g4["split_table"],  max_k)
    cc3_table_full = _pad(cc3["split_table"], max_k)

    g4_n1  = g4_table_full[:, 0].astype(float)
    cc3_n1 = cc3_table_full[:, 0].astype(float)

    g4_bucketed  = _bucket_split_table(g4_table_full,  max_k)
    cc3_bucketed = _bucket_split_table(cc3_table_full, max_k)

    fig = plt.figure(figsize=(11, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.6, 2.0, 1.3], hspace=0.04)
    ax_a   = fig.add_subplot(gs[0])
    ax_b   = fig.add_subplot(gs[1], sharex=ax_a)
    ax_rat = fig.add_subplot(gs[2], sharex=ax_a)

    bw = np.diff(E_BINS) * 0.9
    half = bw / 2 * 0.45
    barw = half * 2 * 0.9

    # ── Panel A: n_split=1, own y-scale ────────────────────────────────
    g4_n1_err  = np.sqrt(np.maximum(g4_n1, 1))
    cc3_n1_err = np.sqrt(np.maximum(cc3_n1, 1))
    ax_a.bar(E_CENTERS - half, g4_n1,  width=barw, color=C_G4,  alpha=0.85,
             label="G4: n_split=1", yerr=g4_n1_err, capsize=3, ecolor="black",
             error_kw=dict(elinewidth=1.0))
    ax_a.bar(E_CENTERS + half, cc3_n1, width=barw, color=C_CC3, alpha=0.85,
             label="CC3: n_split=1", yerr=cc3_n1_err, capsize=3, ecolor="black",
             error_kw=dict(elinewidth=1.0))
    ax_a.set_ylabel("N photons\n(n_split=1, clean)")
    ax_a.set_title("Panel A — Clean (unsplit) population, own y-scale", fontsize=10)
    ax_a.legend(fontsize=8)
    ax_a.tick_params(labelbottom=False)

    # ── Panel B: stacked {2,3,4,5+}, own y-scale ───────────────────────
    g4_bottom  = np.zeros(N_E_BINS)
    cc3_bottom = np.zeros(N_E_BINS)
    for i, cat in enumerate(_SPLIT_CATS):
        g4_counts  = g4_bucketed[:, i]
        cc3_counts = cc3_bucketed[:, i]
        ax_b.bar(E_CENTERS - half, g4_counts, width=barw, bottom=g4_bottom,
                 color=C_G4, alpha=0.5 + 0.12 * i, hatch=_SPLIT_HATCHES2[i],
                 edgecolor="white", linewidth=0.5,
                 label=f"G4: {cat}")
        ax_b.bar(E_CENTERS + half, cc3_counts, width=barw, bottom=cc3_bottom,
                 color=C_CC3, alpha=0.5 + 0.12 * i, hatch=_SPLIT_HATCHES2[i],
                 edgecolor="white", linewidth=0.5,
                 label=f"CC3: {cat}")
        g4_bottom  = g4_bottom + g4_counts
        cc3_bottom = cc3_bottom + cc3_counts

    ax_b.set_ylabel("N photons\n(n_split\u22652, split)")
    ax_b.set_title("Panel B — Split population, fixed categories {2,3,4,5+}, own y-scale", fontsize=10)
    ax_b.legend(fontsize=7, ncol=2, loc="upper right")
    ax_b.tick_params(labelbottom=False)

    # ── Ratio panel: one line per category, distinct color + marker ───
    for i, cat in enumerate(_SPLIT_CATS):
        g4_c  = g4_bucketed[:, i]
        cc3_c = cc3_bucketed[:, i]
        g4_err  = np.sqrt(np.maximum(g4_c, 1))
        cc3_err = np.sqrt(np.maximum(cc3_c, 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = cc3_c / g4_c
            rel_err = np.sqrt((cc3_err / np.where(cc3_c > 0, cc3_c, np.nan)) ** 2 +
                              (g4_err  / np.where(g4_c  > 0, g4_c,  np.nan)) ** 2)
            ratio_err = np.abs(ratio) * rel_err
        valid = (g4_c >= 5) & (cc3_c >= 5)
        ratio_plot = np.where(valid, ratio, np.nan)
        ax_rat.errorbar(E_CENTERS, ratio_plot, yerr=ratio_err,
                         marker=_SPLIT_MARKERS[i], markersize=6, linewidth=1.4,
                         capsize=3, color=_SPLIT_COLORS[i], label=cat)

    ax_rat.axhline(1.0, color="black", lw=1.0, ls=":", alpha=0.6)
    ax_rat.set_ylabel("CC3 / G4\n(per category)")
    ax_rat.set_xlabel("MC Truth Photon Energy  $E_\\mathrm{mc}$  [GeV]")
    ax_rat.set_xlim(E_BINS[0], E_BINS[-1])
    ax_rat.legend(fontsize=8, ncol=4, loc="upper center")

    fig.suptitle(
        "Shower Splitting: how many reco PFOs does one truth photon's shower become?\n"
        "n_split = count of confident-weight reco links to a single MC truth photon "
        "(n_split=1: clean; n_split\u22652: PandoraPFA split the shower into multiple PFOs)\n"
        f"{SIGNAL_SUBTITLE} \u2014 decoded-weight cut ({WEIGHT_CUT}) \u2014 n_split=0 (lost) excluded, "
        "see energy-dependent fake-rate plot instead \u2014 Panel A and B use independent "
        "y-scales since clean photons vastly outnumber split ones \u2014 ratio panel: points "
        "with <5 counts in either sample suppressed",
        fontsize=9
    )
    out = f"{PLOT_DIR}/fig3_splitting_multiplicity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  \u2713  Saved: fig3_splitting_multiplicity.png")


# ── Figure 4: Fake rate, CHANGE H: now energy-dependent ──────────────────
def make_fig_fake_rate(g4, cc3):
    """
    CHANGE H: fake rate = N(reco PDG==22 PFOs with no genuine-photon match
    above WEIGHT_CUT) / N(all reco PDG==22 PFOs), now computed PER RECO
    ENERGY BIN instead of as one global scalar. Binomial-style error bars
    (sqrt(p(1-p)/N)) shown per bin since low-N bins would otherwise look
    artificially precise.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for data, kw in [(g4, G4_KW), (cc3, CC3_KW)]:
        n_all  = data["n_all_photon_pfos_binned"].astype(float)
        n_fake = data["n_fake_photon_pfos_binned"].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = n_fake / n_all
            err  = np.sqrt(rate * (1 - rate) / n_all)
        rate = np.where(n_all > 0, rate, np.nan)
        plot_valid_err(ax, FAKE_E_CENTERS, rate, err, **kw)

    ax.set_xlabel("Reconstructed Photon PFO Energy  $E_\\mathrm{reco}$  [GeV]")
    ax.set_ylabel("Fake rate")
    ax.set_ylim(bottom=0)
    ax.set_xlim(FAKE_E_BINS[0], FAKE_E_BINS[-1])
    ax.legend(framealpha=0.5)
    ax.set_title(
        "Photon Fake Rate vs Reco Energy — G4 vs CC3\n"
        "Fake = reco PDG==22 PFO with no surviving-weight match to a genuine "
        "(genStatus==1, PDG==22) MC particle\n"
        f"No Signal B window — decoded-weight gate (cut={WEIGHT_CUT}) — "
        "error bars: binomial sqrt(p(1-p)/N)",
        fontsize=9
    )
    ax.text(0.01, 1.01, f"ILD ILDConfig reco — {RUN_LABEL}",
            transform=ax.transAxes, fontsize=8, color="gray")

    plt.tight_layout()
    out = f"{PLOT_DIR}/fig4_fake_rate.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  \u2713  Saved: fig4_fake_rate.png")


# ── STUBS: Plot 2b, 2d (sim level) — DO NOT FILL IN YET, unchanged from v3
def make_fig_cluster_width_sim(g4_sim, cc3_sim):
    raise NotImplementedError(
        "Plot 2b (sim-level cluster width) pending sim-hit attribution check."
    )


def make_fig_radial_profile_sim(g4_sim, cc3_sim):
    raise NotImplementedError(
        "Plot 2d (sim-level radial profile) pending sim-hit attribution check."
    )


# ── ENTRY POINT ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  single_photon_observables_v4.py")
print(f"  Dataset: {DATASET}  ({RUN_LABEL})")
print("  Plots implemented this pass: 1, 2a, 2c, 3, 4")
print("  Plots 2b, 2d: STUBBED, pending sim-hit attribution check")
print("  v4 changes: H (energy-dependent fake rate), I (radial bin width")
print("  2.5mm + log-scale main panel), J2 (splitting plot: two")
print("  independent y-scales, fixed categories {2,3,4,5+}, distinct")
print("  color+marker per category in ratio panel), K (width ratio panel)")
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
print(f"  Done.\n")
