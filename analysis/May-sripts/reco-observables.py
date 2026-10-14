"""
single_photon_observables.py
=============================
Single-pass analysis producing 4 figures (5 observables) for the
single-photon benchmarking chapter.

Physics questions answered:
  1. Energy response  : Is there an energy scale bias in CC3 vs G4?
  2. Energy resolution: Does CC3 give noisier E_reco near 10 GeV?
  3. Cluster width    : Does CC3 produce wider/narrower ECAL clusters?
  4. Split multiplicity: Is the excess reco PFO count energy-dependent?
                         Conditioned on reconstruction (N_owned >= 1) —
                         measures splitting, not efficiency.
  5. Reco efficiency  : What fraction of truth photons are reconstructed?

Signal: Signal B — genStatus==1, PDG==22, E_mc>=10 GeV, parent PDG!=11
Link strategy: Strategy A — highest-weight match wins, both directions.
Efficiency: truth photon counts as reconstructed only if best-owned reco PFO has PDG==22.

CC3 note: NO isHandledByFastSim() filter applied — full CC3 sample (fast-sim
+ G4-handled) is compared against full G4 sample for a population-matched
headline comparison. The is_fs flag is stored for optional sub-analysis.

Run:
    source ~/source.sh
    python3 single_photon_observables.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import os
from collections import defaultdict

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
    f"~/thesis-ml-sim/plots/single_photon_observables_{'1k' if USE_1K else '10k-new-bins'}_v2"
)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── PHYSICS CONSTANTS ─────────────────────────────────────────────────────────
PHOTON_PDG   = 22
ELECTRON_PDG = 11
E_MC_MIN     = 10.0

#E_BINS    = np.array([10, 15, 20, 30, 50, 80, 125], dtype=float)
E_BINS = np.array([0,10, 20, 30, 40, 60, 80, 100, 125], dtype=float)
E_CENTERS = 0.5 * (E_BINS[:-1] + E_BINS[1:])

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

# ── HELPERS ───────────────────────────────────────────────────────────────────

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


def compute_cluster_width(cluster):
    """
    Energy-weighted transverse RMS of calorimeter hits in a cluster.
    Transverse = perpendicular to shower axis (unit vector from IP to CoG).
    Returns NaN if cluster has < 2 hits or total energy == 0.
    """
    hits = cluster.getHits()
    if len(hits) < 2:
        return float("nan")

    xs = np.array([h.getPosition().x for h in hits], dtype=float)
    ys = np.array([h.getPosition().y for h in hits], dtype=float)
    zs = np.array([h.getPosition().z for h in hits], dtype=float)
    es = np.array([h.getEnergy()      for h in hits], dtype=float)

    e_tot = es.sum()
    if e_tot == 0:
        return float("nan")

    cx = np.dot(es, xs) / e_tot
    cy = np.dot(es, ys) / e_tot
    cz = np.dot(es, zs) / e_tot

    c_mag = math.sqrt(cx**2 + cy**2 + cz**2)
    if c_mag == 0:
        return float("nan")
    ax, ay, az = cx / c_mag, cy / c_mag, cz / c_mag

    d_perp_sq = np.zeros(len(hits))
    for i in range(len(hits)):
        rx, ry, rz = xs[i], ys[i], zs[i]
        r_dot_a    = rx * ax + ry * ay + rz * az
        r_sq       = rx**2 + ry**2 + rz**2
        d_perp_sq[i] = max(r_sq - r_dot_a**2, 0.0)

    return math.sqrt(np.dot(es, d_perp_sq) / e_tot)


def bin_index(e_mc):
    idx = np.searchsorted(E_BINS, e_mc, side="right") - 1
    if idx < 0 or idx >= len(E_CENTERS):
        return -1
    return int(idx)

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

def run(filepath, label, is_cc3=False):
    """
    Single pass over events. For each Signal B MC truth photon:
      - Always fills: efficiency denominator, splitting multiplicity
      - If reconstructed as photon PFO: fills response, resolution, cluster width

    FIX 1: No isHandledByFastSim() filter — full CC3 sample used.
            is_fs stored per photon for optional downstream sub-analysis.
    FIX 2: Multiplicity conditioned on N_owned >= 1 (pure splitting measure).
    FIX 3: Error bars computed for cluster width and multiplicity (sigma/sqrt(N)).
    """
    from podio import root_io

    print(f"\n{'='*60}")
    print(f"  Processing: {label}  ({RUN_LABEL})")
    print(f"{'='*60}")

    reader = root_io.Reader(filepath)

    n_bins = len(E_CENTERS)

    response_ratios = defaultdict(list)
    widths          = defaultdict(list)
    eff_denom       = np.zeros(n_bins, dtype=int)
    eff_numer       = np.zeros(n_bins, dtype=int)
    # FIX 2: only append when N_owned >= 1
    mult_counts     = defaultdict(list)

    n_events_read = 0

    for event in reader.get("events"):
        if n_events_read >= MAX_EVENTS:
            break
        n_events_read += 1

        # Strategy A: best-weight MC match per reco PFO
        reco_to_mc = {}
        for link in event.get("RecoMCTruthLink"):
            reco   = link.getFrom()
            mc     = link.getTo()
            weight = link.getWeight()
            rid    = reco.getObjectID().index
            if rid not in reco_to_mc or weight > reco_to_mc[rid][1]:
                reco_to_mc[rid] = (mc, weight)

        # Strategy A inverted: reco PFOs owned by each MC particle
        # Owned = MC particle is best-weight match for that reco PFO AND reco PDG==22
        mc_to_owned = defaultdict(list)
        for pfo in event.get("PandoraPFOs"):
            if pfo.getPDG() != PHOTON_PDG:
                continue
            rid = pfo.getObjectID().index
            if rid not in reco_to_mc:
                continue
            mc, _ = reco_to_mc[rid]
            mc_to_owned[mc.getObjectID().index].append(pfo)

        # Loop over MC truth photons — Signal B, NO isHandledByFastSim filter
        for mc in event.get("MCParticles"):
            if not signal_b_cuts(mc):
                continue
            # FIX 1: removed isHandledByFastSim() filter here.
            # is_fs available below for optional use but does not gate selection.

            e_mc  = mc.getEnergy()
            mc_id = mc.getObjectID().index
            b     = bin_index(e_mc)
            if b < 0:
                continue

            owned_pfos = mc_to_owned.get(mc_id, [])
            n_owned    = len(owned_pfos)

            # Efficiency denominator — every Signal B truth photon
            eff_denom[b] += 1

            # FIX 2: multiplicity only when photon was actually reconstructed
            # This measures splitting (>1 PFO per found photon), not efficiency loss.
            if n_owned >= 1:
                mult_counts[b].append(n_owned)

            if n_owned == 0:
                continue

            # Efficiency numerator
            eff_numer[b] += 1

            # Primary reco photon = highest-energy owned PFO
            best_pfo = max(owned_pfos, key=lambda p: p.getEnergy())
            e_reco   = best_pfo.getEnergy()

            # Response
            response_ratios[b].append(e_reco / e_mc)

            # Cluster width
            clusters = best_pfo.getClusters()
            if len(clusters) > 0:
                width = compute_cluster_width(clusters[0])
                if not math.isnan(width):
                    widths[b].append(width)

    print(f"  Events processed : {n_events_read}")
    print(f"  Total truth photons selected (Signal B): {eff_denom.sum()}")
    print(f"  Total reconstructed as photon PFO      : {eff_numer.sum()}")

    # Compute per-bin statistics
    resp_mean  = np.full(n_bins, np.nan)
    resp_sigma = np.full(n_bins, np.nan)
    width_mean = np.full(n_bins, np.nan)
    width_err  = np.full(n_bins, np.nan)   # FIX 3
    mult_mean  = np.full(n_bins, np.nan)
    mult_err   = np.full(n_bins, np.nan)   # FIX 3
    eff        = np.full(n_bins, np.nan)
    eff_err    = np.full(n_bins, np.nan)

    for b in range(n_bins):
        r = response_ratios[b]
        if len(r) >= 5:
            resp_mean[b]  = np.mean(r)
            resp_sigma[b] = np.std(r)

        # FIX 3: cluster width with sigma/sqrt(N) error
        w = widths[b]
        if len(w) >= 5:
            width_mean[b] = np.mean(w)
            width_err[b]  = np.std(w) / math.sqrt(len(w))

        # FIX 3: multiplicity with sigma/sqrt(N) error
        m = mult_counts[b]
        if len(m) >= 5:
            mult_mean[b] = np.mean(m)
            mult_err[b]  = np.std(m) / math.sqrt(len(m))

        nd = eff_denom[b]
        nn = eff_numer[b]
        if nd > 0:
            p          = nn / nd
            eff[b]     = p
            eff_err[b] = math.sqrt(p * (1 - p) / nd) if nd > 0 else np.nan

        print(f"  Bin [{E_BINS[b]:.0f},{E_BINS[b+1]:.0f}) GeV: "
              f"N_truth={nd:4d}  N_reco={nn:4d}  "
              f"response={resp_mean[b]:.3f}  "
              f"width={width_mean[b]:.1f} mm  "
              f"mult={mult_mean[b]:.3f}  "
              f"eff={eff[b]:.3f}")

    return {
        "resp_mean" : resp_mean,
        "resp_sigma": resp_sigma,
        "width_mean": width_mean,
        "width_err" : width_err,
        "mult_mean" : mult_mean,
        "mult_err"  : mult_err,
        "eff"       : eff,
        "eff_err"   : eff_err,
        "eff_denom" : eff_denom,
    }


# ── PLOTTING ──────────────────────────────────────────────────────────────────
G4_KW  = dict(color=C_G4,  marker="o", linewidth=2.0, linestyle="-",  markersize=6, label="Geant4")
CC3_KW = dict(color=C_CC3, marker="s", linewidth=2.0, linestyle="--", markersize=6, label="CC3")

SIGNAL_SUBTITLE = "Signal B: genStatus==1, E_mc≥10 GeV, parent PDG≠11 (not ISR)"


def plot_valid(ax, x, y, **kwargs):
    mask = ~np.isnan(y)
    ax.plot(x[mask], y[mask], **kwargs)


def plot_valid_err(ax, x, y, yerr, **kwargs):
    """Plot with error bars, skipping NaN points."""
    mask = ~np.isnan(y)
    kw = {k: v for k, v in kwargs.items() if k != "linewidth"}
    ax.errorbar(x[mask], y[mask], yerr=yerr[mask],
                lw=kwargs.get("linewidth", 2.0), capsize=4, **kw)


# ── Figure 1: Energy response and resolution ──────────────────────────────────
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
    ax_resp.text(0.01, 1.01, f"ILD ILDConfig reco — {RUN_LABEL} — raw (not normalised)",
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
    print(f"  ✓  Saved: fig1_energy_response_resolution.png")


# ── Figure 2: Cluster width ───────────────────────────────────────────────────
def make_fig_cluster_width(g4, cc3):
    fig, ax = plt.subplots(figsize=(8, 5))

    # FIX 3: error bars on cluster width
    plot_valid_err(ax, E_CENTERS, g4["width_mean"],  g4["width_err"],  **G4_KW)
    plot_valid_err(ax, E_CENTERS, cc3["width_mean"], cc3["width_err"], **CC3_KW)

    ax.set_xlabel("MC Truth Photon Energy  $E_\\mathrm{mc}$  [GeV]")
    ax.set_ylabel("Mean cluster transverse width  $\\langle\\sigma_\\perp\\rangle$  [mm]")
    ax.set_title(
        "ECAL Cluster Transverse Width vs $E_\\mathrm{mc}$ — G4 vs CC3\n"
        f"{SIGNAL_SUBTITLE}\n"
        "Energy-weighted RMS of hit transverse distances from shower axis",
        fontsize=10
    )
    ax.legend(framealpha=0.5)
    ax.text(0.01, 1.01, f"ILD ILDConfig reco — {RUN_LABEL}",
            transform=ax.transAxes, fontsize=8, color="gray")

    plt.tight_layout()
    out = f"{PLOT_DIR}/fig2_cluster_width.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  Saved: fig2_cluster_width.png")


# ── Figure 3: Splitting multiplicity ─────────────────────────────────────────
def make_fig_multiplicity(g4, cc3):
    fig, ax = plt.subplots(figsize=(8, 5))

    # FIX 3: error bars on multiplicity
    plot_valid_err(ax, E_CENTERS, g4["mult_mean"],  g4["mult_err"],  **G4_KW)
    plot_valid_err(ax, E_CENTERS, cc3["mult_mean"], cc3["mult_err"], **CC3_KW)

    ax.axhline(1.0, color="black", lw=1.0, ls=":", alpha=0.6,
               label="No splitting (1 PFO per truth photon)")
    ax.set_xlabel("MC Truth Photon Energy  $E_\\mathrm{mc}$  [GeV]")
    ax.set_ylabel("Mean reco photon PFOs per reconstructed truth photon")
    ax.set_title(
        "Shower Splitting Multiplicity vs $E_\\mathrm{mc}$ — G4 vs CC3\n"
        f"{SIGNAL_SUBTITLE}\n"
        "Conditioned on reconstruction (N_owned≥1) — pure splitting measure",
        fontsize=10
    )
    ax.legend(framealpha=0.5)
    ax.text(0.01, 1.01, f"ILD ILDConfig reco — {RUN_LABEL}",
            transform=ax.transAxes, fontsize=8, color="gray")

    plt.tight_layout()
    out = f"{PLOT_DIR}/fig3_splitting_multiplicity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  Saved: fig3_splitting_multiplicity.png")


# ── Figure 4: Reconstruction efficiency ──────────────────────────────────────
def make_fig_efficiency(g4, cc3):
    fig, ax = plt.subplots(figsize=(8, 5))

    for d, kw in [(g4, G4_KW), (cc3, CC3_KW)]:
        mask = ~np.isnan(d["eff"])
        ax.errorbar(
            E_CENTERS[mask], d["eff"][mask],
            yerr=d["eff_err"][mask],
            **{k: v for k, v in kw.items() if k != "linewidth"},
            lw=kw["linewidth"],
            capsize=4,
        )

    ax.set_xlabel("MC Truth Photon Energy  $E_\\mathrm{mc}$  [GeV]")
    ax.set_ylabel("Reconstruction efficiency  $\\varepsilon(E_\\mathrm{mc})$")
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, color="black", lw=0.8, ls=":", alpha=0.5)
    ax.set_title(
        "Photon Reconstruction Efficiency vs $E_\\mathrm{mc}$ — G4 vs CC3\n"
        f"{SIGNAL_SUBTITLE}\n"
        "Reconstructed = truth photon has ≥1 owned reco PFO with PDG==22",
        fontsize=10
    )
    ax.legend(framealpha=0.5)
    ax.text(0.01, 1.01, f"ILD ILDConfig reco — {RUN_LABEL} — binomial errors",
            transform=ax.transAxes, fontsize=8, color="gray")

    for b in range(len(E_CENTERS)):
        nd_g4  = g4["eff_denom"][b]
        nd_cc3 = cc3["eff_denom"][b]
        if nd_g4 > 0:
            ax.text(E_CENTERS[b], 0.04,
                    f"G4:{nd_g4}\nCC3:{nd_cc3}",
                    ha="center", fontsize=7, color="gray")

    plt.tight_layout()
    out = f"{PLOT_DIR}/fig4_reco_efficiency.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  Saved: fig4_reco_efficiency.png")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  single_photon_observables.py  v2")
print("  Single-photon benchmarking: 5 observables, 4 figures")
print("  CC3: full sample (no isHandledByFastSim filter)")
print("  Multiplicity: conditioned on reconstruction")
print("  Error bars: sigma/sqrt(N) on width and multiplicity")
print("="*60)

g4_data  = run(G4_FILE,  "Geant4", is_cc3=False)
cc3_data = run(CC3_FILE, "CC3",    is_cc3=True)

print("\nGenerating figures...")
make_fig_response(g4_data, cc3_data)
make_fig_cluster_width(g4_data, cc3_data)
make_fig_multiplicity(g4_data, cc3_data)
make_fig_efficiency(g4_data, cc3_data)

print(f"\n  All 4 figures saved to: {PLOT_DIR}")
print(f"  Done.\n")
