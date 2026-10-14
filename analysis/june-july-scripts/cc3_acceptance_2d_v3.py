"""
cc3_acceptance_full_v3.py
==========================
Full geometric acceptance analysis of the CaloClouds3 (CC3) fast-sim model.

1. 2D fast-sim fraction map (theta vs phi)  — reference, same as before
2. Theta histograms, phi-crack contamination removed:
     2a. full theta   (phi-safe cut)
     2b. barrel only  (phi-safe AND theta in [37.5,142.5])
     2c. endcap only  (phi-safe AND theta outside [33,147])
3. Phi histograms, theta-buffer contamination removed:
     3a. full phi         (theta-safe cut)
     3b. barrel-theta only
     3c. endcap-theta only

Run:
    source ~/source.sh
    python3 cc3_acceptance_full_v3.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from podio import root_io
import edm4hep
import os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BIN_WIDTH_THETA = 5.0
BIN_WIDTH_PHI   = 5.0
MIN_ENTRIES_PER_BIN = 3

CC_FILE  = "/data/dust/user/alimuham/thesis/sim/tau-pi0-caloclouds-100kevents-sim.edm4hep.root"
PLOT_DIR = os.path.expanduser("~/thesis-ml-sim/plots/cc3_acceptance_v3")
os.makedirs(PLOT_DIR, exist_ok=True)

PHOTON_PDG  = 22
E_THRESHOLD = 10.0

# Theta boundaries confirmed from 1D scan (dodgerblue lines, original script)
THETA_BEAM_EDGE_LO = 10.0
THETA_EC_BARREL_LO = 37.5
THETA_BARREL_EC_HI = 142.5
THETA_BEAM_EDGE_HI = 170.0

# McKeown BIB-AE reference boundaries (DESY-THESIS-2024-008)
THETA_ENDCAP_EDGE_LO  = 33.0
THETA_BARREL_START    = 40.0
THETA_BARREL_END      = 140.0
THETA_ENDCAP_EDGE_HI  = 147.0

# Phi octagon module boundaries — ASSUMED, unverified against geometry XML
PHI_MODULE_LINES = [-157.5 + 45 * k for k in range(8)]
PHI_MARGIN = 9.0   # deg safety margin for phi-safe cut

plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 12,
    "axes.titlesize"   : 11,
    "axes.labelsize"   : 12,
    "figure.dpi"       : 150,
})

CC3_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "cc3_acceptance",
    ["#ffffff", "#f4a582", "#d6604d", "#8b0000"],
    N=256
)
CC3_CMAP.set_bad(color="#dddddd")

# ─────────────────────────────────────────────────────────────────────────────
def is_from_elecpos(particle):
    return any(abs(p.getPDG()) == 11 for p in particle.getParents())

def load_cc3(path):
    reader = root_io.Reader(path)
    events = reader.get("events")

    theta_list, phi_list, is_fastsim_list = [], [], []

    n_events = 0
    for event in events:
        n_events += 1
        for p in event.get("MCParticles"):
            if p.getPDG() != PHOTON_PDG:
                continue
            if p.getGeneratorStatus() != 1:
                continue
            if p.getEnergy() < E_THRESHOLD:
                continue
            if is_from_elecpos(p):
                continue

            p4 = edm4hep.utils.p4(p)
            theta_list.append(np.degrees(float(p4.theta())))
            phi_list.append(np.degrees(float(p4.phi())))
            is_fastsim_list.append(bool(p.isHandledByFastSim()))

    print(f"  Events processed : {n_events}")
    print(f"  Signal photons   : {len(theta_list)}  "
          f"(genStat=1, E>={E_THRESHOLD} GeV, ISR rejected)")
    print(f"  Fast-sim handled : {int(np.sum(is_fastsim_list))}  "
          f"({100*np.mean(is_fastsim_list):.1f}%)")

    return (np.array(theta_list), np.array(phi_list),
            np.array(is_fastsim_list, dtype=bool))

def build_2d_map(theta, phi, is_fastsim, bins_theta, bins_phi):
    n_t = len(bins_theta) - 1
    n_p = len(bins_phi)   - 1
    total_map   = np.zeros((n_p, n_t), dtype=float)
    handled_map = np.zeros((n_p, n_t), dtype=float)

    for th, ph, fs in zip(theta, phi, is_fastsim):
        it = min(np.searchsorted(bins_theta, th, side="right") - 1, n_t - 1)
        ip = min(np.searchsorted(bins_phi,   ph, side="right") - 1, n_p - 1)
        if 0 <= it < n_t and 0 <= ip < n_p:
            total_map[ip, it]   += 1
            handled_map[ip, it] += float(fs)

    with np.errstate(invalid="ignore", divide="ignore"):
        fraction_map = np.where(total_map > 0, handled_map / total_map, np.nan)
    masked_fraction = np.ma.masked_where(total_map < MIN_ENTRIES_PER_BIN, fraction_map)
    return masked_fraction, total_map

def plot_2d(fraction_map, total_map, bins_theta, bins_phi, n_signal, n_handled):
    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.pcolormesh(bins_theta, bins_phi, fraction_map,
                        cmap=CC3_CMAP, vmin=0.0, vmax=1.0, shading="flat")
    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("Fast-sim fraction", fontsize=11)
    cbar.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])

    theta_boundaries = [
        (THETA_BEAM_EDGE_LO, "beam\nedge"),
        (THETA_EC_BARREL_LO, "EC/barrel\ncrack"),
        (THETA_BARREL_EC_HI, "barrel/EC\ncrack"),
        (THETA_BEAM_EDGE_HI, "beam\nedge"),
    ]
    for xv, lbl in theta_boundaries:
        ax.axvline(xv, color="dodgerblue", linewidth=1.2, linestyle="--", alpha=0.8)
        ax.text(xv + 0.5, 175, lbl, color="dodgerblue", fontsize=7, va="top", ha="left")

    for tv in [THETA_ENDCAP_EDGE_LO, THETA_BARREL_START, THETA_BARREL_END, THETA_ENDCAP_EDGE_HI]:
        ax.axvline(tv, color="purple", linewidth=1.0, linestyle=":", alpha=0.7)

    for i, yv in enumerate(PHI_MODULE_LINES):
        ax.axhline(yv, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.axhline(PHI_MODULE_LINES[0], color="gray", linewidth=0.8, linestyle=":",
               alpha=0.7, label="φ module boundaries\n(assumed, unverified)")

    ax.set_xlabel("MC photon θ [deg]")
    ax.set_ylabel("MC photon φ [deg]")
    ax.set_xlim(bins_theta[0], bins_theta[-1])
    ax.set_ylim(bins_phi[0], bins_phi[-1])
    ax.set_title(
        f"CC3 fast-sim acceptance map — θ vs φ\n"
        f"Signal: genStat=1, E≥{E_THRESHOLD} GeV, parent PDG ≠ ±11 (ISR rejected)\n"
        f"Bin: {BIN_WIDTH_THETA:.0f}° × {BIN_WIDTH_PHI:.0f}°  |  "
        f"N_signal={n_signal}, N_handled={n_handled} ({100*n_handled/n_signal:.1f}%)\n"
        f"ILD sim V2 — 100k events  |  Gray = < {MIN_ENTRIES_PER_BIN} photons in bin",
        fontsize=10)
    ax.set_xticks(np.arange(0, 181, 15))
    ax.set_yticks(np.arange(-180, 181, 45))
    ax.legend(loc="lower right", fontsize=8, framealpha=0.6)
    ax.text(0.01, 1.005,
            "Blue dashed = θ boundaries (1D scan)  |  Purple dotted = McKeown 33/40/140/147°  |  "
            "Gray dotted = φ boundaries ASSUMED",
            transform=ax.transAxes, fontsize=7, color="gray", va="bottom")

    plt.tight_layout()
    fig.savefig(f"{PLOT_DIR}/01_acceptance_2d.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓  01_acceptance_2d.png")

# ─── Isolation masks ─────────────────────────────────────────────────────────
def phi_safe_mask(phi_arr, margin=PHI_MARGIN):
    safe = np.ones_like(phi_arr, dtype=bool)
    for edge in PHI_MODULE_LINES:
        d = np.abs(((phi_arr - edge + 180) % 360) - 180)
        safe &= (d > margin)
    return safe

def theta_safe_mask(theta_arr):
    in_buf_lo = (theta_arr >= THETA_ENDCAP_EDGE_LO) & (theta_arr <= THETA_BARREL_START)
    in_buf_hi = (theta_arr >= THETA_BARREL_END) & (theta_arr <= THETA_ENDCAP_EDGE_HI)
    return ~(in_buf_lo | in_buf_hi)

def barrel_mask(theta_arr):
    return (theta_arr >= THETA_EC_BARREL_LO) & (theta_arr <= THETA_BARREL_EC_HI)

def endcap_mask(theta_arr):
    return (theta_arr <= THETA_ENDCAP_EDGE_LO) | (theta_arr >= THETA_ENDCAP_EDGE_HI)

def fastsim_fraction_1d(values, is_fs, bins):
    """UNCHANGED analysis logic — now also returns n_fs (was already computed internally,
    just wasn't being returned before)."""
    centers = 0.5 * (bins[:-1] + bins[1:])
    n_tot, _ = np.histogram(values, bins=bins)
    n_fs,  _ = np.histogram(values[is_fs], bins=bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(n_tot > 0, n_fs / n_tot, np.nan)
    err = np.where(n_tot > 0, np.sqrt(frac * (1 - frac) / np.maximum(n_tot, 1)), np.nan)
    return centers, frac, err, n_tot, n_fs

# ─── Plotting (visual style updated to match reference figure) ──────────────
def plot_fraction_1d(centers, frac, err, n_tot, n_fs, xlabel, title, filename,
                      vlines=None, vline_color="dodgerblue", vline_labels=None):
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]})
    fig.subplots_adjust(hspace=0.08)

    width = centers[1] - centers[0]
    # Total signal photons (gray) + fast-sim handled overlay (orange) — two-layer overlay
    ax_top.bar(centers, n_tot, width=width * 0.9, color="gray", alpha=0.6,
               label=f"Signal photons (total, N={int(np.nansum(n_tot))})")
    ax_top.bar(centers, n_fs, width=width * 0.9, color="#f4a582", alpha=0.9,
               label=f"Fast-sim handled (N={int(np.nansum(n_fs))})")
    ax_top.set_ylabel("Photons / bin")
    ax_top.set_title(title, fontsize=10)
    ax_top.legend(fontsize=9, loc="upper right")

    ax_bot.errorbar(centers, frac, yerr=err, fmt="o", color="#d6604d",
                     markersize=4, label="Fast-sim fraction")
    ax_bot.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50% reference")
    ax_bot.set_ylim(-0.05, 1.05)
    ax_bot.set_ylabel("Fast-sim fraction")
    ax_bot.set_xlabel(xlabel)
    ax_bot.legend(fontsize=9, loc="upper right")

    if vlines:
        for ax in (ax_top, ax_bot):
            for v in vlines:
                ax.axvline(v, color=vline_color, linestyle=":", linewidth=1.0, alpha=0.8)
        if vline_labels:
            ymax = ax_top.get_ylim()[1]
            for v, lbl in zip(vlines, vline_labels):
                ax_top.text(v, ymax * 0.95, lbl, color=vline_color, fontsize=8,
                            rotation=90, va="top", ha="right")

    plt.tight_layout()
    fig.savefig(f"{PLOT_DIR}/{filename}", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {filename}")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  CC3 Geometric Acceptance — Full Analysis (100k events)")
print("="*65)

print("\nLoading CC3 sim file...")
theta, phi, is_fastsim = load_cc3(CC_FILE)

bins_theta = np.arange(0, 180 + BIN_WIDTH_THETA, BIN_WIDTH_THETA)
bins_phi   = np.arange(-180, 180 + BIN_WIDTH_PHI, BIN_WIDTH_PHI)

print("\nBuilding 2D acceptance map...")
fraction_map, total_map = build_2d_map(theta, phi, is_fastsim, bins_theta, bins_phi)
n_signal, n_handled = len(theta), int(np.sum(is_fastsim))

print(f"  θ bins : {len(bins_theta)-1}  |  φ bins : {len(bins_phi)-1}")
print(f"  Low-stats (masked) cells : {int(np.sum(total_map < MIN_ENTRIES_PER_BIN))}")

print("\nPlotting 2D map...")
plot_2d(fraction_map, total_map, bins_theta, bins_phi, n_signal, n_handled)

print("\nBuilding theta/phi isolation histograms...")
THETA_BINS_1D = np.arange(0, 181, 2.5)
PHI_BINS_1D   = np.arange(-180, 181, 5.0)

m_phisafe   = phi_safe_mask(phi)
m_thetasafe = theta_safe_mask(theta)

for tag, extra_mask, title, tvlines, tlabels in [
    ("2a_theta_full_phisafe.png", np.ones_like(theta, dtype=bool),
     "θ acceptance (φ-crack removed) — full range",
     [33, 40, 140, 147], ["endcap edge", "EC-barrel", "barrel-EC", "endcap edge"]),
    ("2b_theta_barrel_phisafe.png", barrel_mask(theta),
     "θ acceptance (φ-crack removed) — barrel only [37.5°,142.5°]",
     [33, 40, 140, 147], ["endcap edge", "EC-barrel", "barrel-EC", "endcap edge"]),
    ("2c_theta_endcap_phisafe.png", endcap_mask(theta),
     "θ acceptance (φ-crack removed) — endcap only [<33° or >147°]",
     [33, 40, 140, 147], ["endcap edge", "EC-barrel", "barrel-EC", "endcap edge"]),
]:
    mask = m_phisafe & extra_mask
    c, f, e, n_tot, n_fs = fastsim_fraction_1d(theta[mask], is_fastsim[mask], THETA_BINS_1D)
    plot_fraction_1d(c, f, e, n_tot, n_fs, "MC photon θ [deg]", title, tag,
                      vlines=tvlines, vline_color="dodgerblue", vline_labels=tlabels)

for tag, extra_mask, title in [
    ("3a_phi_full_thetasafe.png", np.ones_like(theta, dtype=bool),
     "φ acceptance (θ-buffer removed) — full range"),
    ("3b_phi_barrel_thetasafe.png", barrel_mask(theta),
     "φ acceptance (θ-buffer removed) — barrel θ only"),
    ("3c_phi_endcap_thetasafe.png", endcap_mask(theta),
     "φ acceptance (θ-buffer removed) — endcap θ only"),
]:
    mask = m_thetasafe & extra_mask
    c, f, e, n_tot, n_fs = fastsim_fraction_1d(phi[mask], is_fastsim[mask], PHI_BINS_1D)
    plot_fraction_1d(c, f, e, n_tot, n_fs, "MC photon φ [deg]", title, tag,
                      vlines=PHI_MODULE_LINES, vline_color="gray")

print(f"\nAll plots saved to: {PLOT_DIR}\n")
