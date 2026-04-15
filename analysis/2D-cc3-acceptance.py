"""
cc3_acceptance_2d.py
====================
2D geometric acceptance map of the CaloClouds3 fast-sim model.

Produces a 2D histogram: θ (x-axis) vs φ (y-axis), color = fast-sim handled fraction.
Dead zones (fraction = 0) appear white. Fully active zones appear dark red/orange.

Same signal selection as cc3_acceptance_scan.py:
  - PDG == 22 (photon)
  - generatorStatus == 1
  - E >= 10 GeV
  - parent PDG != ±11  (ISR rejection — direct parent only)

Assumptions stated explicitly:
  - θ barrel-endcap boundaries marked at 35° and 145° (read from 1D scan results)
  - φ octagon module boundaries assumed at -157.5° + every 45°  ← UNVERIFIED
    against ILD geometry XML. These are approximate markers only.

Output: ~/thesis-ml-sim/plots/cc3_acceptance/acceptance_2d.png

Configuration:
  BIN_WIDTH_THETA and BIN_WIDTH_PHI at the top — same as 1D scan script.

Run with:
    source ~/source.sh
    python3 cc3_acceptance_2d.py
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
BIN_WIDTH_THETA = 5.0    # degrees — change for finer/coarser resolution
BIN_WIDTH_PHI   = 5.0    # degrees

MIN_ENTRIES_PER_BIN = 3  # bins with fewer photons shown as hatched (low stats)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
CC_FILE  = "/data/dust/user/alimuham/thesis/sim/tau-pi0-caloclouds-V2-sim.edm4hep.root"
PLOT_DIR = os.path.expanduser("~/thesis-ml-sim/plots/cc3_acceptance")
os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PHOTON_PDG  = 22
E_THRESHOLD = 10.0   # GeV

# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 12,
    "axes.titlesize"   : 11,
    "axes.labelsize"   : 12,
    "figure.dpi"       : 150,
})

# ─────────────────────────────────────────────────────────────────────────────
# COLORMAP
# White (0) → light orange → dark red (1)
# Consistent with CC3 color scheme. Dead zones are immediately obvious as white.
# ─────────────────────────────────────────────────────────────────────────────
CC3_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "cc3_acceptance",
    ["#ffffff", "#f4a582", "#d6604d", "#8b0000"],
    N=256
)
CC3_CMAP.set_bad(color="#dddddd")   # gray for masked (low-stats) bins

# ─────────────────────────────────────────────────────────────────────────────
# PARENT PDG CHECK (one level only — ISR photons are direct e+/e- daughters)
# ─────────────────────────────────────────────────────────────────────────────
def is_from_elecpos(particle):
    return any(abs(p.getPDG()) == 11 for p in particle.getParents())

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
def load_cc3(path):
    reader = root_io.Reader(path)
    events = reader.get("events")

    theta_list      = []
    phi_list        = []
    is_fastsim_list = []

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

    return (np.array(theta_list),
            np.array(phi_list),
            np.array(is_fastsim_list, dtype=bool))

# ─────────────────────────────────────────────────────────────────────────────
# BUILD 2D ACCEPTANCE MAP
# Returns: fraction_map (masked array), total_map, bin edges
# Bins with < MIN_ENTRIES_PER_BIN are masked (gray in plot)
# ─────────────────────────────────────────────────────────────────────────────
def build_2d_map(theta, phi, is_fastsim, bins_theta, bins_phi):
    n_t = len(bins_theta) - 1
    n_p = len(bins_phi)   - 1

    total_map   = np.zeros((n_p, n_t), dtype=float)
    handled_map = np.zeros((n_p, n_t), dtype=float)

    for th, ph, fs in zip(theta, phi, is_fastsim):
        it = np.searchsorted(bins_theta, th, side="right") - 1
        ip = np.searchsorted(bins_phi,   ph, side="right") - 1

        # Clamp to valid range (catches edge values exactly at max)
        it = min(it, n_t - 1)
        ip = min(ip, n_p - 1)

        if 0 <= it < n_t and 0 <= ip < n_p:
            total_map[ip, it]   += 1
            handled_map[ip, it] += float(fs)

    # Compute fraction, mask low-stats bins
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction_map = np.where(total_map > 0, handled_map / total_map, np.nan)

    masked_fraction = np.ma.masked_where(
        total_map < MIN_ENTRIES_PER_BIN, fraction_map
    )

    return masked_fraction, total_map

# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────
def plot_2d(fraction_map, total_map, bins_theta, bins_phi, n_signal, n_handled):

    fig, ax = plt.subplots(figsize=(13, 7))

    # pcolormesh expects (phi edges) x (theta edges)
    im = ax.pcolormesh(
        bins_theta, bins_phi, fraction_map,
        cmap=CC3_CMAP, vmin=0.0, vmax=1.0,
        shading="flat"
    )

    # ── Colorbar ─────────────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("Fast-sim fraction", fontsize=11)
    cbar.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])

    # ── θ boundary markers (from 1D scan — confirmed) ─────────────────────
    # Zeros confirmed at: 0-10°, 35-40°, 140-145°, 170-180°
    theta_boundaries = [
        (10.0,  "beam\nedge", "left"),
        (37.5,  "EC/barrel\ncrack", "left"),
        (142.5, "barrel/EC\ncrack", "left"),
        (170.0, "beam\nedge", "left"),
    ]
    for xv, lbl, ha in theta_boundaries:
        ax.axvline(xv, color="dodgerblue", linewidth=1.2,
                   linestyle="--", alpha=0.8)
        ax.text(xv + 0.5, 175, lbl, color="dodgerblue",
                fontsize=7, va="top", ha="left")

    # ── φ octagon boundary markers (ASSUMED — not verified against geometry) ─
    # Assumption: ILD ECal barrel is a regular octagon.
    # Assumed boundaries at -157.5° + k*45° for k=0..7.
    # *** These positions are NOT verified against ILD_l5_v02 geometry XML. ***
    phi_assumed = [-157.5 + 45*k for k in range(8)]
    for i, yv in enumerate(phi_assumed):
        lbl = "assumed\nmodule boundary" if i == 0 else ""
        ax.axhline(yv, color="gray", linewidth=0.8,
                   linestyle=":", alpha=0.7)
    # Single legend entry for all φ lines
    ax.axhline(phi_assumed[0], color="gray", linewidth=0.8,
               linestyle=":", alpha=0.7,
               label="φ module boundaries\n(assumed, unverified)")

    # ── Labels and title ─────────────────────────────────────────────────────
    ax.set_xlabel("MC photon θ [deg]")
    ax.set_ylabel("MC photon φ [deg]")
    ax.set_xlim(bins_theta[0],  bins_theta[-1])
    ax.set_ylim(bins_phi[0],    bins_phi[-1])

    ax.set_title(
        f"CC3 fast-sim acceptance map — θ vs φ\n"
        f"Signal: genStat=1, E≥{E_THRESHOLD} GeV, parent PDG ≠ ±11 (ISR rejected)\n"
        f"Bin: {BIN_WIDTH_THETA:.0f}° × {BIN_WIDTH_PHI:.0f}°  |  "
        f"N_signal={n_signal}, N_handled={n_handled} ({100*n_handled/n_signal:.1f}%)\n"
        f"ILD sim V2 — 1k events (preliminary)  |  "
        f"Gray = < {MIN_ENTRIES_PER_BIN} photons in bin",
        fontsize=10
    )

    # ── θ tick marks every 15° ────────────────────────────────────────────────
    ax.set_xticks(np.arange(0, 181, 15))
    ax.set_yticks(np.arange(-180, 181, 45))

    ax.legend(loc="lower right", fontsize=8, framealpha=0.6)

    ax.text(0.01, 1.005,
            "Blue dashed = θ boundaries confirmed from 1D scan  |  "
            "Gray dotted = φ boundaries ASSUMED (verify against geometry XML)",
            transform=ax.transAxes, fontsize=7, color="gray", va="bottom")

    plt.tight_layout()
    outpath = f"{PLOT_DIR}/acceptance_2d.png"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  acceptance_2d.png")
    return outpath

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  CC3 Geometric Acceptance — 2D Map")
print("="*65)

print("\nLoading CC3 sim file...")
theta, phi, is_fastsim = load_cc3(CC_FILE)

bins_theta = np.arange(0,    180 + BIN_WIDTH_THETA, BIN_WIDTH_THETA)
bins_phi   = np.arange(-180, 180 + BIN_WIDTH_PHI,   BIN_WIDTH_PHI)

print("\nBuilding 2D acceptance map...")
fraction_map, total_map = build_2d_map(theta, phi, is_fastsim,
                                        bins_theta, bins_phi)

n_signal  = len(theta)
n_handled = int(np.sum(is_fastsim))

print(f"  θ bins : {len(bins_theta)-1}  ({BIN_WIDTH_THETA:.0f}° each)")
print(f"  φ bins : {len(bins_phi)-1}  ({BIN_WIDTH_PHI:.0f}° each)")
print(f"  Total cells        : {(len(bins_theta)-1) * (len(bins_phi)-1)}")
print(f"  Low-stats (masked) : "
      f"{int(np.sum(total_map < MIN_ENTRIES_PER_BIN))} cells "
      f"(< {MIN_ENTRIES_PER_BIN} photons)")

print("\nPlotting...")
plot_2d(fraction_map, total_map, bins_theta, bins_phi, n_signal, n_handled)

print(f"\n  Saved to: {PLOT_DIR}/acceptance_2d.png\n")
