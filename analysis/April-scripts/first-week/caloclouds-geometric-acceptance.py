"""
cc3_acceptance_scan.py
======================
Empirical geometric acceptance scan of the CaloClouds3 fast-sim model.

Question: In which angular regions (θ, φ) does CC3 fail to fast-simulate
eligible photons, and what fraction of eligible photons are lost?

Sample selection (applied to CC3 sim file only):
  - PDG == 22 (photon)
  - generatorStatus == 1
  - E >= 10 GeV  (CC3 trigger threshold)
  - NOT from e+/e- parent  (ISR rejection — direct parent PDG != ±11)

Two signal definitions are computed and printed for comparison:
  A) ISR-rejected  : parent PDG != ±11  (includes π⁰ daughters + τ FSR)
  B) π⁰-only       : parent PDG == 111  (stricter, for cross-check)

Outputs:
  - Terminal table: bin | total signal | fast-sim handled | fraction  (for both θ and φ)
  - Two-panel figure for θ acceptance
  - Two-panel figure for φ acceptance
  Saved to: ~/thesis-ml-sim/plots/cc3_acceptance/

Configuration:
  Change BIN_WIDTH_THETA and BIN_WIDTH_PHI at the top to adjust scan resolution.

Run with:
    source ~/source.sh
    python3 cc3_acceptance_scan.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio import root_io
import edm4hep
import os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — change these to adjust scan resolution
# ─────────────────────────────────────────────────────────────────────────────
BIN_WIDTH_THETA = 5.0   # degrees — increase for coarser, decrease for finer
BIN_WIDTH_PHI   = 5.0   # degrees

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
CC_FILE  = "/data/dust/user/alimuham/thesis/sim/tau-pi0-caloclouds-V2-sim.edm4hep.root"
PLOT_DIR = os.path.expanduser("~/thesis-ml-sim/plots/cc3_acceptance")
os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PHOTON_PDG   = 22
PI0_PDG      = 111
E_THRESHOLD  = 10.0   # GeV

# Colors — consistent with approved scheme
C_TOTAL  = "#aaaaaa"   # gray  — all eligible signal photons
C_FS     = "#f4a582"   # orange — fast-sim handled
C_FRAC   = "#d6604d"   # red    — fraction line

# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 12,
    "axes.titlesize"   : 12,
    "axes.labelsize"   : 12,
    "legend.fontsize"  : 10,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 150,
})

# ─────────────────────────────────────────────────────────────────────────────
# PARENT PDG CHECKER — one level up only (ISR photons are direct e+/e- daughters)
# ─────────────────────────────────────────────────────────────────────────────
def get_parent_pdgs(particle):
    """Return set of PDG codes of immediate parents."""
    return {abs(p.getPDG()) for p in particle.getParents()}

# ─────────────────────────────────────────────────────────────────────────────
# PI0 ANCESTOR CRAWLER — full chain (reused from angular_distributions_sim.py)
# ─────────────────────────────────────────────────────────────────────────────
def has_pi0_ancestor(particle):
    """Walk full parent chain. Return True if any ancestor has PDG == 111."""
    visited = set()
    stack = list(particle.getParents())
    while stack:
        parent = stack.pop()
        uid = parent.getObjectID().index
        if uid in visited:
            continue
        visited.add(uid)
        if abs(parent.getPDG()) == PI0_PDG:
            return True
        stack.extend(parent.getParents())
    return False

# ─────────────────────────────────────────────────────────────────────────────
# LOAD CC3 FILE
# Returns arrays for all photons passing base cuts (E >= 10 GeV, genStat == 1)
# ─────────────────────────────────────────────────────────────────────────────
def load_cc3(path):
    reader = root_io.Reader(path)
    events = reader.get("events")

    theta_list      = []
    phi_list        = []
    is_fastsim_list = []
    is_isr_list     = []   # True if direct parent is e+/e-
    is_pi0_list     = []   # True if has π⁰ ancestor anywhere in chain

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

            p4    = edm4hep.utils.p4(p)
            theta = np.degrees(float(p4.theta()))
            phi   = np.degrees(float(p4.phi()))

            parent_pdgs = get_parent_pdgs(p)
            is_isr      = 11 in parent_pdgs   # e+ or e- direct parent

            theta_list.append(theta)
            phi_list.append(phi)
            is_fastsim_list.append(bool(p.isHandledByFastSim()))
            is_isr_list.append(is_isr)
            is_pi0_list.append(has_pi0_ancestor(p))

    print(f"  Events processed : {n_events}")
    print(f"  Base sample      : {len(theta_list)}  photons  "
          f"(genStat==1, E>={E_THRESHOLD} GeV)")

    return {
        "theta"     : np.array(theta_list),
        "phi"       : np.array(phi_list),
        "is_fastsim": np.array(is_fastsim_list, dtype=bool),
        "is_isr"    : np.array(is_isr_list,     dtype=bool),
        "is_pi0"    : np.array(is_pi0_list,     dtype=bool),
    }

# ─────────────────────────────────────────────────────────────────────────────
# SCAN + PRINT TABLE
# ─────────────────────────────────────────────────────────────────────────────
def scan_and_table(angle_arr, is_fastsim_arr, signal_mask,
                   bins, angle_name):
    """
    For each bin: count total signal photons and fast-sim handled photons.
    Returns (bin_centers, totals, handled, fractions, frac_errors).
    Also prints a formatted table to terminal.
    """
    sig_angles = angle_arr[signal_mask]
    sig_fs     = is_fastsim_arr[signal_mask]

    bin_centers = []
    totals      = []
    handled_arr = []
    fractions   = []
    frac_errors = []

    header = (f"\n{'─'*62}\n"
              f"  {angle_name} scan — bin width {bins[1]-bins[0]:.1f}°\n"
              f"{'─'*62}\n"
              f"  {'Bin [deg]':>14}  {'Total':>7}  {'Fast-sim':>9}  {'Fraction':>9}  {'±err':>7}\n"
              f"{'─'*62}")
    print(header)

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        in_bin  = (sig_angles >= lo) & (sig_angles < hi)
        total   = int(np.sum(in_bin))
        handled = int(np.sum(sig_fs[in_bin]))

        if total > 0:
            frac = handled / total
            err  = np.sqrt(frac * (1 - frac) / total)   # binomial error
        else:
            frac = 0.0
            err  = 0.0

        center = 0.5 * (lo + hi)
        bin_centers.append(center)
        totals.append(total)
        handled_arr.append(handled)
        fractions.append(frac)
        frac_errors.append(err)

        print(f"  {lo:6.1f} – {hi:6.1f}°  {total:7d}  {handled:9d}  "
              f"{frac:9.3f}  {err:7.3f}")

    print(f"{'─'*62}")
    total_all   = np.sum(signal_mask)
    handled_all = int(np.sum(is_fastsim_arr[signal_mask]))
    overall     = handled_all / total_all if total_all > 0 else 0.0
    print(f"  {'TOTAL':>14}  {total_all:7d}  {handled_all:9d}  {overall:9.3f}\n")

    return (np.array(bin_centers), np.array(totals),
            np.array(handled_arr), np.array(fractions), np.array(frac_errors))

# ─────────────────────────────────────────────────────────────────────────────
# TWO-PANEL ACCEPTANCE FIGURE
# Top:    bar chart — total signal (gray) + fast-sim handled (orange), absolute counts
# Bottom: fast-sim fraction per bin with binomial error bars
# ─────────────────────────────────────────────────────────────────────────────
def plot_acceptance(bin_centers, bin_width, totals, handled,
                    fractions, frac_errors,
                    angle_name, xlabel, title_suffix, filename,
                    vlines=None):
    """
    vlines: list of (x_value, label) tuples for detector boundary markers.
    """
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 7),
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=True
    )
    fig.subplots_adjust(hspace=0.08)

    w = bin_width * 0.8   # bar width slightly narrower than bin

    # ── Top panel: absolute counts ──────────────────────────────────────────
    ax_top.bar(bin_centers, totals,   width=w, color=C_TOTAL,
               label=f"Signal photons (total, N={int(np.sum(totals))})",
               alpha=0.85, zorder=2)
    ax_top.bar(bin_centers, handled, width=w, color=C_FS,
               label=f"Fast-sim handled (N={int(np.sum(handled))})",
               alpha=0.9, zorder=3)

    ax_top.set_ylabel("Photons / bin")
    ax_top.legend(framealpha=0.5, loc="upper right")
    ax_top.set_title(
        f"CC3 geometric acceptance — {angle_name} scan\n"
        f"{title_suffix}\n"
        f"ILD sim V2 — 1k events (preliminary)",
        fontsize=11
    )

    # ── Bottom panel: fraction ───────────────────────────────────────────────
    ax_bot.errorbar(bin_centers, fractions, yerr=frac_errors,
                    fmt="o", color=C_FRAC, markersize=5,
                    linewidth=1.5, capsize=3,
                    label="Fast-sim fraction")
    ax_bot.axhline(0.5, color="gray", linewidth=0.8, linestyle="--",
                   label="50% reference")
    ax_bot.set_ylim(-0.05, 1.15)
    ax_bot.set_ylabel("Fast-sim fraction")
    ax_bot.set_xlabel(xlabel)
    ax_bot.legend(framealpha=0.5, loc="upper right", fontsize=9)

    # ── Detector boundary markers (both panels) ──────────────────────────────
    if vlines:
        for xv, lbl in vlines:
            for ax in (ax_top, ax_bot):
                ax.axvline(xv, color="steelblue", linewidth=1.0,
                           linestyle=":", alpha=0.8)
            # Label only on top panel, at top
            ax_top.text(xv + 0.5, ax_top.get_ylim()[1] * 0.97,
                        lbl, fontsize=7, color="steelblue",
                        va="top", ha="left", rotation=90)

    plt.tight_layout()
    fig.savefig(f"{PLOT_DIR}/{filename}", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {filename}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  CC3 Geometric Acceptance Scan")
print("="*65)

print("\nLoading CC3 sim file...")
data = load_cc3(CC_FILE)

# ── Signal definitions ────────────────────────────────────────────────────────
mask_isr_rej = ~data["is_isr"]                  # A: all non-ISR (includes τ FSR)
mask_pi0     =  data["is_pi0"]                  # B: π⁰ daughters only (strict)

n_isr_rej = int(np.sum(mask_isr_rej))
n_pi0     = int(np.sum(mask_pi0))
n_isr     = int(np.sum(data["is_isr"]))

print(f"\n  Signal comparison (genStat==1, E>={E_THRESHOLD} GeV):")
print(f"  A) ISR-rejected (parent != e+/e-)  : {n_isr_rej} photons")
print(f"  B) π⁰-only      (has π⁰ ancestor)  : {n_pi0}     photons")
print(f"     ISR photons removed by cut A    : {n_isr}")

# Use ISR-rejected as primary signal (more inclusive, includes τ FSR)
signal_mask = mask_isr_rej
print(f"\n  → Using ISR-rejected sample (A) as primary signal.")

# ── Bin edges ────────────────────────────────────────────────────────────────
bins_theta = np.arange(0,   180 + BIN_WIDTH_THETA, BIN_WIDTH_THETA)
bins_phi   = np.arange(-180, 180 + BIN_WIDTH_PHI,  BIN_WIDTH_PHI)

# ── θ scan ───────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  θ SCAN  (ISR-rejected signal)")
print("="*65)

bc_t, tot_t, han_t, frac_t, err_t = scan_and_table(
    data["theta"], data["is_fastsim"], signal_mask,
    bins_theta, "θ"
)

# Detector boundary markers for θ
# Barrel-endcap transition around 40° and 140° — these are approximate
# and the scan itself will tell us the true boundaries
theta_vlines = [
    (25.0,  "endcap edge"),
    (40.0,  "EC→barrel"),
    (140.0, "barrel→EC"),
    (155.0, "endcap edge"),
]

plot_acceptance(
    bc_t, BIN_WIDTH_THETA, tot_t, han_t, frac_t, err_t,
    angle_name   = "θ",
    xlabel       = "MC photon θ [deg]",
    title_suffix = ("Signal: genStat=1, E≥10 GeV, parent PDG ≠ ±11 (ISR rejected)\n"
                    f"Bin width: {BIN_WIDTH_THETA:.0f}°"),
    filename     = "acceptance_theta.png",
    vlines       = theta_vlines,
)

# ── φ scan ───────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  φ SCAN  (ISR-rejected signal)")
print("="*65)

bc_p, tot_p, han_p, frac_p, err_p = scan_and_table(
    data["phi"], data["is_fastsim"], signal_mask,
    bins_phi, "φ"
)

# ECal barrel octagon: 8 module boundaries, equally spaced every 45°
# starting from φ = -180° + 22.5° = -157.5°, then every 45°
phi_boundaries = [(-180 + 22.5 + 45 * i, "") for i in range(8)]

plot_acceptance(
    bc_p, BIN_WIDTH_PHI, tot_p, han_p, frac_p, err_p,
    angle_name   = "φ",
    xlabel       = "MC photon φ [deg]",
    title_suffix = ("Signal: genStat=1, E≥10 GeV, parent PDG ≠ ±11 (ISR rejected)\n"
                    f"Bin width: {BIN_WIDTH_PHI:.0f}° | Expected: 8 dips from octagonal barrel"),
    filename     = "acceptance_phi.png",
    vlines       = phi_boundaries,
)

# ── π⁰-only cross-check table (printed only, no separate plot) ───────────────
print("\n" + "="*65)
print("  θ SCAN  (π⁰-only cross-check, B)")
print("="*65)
scan_and_table(
    data["theta"], data["is_fastsim"], mask_pi0,
    bins_theta, "θ (π⁰-only)"
)

print(f"\n  Plots saved to: {PLOT_DIR}")
print(f"  acceptance_theta.png")
print(f"  acceptance_phi.png\n")
