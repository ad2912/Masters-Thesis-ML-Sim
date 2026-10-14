"""
phi_margin_convergence_scan.py
================================
General-purpose method to justify an exclusion-margin parameter using a
data-driven convergence scan, instead of picking it by eye.

-------------------------------------------------------------------------
PHYSICS QUESTION
-------------------------------------------------------------------------
We want to measure the CC3 fast-sim handling fraction as a function of
photon theta, in a region of theta that is NOT affected by detector
geometry cracks in phi (the ILD ECal barrel is an 8-fold octagon; photons
travelling close to a module boundary in phi are not a fair test of theta
acceptance).

To do this cleanly we must exclude photons within some angular distance
("margin") of each phi module boundary. The size of that margin is a
free parameter -- and it should be justified by data, not guessed.

-------------------------------------------------------------------------
METHOD (general, reusable for any "exclusion margin" problem)
-------------------------------------------------------------------------
1. Pick a *reference observable* whose value should be stable / unbiased
   once contamination is properly removed. Here: fast-sim fraction in a
   clean interior theta band (60-120 deg), far from any theta-related
   transition, so only the phi-margin choice can move it.

2. Scan the margin parameter over a physically reasonable range.

3. At each margin value, recompute the reference observable with binomial
   (efficiency-style) uncertainty.

4. Convergence criterion: the observable has "converged" once it stops
   changing by more than 1 sigma across several consecutive margin steps.
   The smallest margin satisfying this is the data-justified choice.

5. Falsifiability: if the observable never plateaus, the margin
   hypothesis is wrong -- the effect is not a simple crack-margin issue,
   and must be investigated differently (e.g. genuine model inefficiency).

This method generalizes to ANY situation where you must choose an
exclusion window around a known geometric or instrumental feature
(dead channels, module edges, trigger thresholds, etc).
-------------------------------------------------------------------------

Run:
    source ~/source.sh
    python3 phi_margin_convergence_scan.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio import root_io
import edm4hep
import os

# =============================================================================
# CONFIGURATION -- edit these for your specific case
# =============================================================================

CC_FILE  = "/data/dust/user/alimuham/thesis/sim/tau-pi0-caloclouds-100kevents-sim.edm4hep.root"
PLOT_DIR = os.path.expanduser("~/thesis-ml-sim/plots/margin_convergence")
os.makedirs(PLOT_DIR, exist_ok=True)

PHOTON_PDG  = 22
E_THRESHOLD = 10.0

# Reference band in the *orthogonal* coordinate (theta), chosen to be
# far from any theta-related transition (barrel-endcap crack), so the
# ONLY thing that can move this number is the phi-margin choice.
REFERENCE_BAND_LOW  = 60.0
REFERENCE_BAND_HIGH = 120.0

# Known geometric feature positions (phi module boundaries).
# Generalizable: replace with any list of "known contamination positions".
FEATURE_LINES = [-157.5 + 45 * k for k in range(8)]

# Margin scan range
MARGIN_MIN  = 1.0
MARGIN_MAX  = 20.0
MARGIN_STEP = 1.0

# Convergence criterion
CONVERGENCE_SIGMA   = 1.0   # max allowed change, in units of sigma
CONVERGENCE_STREAK  = 3     # consecutive steps required to declare converged

plt.rcParams.update({
    "font.family"    : "serif",
    "font.size"      : 12,
    "axes.titlesize" : 11,
    "axes.labelsize" : 12,
    "figure.dpi"     : 150,
})

# =============================================================================
# LOADER -- same signal definition as the rest of the acceptance study
# =============================================================================
def is_from_elecpos(particle):
    return any(abs(p.getPDG()) == 11 for p in particle.getParents())

def load_signal_photons(path):
    reader = root_io.Reader(path)
    theta_list, phi_list, is_fs_list = [], [], []

    for event in reader.get("events"):
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
            is_fs_list.append(bool(p.isHandledByFastSim()))

    return (np.array(theta_list), np.array(phi_list),
            np.array(is_fs_list, dtype=bool))

# =============================================================================
# GENERAL-PURPOSE MARGIN SCAN FUNCTION
# =============================================================================
def angular_distance_to_nearest_feature(values, feature_lines, period=360.0):
    """
    For each entry in `values`, compute the angular distance to the
    nearest feature line, correctly handling wrap-around (e.g. phi
    going from +180 to -180).
    """
    values = np.asarray(values)
    dists = np.full(values.shape, np.inf)
    for f in feature_lines:
        d = np.abs(((values - f + period/2) % period) - period/2)
        dists = np.minimum(dists, d)
    return dists

def compute_reference_fraction(theta, phi, is_fs, margin,
                                band_low, band_high, feature_lines):
    """
    Reference observable: fast-sim fraction inside [band_low, band_high]
    in theta, using only photons whose phi is at least `margin` degrees
    from every known feature line.

    Returns: fraction, n_total, binomial_sigma
    """
    in_band  = (theta >= band_low) & (theta <= band_high)
    dist     = angular_distance_to_nearest_feature(phi, feature_lines)
    is_safe  = dist > margin

    mask = in_band & is_safe
    n_total = mask.sum()
    n_fs    = (mask & is_fs).sum()

    if n_total == 0:
        return np.nan, 0, np.nan

    frac  = n_fs / n_total
    sigma = np.sqrt(frac * (1 - frac) / n_total)
    return frac, n_total, sigma

def run_margin_scan(theta, phi, is_fs, margins,
                     band_low, band_high, feature_lines):
    fracs, sigmas, ntots = [], [], []
    for m in margins:
        f, n, s = compute_reference_fraction(
            theta, phi, is_fs, m, band_low, band_high, feature_lines)
        fracs.append(f)
        sigmas.append(s)
        ntots.append(n)
    return np.array(fracs), np.array(sigmas), np.array(ntots)

def find_convergence_margin(margins, fracs, sigmas,
                             sigma_thresh=CONVERGENCE_SIGMA,
                             streak_required=CONVERGENCE_STREAK):
    """
    Walk the scan from small to large margin. Declare convergence at the
    first margin where the fraction stops moving by more than
    `sigma_thresh` sigma for `streak_required` consecutive steps.

    Returns the converged margin, or None if never converged (falsifies
    the "it's just a margin effect" hypothesis).
    """
    streak = 0
    for i in range(1, len(margins)):
        delta = abs(fracs[i] - fracs[i - 1])
        combined_sigma = np.sqrt(sigmas[i]**2 + sigmas[i - 1]**2)
        if combined_sigma == 0 or np.isnan(combined_sigma):
            continue
        if delta < sigma_thresh * combined_sigma:
            streak += 1
            if streak >= streak_required:
                return margins[i - streak_required + 1]
        else:
            streak = 0
    return None

# =============================================================================
# PLOT
# =============================================================================
def plot_convergence(margins, fracs, sigmas, ntots, converged_margin, filename):
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})
    fig.subplots_adjust(hspace=0.08)

    ax_top.errorbar(margins, fracs, yerr=sigmas, fmt="o-", color="#d6604d",
                     markersize=5, capsize=3,
                     label="Reference fast-sim fraction\n"
                           f"(theta in [{REFERENCE_BAND_LOW:.0f},"
                           f"{REFERENCE_BAND_HIGH:.0f}] deg)")
    if converged_margin is not None:
        ax_top.axvline(converged_margin, color="black", linestyle="--",
                       linewidth=1.2,
                       label=f"Converged margin = {converged_margin:.0f} deg")
    ax_top.set_ylabel("Fast-sim fraction")
    ax_top.set_title(
        "Margin convergence scan\n"
        "Question: how wide must the exclusion window around each\n"
        "known geometric feature be before the reference fraction stabilizes?",
        fontsize=10)
    ax_top.legend(fontsize=9, loc="lower right")
    ax_top.set_ylim(0, 1.05)

    ax_bot.plot(margins, ntots, "o-", color="gray", markersize=4)
    ax_bot.set_ylabel("N photons\nin band")
    ax_bot.set_xlabel("Exclusion margin [deg]")

    plt.tight_layout()
    out = f"{PLOT_DIR}/{filename}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")

# =============================================================================
# MAIN
# =============================================================================
print("\n" + "="*65)
print("  Margin convergence scan")
print("="*65)

print("\nLoading signal photons...")
theta, phi, is_fs = load_signal_photons(CC_FILE)
print(f"  Loaded {len(theta)} signal photons "
      f"(genStat=1, E>={E_THRESHOLD} GeV, ISR rejected)")

margins = np.arange(MARGIN_MIN, MARGIN_MAX + MARGIN_STEP, MARGIN_STEP)

print(f"\nScanning margin from {MARGIN_MIN} to {MARGIN_MAX} deg "
      f"in steps of {MARGIN_STEP} deg...")
fracs, sigmas, ntots = run_margin_scan(
    theta, phi, is_fs, margins,
    REFERENCE_BAND_LOW, REFERENCE_BAND_HIGH, FEATURE_LINES)

for m, f, s, n in zip(margins, fracs, sigmas, ntots):
    print(f"  margin={m:5.1f} deg | fraction={f:.4f} +- {s:.4f} | N={n}")

converged_margin = find_convergence_margin(margins, fracs, sigmas)

print()
if converged_margin is not None:
    print(f"  RESULT: fraction converges at margin = {converged_margin:.1f} deg")
    print(f"  This is the data-justified PHI_MARGIN to use downstream.")
else:
    print("  RESULT: fraction never converges within the scanned range.")
    print("  This FALSIFIES the 'simple margin' hypothesis -- the residual")
    print("  loss is not explained by phi-crack proximity alone. Investigate")
    print("  a genuine theta-independent CC3 inefficiency instead.")

plot_convergence(margins, fracs, sigmas, ntots, converged_margin,
                  "margin_convergence_scan.png")

print(f"\nDone. Plot saved to: {PLOT_DIR}/margin_convergence_scan.png\n")
