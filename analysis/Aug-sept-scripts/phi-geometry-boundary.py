# phi_crack_signed_asymmetry_check.py
#
# Purpose: test whether the phi-crack boundary is ASYMMETRIC (reject on one
# rotational side of the crack center, accept all the way to the center on
# the other side) rather than a symmetric two-sided dead zone. The previous
# analysis (phi_crack_width_barrel_only.py) used unsigned |distance|, which
# would blend a permissive side and a strict side into one diluted-looking
# curve -- exactly the unexplained ~40-50% plateau we saw.
#
# Concept: for each photon, find the nearest crack center, then compute a
# SIGNED distance -- positive if the photon sits on one defined rotational
# side of that center, negative if on the other. Plotting fast-sim fraction
# against signed distance (instead of absolute distance) will show a sharp
# asymmetric step if the hypothesis is right, versus a symmetric V/U shape
# if it's wrong.
#
# Same deep-barrel theta restriction as before, to keep theta effects out.
#
# NOT YET RUN.

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio import root_io
import edm4hep
import os

CC3_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root"
MAX_EVENTS = 10000
PLOT_DIR = os.path.expanduser("~/thesis-ml-sim/plots/phi_crack_signed_check")
os.makedirs(PLOT_DIR, exist_ok=True)

PHOTON_PDG = 22
ELECTRON_PDG = 11
E_MC_MIN = 10.0
THETA_DEEP_BARREL_LO = 55.0
THETA_DEEP_BARREL_HI = 125.0

PHI_CRACK_CENTERS_DEG = [-157.5 + 45 * k for k in range(8)]
SIGNED_BIN_WIDTH = 0.5
SIGNED_MAX = 20.0   # scan +/- 20 deg around each crack center
MIN_ENTRIES_PER_BIN = 5


def get_parent_pdg(mc_part):
    parents = mc_part.getParents()
    return abs(parents[0].getPDG()) if len(parents) > 0 else 0


def passes_layer_a(mc_part):
    if mc_part.getGeneratorStatus() != 1:
        return False
    if mc_part.getPDG() != PHOTON_PDG:
        return False
    if mc_part.getEnergy() < E_MC_MIN:
        return False
    if get_parent_pdg(mc_part) == ELECTRON_PDG:
        return False
    return True


def signed_dist_to_nearest_crack(phi_deg):
    """Returns (signed_distance, which_center_index).
    Signed distance is positive if phi is 'ahead of' the crack center going
    counter-clockwise (increasing phi), negative if 'behind' it. This
    convention is arbitrary but MUST be applied consistently -- what matters
    is whether fraction(signed_dist) is asymmetric about 0, not which sign
    label ends up on which physical side."""
    best_center = None
    best_signed = None
    best_abs = 1e9
    for center in PHI_CRACK_CENTERS_DEG:
        raw = ((phi_deg - center + 180) % 360) - 180  # signed, in [-180, 180)
        if abs(raw) < best_abs:
            best_abs = abs(raw)
            best_signed = raw
            best_center = center
    return best_signed, best_center


reader = root_io.Reader(CC3_FILE)

signed_true_cc3 = []
signed_true_g4fb = []

n_events = 0
for event in reader.get("events"):
    if n_events >= MAX_EVENTS:
        break
    n_events += 1
    for mc in event.get("MCParticles"):
        if not passes_layer_a(mc):
            continue
        p4 = edm4hep.utils.p4(mc)
        theta_deg = np.degrees(float(p4.theta()))
        if not (THETA_DEEP_BARREL_LO <= theta_deg <= THETA_DEEP_BARREL_HI):
            continue
        phi_deg = np.degrees(float(p4.phi()))
        signed, _ = signed_dist_to_nearest_crack(phi_deg)
        if abs(signed) > SIGNED_MAX:
            continue
        is_fastsim = bool(mc.isHandledByFastSim())
        if is_fastsim:
            signed_true_cc3.append(signed)
        else:
            signed_true_g4fb.append(signed)

signed_true_cc3 = np.array(signed_true_cc3)
signed_true_g4fb = np.array(signed_true_g4fb)

bins = np.arange(-SIGNED_MAX, SIGNED_MAX + SIGNED_BIN_WIDTH, SIGNED_BIN_WIDTH)
centers = 0.5 * (bins[:-1] + bins[1:])

n_cc3, _ = np.histogram(signed_true_cc3, bins=bins)
n_g4fb, _ = np.histogram(signed_true_g4fb, bins=bins)
n_total = n_cc3 + n_g4fb

with np.errstate(divide="ignore", invalid="ignore"):
    frac = np.where(n_total >= MIN_ENTRIES_PER_BIN, n_cc3 / n_total, np.nan)
    err = np.where(n_total >= MIN_ENTRIES_PER_BIN,
                    np.sqrt(frac * (1 - frac) / np.maximum(n_total, 1)), np.nan)

print(f"Events processed: {n_events}")
print(f"true-CC3 (deep barrel): {len(signed_true_cc3)}, true-G4-fallback: {len(signed_true_g4fb)}")
print("\nFast-sim fraction vs SIGNED distance to nearest crack center:")
print(f"  {'signed_dist':>12} {'n_total':>8} {'fraction':>10}")
for i in range(len(centers)):
    if not np.isnan(frac[i]):
        print(f"  {centers[i]:>12.2f} {n_total[i]:>8d} {frac[i]:>10.3f}")

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.errorbar(centers, frac, yerr=err, fmt="o", color="#d6604d", markersize=4)
ax.axvline(0, color="black", lw=1, ls="--", alpha=0.6, label="crack center")
ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
ax.set_xlabel("Signed distance to nearest crack center [deg]\n"
              "(sign convention arbitrary -- look for asymmetry, not which side is which)")
ax.set_ylabel("Fast-sim fraction")
ax.set_ylim(-0.05, 1.05)
ax.set_title(
    "Testing for an ASYMMETRIC crack boundary\n"
    "If one side is a hard reject and the other permits up to the center,\n"
    "expect a sharp step on one side and a flat high plateau on the other",
    fontsize=10)
ax.legend()
plt.tight_layout()
out = f"{PLOT_DIR}/signed_crack_asymmetry.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out}")
print("\nHow to read this: if the curve is symmetric about 0 (mirror image on")
print("both sides), the crack is symmetric and the earlier ~40-50% plateau needs")
print("a different explanation. If it's a step on one side and flat-high on the")
print("other, the asymmetric-boundary hypothesis is confirmed.")
