# signal_definition_comparison.py
#
# Compares three signal definitions, all in the CC3 file, all restricted to
# the barrel theta window [37.5, 142.5] deg for a fair comparison:
#
#   Def 1 (PREVIOUS): Layer A physics cuts only, no theta/phi restriction at
#       all -- this is literally what pi0_invariant_mass_v5.py uses today.
#       Reported both globally (as actually used) and barrel-restricted (for
#       fair comparison against Def 2 and Def 3).
#   Def 2 (NEW): Layer A + theta-barrel + asymmetric one-sided 8deg phi-crack
#       cut. Our engineered GEOMETRIC PROXY for "should be CC3".
#   Def 3 (FLAGGER-ONLY): Layer A + theta-barrel + isHandledByFastSim()==True.
#       GROUND TRUTH, restricted only by theta (no phi cut at all).
#
# Key comparison: Def 2 vs Def 3, as a confusion matrix --
#   precision (= purity) = |Def2 AND Def3| / |Def2|
#     -- of what our geometric cut calls "clean CC3", how much really is?
#   recall (= retention)  = |Def2 AND Def3| / |Def3|
#     -- of what's really CC3 (by the flag), how much does our geometric
#        cut successfully capture?
#
# NOT YET RUN.

from podio import root_io
import edm4hep
import numpy as np

CC3_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root"
MAX_EVENTS = 10000

PHOTON_PDG = 22
ELECTRON_PDG = 11
E_MC_MIN = 10.0

THETA_SAFE_LO_DEG = 37.5
THETA_SAFE_HI_DEG = 142.5
PHI_CRACK_CENTERS_DEG = [-157.5 + 45 * k for k in range(8)]
PHI_CRACK_REJECT_WIDTH_DEG = 8.0


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


def in_phi_safe_region(phi_deg):
    for center in PHI_CRACK_CENTERS_DEG:
        raw = ((phi_deg - center + 180) % 360) - 180
        if 0.0 <= raw <= PHI_CRACK_REJECT_WIDTH_DEG:
            return False
    return True


reader = root_io.Reader(CC3_FILE)

n_def1_global = 0     # Layer A only, no restriction at all (as pi0 script uses today)
n_def1_barrel = 0     # Layer A + theta-barrel only (for fair comparison)
n_def2 = 0            # Layer A + theta-barrel + phi-crack cut (geometric proxy)
n_def3 = 0            # Layer A + theta-barrel + real flag (ground truth)
n_def2_and_def3 = 0   # overlap: geometric proxy says yes AND flag says yes
n_def2_only = 0       # geometric proxy says yes, flag says no (false positive)
n_def3_only = 0       # flag says yes, geometric proxy says no (false negative / missed)

n_events = 0
for event in reader.get("events"):
    if n_events >= MAX_EVENTS:
        break
    n_events += 1
    for mc in event.get("MCParticles"):
        if not passes_layer_a(mc):
            continue
        n_def1_global += 1

        p4 = edm4hep.utils.p4(mc)
        theta_deg = np.degrees(float(p4.theta()))
        if not (THETA_SAFE_LO_DEG <= theta_deg <= THETA_SAFE_HI_DEG):
            continue
        n_def1_barrel += 1

        phi_deg = np.degrees(float(p4.phi()))
        geom_ok = in_phi_safe_region(phi_deg)
        flag_ok = bool(mc.isHandledByFastSim())

        if geom_ok:
            n_def2 += 1
        if flag_ok:
            n_def3 += 1
        if geom_ok and flag_ok:
            n_def2_and_def3 += 1
        elif geom_ok and not flag_ok:
            n_def2_only += 1
        elif flag_ok and not geom_ok:
            n_def3_only += 1

print(f"Events processed: {n_events}\n")

print("=== Population sizes ===")
print(f"Def 1 (previous, physics only, GLOBAL -- as used today) : {n_def1_global}")
print(f"Def 1 (previous, physics only, barrel-restricted)        : {n_def1_barrel}")
print(f"Def 2 (new: physics + theta-barrel + phi-crack cut)      : {n_def2}")
print(f"Def 3 (flagger-only: physics + theta-barrel + real flag) : {n_def3}")

print("\n=== Def 1 (previous) vs Def 2 (new) ===")
print(f"Global -> barrel+geometry retention: {n_def2}/{n_def1_global} = {100*n_def2/n_def1_global:.1f}%")
print("(this is the overall shrinkage from adding geometry to the old, ungated definition)")

print("\n=== Def 2 (geometric proxy) vs Def 3 (ground truth flag), confusion matrix ===")
print(f"  Both agree (geom says CC3 AND flag says CC3)      : {n_def2_and_def3}")
print(f"  Geom says CC3, but flag says NO  (false positive)  : {n_def2_only}")
print(f"  Flag says CC3, but geom says NO  (false negative)  : {n_def3_only}")

precision = n_def2_and_def3 / max(n_def2, 1)
recall = n_def2_and_def3 / max(n_def3, 1)
print(f"\n  Precision (= purity)  = agree / Def2 = {n_def2_and_def3}/{n_def2} = {100*precision:.1f}%")
print(f"    -> of what our geometric cut calls 'clean CC3', this fraction really is")
print(f"  Recall (= retention)  = agree / Def3 = {n_def2_and_def3}/{n_def3} = {100*recall:.1f}%")
print(f"    -> of what's really CC3 (by the flag), our geometric cut successfully captures this fraction")
print(f"\nInterpretation: if both numbers are high (~90%+), the geometric proxy is a good")
print(f"stand-in for the real flag. If precision is high but recall is low, our geometric")
print(f"cut is too strict (missing real signal). If recall is high but precision is low,")
print(f"it's too loose (letting contamination through).")
