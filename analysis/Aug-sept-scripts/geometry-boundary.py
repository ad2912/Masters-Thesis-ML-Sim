# geometry_boundary_diagnostic.py
#
# Purpose: measure the ACTUAL theta window and phi-crack half-width used by
# the real CC3 trigger, directly from true-CC3-tagged photons, instead of
# guessing constants a second time. Reuses the same Layer-A pass logic as
# fastsim_confidence_study_v1.py -- no new file-reading approach, just
# different bookkeeping (percentiles instead of a fixed cut).
#
# Concept: every true-CC3 photon (isHandledByFastSim()==True) is, by
# definition, a case the real trigger accepted. So the narrowest theta
# range and the smallest phi-distance-to-crack-center seen among true-CC3
# photons tells us empirically where the real boundaries sit -- no visual
# reading of a plot required.
#
# NOT YET RUN.

import numpy as np
from podio import root_io
import edm4hep

CC3_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root"
MAX_EVENTS = 10000

PHOTON_PDG = 22
ELECTRON_PDG = 11
E_MC_MIN = 10.0

PHI_CRACK_CENTERS_DEG = [-157.5 + 45 * k for k in range(8)]


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


def dist_to_nearest_crack(phi_deg):
    dists = [abs(((phi_deg - c + 180) % 360) - 180) for c in PHI_CRACK_CENTERS_DEG]
    return min(dists)


reader = root_io.Reader(CC3_FILE)

theta_true_cc3 = []
theta_true_g4fb = []
crackdist_true_cc3 = []
crackdist_true_g4fb = []

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
        phi_deg = np.degrees(float(p4.phi()))
        is_fastsim = bool(mc.isHandledByFastSim())
        cdist = dist_to_nearest_crack(phi_deg)
        if is_fastsim:
            theta_true_cc3.append(theta_deg)
            crackdist_true_cc3.append(cdist)
        else:
            theta_true_g4fb.append(theta_deg)
            crackdist_true_g4fb.append(cdist)

theta_true_cc3 = np.array(theta_true_cc3)
theta_true_g4fb = np.array(theta_true_g4fb)
crackdist_true_cc3 = np.array(crackdist_true_cc3)
crackdist_true_g4fb = np.array(crackdist_true_g4fb)

print(f"Events processed: {n_events}")
print(f"\nTrue-CC3 theta [deg]  (N={len(theta_true_cc3)}):")
for p in [0, 1, 5, 50, 95, 99, 100]:
    print(f"  p{p:>3d}: {np.percentile(theta_true_cc3, p):.2f}")

print(f"\nTrue-CC3 distance-to-nearest-crack-center [deg]  (N={len(crackdist_true_cc3)}):")
for p in [0, 1, 5, 10, 50]:
    print(f"  p{p:>3d}: {np.percentile(crackdist_true_cc3, p):.2f}")
print("  -> the p0/p1 value here is the empirical phi-crack HALF-WIDTH")
print("     (smallest distance to a crack center any true-CC3 photon reaches)")

print(f"\nTrue-G4-fallback distance-to-nearest-crack-center [deg]  (N={len(crackdist_true_g4fb)}):")
for p in [0, 5, 10, 25, 50, 75, 90, 95, 100]:
    print(f"  p{p:>3d}: {np.percentile(crackdist_true_g4fb, p):.2f}")
print("  -> compare against true-CC3's p0/p1 above: if G4-fallback photons")
print("     exist well beyond that distance too, the crack isn't the only")
print("     driver of fallback in phi (ties to the residual proximity hypothesis)")
