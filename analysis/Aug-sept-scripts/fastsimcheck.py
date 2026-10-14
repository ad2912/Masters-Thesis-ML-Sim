# quick_check_isHandledByFastSim.py
#
# Read-only diagnostic. Does NOT change WEIGHT_CUT or any existing script
# logic. Two goals:
#   (1) verify MCParticle.isHandledByFastSim() is callable on MCParticles
#       read from a REC file, and whether its True/False pattern makes
#       geometric sense (no crack-angle assumptions hardcoded -- print raw
#       phi/cosTheta for you to eyeball against the geometry XML).
#   (2) pin down the correct track_weight divisor from the raw weight
#       data's own ceiling, the same way cluster_weight was verified
#       against real data structure on 2026-08-03 (reasoning contract B2).
#   (3) print n_matched_PFO per truth photon as a splitting proxy, since
#       Pandora splitting could make "how many reco objects" a separate
#       question from "was this fast-sim handled".
#
# NOT YET RUN -- share the printed output after running this on NAF.

from podio import root_io
import math
import numpy as np

CC3_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-caloclouds-100kevents-reco-test_REC.edm4hep.root"
MAX_EVENTS = 200
E_MC_MIN = 10.0
COS_THETA_BARREL_LOOSE = 0.85  # looser than the ~0.77 canonical cut, to catch near-edge cases too
PHOTON_PDG = 22
MAX_PRINT = 40

reader = root_io.Reader(CC3_FILE)

header = "{:>4} {:>7} {:>9} {:>8} {:>19} {:>14}".format(
    "idx", "E_mc", "cosTheta", "phi_deg", "isHandledByFastSim", "n_matched_PFO"
)
print(header)

n_events = 0
n_printed = 0
raw_weight_values = []
method_missing = False

for event in reader.get("events"):
    if n_events >= MAX_EVENTS or n_printed >= MAX_PRINT:
        break
    n_events += 1

    for link in event.get("RecoMCTruthLink"):
        raw_weight_values.append(link.getWeight())

    reco_to_mc_count = {}
    for link in event.get("RecoMCTruthLink"):
        mc_id = link.getTo().getObjectID().index
        reco_to_mc_count[mc_id] = reco_to_mc_count.get(mc_id, 0) + 1

    for mc in event.get("MCParticles"):
        if n_printed >= MAX_PRINT:
            break
        if mc.getGeneratorStatus() != 1 or mc.getPDG() != PHOTON_PDG:
            continue
        e_mc = mc.getEnergy()
        if e_mc < E_MC_MIN:
            continue
        mom = mc.getMomentum()
        p = math.sqrt(mom.x**2 + mom.y**2 + mom.z**2)
        if p == 0:
            continue
        cos_theta = mom.z / p
        if abs(cos_theta) > COS_THETA_BARREL_LOOSE:
            continue

        phi_deg = math.degrees(math.atan2(mom.y, mom.x)) % 360

        try:
            is_fastsim = mc.isHandledByFastSim()
            fastsim_str = str(is_fastsim)
        except AttributeError as e:
            fastsim_str = "ERROR: {}".format(e)
            method_missing = True

        mc_id = mc.getObjectID().index
        n_matched = reco_to_mc_count.get(mc_id, 0)

        row = "{:>4} {:7.2f} {:9.4f} {:8.2f} {:>19} {:>14d}".format(
            n_printed, e_mc, cos_theta, phi_deg, fastsim_str, n_matched
        )
        print(row)
        n_printed += 1

print("")
print("Events scanned: {}, photons printed: {}".format(n_events, n_printed))
if method_missing:
    print("WARNING: isHandledByFastSim() raised AttributeError -- not callable on REC-level MCParticles.")

raw_arr = np.array(raw_weight_values)
print("")
print("Raw RecoMCTruthLink weight stats (N={}):".format(len(raw_arr)))
print("  min={:.1f}  max={:.1f}".format(raw_arr.min(), raw_arr.max()))

cluster_permille_vals = (raw_arr.astype(np.int64) // 10000)
remainder_vals = (raw_arr.astype(np.int64) % 10000)
print("")
print("  cluster_permille (raw // 10000): min={}, max={}".format(
    cluster_permille_vals.min(), cluster_permille_vals.max()))
print("  remainder        (raw % 10000) : min={}, max={}".format(
    remainder_vals.min(), remainder_vals.max()))
print("  -> if remainder max is <=1000: divide by 1000.0 for track_weight in [0,1] (v5's formula)")
print("  -> if remainder max is <=10000: divide by 10000.0 for track_weight in [0,1] (v4's formula)")
print("  -> whichever ceiling actually appears in the data is correct (per contract B2)")
