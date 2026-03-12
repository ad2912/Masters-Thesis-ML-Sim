"""
count_photons.py
================
Quick diagnostic: count reconstructed photons in all reco files.

For Geant4: just counts reco photons (PDG==22 in PandoraPFOs)
For CaloClouds: counts total, ML-tagged, and G4-type reco photons

ML tagging logic:
  - For each reco photon, trace its cluster hits back to sim hits
    via the ECAL SimRec relations
  - Count how many sim hits have at least one contribution with steplength == 0
  - If that count > 10: photon is ML-simulated (CaloClouds)
  - If count <= 10: photon was Geant4-simulated (even inside CC file)

Run with:
    source ~/source.sh
    python3 count_photons.py
"""

from podio import root_io

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
REC_G4_P1 = "/data/dust/user/alimuham/thesis/reco/tau_geant4_part1_REC.edm4hep.root"
REC_G4_P2 = "/data/dust/user/alimuham/thesis/reco/tau_geant4_part2_REC.edm4hep.root"
REC_G4_P3 = "/data/dust/user/alimuham/thesis/reco/tau_geant4_part3_REC.edm4hep.root"
REC_CC    = "/data/dust/user/alimuham/thesis/reco/tau_caloclouds_REC.edm4hep.root"

PHOTON_PDG   = 22
ML_THRESHOLD = 10

ECAL_RELATIONS = [
    "EcalBarrelRelationsSimRec",
    "EcalEndcapsRelationsSimRec",
    "EcalEndcapRingRelationsSimRec",
]

# ─────────────────────────────────────────────────────────────────────────────
# COUNT G4 RECO PHOTONS (one file)
# ─────────────────────────────────────────────────────────────────────────────
def count_g4(path, label):
    reader = root_io.Reader(path)
    n_events  = 0
    n_photons = 0
    for event in reader.get("events"):
        n_events += 1
        for pfo in event.get("PandoraPFOs"):
            if pfo.getPDG() == PHOTON_PDG:
                n_photons += 1
    print(f"  {label}")
    print(f"    Events  : {n_events}")
    print(f"    Photons : {n_photons}")
    print(f"    Per event (mean): {n_photons/max(n_events,1):.2f}")
    return n_photons, n_events

# ─────────────────────────────────────────────────────────────────────────────
# COUNT CC RECO PHOTONS WITH ML TAGGING
# ─────────────────────────────────────────────────────────────────────────────
def count_cc(path, label):
    reader = root_io.Reader(path)
    n_events    = 0
    n_ml        = 0
    n_g4type    = 0
    n_unmatched = 0

    for event in reader.get("events"):
        n_events += 1

        hit_lookup = {}
        for rel_name in ECAL_RELATIONS:
            try:
                for rel in event.get(rel_name):
                    hit_lookup[rel.getFrom().getObjectID().index] = rel.getTo()
            except Exception:
                pass

        for pfo in event.get("PandoraPFOs"):
            if pfo.getPDG() != PHOTON_PDG:
                continue

            ml_count = 0
            for cluster in pfo.getClusters():
                for rec_hit in cluster.getHits():
                    sim_hit = hit_lookup.get(rec_hit.getObjectID().index)
                    if sim_hit is None:
                        n_unmatched += 1
                        continue
                    for contrib in sim_hit.getContributions():
                        if contrib.getStepLength() == 0:
                            ml_count += 1
                            break  # one zero-step contrib per sim hit is enough

            if ml_count > ML_THRESHOLD:
                n_ml += 1
            else:
                n_g4type += 1

    total = n_ml + n_g4type
    print(f"  {label}")
    print(f"    Events          : {n_events}")
    print(f"    Total photons   : {total}")
    print(f"    Per event (mean): {total/max(n_events,1):.2f}")
    print(f"    ML-tagged       : {n_ml}  ({100*n_ml/max(total,1):.1f}%)")
    print(f"    G4-type         : {n_g4type}  ({100*n_g4type/max(total,1):.1f}%)")
    if n_unmatched > 0:
        print(f"    ⚠ Unmatched hits: {n_unmatched}  "
              f"(hits with no SimRec relation — ML count may be underestimated)")
    return total, n_ml, n_g4type, n_events


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  GEANT4 RECO PHOTON COUNTS")
print("="*60)
g4_n1, ev1 = count_g4(REC_G4_P1, "G4 part1")
print()
g4_n2, ev2 = count_g4(REC_G4_P2, "G4 part2")
print()
g4_n3, ev3 = count_g4(REC_G4_P3, "G4 part3")

g4_total  = g4_n1 + g4_n2 + g4_n3
ev_total  = ev1   + ev2   + ev3
print(f"\n  ── G4 MERGED ──────────────────────────")
print(f"    Events  : {ev_total}")
print(f"    Photons : {g4_total}")
print(f"    Per event (mean): {g4_total/max(ev_total,1):.2f}")

print("\n" + "="*60)
print("  CALOCLOUDS RECO PHOTON COUNTS")
print("="*60)
cc_total, n_ml, n_g4type, cc_events = count_cc(REC_CC, "CaloClouds")

print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
print(f"  {'Full Geant4':<30} {g4_total:>5} photons in {ev_total} events")
print(f"  {'CaloClouds total':<30} {cc_total:>5} photons in {cc_events} events")
print(f"    {'└─ ML-simulated':<28} {n_ml:>5}  ({100*n_ml/max(cc_total,1):.1f}%)")
print(f"    {'└─ G4-type':<28} {n_g4type:>5}  ({100*n_g4type/max(cc_total,1):.1f}%)")
print()
print(f"  Expected ML triggers from SIM scan : ~8577")
print(f"  Actual ML-tagged reco photons      : {n_ml}")
print(f"  → These are different quantities: one MC photon can")
print(f"    produce one reco photon, or be merged, or be lost.")
