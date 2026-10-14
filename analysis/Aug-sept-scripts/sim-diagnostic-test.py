"""
sim_hit_attribution_check.py
=============================
Minimal, targeted check before writing sim-level cluster width / radial
profile code (Plots 2b, 2d in single_photon_observables.py revision).

Two things to confirm, both cheap, first few events only:

  CHECK A: Does CaloHitContribution expose getParticle() and getEnergy(),
           and does getParticle() resolve to a real MCParticle we can
           match against Signal B cuts? This is the candidate path for
           attributing sim-hit energy to a specific MC truth photon.

  CHECK B: Is EcalBarrelCollection (subset, in the REC file) really just
           a merged view of the four split sim-stage collections
           (ECalBarrelScHitsEven/Odd, ECalBarrelSiHitsEven/Odd)?
           Checked by comparing hit counts for a matching event index
           between the sim-stage file and the REC file.

FIX (vs. previous version): podio's getContributions() returns a
RelationRange, which supports iteration and single-integer indexing
but NOT Python slice syntax (contribs[:3] raises TypeError, since the
slice object is passed straight to the C++ operator[] which expects an
int). Fixed here with itertools.islice instead of slicing.

Run:
    source ~/source.sh
    python3 sim_hit_attribution_check.py
"""

from itertools import islice
from podio import root_io

SIM_FILE = "/data/dust/user/alimuham/thesis/sim/tau-pi0-caloclouds-100kevents-sim.edm4hep.root"
REC_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root"

N_EVENTS_TO_CHECK = 3

print("="*70)
print("CHECK A: CaloHitContribution -> MCParticle attribution")
print("="*70)

reader_sim = root_io.Reader(SIM_FILE)
n_checked = 0
for event in reader_sim.get("events"):
    if n_checked >= N_EVENTS_TO_CHECK:
        break
    n_checked += 1
    print(f"\n--- sim event {n_checked} ---")

    sim_hit_collections = [
        "ECalBarrelScHitsEven", "ECalBarrelScHitsOdd",
        "ECalBarrelSiHitsEven", "ECalBarrelSiHitsOdd",
    ]

    for hit_coll_name in sim_hit_collections:
        hits = event.get(hit_coll_name)
        print(f"  {hit_coll_name}: {len(hits)} hits")
        if len(hits) == 0:
            continue

        # find the first hit that actually has >=1 contribution and E>0,
        # since hit[0] can be a zero-energy/no-contribution entry (as seen
        # in the previous run: hit[0] energy=0.00000 but still "1 contribution")
        h0 = hits[0]
        print(f"    hit[0] energy={h0.getEnergy():.5f}  "
              f"pos=({h0.getPosition().x:.1f},{h0.getPosition().y:.1f},{h0.getPosition().z:.1f})")

        contribs = h0.getContributions()
        n_contribs = len(contribs)
        print(f"    hit[0] has {n_contribs} contributions")

        # FIX: use islice instead of contribs[:3] (RelationRange has no slice support)
        for i, contrib in enumerate(islice(contribs, 3)):
            try:
                part = contrib.getParticle()
                pdg = part.getPDG()
                gen_status = part.getGeneratorStatus()
                e_contrib = contrib.getEnergy()
                print(f"      contrib[{i}]: E={e_contrib:.5f}  "
                      f"particle PDG={pdg}  genStatus={gen_status}")
            except Exception as e:
                print(f"      contrib[{i}]: FAILED to resolve particle -- {e}")

        # Also scan a few more hits for a non-zero-energy example, so the
        # attribution check isn't accidentally run only on an edge-case hit.
        nonzero_found = False
        for hit in islice(hits, 1, 20):
            if hit.getEnergy() > 0:
                nonzero_found = True
                contribs2 = hit.getContributions()
                print(f"    [nonzero-energy hit check] energy={hit.getEnergy():.5f}  "
                      f"n_contributions={len(contribs2)}")
                for i, contrib in enumerate(islice(contribs2, 3)):
                    try:
                        part = contrib.getParticle()
                        print(f"      contrib[{i}]: E={contrib.getEnergy():.5f}  "
                              f"particle PDG={part.getPDG()}  "
                              f"genStatus={part.getGeneratorStatus()}")
                    except Exception as e:
                        print(f"      contrib[{i}]: FAILED -- {e}")
                break
        if not nonzero_found:
            print(f"    [nonzero-energy hit check] none found in first 20 hits of this collection")

print("\n" + "="*70)
print("CHECK B: EcalBarrelCollection (REC, subset) vs split sim-stage collections")
print("="*70)

reader_rec = root_io.Reader(REC_FILE)
n_checked = 0
for event in reader_rec.get("events"):
    if n_checked >= N_EVENTS_TO_CHECK:
        break
    n_checked += 1
    subset_hits = event.get("EcalBarrelCollection")
    print(f"\n--- REC event {n_checked} ---")
    print(f"  EcalBarrelCollection (subset) size: {len(subset_hits)}")

print("\nNOTE: REC file event N and sim-stage file event N are NOT guaranteed")
print("to be the same physical event (different files, possibly different")
print("event ordering/filtering) -- this check only confirms field access")
print("and rough count magnitude, NOT an exact per-event cross-match.")
print("If exact per-event correspondence matters, we need matching Event")
print("Number / run ID metadata, not just positional indexing.")
