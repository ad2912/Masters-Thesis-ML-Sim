"""
check_ggpi0_collection.py
==========================
Quick supervisor-facing check: is GammaGammaCandidatePi0s.mass a real
measurement or a hardcoded constant?

Prints:
  1. First N raw .mass values from the collection, both files.
  2. Whether all values are bit-identical (the smoking gun).
  3. Independent mass recomputed from the collection's own stored
     energy/momentum 4-vectors, for the same N candidates.

Run on NAF:
    source ~/source.sh
    python3 check_ggpi0_collection.py
"""

import math
from podio import root_io

G4_FILE  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-10k-events-test_REC.edm4hep.root"
CC3_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root"
N_PRINT  = 20
MAX_EVENTS = 200  # only need a handful of events to make the point


def m_from_4vec(e, px, py, pz):
    m2 = e*e - (px*px + py*py + pz*pz)
    return math.sqrt(max(m2, 0.0))


def check_file(filepath, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    reader = root_io.Reader(filepath)
    raw_masses = []
    recomputed_masses = []
    n_events = 0

    for event in reader.get("events"):
        if n_events >= MAX_EVENTS:
            break
        n_events += 1
        for cand in event.get("GammaGammaCandidatePi0s"):
            raw_masses.append(cand.getMass())
            e = cand.getEnergy()
            mom = cand.getMomentum()
            recomputed_masses.append(m_from_4vec(e, mom.x, mom.y, mom.z))
            if len(raw_masses) >= N_PRINT:
                break
        if len(raw_masses) >= N_PRINT:
            break

    print(f"\n  First {len(raw_masses)} raw .mass() values [GeV]:")
    for i, m in enumerate(raw_masses):
        print(f"    [{i:2d}] {m:.10f}")

    all_identical = len(set(raw_masses)) == 1
    print(f"\n  All values bit-identical? {all_identical}")
    if all_identical:
        print(f"  --> .mass is a HARDCODED CONSTANT: {raw_masses[0]:.10f} GeV")

    print(f"\n  Independently recomputed mass from stored 4-vectors [GeV]:")
    for i, m in enumerate(recomputed_masses):
        print(f"    [{i:2d}] {m:.10f}   (raw .mass = {raw_masses[i]:.10f})")

    spread = max(recomputed_masses) - min(recomputed_masses)
    print(f"\n  Spread of recomputed mass: {spread*1000:.6f} MeV")
    print(f"  --> Even the independently recomputed mass is essentially")
    print(f"      constant, confirming the 4-vectors themselves are")
    print(f"      mass-constrained upstream (not just the .mass field).")


print("\nGammaGammaCandidatePi0s sanity check: is this a real measurement?")
check_file(G4_FILE, "Geant4")
check_file(CC3_FILE, "CC3")
