#!/usr/bin/env python3

import sys
import math
from podio.root_io import Reader

if len(sys.argv) < 2:
    print("Usage: python3 inspect_sim.py filename")
    sys.exit(1)

filename = sys.argv[1]
reader = Reader(filename)

print("\n===== SIM FILE INSPECTION =====\n")

for ievt, event in enumerate(reader.get("events")):

    print(f"\n================ Event {ievt} ================")

    # -------------------------------------------------
    # 1) MC TRUTH
    # -------------------------------------------------
    mc_particles = event.get("MCParticles")

    for p in mc_particles:
        # generator status 1 = stable final state particle
        if p.getGeneratorStatus() == 1:

            energy = p.getEnergy() / 1000.0  # MeV → GeV
            mom = p.getMomentum()
            px, py, pz = mom.x, mom.y, mom.z

            p_mag = math.sqrt(px**2 + py**2 + pz**2)

            theta = math.acos(pz / p_mag) if p_mag != 0 else 0
            phi = math.atan2(py, px)

            print("MC final-state particle:")
            print(f"  Energy = {energy:.3f} GeV")
            print(f"  theta  = {theta:.3f} rad")
            print(f"  phi    = {phi:.3f} rad")

    # -------------------------------------------------
    # 2) ECAL ENERGY
    # -------------------------------------------------
    ecal_energy = 0.0

    if "ECalBarrelCollection" in event.getAvailableCollections():
        for hit in event.get("ECalBarrelCollection"):
            ecal_energy += hit.getEnergy()

    if "ECalEndcapCollection" in event.getAvailableCollections():
        for hit in event.get("ECalEndcapCollection"):
            ecal_energy += hit.getEnergy()

    # -------------------------------------------------
    # 3) HCAL ENERGY
    # -------------------------------------------------
    hcal_energy = 0.0

    if "HCalBarrelCollection" in event.getAvailableCollections():
        for hit in event.get("HCalBarrelCollection"):
            hcal_energy += hit.getEnergy()

    if "HCalEndcapCollection" in event.getAvailableCollections():
        for hit in event.get("HCalEndcapCollection"):
            hcal_energy += hit.getEnergy()

    # Convert to GeV
    ecal_energy /= 1000.0
    hcal_energy /= 1000.0
    total_energy = ecal_energy + hcal_energy

    print("\nCalorimeter response:")
    print(f"  ECAL energy  = {ecal_energy:.3f} GeV")
    print(f"  HCAL energy  = {hcal_energy:.3f} GeV")
    print(f"  TOTAL calo   = {total_energy:.3f} GeV")

print("\n===== DONE =====\n")
