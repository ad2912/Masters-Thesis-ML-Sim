import numpy as np
from podio import root_io
import os
import edm4hep

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SIM_FILE = "/data/dust/user/alimuham/thesis/sim/tau-pi0-caloclouds-V2-sim.edm4hep.root"
PHOTON_PDG = 22
ML_THRESHOLD = 10 

# Using the EXACT collections from your working verify_sim_files.py
ECAL_CONTRIB_COLLECTIONS = [
    "ECalBarrelSiHitsEvenContributions",
    "ECalBarrelSiHitsOddContributions",
    "ECalBarrelScHitsEvenContributions",
    "ECalBarrelScHitsOddContributions",
    "ECalEndcapSiHitsEvenContributions",
    "ECalEndcapSiHitsOddContributions",
    "ECalEndcapScHitsEvenContributions",
    "ECalEndcapScHitsOddContributions",
]

def run_comparison():
    if not os.path.exists(SIM_FILE):
        print(f"Error: {SIM_FILE} not found.")
        return

    reader = root_io.Reader(SIM_FILE)
    
    total_photons = 0
    new_method_count = 0
    old_method_count = 0
    both_match_count = 0

    print(f"Processing: {SIM_FILE}...")

    for event in reader.get("events"):
        mcp_collection = event.get("MCParticles")
        
        # --- 1. Build a map of 0-step hits per particle ---
        # We look into the CONTRIB collections directly (as in your verify script)
        particle_zero_step_map = {}

        for coll_name in ECAL_CONTRIB_COLLECTIONS:
            try:
                contribs = event.get(coll_name)
                for c in contribs:
                    if c.getStepLength() == 0:
                        part = c.getParticle()
                        if part:
                            # Use index for reliable matching
                            idx = part.getObjectID().index
                            particle_zero_step_map[idx] = particle_zero_step_map.get(idx, 0) + 1
            except Exception:
                continue # Collection might not exist in this event

        # --- 2. Check the Photons ---
        for p in mcp_collection:
            # Look for final-state photons
            if p.getPDG() != PHOTON_PDG or p.getGeneratorStatus() != 1:
                continue
            
            total_photons += 1
            p_idx = p.getObjectID().index

            # Method A: The Flag (New)
            is_new = p.isHandledByFastSim()
            if is_new:
                new_method_count += 1

            # Method B: The Hits (Old)
            zero_step_count = particle_zero_step_map.get(p_idx, 0)
            is_old = (zero_step_count >= ML_THRESHOLD)
            if is_old:
                old_method_count += 1

            # Track consistency
            if is_new == is_old:
                both_match_count += 1

    # ─────────────────────────────────────────────────────────────────────────────
    # OUTPUT
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("  SIMULATION PHOTON TAGGING COMPARISON")
    print("="*50)
    print(f"Total Photons Analyzed:        {total_photons}")
    print("-" * 50)
    print(f"Tagged by 'isHandledByFastSim': {new_method_count}")
    print(f"Tagged by '0-step hits' (>={ML_THRESHOLD}): {old_method_count}")
    print("-" * 50)
    
    if total_photons > 0:
        match_rate = (both_match_count / total_photons) * 100
        print(f"Consistency Match Rate:        {match_rate:.2f}%")
    
    # Extra Touch: Efficiency Check
    if new_method_count > 0:
        found_pct = (old_method_count / new_method_count) * 100
        print(f"\nNote: {found_pct:.1f}% of flag-tagged particles also show 0-step hits.")
    print("="*50 + "\n")

run_comparison()
