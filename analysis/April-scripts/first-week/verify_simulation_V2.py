import os
from podio import root_io

# --- File Paths ---
GEN = "/afs/desy.de/user/a/alimuham/thesis-ml-sim/steering/tau_pi0_10GeV_filtered.edm4hep.root"
G4  = "/data/dust/user/alimuham/thesis/sim/tau-pi0-geant4-V2-sim.edm4hep.root"
CC  = "/data/dust/user/alimuham/thesis/sim/tau-pi0-caloclouds-V2-sim.edm4hep.root"

# ECal collections where fast sim deposits hits (used for old steplength tagging)
ECAL_COLLECTIONS = [
    "ECalBarrelSiHitsEvenContributions",
    "ECalBarrelSiHitsOddContributions",
    "ECalEndcapSiHitsEvenContributions",
    "ECalEndcapSiHitsOddContributions",
    "ECalBarrelScHitsEvenContributions",
    "ECalBarrelScHitsOddContributions",
    "ECalEndcapScHitsEvenContributions",
    "ECalEndcapScHitsOddContributions",
    "EcalEndcapRingCollectionContributions",
]

MAX_EVENTS = 1000


def build_fastsim_hitmap(event):
    """
    Old tagging method: a particle is considered fast-simulated if it has
    >= 10 hits with step_length == 0 across all ECal collections.
    Returns a set of MCParticle object indices tagged as fast-simulated.
    """
    hit_map = {}
    for col_name in ECAL_COLLECTIONS:
        try:
            for contrib in event.get(col_name):
                if contrib.getStepLength() == 0:
                    idx = contrib.getParticle().getObjectID().index
                    hit_map[idx] = hit_map.get(idx, 0) + 1
        except Exception:
            continue
    return {idx for idx, count in hit_map.items() if count >= 10}


def get_stats(path, is_cc=False):
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        return None

    reader = root_io.Reader(path)
    events = reader.get("events")

    stats = {
        "events":           0,
        "total_photons":    0,   # all photons regardless of status or energy
        "genstatus1":       0,   # photons with generatorStatus == 1
        "above10gev":       0,   # photons with E >= 10 GeV (any status)
        "fastsim_flag":     0,   # isHandledByFastSim() == True (CC only)
        "fastsim_hits":     0,   # old steplength method (CC only)
    }

    for ev in events:
        if stats["events"] >= MAX_EVENTS:
            break
        stats["events"] += 1

        # Build old-method fast sim map for this event (CC only)
        fastsim_indices = build_fastsim_hitmap(ev) if is_cc else set()

        for p in ev.get("MCParticles"):
            if p.getPDG() != 22:  # photons only
                continue

            stats["total_photons"] += 1

            if p.getGeneratorStatus() == 1:
                stats["genstatus1"] += 1

            if p.getEnergy() >= 10.0:
                stats["above10gev"] += 1

            if is_cc:
                if p.isHandledByFastSim():
                    stats["fastsim_flag"] += 1
                if p.getObjectID().index in fastsim_indices:
                    stats["fastsim_hits"] += 1

    return stats


# --- Run ---
print("Reading files...")
s_gen = get_stats(GEN)
s_g4  = get_stats(G4)
s_cc  = get_stats(CC, is_cc=True)

if not all([s_gen, s_g4, s_cc]):
    print("One or more files could not be read. Check paths above.")
else:
    w = 32
    col = 14
    sep = "-" * (w + 3 * col + 6)

    print(f"\n{'Metric':<{w}} | {'Generator':<{col}} | {'Geant4':<{col}} | {'CaloClouds':<{col}}")
    print(sep)
    print(f"{'Events analyzed':<{w}} | {s_gen['events']:<{col}} | {s_g4['events']:<{col}} | {s_cc['events']:<{col}}")
    print(sep)
    print(f"{'Total MC photons':<{w}} | {s_gen['total_photons']:<{col}} | {s_g4['total_photons']:<{col}} | {s_cc['total_photons']:<{col}}")
    print(f"{'genStatus == 1 photons':<{w}} | {s_gen['genstatus1']:<{col}} | {s_g4['genstatus1']:<{col}} | {s_cc['genstatus1']:<{col}}")
    print(f"{'Photons with E >= 10 GeV':<{w}} | {s_gen['above10gev']:<{col}} | {s_g4['above10gev']:<{col}} | {s_cc['above10gev']:<{col}}")
    print(sep)
    print(f"{'Fast sim: isHandledByFastSim':<{w}} | {'-':<{col}} | {'-':<{col}} | {s_cc['fastsim_flag']:<{col}}")
    print(f"{'Fast sim: steplength method':<{w}} | {'-':<{col}} | {'-':<{col}} | {s_cc['fastsim_hits']:<{col}}")
    print(sep)
    print()
    print("Notes:")
    print("  genStatus == 1  : particle came directly from the event generator (final state)")
    print("  E >= 10 GeV     : independent of genStatus — all photons above ML trigger threshold")
    print("  isHandledByFastSim : new canonical method, requires nightlies build")
    print("  steplength method  : step_length==0 with >=10 hits in ECal collections (old method)")
