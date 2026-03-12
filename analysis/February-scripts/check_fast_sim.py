import os
from podio import root_io

# 1. Setup path
dust_path = os.environ.get("DUST", "/data/dust/user/alimuham/thesis/sim")
file_name = "tau_pi0_SIM_geant4.edm4hep.root"
full_path = os.path.join(dust_path, file_name)

if not os.path.exists(full_path):
    print(f"Error: {full_path} not found.")
    exit(1)

reader = root_io.Reader(full_path)
events = reader.get("events")

# 2. Define all Calorimeter collections (Barrel + Endcap)
# Checking both ensures we don't miss triggers based on geometry
collections = [
    "ECalBarrelSiHitsEvenContributions", "ECalBarrelSiHitsOddContributions",
    "ECalBarrelScHitsEvenContributions", "ECalBarrelScHitsOddContributions",
    "ECalEndcapSiHitsEvenContributions", "ECalEndcapSiHitsOddContributions",
    "ECalEndcapScHitsEvenContributions", "ECalEndcapScHitsOddContributions"
]

events_with_ml = 0
total_events = len(events)

print(f"Scanning {total_events} events for ML trigger statistics...\n")
print(f"{'Event #':<10} | {'ML Hits':<10} | {'Status':<10}")
print("-" * 35)

for i, event in enumerate(events):
    ml_hits_in_event = 0
    
    for coll_name in collections:
        try:
            contribs = event.get(coll_name)
            for c in contribs:
                if c.getStepLength() == 0:
                    ml_hits_in_event += 1
        except:
            continue
    
    if ml_hits_in_event > 0:
        events_with_ml += 1
        status = "✅ ML"
        print(f"DEBUG: Found {ml_hits_in_event} zero-step hits in Event #{i}")
        print(f"   --> Artifact in {coll_name}: Energy = {c.getEnergy():.6f} GeV")
    else:
        status = "❌ G4 Only"

    # Print every 50 events so the terminal isn't a mess, 
    # but still shows progress.
    if i % 50 == 0 or ml_hits_in_event > 0:
        print(f"{i:<10} | {ml_hits_in_event:<10} | {status:<10}")

# 3. Final Summary
print("-" * 35)
trigger_rate = (events_with_ml / total_events) * 100
print(f"Total Events Scanned: {total_events}")
print(f"Events Triggered ML:  {events_with_ml}")
print(f"ML Trigger Rate:      {trigger_rate:.2f}%")
