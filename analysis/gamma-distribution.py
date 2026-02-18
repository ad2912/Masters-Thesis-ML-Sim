import podio
import numpy as np
import matplotlib.pyplot as plt

# Load the file
reader = podio.root_io.Reader("photon_gun_SIM.edm4hep.root")

# Collections from your podio-dump
ecal_collections = [
    "ECalBarrelScHitsEven", "ECalBarrelScHitsOdd",
    "ECalBarrelSiHitsEven", "ECalBarrelSiHitsOdd"
]

event_energies = []
all_hit_x = []
all_hit_e = []

for event in reader.get("events"):
    total_energy = 0.0
    for col_name in ecal_collections:
        hits = event.get(col_name)
        for hit in hits:
            e = hit.getEnergy() # Usually in GeV
            pos = hit.getPosition() # Vector with .x, .y, .z
            
            total_energy += e
            all_hit_x.append(pos.x)
            all_hit_e.append(e)
            
    event_energies.append(total_energy)

# Filter hits to only include the ECal region (X > 1800)
# This removes tracker noise and backscatter
ecal_hits_x = [x for x in all_hit_x if x > 1800]
ecal_hits_e = [e for i, e in enumerate(all_hit_e) if all_hit_x[i] > 1800]

# --- PLOTTING ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Energy Sum (10 events won't look like a Gaussian, but should be > 0.1 GeV)
ax1.hist(event_energies, bins=10, color='skyblue', edgecolor='black')
ax1.set_title("Total Energy per Event (10 Events)")
ax1.set_xlabel("Energy Sum [GeV]")

# Plot 2: Longitudinal Profile (Zoomed & Fine Binned)
# We set range=(1800, 2100) and increase bins to 100 for "finer" detail
ax2.hist(ecal_hits_x, bins=100, range=(1800, 2100), weights=ecal_hits_e, 
         color='salmon', alpha=0.8, edgecolor='brown', linewidth=0.5)

ax2.set_title("ECal Longitudinal Profile (Zoomed)")
ax2.set_xlabel("X Position (Depth) [mm]")
ax2.set_ylabel("Energy Deposition")
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
