import podio
import numpy as np
import matplotlib.pyplot as plt

reader = podio.root_io.Reader("photon_gun_SIM.edm4hep.root")

ecal_collections = [
    "ECalBarrelScHitsEven",
    "ECalBarrelScHitsOdd",
    "ECalBarrelSiHitsEven",
    "ECalBarrelSiHitsOdd",
]

event_energies = []

for event in reader.get("events"):

    total_energy = 0.0

    for col_name in ecal_collections:
        try:
            hits = event.get(col_name)
            for hit in hits:
                total_energy += hit.getEnergy()
        except KeyError:
            # collection not present in this event
            continue

    event_energies.append(total_energy)

event_energies = np.array(event_energies)

plt.figure()
plt.hist(event_energies, bins=20)
plt.xlabel("Total ECAL Energy per Event [GeV]")
plt.ylabel("Number of Events")
plt.title("Photon Gun - ECAL Energy Response")
plt.show()
