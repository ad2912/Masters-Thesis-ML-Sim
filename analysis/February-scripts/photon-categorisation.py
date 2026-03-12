#to check number of ML sim photons and Geant4 sim photons
from podio import root_io
import numpy as np

input_file = "/data/dust/user/alimuham/thesis/reco/tau_pi0_RECO_geant4_REC.edm4hep.root"
reader = root_io.Reader(input_file)

ml_photon_energies = []
g4_photon_energies = []

for event in reader.get("events"):
    pfos = event.get("PandoraPFOs")
    
    # Load relations
    barrel_rels = event.get("EcalBarrelRelationsSimRec")
    endcap_rels = event.get("EcalEndcapsRelationsSimRec")
    
    # Build lookup using .getFrom() and .getTo()
    hit_lookup = {}
    for rel in barrel_rels:
        hit_lookup[rel.getFrom().getObjectID().index] = rel.getTo()
    for rel in endcap_rels:
        hit_lookup[rel.getFrom().getObjectID().index] = rel.getTo()

    for pfo in pfos:
    
        if pfo.getPDG() != 22:
            continue
            
        ml_hit_count = 0
        for cluster in pfo.getClusters():
            for rec_hit in cluster.getHits():
                sim_hit = hit_lookup.get(rec_hit.getObjectID().index)
                if sim_hit:
                    for contrib in sim_hit.getContributions():
                        # The robust check
                        if contrib.getStepLength() == 0:
                            ml_hit_count += 1
                            break 

        if ml_hit_count > 10:
            ml_photon_energies.append(pfo.getEnergy())
        else:
            g4_photon_energies.append(pfo.getEnergy())

print(f"Categorization Complete!")
print(f"Found {len(ml_photon_energies)} Robust ML Photons")
print(f"Found {len(g4_photon_energies)} Geant4 Photons")
