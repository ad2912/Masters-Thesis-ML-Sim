from podio import root_io
import matplotlib.pyplot as plt

input_file = "/data/dust/user/alimuham/thesis/reco/tau_pi0_RECO_caloclouds_REC.edm4hep.root"
reader = root_io.Reader(input_file)

# Categorization storage
# Format: { (Is_ML, Parent_PDG): [energies] }
results = {
    "ML_from_Pi0": [],
    "ML_from_Tau": [],
    "G4_from_Pi0": [],
    "G4_from_Tau": []
}

for event in reader.get("events"):
    pfos = event.get("PandoraPFOs")
    
    # Bridge collections
    try:
        barrel_rels = event.get("EcalBarrelRelationsSimRec")
        endcap_rels = event.get("EcalEndcapsRelationsSimRec")
        reco_mc_links = event.get("RecoMCTruthLink")
    except: continue

    # Mapping
    hit_lookup = {rel.getFrom().getObjectID().index: rel.getTo() for rel in barrel_rels}
    hit_lookup.update({rel.getFrom().getObjectID().index: rel.getTo() for rel in endcap_rels})
    pfo_to_mc = {link.getFrom().getObjectID().index: link.getTo() for link in reco_mc_links}

    for pfo in pfos:
        if pfo.getPDG() != 22: continue
        
        # 1. Determine if ML (Thomas's Rule)
        ml_hits = 0
        for cluster in pfo.getClusters():
            for rec_hit in cluster.getHits():
                sim_hit = hit_lookup.get(rec_hit.getObjectID().index)
                if sim_hit:
                    for contrib in sim_hit.getContributions():
                        if contrib.getStepLength() == 0:
                            ml_hits += 1
                            break
        is_ml = ml_hits > 10

        # 2. Determine Physics Origin
        mc_particle = pfo_to_mc.get(pfo.getObjectID().index)
        parent_type = "Other"
        if mc_particle and len(mc_particle.getParents()) > 0:
            parent_pdg = abs(mc_particle.getParents()[0].getPDG())
            if parent_pdg == 111: parent_type = "Pi0"
            elif parent_pdg == 15: parent_type = "Tau"

        # 3. Store
        key = f"{'ML' if is_ml else 'G4'}_from_{parent_type}"
        if key in results:
            results[key].append(pfo.getEnergy())

# --- Print Summary for Thomas ---
print(f"{'Category':<15} | {'Count':<10} | {'Mean Energy':<12}")
print("-" * 45)
for cat, energies in results.items():
    mean_e = sum(energies)/len(energies) if energies else 0
    print(f"{cat:<15} | {len(energies):<10} | {mean_e:<12.2f} GeV")
