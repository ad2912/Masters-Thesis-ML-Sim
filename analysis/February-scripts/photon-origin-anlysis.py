from podio import root_io
import matplotlib.pyplot as plt
import numpy as np

input_file = "/data/dust/user/alimuham/thesis/reco/tau_pi0_RECO_caloclouds_REC.edm4hep.root"
reader = root_io.Reader(input_file)

# We update the dictionary to match the recursive categories
data_to_plot = {
    "G4 from Pi0": [],
    "ML from Pi0": [],
    "G4 from Tau (FSR)": [],
    "ML from Tau (FSR)": [],
    "Other/Fake": []
}

def get_tau_ancestor_and_mother(mc_particle):
    """Climbs tree to find Tau and returns the immediate mother PDG."""
    curr = mc_particle
    immediate_mother_pdg = 0
    if len(curr.getParents()) > 0:
        immediate_mother_pdg = abs(curr.getParents()[0].getPDG())
        
    found_tau = False
    for _ in range(10):
        parents = curr.getParents()
        if len(parents) == 0: break
        parent = parents[0]
        if abs(parent.getPDG()) == 15:
            found_tau = True
            break
        curr = parent
    return found_tau, immediate_mother_pdg

print(f"Analyzing {input_file} with recursive logic...")

for event in reader.get("events"):
    try:
        pfos = event.get("PandoraPFOs")
        barrel_rels = event.get("EcalBarrelRelationsSimRec")
        endcap_rels = event.get("EcalEndcapsRelationsSimRec")
        reco_mc_links = event.get("RecoMCTruthLink")
    except: continue

    hit_lookup = {rel.getFrom().getObjectID().index: rel.getTo() for rel in barrel_rels}
    hit_lookup.update({rel.getFrom().getObjectID().index: rel.getTo() for rel in endcap_rels})
    pfo_to_mc = {link.getFrom().getObjectID().index: link.getTo() for link in reco_mc_links}

    for pfo in pfos:
        if pfo.getPDG() != 22: continue
        
        energy_reco = pfo.getEnergy()

        # 1. Software Logic
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
        engine = "ML" if is_ml else "G4"

        # 2. Recursive Physics Logic
        mc_particle = pfo_to_mc.get(pfo.getObjectID().index)
        label = "Other/Fake" # Default
        
        if mc_particle:
            has_tau, mom_pdg = get_tau_ancestor_and_mother(mc_particle)
            if has_tau:
                if mom_pdg == 111:
                    label = f"{engine} from Pi0"
                else:
                    label = f"{engine} from Tau (FSR)"

        data_to_plot[label].append(energy_reco)

# --- PLOTTING ---
plt.figure(figsize=(10, 7))
categories = ["G4 from Pi0", "ML from Pi0", "G4 from Tau (FSR)", "ML from Tau (FSR)", "Other/Fake"]
colors = ['#1f77b4', '#ff7f0e', '#aec7e8', '#ffbb78', '#7f7f7f'] 

plt.hist([data_to_plot[cat] for cat in categories], 
         bins=50, range=(0, 50), stacked=True, 
         label=categories, color=colors, alpha=0.8)

plt.yscale('log')
plt.xlabel("Reconstructed Energy [GeV]")
plt.ylabel("Number of Photons")
plt.title("RECURSIVE Photon Energy Spectrum: Physics vs. Software")
plt.legend()
plt.savefig("photon_physics_origin_stacked_RECURSIVE.png")
print("Done! Check 'photon_physics_origin_stacked_RECURSIVE.png'")
