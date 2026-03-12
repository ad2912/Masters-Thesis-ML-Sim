from podio import root_io
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
input_file = "/data/dust/user/alimuham/thesis/reco/tau_pi0_RECO_caloclouds_REC.edm4hep.root"
reader = root_io.Reader(input_file)

# Data storage
ml_pfo_energies = []
g4_pfo_energies = []
event_total_ml_energy = []
resolution_ml = []
resolution_g4 = []

print(f"Analyzing {input_file}...")

for event in reader.get("events"):
    pfos = event.get("PandoraPFOs")
    
    # 1. Setup Relations for Rec->Sim and Rec->MC
    try:
        barrel_rels = event.get("EcalBarrelRelationsSimRec")
        endcap_rels = event.get("EcalEndcapsRelationsSimRec")
        # Standard EDM4hep collection for PFO -> MCParticle matching
        # If this name differs in your file, check podio-dump
        reco_mc_links = event.get("RecoMCTruthLink") 
    except:
        barrel_rels, endcap_rels, reco_mc_links = [], [], []

    # Map: RecHit Index -> SimHit
    hit_lookup = {}
    for rel in barrel_rels: hit_lookup[rel.getFrom().getObjectID().index] = rel.getTo()
    for rel in endcap_rels: hit_lookup[rel.getFrom().getObjectID().index] = rel.getTo()

    # Map: PFO Index -> MCParticle (for Truth comparison)
    pfo_to_mc = {}
    for link in reco_mc_links:
        pfo_to_mc[link.getFrom().getObjectID().index] = link.getTo()

    current_event_ml_energy = 0
    
    for pfo in pfos:
        if pfo.getPDG() != 22: continue
        
        # 2. Identify ML vs G4 (Thomas's Rule)
        ml_hit_count = 0
        for cluster in pfo.getClusters():
            for rec_hit in cluster.getHits():
                sim_hit = hit_lookup.get(rec_hit.getObjectID().index)
                if sim_hit:
                    for contrib in sim_hit.getContributions():
                        if contrib.getStepLength() == 0:
                            ml_hit_count += 1
                            break
        
        energy_reco = pfo.getEnergy()
        is_ml = ml_hit_count > 10
        
        # 3. Truth Matching & Resolution
        mc_particle = pfo_to_mc.get(pfo.getObjectID().index)
        if mc_particle:
            energy_true = mc_particle.getEnergy()
            res = energy_reco / energy_true
            if is_ml: resolution_ml.append(res)
            else: resolution_g4.append(res)

        # 4. Store Results
        if is_ml:
            ml_pfo_energies.append(energy_reco)
            current_event_ml_energy += energy_reco
        else:
            g4_pfo_energies.append(energy_reco)

    if current_event_ml_energy > 0:
        event_total_ml_energy.append(current_event_ml_energy)

# --- PLOTTING ---
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Individual PFO Energy (What you saw before)
axs[0].hist(g4_pfo_energies, bins=50, range=(0,50), alpha=0.5, label='G4 PFOs', color='blue')
axs[0].hist(ml_pfo_energies, bins=50, range=(0,50), alpha=0.6, label='ML PFOs', color='orange')
axs[0].set_title("Individual Photon Energy")
axs[0].set_yscale('log')
axs[0].set_xlabel("Energy [GeV]")
axs[0].legend()

# Plot 2: Total ML Energy Per Event (The Trigger Check)
axs[1].hist(event_total_ml_energy, bins=50, range=(0,50), color='green', alpha=0.7)
axs[1].axvline(10, color='red', linestyle='--', label='Trigger Threshold')
axs[1].set_title("Total ML Energy per Event")
axs[1].set_xlabel("Summed ML Energy [GeV]")
axs[1].legend()

# Plot 3: Energy Resolution (Reco / Truth)
axs[2].hist(resolution_g4, bins=50, range=(0.5, 1.5), alpha=0.5, label='G4', color='blue')
axs[2].hist(resolution_ml, bins=50, range=(0.5, 1.5), alpha=0.6, label='ML', color='orange')
axs[2].set_title("Energy Resolution ($E_{reco} / E_{true}$)")
axs[2].set_xlabel("Resolution")
axs[2].legend()

plt.tight_layout()
plt.savefig("photon_analysis_full_caloclouds.png")
print("Analysis complete. Saved 'photon_analysis_full_caloclouds.png'")
