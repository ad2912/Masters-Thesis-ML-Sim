from podio import root_io
import matplotlib.pyplot as plt
import numpy as np
import sys

# --- CONFIGURATION ---
# Change this path to switch between files
#file_path = "/data/dust/user/alimuham/thesis/reco/tau_pi0_RECO_geant4_REC.edm4hep.root"
file_path = "/data/dust/user/alimuham/thesis/reco/tau_pi0_RECO_caloclouds_REC.edm4hep.root"

def run_analysis(input_file):
    print(f"Reading file: {input_file}")
    reader = root_io.Reader(input_file)
    
    ml_energies = []
    g4_energies = []

    for event in reader.get("events"):
        pfos = event.get("PandoraPFOs")
        
        # Load Relations to bridge Rec -> Sim
        # Note: Some events might lack one of these if hits are only in one region
        try:
            barrel_rels = event.get("EcalBarrelRelationsSimRec")
        except: barrel_rels = []
        try:
            endcap_rels = event.get("EcalEndcapsRelationsSimRec")
        except: endcap_rels = []
        
        # Build the lookup dictionary {RecHit_Index: SimHit}
        # Using .getFrom() and .getTo() for PODIO Link API
        hit_lookup = {}
        for rel in barrel_rels:
            hit_lookup[rel.getFrom().getObjectID().index] = rel.getTo()
        for rel in endcap_rels:
            hit_lookup[rel.getFrom().getObjectID().index] = rel.getTo()

        for pfo in pfos:
            # Filter for Photons (PDG 22)
            if pfo.getPDG() != 22:
                continue
                
            ml_hit_count = 0
            for cluster in pfo.getClusters():
                for rec_hit in cluster.getHits():
                    sim_hit = hit_lookup.get(rec_hit.getObjectID().index)
                    if sim_hit:
                        # Check contributions for StepLength == 0 (ML Signature)
                        for contrib in sim_hit.getContributions():
                            if contrib.getStepLength() == 0:
                                ml_hit_count += 1
                                break # Count once per hit
            
            # CATEGORIZATION ( 10-hit Rule)
            energy = pfo.getEnergy()
            if ml_hit_count > 10:
                ml_energies.append(energy)
            else:
                g4_energies.append(energy)

    return ml_energies, g4_energies

# Run Analysis
ml_e, g4_e = run_analysis(file_path)

# --- STATISTICS ---
print("\n" + "="*30)
print(f"RESULTS FOR: {file_path.split('/')[-1]}")
print(f"ML Photons: {len(ml_e)}")
if ml_e:
    print(f"  - Energy Range: {min(ml_e):.2f} to {max(ml_e):.2f} GeV")
    print(f"  - Mean Energy: {np.mean(ml_e):.2f} GeV")

print(f"G4 Photons: {len(g4_e)}")
if g4_e:
    print(f"  - Energy Range: {min(g4_e):.2f} to {max(g4_e):.2f} GeV")
    print(f"  - Mean Energy: {np.mean(g4_e):.2f} GeV")
print("="*30 + "\n")

# --- PLOTTING ---
plt.figure(figsize=(10, 6))

# Define bins for 0 to 50 GeV (typical for tau decays)
bins = np.linspace(0, 50, 50)

# Plot Geant4 Photons
plt.hist(g4_e, bins=bins, alpha=0.5, label=f'Geant4 ({len(g4_e)})', color='blue', edgecolor='darkblue')

# Plot ML Photons
if ml_e:
    plt.hist(ml_e, bins=bins, alpha=0.6, label=f'CaloClouds ({len(ml_e)})', color='orange', edgecolor='darkorange')

plt.title(f"Photon Energy Distribution ($E_{{reco}}$)\nFile: {file_path.split('/')[-1]}")
plt.xlabel("Reconstructed Energy [GeV]")
plt.ylabel("Counts")
plt.yscale('log') # Log scale is better for seeing the 10 GeV transition
plt.grid(axis='y', alpha=0.3)
plt.legend()

output_plot = "photon_energy_dist.png"
plt.savefig(output_plot, dpi=300)
print(f"Plot saved as {output_plot}")
