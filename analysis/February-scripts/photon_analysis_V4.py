from podio import root_io
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
input_file = "/data/dust/user/alimuham/thesis/reco/tau_pi0_RECO_caloclouds_REC.edm4hep.root"
reader = root_io.Reader(input_file)

# Separate storage for Thomas's 3 plots
data = {
    "Pi0": {"ML": [], "G4": []},
    "FSR": {"ML": [], "G4": []},
    "Other": {"ML": [], "G4": []},
    "Residuals_ML": []
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

print(f"Analyzing {input_file} ")

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

        # --- 1. DEFINE ENGINE (The fix for your error) ---
        ml_hits = 0
        for cluster in pfo.getClusters():
            for rec_hit in cluster.getHits():
                sim_hit = hit_lookup.get(rec_hit.getObjectID().index)
                if sim_hit:
                    for contrib in sim_hit.getContributions():
                        if contrib.getStepLength() == 0:
                            ml_hits += 1
                            break
        engine = "ML" if ml_hits > 10 else "G4"

        # --- 2. DEFINE ORIGIN (Recursive Logic) ---
        mc_particle = pfo_to_mc.get(pfo.getObjectID().index)
        origin = "Other"
        
        if mc_particle:
            has_tau, mom_pdg = get_tau_ancestor_and_mother(mc_particle)
            if has_tau:
                origin = "Pi0" if mom_pdg == 111 else "FSR"

            # --- 3. COLLECT RESIDUALS (Thomas's Request #1) ---
            if engine == "ML":
                energy_true = mc_particle.getEnergy()
                data["Residuals_ML"].append(energy_reco - energy_true)

        # --- 4. CATEGORIZE FOR SEPARATE PLOTS (Thomas's Request #2) ---
        data[origin][engine].append(energy_reco)

# --- PLOTTING ---

# Plot 1: Normalized Residuals
plt.figure(figsize=(8, 6))
# Using density=True for normalization
plt.hist(data["Residuals_ML"], bins=50, range=(-5, 5), density=True, color='orange', alpha=0.7)
plt.axvline(0, color='black', linestyle='--') # Mark the "perfect" 0 point
plt.title("Normalized Energy Residuals (ML Only)\n$E_{reco} - E_{true}$")
plt.xlabel("$\Delta E$ [GeV]")
plt.ylabel("Probability Density")
plt.savefig("ml_energy_residuals_normalized.png")

# Plot 2: Three-Panel Comparison
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
titles = ["Photons from Pi0", "Photons from Tau FSR", "Other/Fake Photons"]
keys = ["Pi0", "FSR", "Other"]

for i, key in enumerate(keys):
    # To compare shapes fairly, we could also normalize these 
    axs[i].hist(data[key]["G4"], bins=30, range=(0, 50), label="Geant4", alpha=0.5, color='blue')
    axs[i].hist(data[key]["ML"], bins=30, range=(0, 50), label="CaloClouds", alpha=0.7, color='orange')
    axs[i].set_title(titles[i])
    axs[i].set_yscale('log')
    axs[i].set_xlabel("Energy [GeV]")
    axs[i].legend()

plt.tight_layout()
plt.savefig("physics_comparison_separated.png")
print("Plots generated: 'ml_energy_residuals_normalized.png' and 'physics_comparison_separated.png'")
