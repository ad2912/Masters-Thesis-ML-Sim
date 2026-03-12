from podio import root_io
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
input_file = "/data/dust/user/alimuham/thesis/reco/tau_pi0_RECO_caloclouds_REC.edm4hep.root"
reader = root_io.Reader(input_file)

# Data storage
data = {
    "Pi0": {"ML": [], "G4": []},
    "FSR": {"ML": [], "G4": []},
    "Other": {"ML": [], "G4": []},
    "Event_Residuals_ML": [],
    "Resolution_G4": [],
    "Resolution_ML": []
}

# Stats Counters
total_ml_pfo_count = 0
total_trigger_truth_photons = 0
truth_indices_simulated_by_ml = set()
truth_indices_simulated_by_g4 = set()

def get_tau_ancestor_and_mother(mc_particle):
    """Climbs decay tree to find Tau and immediate mother."""
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

print(f"Running Final Thomas-Validation on: {input_file}")

for event in reader.get("events"):
    try:
        pfos = event.get("PandoraPFOs")
        mcp_coll = event.get("MCParticles")
        barrel_rels = event.get("EcalBarrelRelationsSimRec")
        endcap_rels = event.get("EcalEndcapsRelationsSimRec")
        reco_mc_links = event.get("RecoMCTruthLink")
    except: continue

    hit_lookup = {rel.getFrom().getObjectID().index: rel.getTo() for rel in barrel_rels}
    hit_lookup.update({rel.getFrom().getObjectID().index: rel.getTo() for rel in endcap_rels})
    pfo_to_mc = {link.getFrom().getObjectID().index: link.getTo() for link in reco_mc_links}

    event_ml_reco_sum = 0
    event_has_ml = False

    # 1. PFO LOOP
    for pfo in pfos:
        if pfo.getPDG() != 22: continue
        
        # StepLength check for engine
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
        
        if is_ml:
            event_ml_reco_sum += pfo.getEnergy()
            event_has_ml = True
            total_ml_pfo_count += 1

        # Track which truth particle this PFO came from
        mc_particle = pfo_to_mc.get(pfo.getObjectID().index)
        if mc_particle:
            idx = mc_particle.getObjectID().index
            if is_ml: truth_indices_simulated_by_ml.add(idx)
            else: truth_indices_simulated_by_g4.add(idx)

            # Physics Categorization
            has_tau, mom_pdg = get_tau_ancestor_and_mother(mc_particle)
            origin = "Other"
            if has_tau:
                origin = "Pi0" if mom_pdg == 111 else "FSR"
            
            data[origin][engine].append(pfo.getEnergy())
            data[f"Resolution_{engine}"].append(pfo.getEnergy() / mc_particle.getEnergy())

    # 2. TRUTH TRIGGER CHECK (Strictly >= 10 GeV)
    event_truth_trigger_sum = 0
    event_truth_count = 0
    for mcp in mcp_coll:
        if mcp.getPDG() == 22 and mcp.getGeneratorStatus() == 1 and mcp.getEnergy() >= 10.0:
            event_truth_trigger_sum += mcp.getEnergy()
            event_truth_count += 1
            total_trigger_truth_photons += 1

    if event_has_ml and event_truth_count > 0:
        data["Event_Residuals_ML"].append(event_ml_reco_sum - event_truth_trigger_sum)

# --- RESULTS & DOUBLE SIMULATION CHECK ---
double_sim_count = len(truth_indices_simulated_by_ml.intersection(truth_indices_simulated_by_g4))
frag_ratio = total_ml_pfo_count / total_trigger_truth_photons if total_trigger_truth_photons > 0 else 0

print("\n" + "="*45)
print("FINAL VALIDATION REPORT")
print("-" * 45)
print(f"Total Truth Photons (>=10GeV):    {total_trigger_truth_photons}")
print(f"Total ML PFOs Reconstructed:      {total_ml_pfo_count}")
print(f"Average Fragments per Photon:     {frag_ratio:.3f}")
print(f"Double Simulated Particles:       {double_sim_count}")
print("="*45 + "\n")

# --- PLOTTING ---
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
titles = ["Pi0 Photons", "Tau FSR Photons", "Other/Fake"]
keys = ["Pi0", "FSR", "Other"]
for i, key in enumerate(keys):
    axs[i].hist(data[key]["G4"], bins=40, range=(0, 50), label="G4", alpha=0.5, density=True)
    axs[i].hist(data[key]["ML"], bins=40, range=(0, 50), label="ML", alpha=0.6, density=True)
    axs[i].set_title(titles[i]); axs[i].set_yscale('log'); axs[i].legend()

plt.savefig("physics_origin_final.png")

plt.figure()
plt.hist(data["Resolution_G4"], bins=80, range=(0.5, 1.5), alpha=0.5, label="G4", density=True)
plt.hist(data["Resolution_ML"], bins=80, range=(0.5, 1.5), alpha=0.5, label="ML", density=True)
plt.title("Normalized Resolution Comparison"); plt.legend()
plt.savefig("resolution_final.png")
