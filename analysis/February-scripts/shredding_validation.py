from podio import root_io
import matplotlib.pyplot as plt
import numpy as np

input_file = "/data/dust/user/alimuham/thesis/reco/tau_pi0_RECO_caloclouds_REC.edm4hep.root"
reader = root_io.Reader(input_file)

summed_ml_energies = [] 
shredding_counts = []

print(f"Running Thomas-Corrected Validation: {input_file}")

for event in reader.get("events"):
    pfos = event.get("PandoraPFOs")
    barrel_rels = event.get("EcalBarrelRelationsSimRec")
    # Adding Endcaps just in case, to be safe
    endcap_rels = event.get("EcalEndcapsRelationsSimRec")
    
    if not pfos or not barrel_rels: continue

    # Build lookup: RecoHit Index -> SimHit
    hit_lookup = {rel.getFrom().getObjectID().index: rel.getTo() for rel in barrel_rels}
    if endcap_rels:
        hit_lookup.update({rel.getFrom().getObjectID().index: rel.getTo() for rel in endcap_rels})

    # Grouping dictionary: { MC_Particle_Object_Pointer : [list of energies] }
    # We use the object itself or its ID to group fragments
    mcp_groups = {}

    for pfo in pfos:
        if pfo.getPDG() != 22: continue

        ml_hits = 0
        found_mc_particle = None
        
        for cluster in pfo.getClusters():
            for rec_hit in cluster.getHits():
                sim_hit = hit_lookup.get(rec_hit.getObjectID().index)
                
                if sim_hit:
                    for contrib in sim_hit.getContributions():
                        # CHECK 1: Is it ML?
                        if contrib.getStepLength() == 0:
                            ml_hits += 10
                            # CHECK 2: Get the TRUE MC Particle from the contribution
                            # This is what Thomas wants!
                            if not found_mc_particle:
                                found_mc_particle = contrib.getParticle()
                            break # Found ML contribution for this hit
        
        # Applying the hit threshold (Thomas suggested >1 or >10)
        if ml_hits > 1 and found_mc_particle:
            mcp_idx = found_mc_particle.getObjectID().index
            if mcp_idx not in mcp_groups:
                mcp_groups[mcp_idx] = []
            mcp_groups[mcp_idx].append(pfo.getEnergy())

    # Sum up fragments for each MC Particle identified via contributions
    for mcp_idx, energies in mcp_groups.items():
        summed_ml_energies.append(sum(energies))
        shredding_counts.append(len(energies))

# Plotting
if summed_ml_energies:
    plt.figure(figsize=(10, 6))
    plt.hist(summed_ml_energies, bins=60, range=(0, 60), color='forestgreen', alpha=0.7)
    plt.axvline(10, color='red', linestyle='--', label="10 GeV Trigger")
    plt.yscale('log')
    plt.title("Per-Particle Summed ML Energy (Contribution-Linked)")
    plt.xlabel("Summed Reco Energy [GeV]")
    plt.savefig("per_particle_sum_contribution_link.png")
    print(f"Done! Found {len(summed_ml_energies)} matched MC particles.")
    print(f"Average fragments per photon: {np.mean(shredding_counts):.2f}")
else:
    print("No data found. The StepLength or Particle link in contributions might be empty.")
