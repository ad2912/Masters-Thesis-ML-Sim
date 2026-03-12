from podio import root_io
import matplotlib.pyplot as plt
import numpy as np

input_file = "/data/dust/user/alimuham/thesis/reco/tau_pi0_RECO_caloclouds_REC.edm4hep.root"
reader = root_io.Reader(input_file)

hit_level_sums = [] 
pfo_level_sums = []

# These are the exact names from your podio-dump that have Size > 0
sim_names = ["EcalBarrelCollection", "ECalBarrelSiHitsEven", "ECalBarrelSiHitsOdd"]

print(f"Executing Godly Move on {input_file}")

for event in reader.get("events"):
    pfos = event.get("PandoraPFOs")
    barrel_rels = event.get("EcalBarrelRelationsSimRec")
    
    if not pfos or not barrel_rels:
        continue
    # 1. Collect all available SimHits safely
    all_sim_hits = []
    for name in sim_names:
        try:
            coll = event.get(name)
            if coll:
                all_sim_hits.extend(coll)
        except KeyError:
            continue # Skip if the name isn't in this specific frame

    # 2. Map RecoHit Index -> SimHit Object
    hit_to_sim = {rel.getFrom().getObjectID().index: rel.getTo() for rel in barrel_rels}
    
    mcp_hit_energy = {} 
    mcp_pfo_energy = {} 

    # --- STEP A: THE TRUTH HITS (The Godly Move) ---
    for hit in all_sim_hits:
        # Check if the hit object is valid before calling contributions
        if not hit: continue
        for contrib in hit.getContributions():
            if contrib.getStepLength() == 0: # ML Tag
                mcp = contrib.getParticle()
                if mcp:
                    idx = mcp.getObjectID().index
                    mcp_hit_energy[idx] = mcp_hit_energy.get(idx, 0) + contrib.getEnergy()

    # --- STEP B: THE RECO CHECK (Pandora PFOs with 10-hit filter) ---
    for pfo in pfos:
        if pfo.getPDG() != 22: continue
        
        found_mcp_idx = None
        ml_hits_count = 0 
        
        for cluster in pfo.getClusters():
            for rec_hit in cluster.getHits():
                sim_hit = hit_to_sim.get(rec_hit.getObjectID().index)
                if sim_hit:
                    for c in sim_hit.getContributions():
                        if c.getStepLength() == 0:
                            ml_hits_count += 1
                            if found_mcp_idx is None and c.getParticle():
                                found_mcp_idx = c.getParticle().getObjectID().index
                            break 
        
        # APPLY THE 10-HIT THRESHOLD
        if ml_hits_count > 10 and found_mcp_idx is not None:
            mcp_pfo_energy[found_mcp_idx] = mcp_pfo_energy.get(found_mcp_idx, 0) + pfo.getEnergy()

    # Match up per MC Particle
    for idx, hit_e in mcp_hit_energy.items():
        if hit_e > 0.1:
            hit_level_sums.append(hit_e)
            pfo_level_sums.append(mcp_pfo_energy.get(idx, 0))

# --- PLOTTING ---
plt.figure(figsize=(12, 7))
plt.hist(hit_level_sums, bins=60, range=(0, 60), alpha=0.4, label="Total ML Energy (SimTruth)", color='blue')
plt.hist(pfo_level_sums, bins=60, range=(0, 60), alpha=0.6, label="Reconstructed Energy (>10 ML hits)", color='forestgreen')
plt.axvline(10, color='red', linestyle='--', label="10 GeV Threshold")
plt.yscale('log')
plt.title("Comparison: Sim Hits vs. Filtered PFOs")
plt.xlabel("Energy [GeV]")
plt.ylabel("Counts")
plt.legend()
plt.grid(alpha=0.2)
plt.savefig("godly_move_final.png")

print(f"Done. Analyzed {len(hit_level_sums)} particles. Plot saved as 'godly_move_final.png'")
