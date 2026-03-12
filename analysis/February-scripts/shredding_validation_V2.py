from podio import root_io
import matplotlib.pyplot as plt
import numpy as np

input_file = "/data/dust/user/alimuham/thesis/reco/tau_pi0_RECO_caloclouds_REC.edm4hep.root"
reader = root_io.Reader(input_file)

ml_gaps = []    
total_gaps = [] 
theta_angles = []

print(f"DEBUGGING Engine-Handoff Analysis: {input_file}")

# Counter for diagnostics
stats = {"total_pfos": 0, "photons": 0, "ml_tagged": 0, "linked_to_mc": 0}

for event in reader.get("events"):
    pfos = event.get("PandoraPFOs")
    reco_mc_links = event.get("RecoMCTruthLink")
    
    # Check for BOTH Barrel and Endcap relations
    barrel_rels = event.get("EcalBarrelRelationsSimRec")
    endcap_rels = event.get("EcalEndcapsRelationsSimRec")
    
    if not pfos or not reco_mc_links: continue

    # Build a combined lookup for hits
    hit_lookup = {}
    if barrel_rels:
        hit_lookup.update({rel.getFrom().getObjectID().index: rel.getTo() for rel in barrel_rels})
    if endcap_rels:
        hit_lookup.update({rel.getFrom().getObjectID().index: rel.getTo() for rel in endcap_rels})

    # Build the link lookup
    pfo_to_mc = {link.getFrom().getObjectID().index: link.getTo() for link in reco_mc_links}

    mcp_tracker = {}

    for pfo in pfos:
        stats["total_pfos"] += 1
        if pfo.getPDG() != 22: continue
        stats["photons"] += 1

        ml_hits = 0
        for cluster in pfo.getClusters():
            for hit in cluster.getHits():
                sim_hit = hit_lookup.get(hit.getObjectID().index)
                if sim_hit:
                    if any(c.getStepLength() == 0 for c in sim_hit.getContributions()):
                        ml_hits += 1
                        break
        
        is_ml = ml_hits > 10 
        if is_ml: stats["ml_tagged"] += 1
        
        mc_particle = pfo_to_mc.get(pfo.getObjectID().index)
        if mc_particle:
            if is_ml: stats["linked_to_mc"] += 1
            idx = mc_particle.getObjectID().index
            if idx not in mcp_tracker:
                mom = mc_particle.getMomentum()
                mag = np.sqrt(mom.x**2 + mom.y**2 + mom.z**2)
                theta = np.degrees(np.arccos(mom.z / mag)) if mag > 0 else 0
                mcp_tracker[idx] = {'ml_sum': 0, 'g4_sum': 0, 'mc_e': mc_particle.getEnergy(), 'theta': theta}
            
            if is_ml:
                mcp_tracker[idx]['ml_sum'] += pfo.getEnergy()
            else:
                mcp_tracker[idx]['g4_sum'] += pfo.getEnergy()

    for idx, d in mcp_tracker.items():
        if d['ml_sum'] > 0: 
            ml_gaps.append(d['mc_e'] - d['ml_sum'])
            total_gaps.append(d['mc_e'] - (d['ml_sum'] + d['g4_sum']))
            theta_angles.append(d['theta'])

print("\n--- DIAGNOSTIC SUMMARY ---")
print(f"Total PFOs checked: {stats['total_pfos']}")
print(f"Total Photons (PDG 22): {stats['photons']}")
print(f"Photons tagged as ML (>10 hits): {stats['ml_tagged']}")
print(f"ML Photons successfully linked to MC: {stats['linked_to_mc']}")
print(f"Data points for plotting: {len(ml_gaps)}")

if len(ml_gaps) > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.hist(ml_gaps, bins=50, range=(-5, 20), color='red', alpha=0.7, label="ML Gap")
    ax1.hist(total_gaps, bins=50, range=(-5, 20), color='green', alpha=0.5, label="Total Gap")
    ax1.set_yscale('log')
    ax1.legend()
    
    ax2.scatter(theta_angles, ml_gaps, alpha=0.5, s=5)
    ax2.set_xlabel("Theta")
    ax2.set_ylabel("E_mc - E_ml")
    
    plt.savefig("thomas_validation_fixed.png")
    print("\nPlots saved to 'thomas_validation_fixed.png'")
else:
    print("\nERROR: No data to plot. Check if ML Photons were linked to MC.")
