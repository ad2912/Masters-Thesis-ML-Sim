from podio import root_io
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
input_file = "/data/dust/user/alimuham/thesis/reco/tau_pi0_RECO_caloclouds_REC.edm4hep.root"
reader = root_io.Reader(input_file)

# We categorize by (Software_Engine) + (Physics_Origin)
stats = {
    "G4: Pi0 from Tau": 0,
    "ML: Pi0 from Tau": 0,
    "G4: Direct Tau FSR": 0,
    "ML: Direct Tau FSR": 0,
    "G4: Background/Material": 0,
    "ML: Background/Material": 0,
    "Ghost/Unknown": 0
}

def get_tau_ancestor(mc_particle):
    """
    Climbs the family tree recursively. 
    Returns True if any ancestor is a Tau (PDG 15).
    """
    curr = mc_particle
    # Limit to 10 generations to avoid infinite loops in complex events
    for _ in range(10):
        parents = curr.getParents()
        if len(parents) == 0:
            return False
        
        # In decay chains, we usually care about the first parent
        parent = parents[0]
        if abs(parent.getPDG()) == 15:
            return True
        curr = parent
    return False

print(f"Analyzing {input_file}...")

for event in reader.get("events"):
    try:
        pfos = event.get("PandoraPFOs")
        barrel_rels = event.get("EcalBarrelRelationsSimRec")
        endcap_rels = event.get("EcalEndcapsRelationsSimRec")
        reco_mc_links = event.get("RecoMCTruthLink")
    except:
        continue

    # 1. Build lookup tables for this event
    hit_lookup = {rel.getFrom().getObjectID().index: rel.getTo() for rel in barrel_rels}
    hit_lookup.update({rel.getFrom().getObjectID().index: rel.getTo() for rel in endcap_rels})
    pfo_to_mc = {link.getFrom().getObjectID().index: link.getTo() for link in reco_mc_links}

    for pfo in pfos:
        if pfo.getPDG() != 22: continue # Photon check
        
        # 2. SOFTWARE TAG (Thomas's Rule)
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

        # 3. PHYSICS TAG (Recursive Ancestry)
        mc_particle = pfo_to_mc.get(pfo.getObjectID().index)
        if not mc_particle:
            stats["Ghost/Unknown"] += 1
            continue

        if get_tau_ancestor(mc_particle):
            # Check immediate mother to see if it's a Pi0 decay or direct FSR
            parents = mc_particle.getParents()
            mother_pdg = abs(parents[0].getPDG()) if len(parents) > 0 else 0
            
            if mother_pdg == 111:
                label = f"{engine}: Pi0 from Tau"
            else:
                label = f"{engine}: Direct Tau FSR"
        else:
            label = f"{engine}: Background/Material"
        
        stats[label] += 1

# --- PLOTTING ---
plt.figure(figsize=(14, 7))

# Define order and colors
# Blues for G4, Oranges for ML, Gray for Ghost
categories = [
    "G4: Pi0 from Tau", "ML: Pi0 from Tau",
    "G4: Direct Tau FSR", "ML: Direct Tau FSR",
    "G4: Background/Material", "ML: Background/Material",
    "Ghost/Unknown"
]
colors = ['#1f77b4', '#ff7f0e', '#aec7e8', '#ffbb78', '#34495e', '#7f8c8d', '#bdc3c7']

counts = [stats[cat] for cat in categories]
bars = plt.bar(categories, counts, color=colors)

plt.ylabel("Number of Photons", fontsize=12)
plt.title("Robust Physics & Software Origin Characterization", fontsize=14)
plt.xticks(rotation=25, ha='right')
plt.yscale('log') # Log scale is helpful since Pi0 counts are usually much higher than FSR

# Add numbers on top of bars
for bar in bars:
    yval = bar.get_height()
    if yval > 0:
        plt.text(bar.get_x() + bar.get_width()/2, yval * 1.1, int(yval), ha='center', va='bottom')

plt.tight_layout()
plt.savefig("robust_photon_characterization.png")
print("\nDone! Results saved to 'robust_photon_characterization.png'")
