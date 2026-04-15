import numpy as np
from podio import root_io
import edm4hep
import os

# ─── Config ───────────────────────────────────────────────────────────────────
GEN_FILE = "/data/dust/user/alimuham/thesis/InputFiles/tau_pi0_10GeV_filtered_500kevents.edm4hep.root"
G4_FILE  = "/data/dust/user/alimuham/thesis/sim/tau-pi0-geant4-100kevents-sim.edm4hep.root"
CC3_FILE = "/data/dust/user/alimuham/thesis/sim/tau-pi0-caloclouds-100kevents-sim.edm4hep.root"

MAX_EVENTS = 100_000
PHOTON_PDG = 22
PI0_PDG    = 111
TAU_PDG    = 15
E_PDG      = 11

def produce_master_ntuple(input_path, output_name, max_events=MAX_EVENTS):
    if not os.path.exists(input_path):
        print(f"Skipping {input_path} (not found)")
        return
    
    print(f"Processing {input_path}...")
    reader = root_io.Reader(input_path)
    
    data = {"e": [], "theta": [], "phi": [], "stat": [], "fs": [], "cat": []}
    
    events = reader.get("events")
    for i, event in enumerate(events):
        if i >= max_events: break
        
        particles = list(event.get("MCParticles"))
        pi0_children = {child.getObjectID().index for p in particles 
                        if abs(p.getPDG()) == PI0_PDG for child in p.getDaughters()}

        for p in particles:
            if p.getPDG() != PHOTON_PDG: continue

            p4 = edm4hep.utils.p4(p)
            data["e"].append(p.getEnergy())
            data["theta"].append(np.degrees(float(p4.theta())))
            data["phi"].append(np.degrees(float(p4.phi())))
            data["stat"].append(p.getGeneratorStatus())

            # Check Fast-Sim handling
            has_fs = False
            if hasattr(p, "isHandledByFastSim"):
                try: has_fs = bool(p.isHandledByFastSim())
                except: pass
            data["fs"].append(has_fs)

            # Categorization logic
            parent_pdgs = {abs(par.getPDG()) for par in p.getParents()}
            idx = p.getObjectID().index
            if idx in pi0_children: cat = 0
            elif E_PDG in parent_pdgs: cat = 1
            elif TAU_PDG in parent_pdgs: cat = 2
            else: cat = 3
            data["cat"].append(cat)

    np.savez_compressed(output_name, **{k: np.array(v) for k, v in data.items()})
    print(f"Saved to {output_name}")

if __name__ == "__main__":
    produce_master_ntuple(GEN_FILE, "gen_master.npz")
    produce_master_ntuple(G4_FILE,  "g4_master.npz")
    produce_master_ntuple(CC3_FILE, "cc3_master.npz")
