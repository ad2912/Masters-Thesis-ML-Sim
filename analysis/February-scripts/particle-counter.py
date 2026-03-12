from podio import root_io
from collections import Counter

# Path to your file
file = "../results/sim/tautest2_gun_SIM_geant4.edm4hep.root"
reader = root_io.Reader(file)

# Dictionary to map common PDG IDs to names for readability
pdg_names = {
    11: "e-", -11: "e+",
    12: "nu_e", -12: "anti_nu_e",
    13: "mu-", -13: "mu+",
    14: "nu_mu", -14: "anti_nu_mu",
    16: "nu_tau", -16: "anti_nu_tau",
    22: "photon",
    111: "pi0",
    211: "pi+", -211: "pi-",
    130: "K_long",
    310: "K_short",
    321: "K+", -321: "K-",
    2112: "neutron",
    2212: "proton",
    # Add any others you see in your output
}

particle_counts = Counter()
event_count = 0

# Loop through events
for event in reader.get("events"):
    event_count += 1
    mc_particles = event.get("MCParticles")
    
    for mc in mc_particles:
        pdg = mc.getPDG()
        particle_counts[pdg] += 1

print(f"--- Statistics for {event_count} events ---")
print(f"{'PDG':>8} | {'Name':>12} | {'Total Count':>12} | {'Avg/Event':>10}")
print("-" * 50)

# Sort by frequency (most common particles first)
for pdg, count in particle_counts.most_common():
    name = pdg_names.get(pdg, "unknown")
    avg = count / event_count
    print(f"{pdg:8d} | {name:>12} | {count:12d} | {avg:10.2f}")
