from podio import root_io

input_file = "elec_pos_10k.edm4hep.root"
output_file = "tau_pi0_10GeV_filtered.edm4hep.root"

def has_high_energy_pi0_gamma(mc_particle, threshold=10.0):
    """
    Crawls the tau decay tree. 
    1. Finds a pi0.
    2. Checks the pi0's daughters(photons).
    3. Returns True if at least one photon has Energy >= threshold.
    """
    stack = list(mc_particle.getDaughters())
    while stack:
        curr = stack.pop()
        
        # 1. Look for the pi0
        if curr.getPDG() == 111:
            # 2. Check the pi0 daughters (should be photons)
            for daughter in curr.getDaughters():
                if daughter.getPDG() == 22: # It's a photon
                    # 3. Check Energy
                    if daughter.getEnergy() >= threshold:
                        return True
        
        # Keep digging if we haven't found a 10GeV photon yet
        stack.extend(curr.getDaughters())
    return False

# Setup Reader and Writer
reader = root_io.Reader(input_file)
writer = root_io.Writer(output_file)

print(f"Starting Skim with >={10} GeV Photon requirement...")

keep_count = 0
total_processed = 0

for event in reader.get("events"):
    total_processed += 1
    mc_particles = event.get("MCParticles")
    
    keep_this_event = False
    for mc in mc_particles:
        # Identify decaying Taus
        if abs(mc.getPDG()) == 15 and len(mc.getDaughters()) > 0:
            if has_high_energy_pi0_gamma(mc, 10.0):
                keep_this_event = True
                break
    
    if keep_this_event:
        writer.write_frame(event, "events")
        keep_count += 1

print("-" * 35)
print("Skimming Finished!")
print(f"Total processed: {total_processed}")
print(f"Saved (Pi0 + 10GeV Gamma): {keep_count}")
#below part doesnt include the 10 GeV cut
'''from podio import root_io

# Input and Output paths
input_file = "elec_pos_10k.edm4hep.root"
output_file = "tau_pi0_filtered_signal.edm4hep.root"

def has_pi0_from_tau(mc_particle):
    stack = list(mc_particle.getDaughters())
    while stack:
        curr = stack.pop()
        if curr.getPDG() == 111:
            return True
        stack.extend(curr.getDaughters())
    return False

# 1. Open the reader
reader = root_io.Reader(input_file)

# 2. Open the writer manually (No 'with' statement)
writer = root_io.Writer(output_file)

print(f"Starting skim of {input_file}...")

keep_count = 0
total_processed = 0

# 3. Process the events
for event in reader.get("events"):
    total_processed += 1
    mc_particles = event.get("MCParticles")
    
    keep_this_event = False
    
    for mc in mc_particles:
        # Check for decaying Tau
        if abs(mc.getPDG()) == 15 and len(mc.getDaughters()) > 0:
            if has_pi0_from_tau(mc):
                keep_this_event = True
                break
    
    if keep_this_event:
        # Using the frame-based writing required by Podio 1.7.0
        writer.write_frame(event, "events")
        keep_count += 1

# Note: No writer.close() or writer.finish() here because the 
# Python bindings for 1.7.0 handle it automatically on script exit.
print("-" * 35)
print("Skimming finished!")
print(f"Total events processed: {total_processed}")
print(f"Signal events saved:    {keep_count}")
print(f"Output saved to:        {output_file}")
'''
