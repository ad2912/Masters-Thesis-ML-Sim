from podio import root_io

# Input and Output paths
input_file = "../steering/elec_pos_generatorfile.edm4hep.root"
reader = root_io.Reader(input_file)

# Counters for your presentation slide
total_events = 0
primary_taus = 0
taus_with_pi0 = 0
ml_trigger_photons = 0  # Count photons >= 10 GeV from Pi0s
all_pi0_photons = []    # To calculate average energy if needed

print(f"Starting analysis of {input_file}...")

for event in reader.get("events"):
    total_events += 1
    mc_particles = event.get("MCParticles")
    
    for mc in mc_particles:
        # 1. Identify Primary Taus (Status 2, avoid double-counting radiated taus)
        if abs(mc.getPDG()) == 15 and mc.getGeneratorStatus() == 2:
            is_primary = True
            for parent in mc.getParents():
                if abs(parent.getPDG()) == 15:
                    is_primary = False
                    break
            
            if not is_primary:
                continue
                
            primary_taus += 1
            
            # 2. Tree-Crawler to find the Pi0 daughter
            stack = list(mc.getDaughters())
            found_pi0_for_this_tau = False
            
            while stack:
                current = stack.pop()
                
                # Check for Pi0
                if current.getPDG() == 111:
                    found_pi0_for_this_tau = True
                    
                    # 3. NEW: Count and check Energy of daughter Photons
                    for daughter in current.getDaughters():
                        if daughter.getPDG() == 22: # It's a photon
                            energy = daughter.getEnergy()
                            all_pi0_photons.append(energy)
                            
                            if energy >= 10.0:
                                ml_trigger_photons += 1
                    
                    # Once we find the Pi0 for this Tau, we move to the next Tau
                    break 
                
                # Add daughters to stack to keep crawling deeper
                stack.extend(current.getDaughters())
            
            if found_pi0_for_this_tau:
                taus_with_pi0 += 1

# --- Final Stats for the FTX Meeting ---
print("\n" + "="*40)
print("     GENERATOR FILE ANALYSIS RESULTS")
print("="*40)
print(f"Total Events Processed:      {total_events}")
print(f"Total Primary Taus Found:    {primary_taus}")
print(f"Taus Decaying to Pi0:        {taus_with_pi0}")
print(f"Pi0 Photons >= 10 GeV:       {ml_trigger_photons}")

if primary_taus > 0:
    efficiency = (taus_with_pi0 / primary_taus) * 100
    print(f"\nPi0 Channel Fraction:        {efficiency:.2f}%")

if taus_with_pi0 > 0:
    ml_potential = (ml_trigger_photons / (taus_with_pi0 * 2)) * 100
    print(f"Photons hitting ML Trigger:  {ml_potential:.2f}% (out of all Pi0 photons)")
print("="*40)
