from podio import root_io

file = "../steering/tau_pi0_10GeV_filtered.edm4hep.root"
reader = root_io.Reader(file)

event_count = 0
events_with_taus = 0
events_with_tau_pi0 = 0

for event in reader.get("events"):
    event_count += 1
    mc_particles = event.get("MCParticles")
    
    has_event_tau = False
    has_event_pi0_from_tau = False
    
    for mc in mc_particles:
        # Step 1: Identify Primary Taus
        if abs(mc.getPDG()) == 15 and mc.getGeneratorStatus() == 2:
            # Avoid counting radiated taus (only look at the "original" tau)
            is_primary = True
            for parent in mc.getParents():
                if abs(parent.getPDG()) == 15:
                    is_primary = False
                    break
            
            if not is_primary:
                continue
            
            # If we found at least one primary Tau, mark the event
            has_event_tau = True
            
            # Step 2: The "Tree-Crawler" logic to find a pi0 from this specific Tau
            stack = list(mc.getDaughters())
            while stack:
                current = stack.pop()
                if current.getPDG() == 111: # Found a pi0!
                    has_event_pi0_from_tau = True
                    break
                stack.extend(current.getDaughters())
        
        # Optimization: if we already found a pi0 for this event, we can stop looking at particles
        if has_event_pi0_from_tau:
            break

    if has_event_tau:
        events_with_taus += 1
    if has_event_pi0_from_tau:
        events_with_tau_pi0 += 1

# ---  "Valuable Numbers" Output ---
print(f"--- Statistics for {event_count} Total Events ---")
print(f"1. Events containing Taus: {events_with_taus}")
print(f"2. Events where a Tau decays into at least one pi0: {events_with_tau_pi0}")

if events_with_taus > 0:
    event_efficiency = (events_with_tau_pi0 / events_with_taus) * 100
    print(f"\nYour 'Event-Level' Usable Fraction: {event_efficiency:.2f}%")
#    print(f"Meaning: For every 100 Tau events, {event_efficiency:.1f} of them are useful for your study.")


#counts taus and taus decaying to pions, not events 
'''
from podio import root_io
from collections import Counter

file = "../steering/elec_pos_generatorfile.edm4hep.root"
reader = root_io.Reader(file)

total_taus = 0
taus_with_pi0 = 0
event_count = 0

for event in reader.get("events"):
    event_count += 1
    mc_particles = event.get("MCParticles")
    
    for mc in mc_particles:
        # Step 1: Find the "Source" Taus
        # We look for PDG 15 and status 2 (decaying). 
        # We check if the parent isn't also a Tau to avoid double-counting radiated taus.
        if abs(mc.getPDG()) == 15 and mc.getGeneratorStatus() == 2:
            is_primary = True
            for parent in mc.getParents():
                if abs(parent.getPDG()) == 15:
                    is_primary = False
                    break
            
            if not is_primary:
                continue
                
            total_taus += 1
            
            # Step 2: The "Tree-Crawler" logic
            # We put immediate daughters in a list and keep looking deeper
            found_pi0 = False
            stack = list(mc.getDaughters())
            
            while stack:
                current = stack.pop()
                if current.getPDG() == 111: # Found a pi0!
                    found_pi0 = True
                    break
                # If not a pi0, add its children to the stack to check them next
                stack.extend(current.getDaughters())
            
            if found_pi0:
                taus_with_pi0 += 1

# --- The "Decision" Output ---
print(f"--- Analysis of {event_count} Events ---")
print(f"Total Primary Taus found: {total_taus}")
print(f"Taus that eventually produce a pi0: {taus_with_pi0}")

if total_taus > 0:
    efficiency = (taus_with_pi0 / total_taus) * 100
    print(f"\nYour 'Usable' Fraction: {efficiency:.2f}%")
    
    if efficiency < 20:
        print("Note: This looks low. You might have a lot of leptonic decays in this file.")
'''
