#from podio import root_io

#file = "../results/reco/tau_geant4_REC.edm4hep.root"

#reader = root_io.Reader(file)

#pdg_set = set()
#counter = 0

#for event in reader.get("events"):
#    mc_particles = event.get("MCParticles")
#    for mc in mc_particles:
#        pdg_set.add(mc.getPDG())
#        counter += 1

#print("Total MCParticles scanned:", counter)
#print("Unique PDGs found:")
#print(sorted(pdg_set))

from podio import root_io

file ="../results/reco/tau_geant4_REC.edm4hep.root"

reader = root_io.Reader(file)

for event in reader.get("events"):
    for mc in event.get("MCParticles"):
        if abs(mc.getPDG()) == 11:
            print("Electron energy:", mc.getEnergy())
            print("Generator status:", mc.getGeneratorStatus())
            break
    break
