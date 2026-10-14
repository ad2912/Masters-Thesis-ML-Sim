import numpy as np
from podio import root_io

#G4_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-1000events-test_REC.edm4hep.root"
CC3_FILE="/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-1000events-test_REC.edm4hep.root"
reader = root_io.Reader(CC3_FILE)

raw_weights = []
n_events = 0
for event in reader.get("events"):
    n_events += 1
    for link in event.get("RecoMCTruthLink"):
        raw_weights.append(link.getWeight())
    if n_events >= 200:
        break

raw_weights = np.array(raw_weights)


print(f"Events checked   : {n_events}")
print(f"N links          : {len(raw_weights)}")
print(f"min              : {raw_weights.min()}")
print(f"max              : {raw_weights.max()}")
print(f"percentiles      : {np.percentile(raw_weights, [1,25,50,75,99])}")
print(f"first 15 values  : {raw_weights[:15]}")
print(f"values > 1.0     : {np.sum(raw_weights > 1.0)}  ({100*np.mean(raw_weights>1.0):.1f}%)")
print(f"values > 1000    : {np.sum(raw_weights > 1000)}")
print(f"values > 1e6     : {np.sum(raw_weights > 1e6)}")
