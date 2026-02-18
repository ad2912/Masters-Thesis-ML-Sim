# project_state.md

## Thesis Goal
Study the impact of ML-based calorimeter shower simulation (DDML / CaloClouds3) compared to Geant4 within the ILD detector framework.

Primary question: How do physics observables differ when using ML-based fast simulation vs full Geant4?

## Software Stack
- Framework: Key4hep (release 2026-02-01)
- Geometry: ILD_l5_v02
- Simulation: DDSim (Geant4 via DD4hep)
- ML plugin: DDML (libDDML.so, built from source)
- ML model: CaloClouds3 (CC3_SF_2A.pt) — point-cloud based ECAL shower model
- Data format: EDM4hep / podio
- Analysis: Python (matplotlib)

## Current Status
DDML environment is fully working. CaloClouds3 ML simulation runs successfully.
Both ML and Geant4 ROOT output files exist. Basic ECAL energy plots confirmed working.

Ready to move into systematic comparison analysis.

## Project Structure
```
~/thesis-ml-sim/
├── steering/        # ddsim steer files
├── analysis/        # python analysis + plotting scripts
├── results/         # ROOT output files (.edm4hep.root)
├── models/          # symlinks to .pt model weight files
├── plots/           # saved figures
├── cognitive-state/ # these markdown files
└── docs/            # setup documentation
```

## Key Paths
- Project root:  /afs/desy.de/user/a/alimuham/thesis-ml-sim/
- DDML install:  /afs/desy.de/user/a/alimuham/key4hep_tut_ild_reco/DDML/install/
- Geometry file: $k4geo_DIR/ILD/compact/ILD_l5_v02/ILD_l5_v02.xml

## Current Dataset
- Process: single photon gun, direction (1,0,0), 10 GeV, 10 events
- photon_gun_SIM.edm4hep.root          — Geant4 baseline
- photon_gun_SIM_caloclouds.edm4hep.root — CaloClouds3 ML

## Near-Term Milestones
- [ ] Systematic comparison plots: ECAL energy, hit multiplicity, shower shape
- [ ] Increase statistics (100+ events) for meaningful distributions
- [ ] Understand CaloClouds3 model architecture and what it approximates
- [ ] Document physics differences observed
