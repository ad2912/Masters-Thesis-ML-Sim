# environment.md

## Machine & Access
- Local OS: Linux
- Remote: NAF cluster (naf-ilc22.desy.de)
- Login: `ssh -X alimuham@naf-ilc.desy.de`

## Startup Ritual (every session)
```bash
source ~/source.sh
```
This sources Key4hep, DDML, fixes LD_LIBRARY_PATH for torch+onnx, sets aliases, and cds into project root.

## Aliases (set by source.sh)
| Alias       | Goes to                        |
|-------------|-------------------------------|
| cdwork      | ~/thesis-ml-sim/              |
| cdprod      | ~/thesis-ml-sim/steering/     |
| cdresults   | ~/thesis-ml-sim/results/      |
| cdanalysis  | ~/thesis-ml-sim/analysis/     |
| cdcog       | ~/thesis-ml-sim/cognitive-state/ |
| runsim-ml   | runs ddsim with CaloClouds steer |
| runsim-g4   | runs ddsim with Geant4 steer  |
| checkenv    | verifies all libs are resolved |

## Running Simulation
```bash
# ML simulation
runsim-ml --outputFile results/photon_gun_SIM_caloclouds.edm4hep.root

# Geant4 baseline
runsim-g4 --outputFile results/photon_gun_SIM_geant4.edm4hep.root
```

## Key4hep Details
- Release: 2026-02-01
- System: RHEL9 / AlmaLinux
- Python: 3.13 (from CVMFS, read-only)

## Critical LD_LIBRARY_PATH additions (handled by source.sh)
These are NOT set by Key4hep or DDML by default but are required:
- `.../py-torch/2.9.1-fl7w5y/.../torch/lib`   → libtorch_cpu.so, libc10.so
- `.../py-onnxruntime/1.22.2-haobv5/lib64`     → libonnxruntime.so.1

## Sanity Check
```bash
checkenv   # should print "OK: all libs resolved"
```

## Inspection
```bash
podio-dump file.root
python3 analysis/inspect-sim.py results/file.root
```
