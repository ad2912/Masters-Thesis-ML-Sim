# current_task.md

## What I am doing right now
Setting up clean project structure, git version control, and environment management for the ML shower simulation study.

## Why
- Old working directory was messy (production/ had steer files, ROOT files, analysis scripts, backups all mixed together)
- Need reproducible environment setup ritual for every NAF session
- Need git backup so work is never lost

## Concrete next steps
1. Run bootstrap.sh to create ~/thesis-ml-sim/ structure
2. Copy source.sh to ~ (home directory) so it's always findable
3. Fix GitHub SSH keys on NAF
4. Init git repo in ~/thesis-ml-sim/ and push to GitHub
5. Run systematic comparison: ML vs Geant4 ECAL plots

## Environment ritual (every SSH session)
```bash
source ~/source.sh
```
That's it. Everything else is handled.

## Confusions / Open Questions
- CaloClouds3 has an energy threshold (9.5 GeV). What happens to photons below threshold? Does Geant4 take over automatically?
- Need to understand the point-cloud output format of CaloClouds3 and how it maps to EDM4hep SimCalorimeterHit
