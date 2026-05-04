#!/bin/bash
JOB_ID=$1
SIM_TYPE=$2  # "g4" or "cc3"
NEVENTS=2000
SKIP=$((JOB_ID * NEVENTS))

# --- Environment setup ---
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh

# --- Paths ---
RECO_DIR=/afs/desy.de/user/a/alimuham/thesis-ml-sim/reconstruction
OUTDIR=/data/dust/user/alimuham/thesis/reco/100k-events

if [ "$SIM_TYPE" == "g4" ]; then
    INPUT=/data/dust/user/alimuham/thesis/sim/tau-pi0-geant4-100kevents-sim.edm4hep.root
    OUTBASE=$OUTDIR/g4/tau-pi0-geant4-reco-job${JOB_ID}
    LOG=$OUTDIR/g4/tau-pi0-geant4-reco-job${JOB_ID}.log
elif [ "$SIM_TYPE" == "cc3" ]; then
    INPUT=/data/dust/user/alimuham/thesis/sim/tau-pi0-caloclouds-100kevents-sim.edm4hep.root
    OUTBASE=$OUTDIR/cc3/tau-pi0-caloclouds-reco-job${JOB_ID}
    LOG=$OUTDIR/cc3/tau-pi0-caloclouds-reco-job${JOB_ID}.log
else
    echo "ERROR: SIM_TYPE must be g4 or cc3, got: $SIM_TYPE"
    exit 1
fi

# --- Timing ---
START=$(date +%s)
echo "Job $JOB_ID | SIM_TYPE=$SIM_TYPE | SKIP=$SKIP | NEVENTS=$NEVENTS" > $LOG
echo "Start: $(date)" >> $LOG

# --- Run reconstruction ---
cd $RECO_DIR

k4run ILDReconstruction.py \
    --inputFiles=$INPUT \
    --outputFileBase=$OUTBASE \
    --IOSvc.FirstEventEntry=$SKIP \
    -n $NEVENTS \
    2>&1 | tee -a $LOG

END=$(date +%s)
echo "End: $(date)" >> $LOG
echo "Wall time: $((END - START)) seconds" >> $LOG
