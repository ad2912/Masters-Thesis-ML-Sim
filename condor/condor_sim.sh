#!/bin/bash

JOB_ID=$1
SIM_TYPE=$2  # "g4" or "cc3"
NEVENTS=2000
SKIP=$((JOB_ID * NEVENTS))

# --- Environment setup (non-interactive, cannot use source.sh) ---
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh

DDML_INSTALL=/afs/desy.de/user/a/alimuham/key4hep_tut_ild_reco/DDML/install
source $DDML_INSTALL/bin/thisDDML.sh

TORCH_PATH=$(dirname $(python3 -c 'import torch; print(torch.__file__)'))
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${TORCH_PATH}/lib

ONNXRUNTIME_PATH=$(dirname $(python3 -c 'import onnxruntime; print(onnxruntime.__file__)'))
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(realpath ${ONNXRUNTIME_PATH}/../../../../)/lib64
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(realpath ${ONNXRUNTIME_PATH}/../../../../)/lib

# --- Paths ---
INPUT=/data/dust/user/alimuham/thesis/InputFiles/tau_pi0_10GeV_filtered_500kevents.edm4hep.root
OUTDIR=/data/dust/user/alimuham/thesis/sim/100k-events
STEERING_DIR=/afs/desy.de/user/a/alimuham/thesis-ml-sim/steering

if [ "$SIM_TYPE" == "g4" ]; then
    STEERING=$STEERING_DIR/tau_ddsim_steer.py
    OUTPUT=$OUTDIR/g4/tau-pi0-geant4-job${JOB_ID}.edm4hep.root
    LOG=$OUTDIR/g4/tau-pi0-geant4-job${JOB_ID}.log
elif [ "$SIM_TYPE" == "cc3" ]; then
    STEERING=$STEERING_DIR/tau_ML_ddsim_steer.py
    OUTPUT=$OUTDIR/cc3/tau-pi0-caloclouds-job${JOB_ID}.edm4hep.root
    LOG=$OUTDIR/cc3/tau-pi0-caloclouds-job${JOB_ID}.log
else
    echo "ERROR: SIM_TYPE must be g4 or cc3, got: $SIM_TYPE"
    exit 1
fi

# --- Timing ---
START=$(date +%s)
echo "Job $JOB_ID | SIM_TYPE=$SIM_TYPE | SKIP=$SKIP | NEVENTS=$NEVENTS" > $LOG
echo "Start: $(date)" >> $LOG

# --- Run simulation ---
ddsim --compactFile $k4geo_DIR/ILD/compact/ILD_l5_v02/ILD_l5_v02.xml \
      --steeringFile $STEERING \
      --inputFiles $INPUT \
      --outputFile $OUTPUT \
      --numberOfEvents $NEVENTS \
      --skipNEvents $SKIP \
      2>&1 | tee -a $LOG

END=$(date +%s)
echo "End: $(date)" >> $LOG
echo "Wall time: $((END - START)) seconds" >> $LOG
