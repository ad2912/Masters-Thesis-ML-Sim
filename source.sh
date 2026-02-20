#!/bin/bash
# ============================================================
#  source.sh — NAF startup ritual for ML shower simulation
#  Usage: source ~/source.sh
#  Author: alimuham @ DESY
# ============================================================

# --- 1. Key4hep environment ---------------------------------
source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2026-02-01

# --- 2. DDML plugin + official setup ------------------------
DDML_INSTALL=/afs/desy.de/user/a/alimuham/key4hep_tut_ild_reco/DDML/install
source $DDML_INSTALL/bin/thisDDML.sh

# --- 3. Missing runtime libs (torch + onnxruntime) ----------
#K4H=/cvmfs/sw.hsf.org/key4hep/releases/2026-02-01/x86_64-almalinux9-gcc14.2.0-opt
#export LD_LIBRARY_PATH=$K4H/py-torch/2.9.1-fl7w5y/lib/python3.13/site-packages/torch/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=$K4H/py-onnxruntime/1.22.2-haobv5/lib64:$LD_LIBRARY_PATH
# --- 3. Missing runtime libs (dynamic, version-independent) ---
TORCH_PATH=$(dirname $(python3 -c 'import torch; print(torch.__file__)'))
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${TORCH_PATH}/lib

ONNXRUNTIME_PATH=$(dirname $(python3 -c 'import onnxruntime; print(onnxruntime.__file__)'))
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(realpath ${ONNXRUNTIME_PATH}/../../../../)/lib64
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(realpath ${ONNXRUNTIME_PATH}/../../../../)/lib


# --- 4. Aliases ---------------------------------------------
alias cdwork='cd /afs/desy.de/user/a/alimuham/thesis-ml-sim'
alias cdprod='cd /afs/desy.de/user/a/alimuham/thesis-ml-sim/steering'
alias cdresults='cd /afs/desy.de/user/a/alimuham/thesis-ml-sim/results'
alias cdanalysis='cd /afs/desy.de/user/a/alimuham/thesis-ml-sim/analysis'
alias cdcog='cd /afs/desy.de/user/a/alimuham/thesis-ml-sim/cognitive-state'

# quick simulation launchers
#alias runsim-ml='ddsim --compactFile $k4geo_DIR/ILD/compact/ILD_l5_v02/ILD_l5_v02.xml --steeringFile /afs/desy.de/user/a/alimuham/thesis-ml-sim/steering/gamma_ML-ddsim_steer.py'
#alias runsim-g4='ddsim --compactFile $k4geo_DIR/ILD/compact/ILD_l5_v02/ILD_l5_v02.xml --steeringFile /afs/desy.de/user/a/alimuham/thesis-ml-sim/steering/gamma-ddsim_steer.py'

# sanity check alias — run this if you suspect env is broken
alias checkenv='ldd $DDML_INSTALL/lib/libDDML.so | grep "not found" && echo "BROKEN: missing libs" || echo "OK: all libs resolved"'

# --- 5. Go to working directory -----------------------------
cd /afs/desy.de/user/a/alimuham/thesis-ml-sim

echo ""
echo "================================================"
echo "  ML Sim environment ready"
echo "  Key4hep : 2026-02-01"
echo "  DDML    : $DDML_INSTALL"
echo "  Workdir : $(pwd)"
echo "================================================"
echo ""
