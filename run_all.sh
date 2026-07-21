#!/bin/bash
# End-to-end driver. Reproduces all results in the paper.
# Usage: bash run_all.sh
set -euo pipefail
source configs/default.env

bash data/download_data.sh

for MODEL in nt dnabert2; do
  if [ "$MODEL" = "nt" ]; then LAYER=14; else LAYER=6; fi
  export MODEL LAYER
  python src/harvest_activations.py
  python src/train_sae.py
done

NPER=$NPER python src/build_binding_windows.py
NPER=$NPER TF=GATA1 python src/build_scramble_windows.py

for MODEL in nt dnabert2; do
  if [ "$MODEL" = "nt" ]; then LAYER=14; else LAYER=6; fi
  for TF in CTCF GATA1 REST SCRAMBLE GATA1SCRAM; do
    export MODEL LAYER TF
    python src/harvest_binding.py
    python src/test_binding.py || echo "[skip test] $MODEL $TF"
    python src/causal_population.py || echo "[null control] $MODEL $TF"
  done
done

python src/make_figures.py
echo "== pipeline complete; see results/ =="
