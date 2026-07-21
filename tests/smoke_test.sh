#!/bin/bash
# ~5-minute CPU smoke test on a small bundled input for one TF.
# Verifies the pipeline runs end-to-end and produces a binding-test JSON.
set -euo pipefail
source configs/default.env
export MODEL=nt LAYER=14 TF=CTCF NPER=200 NWIN=60
echo "[smoke] building small windows"
python src/build_binding_windows.py
echo "[smoke] harvesting + testing (subset)"
python src/harvest_binding.py
python src/test_binding.py
echo "[smoke] OK — compare against tests/expected_output.txt"
