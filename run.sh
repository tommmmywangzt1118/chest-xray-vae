#!/bin/bash
# Train the VAE, then compute FID / IS and make the report figures.
# Usage: DATA=/path/to/chest_xray PY=/path/to/env/bin/python bash run.sh
set -e
DATA=${DATA:-data/chest_xray}
PY=${PY:-python}
OUT=${OUT:-runs/vae}

$PY vae.py --data "$DATA" --out "$OUT" --epochs 300 --batch-size 64 --lr 1e-3 --latent-dim 128 --beta 1.0
$PY evaluate.py --data "$DATA" --run "$OUT" --n 5000
