#!/bin/bash
#SBATCH --job-name=genomic_sae
#SBATCH --account=<your-account>
#SBATCH --partition=<gpu-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=96g
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate
source configs/default.env
# example: MODEL=nt LAYER=14 TF=CTCF python src/causal_population.py
bash run_all.sh
