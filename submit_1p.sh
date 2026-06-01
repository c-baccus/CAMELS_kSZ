#!/bin/bash
#SBATCH --account=torch_pr_499_general
#SBATCH --partition=cpu_short
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=1p_pipeline
#SBATCH --output=run_all_1p_%j.log
#SBATCH --error=run_all_1p_%j.log

module load anaconda3/2025.06
cd /scratch/cjb9346/CAMELS_kSZ
python run_all_1p.py
