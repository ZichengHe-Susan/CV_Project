#!/bin/bash
#SBATCH --job-name=train_coco
#SBATCH -A uva_cs4501_cv
#SBATCH --partition=gpu          # use a CPU partition (or 'standard', 'compute', etc.)
#SBATCH --time=02:00:00          # download + untar is usually <1 h on good bandwidth
#SBATCH --mem=32G                # plenty; images ~25 GB, unzip peaks <10 GB RAM
#SBATCH --cpus-per-task=4
#SBATCH --output=output_evaluate_coco.txt           # Standard output file
#SBATCH --error=error_evaluate_coco.txt             # Standard error file
#SBATCH --gres=gpu:1   

module load gcc/11.4.0 openmpi/4.1.4
module load python/3.11.4
# 3. Activate it
source ~/venvs/msvd311/bin/activate

pip install nltk

python3 evaluate.py