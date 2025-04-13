#!/bin/bash
#SBATCH --job-name=train_coco
#SBATCH -A uva_cs4501_cv
#SBATCH --partition=gpu          # use a CPU partition (or 'standard', 'compute', etc.)
#SBATCH --time=02:00:00          # download + untar is usually <1 h on good bandwidth
#SBATCH --mem=32G                # plenty; images ~25 GB, unzip peaks <10 GB RAM
#SBATCH --cpus-per-task=4
#SBATCH --output=output_train_coco.txt           # Standard output file
#SBATCH --error=error_train_coco.txt             # Standard error file
#SBATCH --gres=gpu:1   

module load gcc/11.4.0 openmpi/4.1.4
module load python/3.11.4

# 2. Create virtual environment
python -m venv ~/venvs/msvd311

# 3. Activate it
source ~/venvs/msvd311/bin/activate

# 4. Install project dependencies
pip install --upgrade pip
pip install torch torchvision transformers ftfy regex tqdm opencv-python-headless git+https://github.com/openai/CLIP.git

python3 trainCOCO.py