
#!/bin/bash
#SBATCH --job-name=download_coco
#SBATCH -A uva_cs4501_cv
#SBATCH --partition=standard
#SBATCH --time=02:00:00          # download + untar is usually <1 h on good bandwidth
#SBATCH --mem=32G                # plenty; images ~25 GB, unzip peaks <10 GB RAM
#SBATCH --cpus-per-task=4        # parallel downloads / untar faster
module load gcc/11.4.0         # First load the prerequisite
module load python/3.11.4      # Then load Python
module load cuda
# Run the get_coco.py script
python3 data/get_coco_local.py                   