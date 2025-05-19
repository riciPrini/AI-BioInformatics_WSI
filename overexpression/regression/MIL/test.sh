#! /bin/bash

#SBATCH --job-name=brca
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH --account=ai4bio2024
#SBATCH --time=00:10:00

#SBATCH --output="/homes/fdoronzio/ai4bio/regression/MIL/test.log"
#SBATCH --error="/homes/fdoronzio/ai4bio/regression/MIL/test.err"


module unload cuda/12.1
module load cuda/11.8
. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate bio2
~/.conda/envs/bio2/bin/python /homes/fdoronzio/ai4bio/regression/MIL/test.py  
