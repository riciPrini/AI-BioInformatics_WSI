#! /bin/bash

#SBATCH --job-name=dataset
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2024
#SBATCH --time=12:00:00

#SBATCH --output="/homes/fdoronzio/cv/fabio.log"
#SBATCH --error="/homes/fdoronzio/cv/fabio.log"

python3 /homes/fdoronzio/cv/reduce_dataset.py
