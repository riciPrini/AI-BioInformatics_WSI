#! /bin/bash

#SBATCH --job-name=style
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2024
#SBATCH --time=22:00:00

#SBATCH --output="/homes/fdoronzio/cv/fabio.log"
#SBATCH --error="/homes/fdoronzio/cv/fabio.log"

python3 /homes/fdoronzio/cv/train.py --train_content_dir  /homes/fdoronzio/cv/content  --train_style_dir /homes/fdoronzio/cv/style