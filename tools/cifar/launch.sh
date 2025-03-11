#!/bin/bash -l
#SBATCH -p swarm_h100,swarm_a100,a100,scavenger_mathsa100,swarm_l4,scavenger_4a100
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 12
#SBATCH --time=03:00:00

module load conda/python3
source activate torchwatcher

python train.py -j 8 --batch-size 64 --epochs 180 --log-dir ./output/logs --output-dir ./output "$@"

