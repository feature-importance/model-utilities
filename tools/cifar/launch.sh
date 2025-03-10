#!/bin/bash -l
#SBATCH -p swarm_h100,swarm_a100,a100,scavenger_mathsa100,swarm_l4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH --time=03:00:00

module load conda/python3
source activate torchwatcher

python train.py --batch-size 128 --epochs 180 --lr "0.1*0.1@[90,140]" --log-dir ./output/logs --output-dir ./output "$@"

