#!/bin/bash -l
#SBATCH -p swarm_h100,swarm_a100
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --time=01:00:00

module load conda/python3
source activate torchwatcher

python train.py --dataset cifar10 --model resnet18_3x3 --seed 0 --batch-size 128 --epochs 180 --lr "0.1*0.1@[90,140]" --log-dir ./output/logs --output-dir ./output

