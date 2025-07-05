#!/bin/bash

for SEED in $(seq 0 2)
do
  for DATASET in cifar10 cifar100
  do
    for MODEL in convnext_base convnext_tiny convnext_small convnext_large
    do
      sbatch launch.sh --dataset "$DATASET" --model "$MODEL" --seed "$SEED" --lr "1e-3*linearWarmup0.01,5<@5;cosineAnnealing0,595" --batch-size 1024 --epochs 600 --opt=adamw
    done
  done
done

