#!/bin/bash

for SEED in $(seq 0 2)
do
  for DATASET in cifar10 cifar100
  do
    for MODEL in convnext_base convnext_tiny convnext_small convnext_large
    do
      sbatch launch.sh --dataset "$DATASET" --model "$MODEL" --seed "$SEED" --lr "0.01*0.1@[90,140]"
    done
  done
done

