#!/bin/bash

for SEED in $(seq 1 3)
do
  for DATASET in cifar10 cifar100
  do
    for MODEL in resnet18_3x3 resnet34_3x3 resnet50_3x3 resnet101_3x3 resnet152_3x3
    do
      sbatch launch.sh --dataset "$DATASET" --model "$MODEL" --seed "$SEED"
    done
    for MODEL in resnet18 resnet34 resnet50 resnet101 resnet152
    do
      sbatch launch.sh --dataset "$DATASET" --model "$MODEL" --seed "$SEED"
    done
  done
done

