#!/bin/bash

for SEED in $(seq 0 2)
do
  for DATASET in cifar10 cifar100
  do
    for MODEL in resnet18_3x3 resnet34_3x3 resnet50_3x3 resnet18 resnet34 resnet50
    do
      sbatch launch.sh --dataset "$DATASET" --model "$MODEL" --seed "$SEED" --lr "0.1*0.1@[90,140]"
    done
    for MODEL in resnet101_3x3 resnet152_3x3 resnet101 resnet152
    do
      sbatch launch.sh --dataset "$DATASET" --model "$MODEL" --seed "$SEED" --lr "0.01<@3,0.1<@90,0.01<@140,0.001"
    done
  done
done

