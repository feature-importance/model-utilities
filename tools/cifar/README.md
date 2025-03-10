# CIFAR10 Training Scripts

## ResNet and ResNet-3x3 Models

    python train.py --dataset cifar10 --model resnet18_3x3 --seed 0 \
        --batch-size 128 --epochs 180 --lr 0.1*0.1@[90,140]


