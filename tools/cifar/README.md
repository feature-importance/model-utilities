# CIFAR10 & 100 Training Scripts

## ResNet and ResNet-3x3 Models
<a name="resnet"></a>

All `resnet` and `resnet_3x3` models were trained with batch size 64, for 
180 epochs using SGD (momentum=0.9). For models <=50 layers we start with a 
learning rate of 0.1 and drop by a factor or 10 at 90 and 140 epochs:

    python train.py --dataset cifar10 --model resnet18_3x3 --seed 0 \
        --batch-size 64 --epochs 180 --lr 0.1*0.1@[90,140]

Deeper models had one epoch of lr warmup at 0.01 before following the same 
schedule:
    
    --lr "0.01<@1,0.1<@90,0.01<@140,0.001"

We've trained 3 seeds of each model on both CIFAR-10 and CIFAR-100:

| Dataset  | Model         | Seed | Val. Acc.   |
| -------- | ------------- | ---- | ----------- |
| cifar10  | resnet18_3x3  | 0    | 0.915699959 |
| cifar10  | resnet18_3x3  | 1    | 0.923299968 |
| cifar10  | resnet18_3x3  | 2    | 0.922099948 |
| cifar10  | resnet34_3x3  | 0    | 0.926799953 |
| cifar10  | resnet34_3x3  | 1    | 0.923500001 |
| cifar10  | resnet34_3x3  | 2    | 0.924600005 |
| cifar10  | resnet50_3x3  | 0    | 0.926299989 |
| cifar10  | resnet50_3x3  | 1    | 0.932899952 |
| cifar10  | resnet50_3x3  | 2    | 0.922699988 |
| cifar10  | resnet101_3x3 | 0    | 0.932399988 |
| cifar10  | resnet101_3x3 | 1    | 0.9199      |
| cifar10  | resnet101_3x3 | 2    | 0.92809999  |
| cifar10  | resnet152_3x3 | 0    | 0.920299947 |
| cifar10  | resnet152_3x3 | 1    | 0.921999991 |
| cifar10  | resnet152_3x3 | 2    | 0.9278      |
| cifar10  | resnet18      | 0    | 0.871599972 |
| cifar10  | resnet18      | 1    | 0.866999984 |
| cifar10  | resnet18      | 2    | 0.869099975 |
| cifar10  | resnet34      | 0    | 0.866599977 |
| cifar10  | resnet34      | 1    | 0.870799959 |
| cifar10  | resnet34      | 2    | 0.869899988 |
| cifar10  | resnet50      | 0    | 0.872099996 |
| cifar10  | resnet50      | 1    | 0.879399955 |
| cifar10  | resnet50      | 2    | 0.879999995 |
| cifar10  | resnet101     | 0    | 0.879099965 |
| cifar10  | resnet101     | 1    | 0.870700002 |
| cifar10  | resnet101     | 2    | 0.871800005 |
| cifar10  | resnet152     | 0    | 0.864600003 |
| cifar10  | resnet152     | 1    | 0.868200004 |
| cifar10  | resnet152     | 2    | 0.87469995  |
| cifar100 | resnet18_3x3  | 0    | 0.656599998 |
| cifar100 | resnet18_3x3  | 1    | 0.664799988 |
| cifar100 | resnet18_3x3  | 2    | 0.659799993 |
| cifar100 | resnet34_3x3  | 0    | 0.656499982 |
| cifar100 | resnet34_3x3  | 1    | 0.638499975 |
| cifar100 | resnet34_3x3  | 2    | 0.65259999  |
| cifar100 | resnet50_3x3  | 0    | 0.64230001  |
| cifar100 | resnet50_3x3  | 1    | 0.671099961 |
| cifar100 | resnet50_3x3  | 2    | 0.65200001  |
| cifar100 | resnet101_3x3 | 0    | 0.648599982 |
| cifar100 | resnet101_3x3 | 1    | 0.613099992 |
| cifar100 | resnet101_3x3 | 2    | 0.666399956 |
| cifar100 | resnet152_3x3 | 0    | 0.657700002 |
| cifar100 | resnet152_3x3 | 1    | 0.647099972 |
| cifar100 | resnet152_3x3 | 2    | 0.619699955 |
| cifar100 | resnet18      | 0    | 0.556299984 |
| cifar100 | resnet18      | 1    | 0.551800013 |
| cifar100 | resnet18      | 2    | 0.559099972 |
| cifar100 | resnet34      | 0    | 0.558200002 |
| cifar100 | resnet34      | 1    | 0.569499969 |
| cifar100 | resnet34      | 2    | 0.568499982 |
| cifar100 | resnet50      | 0    | 0.546999991 |
| cifar100 | resnet50      | 1    | 0.546499968 |
| cifar100 | resnet50      | 2    | 0.526199996 |
| cifar100 | resnet101     | 0    | 0.501599967 |
| cifar100 | resnet101     | 1    | 0.540199995 |
| cifar100 | resnet101     | 2    | 0.543099999 |
| cifar100 | resnet152     | 0    | 0.495599985 |
| cifar100 | resnet152     | 1    | 0.507099986 |
| cifar100 | resnet152     | 2    | 0.552699983 |

# VGG models
<a name="vgg"></a>

# ...