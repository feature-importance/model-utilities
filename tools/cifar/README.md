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

| Dataset  | Model         | Seed | Val. Acc. |
|----------|---------------|------|-----------|
| cifar10  | resnet18_3x3  | 0    | 0.915     |
| cifar10  | resnet18_3x3  | 1    | 0.923     |
| cifar10  | resnet18_3x3  | 2    | 0.922     |
| cifar10  | resnet34_3x3  | 0    | 0.926     |
| cifar10  | resnet34_3x3  | 1    | 0.923     |
| cifar10  | resnet34_3x3  | 2    | 0.924     |
| cifar10  | resnet50_3x3  | 0    | 0.926     |
| cifar10  | resnet50_3x3  | 1    | 0.932     |
| cifar10  | resnet50_3x3  | 2    | 0.922     |
| cifar10  | resnet101_3x3 | 0    | 0.932     |
| cifar10  | resnet101_3x3 | 1    | 0.9       |
| cifar10  | resnet101_3x3 | 2    | 0.928     |
| cifar10  | resnet152_3x3 | 0    | 0.920     |
| cifar10  | resnet152_3x3 | 1    | 0.921     |
| cifar10  | resnet152_3x3 | 2    | 0.9       |
| cifar10  | resnet18      | 0    | 0.871     |
| cifar10  | resnet18      | 1    | 0.866     |
| cifar10  | resnet18      | 2    | 0.869     |
| cifar10  | resnet34      | 0    | 0.866     |
| cifar10  | resnet34      | 1    | 0.870     |
| cifar10  | resnet34      | 2    | 0.869     |
| cifar10  | resnet50      | 0    | 0.872     |
| cifar10  | resnet50      | 1    | 0.879     |
| cifar10  | resnet50      | 2    | 0.879     |
| cifar10  | resnet101     | 0    | 0.879     |
| cifar10  | resnet101     | 1    | 0.870     |
| cifar10  | resnet101     | 2    | 0.871     |
| cifar10  | resnet152     | 0    | 0.864     |
| cifar10  | resnet152     | 1    | 0.868     |
| cifar10  | resnet152     | 2    | 0.874     |
| cifar100 | resnet18_3x3  | 0    | 0.656     |
| cifar100 | resnet18_3x3  | 1    | 0.664     |
| cifar100 | resnet18_3x3  | 2    | 0.659     |
| cifar100 | resnet34_3x3  | 0    | 0.656     |
| cifar100 | resnet34_3x3  | 1    | 0.638     |
| cifar100 | resnet34_3x3  | 2    | 0.652     |
| cifar100 | resnet50_3x3  | 0    | 0.642     |
| cifar100 | resnet50_3x3  | 1    | 0.671     |
| cifar100 | resnet50_3x3  | 2    | 0.652     |
| cifar100 | resnet101_3x3 | 0    | 0.648     |
| cifar100 | resnet101_3x3 | 1    | 0.613     |
| cifar100 | resnet101_3x3 | 2    | 0.666     |
| cifar100 | resnet152_3x3 | 0    | 0.657     |
| cifar100 | resnet152_3x3 | 1    | 0.647     |
| cifar100 | resnet152_3x3 | 2    | 0.619     |
| cifar100 | resnet18      | 0    | 0.556     |
| cifar100 | resnet18      | 1    | 0.551     |
| cifar100 | resnet18      | 2    | 0.559     |
| cifar100 | resnet34      | 0    | 0.558     |
| cifar100 | resnet34      | 1    | 0.569     |
| cifar100 | resnet34      | 2    | 0.568     |
| cifar100 | resnet50      | 0    | 0.546     |
| cifar100 | resnet50      | 1    | 0.546     |
| cifar100 | resnet50      | 2    | 0.526     |
| cifar100 | resnet101     | 0    | 0.501     |
| cifar100 | resnet101     | 1    | 0.540     |
| cifar100 | resnet101     | 2    | 0.543     |
| cifar100 | resnet152     | 0    | 0.495     |
| cifar100 | resnet152     | 1    | 0.507     |
| cifar100 | resnet152     | 2    | 0.552     |


# VGG models
<a name="vgg"></a>

All standard VGG16 and 19 models were trained in a similar way to the 
resnets but with a lower inital lr (these models don't include BN) and are 
expected to be more sensitive:

    python train.py --dataset cifar10 --model vgg16 --seed 0 \
        --batch-size 64 --epochs 180 --lr 0.01*0.1@[90,140]

We've trained 3 seeds of each model on both CIFAR-10 and CIFAR-100:

| Dataset   | Model | Seed | Val. Acc. |
|-----------|-------|------|-----------|
| cifar10   | vgg16 | 0    | 0.916     |
| cifar10   | vgg16 | 1    | 0.916     |
| cifar10   | vgg16 | 2    | 0.916     |
| cifar10   | vgg19 | 0    | 0.912     |
| cifar10   | vgg19 | 1    | 0.914     |
| cifar10   | vgg19 | 2    | 0.912     |
| cifar100  | vgg16 | 0    | 0.653     |
| cifar100  | vgg16 | 1    | 0.655     |
| cifar100  | vgg16 | 2    | 0.661     |
| cifar100  | vgg19 | 0    | 0.662     |
| cifar100  | vgg19 | 1    | 0.657     |
| cifar100  | vgg19 | 2    | 0.659     |

# ...