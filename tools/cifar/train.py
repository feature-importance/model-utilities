"""
Training CIFAR10 models
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torchvision
from model_utilities.models.cifar_resnet import resnet18_3x3, resnet34_3x3, \
    resnet50_3x3, resnet101_3x3, resnet152_3x3
from model_utilities.training.modelfitting import fit_model, set_seed, \
    get_device, parse_learning_rate_arg
from model_utilities.training.utils import save_args
from model_utilities.transforms._cifar_presets import \
    ImageClassificationTraining, ImageClassificationEval
from torchbearer.callbacks import CSVLogger
from torchvision.datasets import CIFAR10, CIFAR100


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description=__doc__, add_help=add_help)
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='dataset')
    parser.add_argument('--model', type=str,
                        default='resnet18_3x3')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-dir', type=str, default=None,
                        help='log directory for tensorboard. CSV logs will be '
                             'saved in the output directory.')
    parser.add_argument('--data-dir', type=str,
                        default="/scratch/jsh2/datasets")
    parser.add_argument("--output-dir", default=".", type=str,
                        help="path to save outputs")

    parser.add_argument("-j", "--workers", default=16, type=int,
                        metavar="N",
                        help="number of data loading workers (default: 16)")

    parser.add_argument("--device", default='auto', type=str,
                        help="device (Use cuda/cpu/mps Default: auto)")

    parser.add_argument("-b", "--batch-size", default=32, type=int,
                        help="images per batch")
    parser.add_argument("--epochs", default=90, type=int,
                        metavar="N", help="number of total epochs to run")
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument("--opt", default="sgd", type=str,
                        help="optimizer")
    parser.add_argument("--lr", default="0.1", type=str,
                        help="learning rate specification")
    parser.add_argument("--momentum", default=0.9, type=float,
                        metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight-decay", default=1e-4,
                        type=float, metavar="W",
                        help="weight decay (default: 1e-4)",
                        dest="weight_decay")

    return parser


def get_model(model_name, num_classes):
    if model_name == 'resnet18_3x3':
        return resnet18_3x3(num_classes=num_classes)
    if model_name == 'resnet34_3x3':
        return resnet34_3x3(num_classes=num_classes)
    if model_name == 'resnet50_3x3':
        return resnet50_3x3(num_classes=num_classes)
    if model_name == 'resnet101_3x3':
        return resnet101_3x3(num_classes=num_classes)
    if model_name == 'resnet152_3x3':
        return resnet152_3x3(num_classes=num_classes)

    return torchvision.models.get_model(model_name, num_classes=num_classes)


def get_optimizer(opt_name, parameters, init_lr, momentum, weight_decay):
    opt_name = opt_name.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters, lr=init_lr, momentum=momentum,
            weight_decay=weight_decay, nesterov="nesterov" in opt_name)
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=init_lr, momentum=momentum,
            weight_decay=weight_decay, eps=0.0316, alpha=0.9)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=init_lr,
                                      weight_decay=weight_decay)
    else:
        raise RuntimeError(
            f"Invalid optimizer {opt_name}. Only SGD, RMSprop and AdamW are "
            f"supported.")

    return optimizer


def main():
    print("Training models with: ")
    print(sys.argv)

    parser = get_args_parser()
    args = parser.parse_args()

    set_seed(args.seed)

    device = get_device(args.device)
    model = get_model(args.model,
                      num_classes=10 if args.dataset == 'cifar10' else 100)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    init_lr, schedule = parse_learning_rate_arg(args.lr)

    opt = get_optimizer(args.opt,
                        filter(lambda p: p.requires_grad, model.parameters()),
                        init_lr, args.momentum, args.weight_decay)

    if args.dataset == 'cifar10':
        train_data = CIFAR10(root=args.data_dir, train=True, download=False,
                             transform=ImageClassificationTraining())
        val_data = CIFAR10(root=args.data_dir, train=False, download=False,
                           transform=ImageClassificationEval())
    else:
        train_data = CIFAR100(root=args.data_dir, train=True, download=False,
                              transform=ImageClassificationTraining())
        val_data = CIFAR100(root=args.data_dir, train=False, download=False,
                            transform=ImageClassificationEval())

    train_data_loader = torch.utils.data.DataLoader(train_data,
                                                    pin_memory=True,
                                                    num_workers=args.workers,
                                                    batch_size=args.batch_size)
    val_data_loader = torch.utils.data.DataLoader(val_data,
                                                  pin_memory=True,
                                                  num_workers=args.workers,
                                                  batch_size=args.batch_size)

    run_id = f"{args.model}-{args.dataset}-seed_{args.seed}"

    model_file = os.path.join(args.output_dir,
                              f"{args.model}-{args.dataset}",
                              f"model_{args.seed}.pt")

    fit_model(model, criterion, opt, train_data_loader, val_data_loader,
              epochs=args.epochs, device='auto', verbose=1, acc='acc',
              model_file=model_file, run_id=run_id, log_dir=args.log_dir,
              resume=args.resume, period=None, schedule=schedule)

    save_args(os.path.dirname(model_file),
              os.path.basename(model_file).replace(".pt", "-cmd.txt"))


if __name__ == '__main__':
    main()
