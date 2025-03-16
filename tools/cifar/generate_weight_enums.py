from torchvision.datasets import CIFAR10, CIFAR100

from model_utilities.models.cifar_resnet import *
from torchvision.models.resnet import *
import os


def get_acc(logfile):
    with open(logfile, 'r') as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
        return float(last_line.split(",")[-1])


def count_params(model, dataset):
    if dataset == 'cifar10':
        nc = 10
    else:
        nc = 100

    klass = globals()[model]
    net = klass(num_classes=nc)
    return sum([c.numel() for c in net.parameters()])


if __name__ == '__main__':
    model_dataset = {}
    for dir in os.listdir('output'):
        if '-' not in dir:
            continue

        model, dataset = dir.split("-")
        if model not in model_dataset:
            model_dataset[model] = []
        model_dataset[model].append(dataset)

    for dataset in ['cifar10', 'cifar100']:
        if dataset == 'cifar10':
            categories = str(CIFAR10(root="/scratch/jsh2/datasets").classes)
            name = "_COMMON_META_CIFAR10"
        else:
            categories = str(CIFAR100(root="/scratch/jsh2/datasets").classes)
            name = "_COMMON_META_CIFAR100"

        print(name + """ = {
    "min_size": (1, 1),
    "categories": """ + categories)
        print("}")
        print()
        print()

    for model in model_dataset.keys():
        ccmodel = model.replace("resnet", "ResNet")

        print(f"class {ccmodel}_Weights(WeightsEnum):")

        best_seed = 0
        best_acc = 0
        for dataset in model_dataset[model]:
            dir = f"{model}-{dataset}"
            nparams = count_params(model, dataset)
            dataset = dataset.upper()

            for seed in range(3):
                acc = get_acc(f"output/{dir}/model_{seed}-log.csv")

                if dataset == 'cifar10':
                    if acc > best_acc:
                        best_acc = acc
                        best_seed = seed

                weights = dataset + "_s" + str(seed)
                url = (f"http://marc.ecs.soton.ac.uk/pytorch-models/model-utilities"
                       f"/{dir}/model_{seed}.pt")

                en = """    """ + weights + """ = Weights(
        url=\"""" + url + """\",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META_""" + dataset + """,
            "num_params": """ + str(nparams) + """,
            "recipe": "https://github.com/feature-importance/model-utilities/tree/main/tools/cifar#resnet",
            "_metrics": {
                \"""" + dataset + """\": {
                    "acc@1": """ + str(f"{acc:.3f}") + """,
                }
            },
            "_docs": \"\"\"These weights reproduce closely the results of the paper using a simple training recipe.\"\"\",
        }
    )"""
                print(en)
        print(f"    DEFAULT = CIFAR10_s{best_seed}")
        print()
        print()
