import sys


def save_args(directory, name='cmd.txt'):
    with open(str(directory) + '/' + name, "w") as f:
        f.write(' '.join(sys.argv))


def save_model_info(model, directory, name='model-info.txt'):
    with open(str(directory) + '/' + name, "w") as f:
        f.write(str(model))
