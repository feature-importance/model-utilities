import os

if __name__ == '__main__':
    for dir in os.listdir('output'):
        if '-' not in dir:
            continue

        model, dataset = dir.split("-")
        dataset = dataset.upper()
        ccmodel = model.replace("resnet", "ResNet")

        print(f"class {ccmodel}_Weights(WeightsEnum):")
        nparams = 0
        for seed in range(3):
            acc = 0
            weights = dataset + "_s" + str(seed)

            en = """    """ + weights + """ = Weights(
        url="",
        transforms=ImageClassificationEval,
        meta={
            **_COMMON_META,
            "num_params": """ + str(nparams) + """,
            "recipe": "https://github.com/feature-importance/model-utilities"
                      "/tree/main/tools/cifar#resnet",
            "_metrics": {
                """"" + dataset + """": {
                    "acc@1": """ + str(acc) + """,
                }
            },
            "_docs": \"\"\"These weights reproduce closely the results of the 
            paper using a simple training recipe.\"\"\",
        },
    )"""
            print(en)
    # DEFAULT = CIFAR10_V1