from tabulate import tabulate
 


def show_models():
    from . import models

    model_names = models.__all__
    numbers = list(range(1, len(model_names)+1))
    print(tabulate({'No.': numbers, 'Model Names': model_names}, headers='keys'))

def show_representations():
    from . import representations

    representation_names = representations.__all__
    numbers = list(range(1, len(representation_names)+1))
    print(tabulate({'No.': numbers, 'Representation Names': representation_names}, headers='keys'))

def show_losses():
    from . import losses

    loss_names = losses.__all__
    numbers = list(range(1, len(loss_names)+1))
    print(tabulate({'No.': numbers, 'Loss Names': loss_names}, headers='keys'))

'''
def show_backbones():
    backbone_names = backbones.__all__
    variants = []
    for name in backbone_names:
        try:
            variants.append(list(eval(f"backbones.{name.lower()}_settings").keys()))
        except:
            variants.append('-')
    print(tabulate({'Backbone Names': backbone_names, 'Variants': variants}, headers='keys'))
'''

def show_datasets():
    from . import datasets

    dataset_names = datasets.__all__
    numbers = list(range(1, len(dataset_names)+1))
    print(tabulate({'No.': numbers, 'Datasets': dataset_names}, headers='keys'))