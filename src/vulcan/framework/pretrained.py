import torch

def get_pretrained_model(pretrained_config):
    if pretrained_config:
        return torch.load(pretrained_config, map_location="cpu")
    return None