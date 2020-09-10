import torch
import torch.nn as nn
import torchvision.models as models

from scaleadv.datasets.imagenet import IMAGENET_NUM_CLASSES


def create_network(num_classes, pretrained=False, weights=None):
    net = models.resnet50(pretrained=pretrained)

    if not pretrained or num_classes != IMAGENET_NUM_CLASSES:
        net.fc = nn.Linear(net.fc.in_features, num_classes, bias=True)

    if weights:
        ckpt = torch.load(weights)
        net.load_state_dict(ckpt)

    return net

