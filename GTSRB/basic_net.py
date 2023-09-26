import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


def GTSRBNet2():
    return torchvision.models.vgg16_bn(num_classes=43)

def GTSRBNet3():
    return torchvision.models.resnet18(num_classes=43)

def GTSRBNet4():
    return torchvision.models.squeezenet1_0(num_classes=43)

def GTSRBNet6():
    return torchvision.models.shufflenet_v2_x1_0(num_classes=43)

def GTSRBNet7():
    return torchvision.models.mobilenet_v2(num_classes=43)


