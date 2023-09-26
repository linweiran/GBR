import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x=self.conv1(x)
        x=F.max_pool2d(x, 2)
        x = F.relu(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class GTSRBNet1(nn.Module):
    #This is the top benign accuracy architecture on Kaggle (https://github.com/poojahira/gtsrb-pytorch)
    def __init__(self):
        super(GTSRBNet1, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(3, 100, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 150, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(250)
        self.conv_drop = nn.Dropout2d()
        #self.fc1 = nn.Linear(250*2*2, 350)
        self.fc1 = nn.Linear(250*4*4, 350)
        self.fc2 = nn.Linear(350, 43)

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
            )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            #nn.Linear(10 * 4 * 4, 32),
            nn.Linear(10 * 8 * 8, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
            )
   
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        
        #xs = xs.view(-1, 10 * 4 * 4)
        xs = xs.view(-1, 10 * 8 * 8)
        
        theta = self.fc_loc(xs)
        
        theta = theta.view(-1, 2, 3)
        
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform forward pass
        x = self.bn1(F.max_pool2d(F.leaky_relu(self.conv1(x)),2))
        x = self.conv_drop(x)
        x = self.bn2(F.max_pool2d(F.leaky_relu(self.conv2(x)),2))
        x = self.conv_drop(x)
        x = self.bn3(F.max_pool2d(F.leaky_relu(self.conv3(x)),2))
        x = self.conv_drop(x)
        #x = x.view(-1, 250*2*2)
        x = x.view(-1, 250*4*4)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x#F.log_softmax(x, dim=1)

def GTSRBNet2():
    return torchvision.models.vgg16_bn(num_classes=43)

def GTSRBNet21():
    return torchvision.models.vgg16(num_classes=43)

def GTSRBNet3():
    return torchvision.models.resnet18(num_classes=43)

def GTSRBNet3D():
    return torchvision.models.resnet18(num_classes=86)

def GTSRBNet4():
    return torchvision.models.squeezenet1_0(num_classes=43)

def GTSRBNet5():
    return torchvision.models.densenet161(num_classes=43)

def GTSRBNet6():
    return torchvision.models.shufflenet_v2_x1_0(num_classes=43)

def GTSRBNet7():
    return torchvision.models.mobilenet_v2(num_classes=43)

def CelebANet21():
    return torchvision.models.vgg16(num_classes=2360)

def CelebANet3():
    return torchvision.models.resnet18(num_classes=2360)

def CelebANet4():
    return torchvision.models.squeezenet1_0(num_classes=2360)

def CelebANet6():
    return torchvision.models.shufflenet_v2_x1_0(num_classes=2360)

def CelebANet7():
    return torchvision.models.mobilenet_v2(num_classes=2360)

