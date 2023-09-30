# -*- coding: utf-8 -*-
'''
We use this file to build the model and copy the weights
We change the fc layers of vggface model to 1024 -> 1024 -> 51.
We only copy the weights in Convolutional layers
'''
__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"

#import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile

class VGG_16(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 1024)
        self.fc7 = nn.Linear(1024, 1024)
        #self.fc8 = nn.Linear(1024, 36) #TO DO change 10 to 51
        self.fc8 = nn.Linear(1024, 60)
        #1024 -> 1024 -> 51

    def load_weights(self, path="./vgg_face_torch/VGG_FACE.t7"):
        """ Function to load luatorch pretrained

        Args:
            path: path for the luatorch pretrained
        """
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                    # only load convolutional layers, learn fc layers by our own examples

    def forward(self, x):
        """ Pytorch forward

        Args:
            x: input image (224x224)

        Returns: class logits

        """
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5, self.training)
        return self.fc8(x)

    # def find_feature(self, x):
    #     x = F.relu(self.conv_1_1(x))
    #     x = F.relu(self.conv_1_2(x))
    #     x = F.max_pool2d(x, 2, 2)
    #     x = F.relu(self.conv_2_1(x))
    #     x = F.relu(self.conv_2_2(x))
    #     x = F.max_pool2d(x, 2, 2)
    #     x = F.relu(self.conv_3_1(x))
    #     x = F.relu(self.conv_3_2(x))
    #     x = F.relu(self.conv_3_3(x))
    #     x = F.max_pool2d(x, 2, 2)
    #     return x



if __name__ == "__main__":
    model = VGG_16()
    model.load_weights()

    # im = cv2.imread("../images/ak.png")

    # #im = cv2.imread("../images/ak.png")
    # im = torch.Tensor(im).permute(2, 0, 1).view(1, 3, 224, 224)
    # import numpy as np

    # model.eval()
    # im -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).view(1, 3, 1, 1)
    # preds = F.softmax(model(im[:,[1,0,2],:,:]), dim=1)
    # #preds = model(im)
    # values, indices = preds.max(-1)
    # print(values,indices)
