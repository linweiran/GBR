'''
We use this file to test the clean accuracy.
Type 'python origin_test {}.pt '  ({} name of your model want to test) to run
'''


import torch
import torch.nn as nn
import torch.optim as optim
from origin_train import data_process
#from origin_train2 import data_process2
import numpy as np
import argparse
from new_vgg_face import VGG_16
import copy
import torchfile
import torchvision
from torchvision import datasets, models, transforms
#from save_image import save_image 
#uncomment to show images 


def test(model,dataloaders,dataset_sizes):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    correct = 0 
    total = 0
    #torch.manual_seed(12345)
    with torch.no_grad():
        for data in dataloaders['test']:
            images, labels = data
            # if labels.data == 0:
            #     continue
            images = images[:,[2,1,0],:,:]   # rgb to bgr
            images = images.to(device).half()
            labels = labels.to(device)
            #print (images.dtype)
            outputs = model(images)
            # save_image('ori_test',images)
            # uncomment to see images 
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #print(correct/total)
    print('Accuracy of the network on the %s test images: %10.5f %%' % (total,100 * correct / total))
    return correct / total

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument("model", type=str, help="test_model")
    args = parser.parse_args()
    model = VGG_16() 
    model.load_state_dict(torch.load('../donemodel/'+args.model))
    #model=model.half()
    print("test model is ", args.model)
    dataloaders,dataset_sizes =data_process(batch_size =1)
    
    test(model,dataloaders,dataset_sizes)
    
    
