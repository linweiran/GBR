# this file is based on https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# Author: Sasank Chilamkurthy

'''
This file basically runs Defending against rectangular Occlusion Attacks (DOA) see paper
Type 'python sticker_retrain.py {}.pt -alpha 4 -iters 50 -out 99 -search 1 -epochs 5' to run
{}.pt is the name of model you want to train with DOA
alpha is learning rate of PGD e.g. 4
iters is the iterations of PGD e.g.50
out is name of your final model e.g.99
search is method of searching, '0' is exhaustive_search, '1' is gradient_based_search"
epochs is the epoch you want to fine tune your network e.g. 5

Note that ROA is a abstract attacking model simulate the "physical" attacks
Thus there is no restriction for the mask to be rectangle
'''


import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
#import cv2
import torchfile
import matplotlib.pyplot as plt
from origin_train import data_process
import argparse
import copy
from origin_test import test
from new_vgg_face import VGG_16
#from sticker_attack import ROA
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2143589489



class ROA(object):
    '''
    Rectangular Occlusion Attacks class
    '''

    def __init__(self, base_classifier, alpha, iters):
        self.base_classifier = base_classifier
        self.alpha = alpha
        self.iters = iters

    def exhaustive_search(self, X, y, width, height, xskip, yskip):
        model = self.base_classifier
        model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X = X.to(device)
        y = y.to(device)
        mean = torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).view(1, 3, 1, 1)
        mean = mean.to(device)

        max_loss = torch.zeros(y.shape[0]).to(y.device) -100
        all_loss = torch.zeros(y.shape[0]).to(y.device)
        xtimes = (224 - width )//xskip
        ytimes = (224 - height)//yskip

        output_j = torch.zeros(y.shape[0])
        output_i = torch.zeros(y.shape[0])

        for i in range(xtimes):
            for j in range(ytimes):

                sticker = X+ mean
                sticker[:,:,yskip*j:(yskip*j+height),xskip*i:(xskip*i+width)] = 255/2
                sticker1 = sticker.detach() - mean.detach()
                all_loss = nn.CrossEntropyLoss(reduction='none')(model(sticker1),y)
                padding_j = torch.zeros(y.shape[0]) + j
                padding_i = torch.zeros(y.shape[0]) + i
                output_j[all_loss > max_loss] = padding_j[all_loss > max_loss]
                output_i[all_loss > max_loss] = padding_i[all_loss > max_loss]
                max_loss = torch.max(max_loss, all_loss)

        # when the max loss is zero, we cannot choose one part to attack, we will randomly choose a positon
        zero_loss =  np.transpose(np.argwhere(max_loss.cpu()==0))

        for ind in zero_loss:
            output_j[ind] = torch.randint(ytimes,(1,))
            output_i[ind] = torch.randint(xtimes,(1,))

        with torch.set_grad_enabled(True):
            return self.cpgd(X,y,width, height, xskip, yskip, output_j, output_i ,mean)


    def gradient_based_search(self,X,y, width, height, xskip, yskip):
        #print(">>> in gbs X is: ", X)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        gradient = torch.zeros_like(X,requires_grad=True).to(device)
        X1 = torch.zeros_like(X,requires_grad=True)
        X = X.to(device)
        X1.data = X.detach().to(device)
        y = y.to(device)
        model = self.base_classifier
        loss = nn.CrossEntropyLoss()(model(X1), y)
        loss.backward() #to do error here
        gradient.data = X1.grad.detach()
        max_val,indice = torch.max(torch.abs(gradient.view(gradient.shape[0], -1)),1)
        gradient = gradient / max_val[:,None,None,None]
        X1.grad.zero_()
        mean = torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).view(1, 3, 1, 1)
        mean = mean.to(device)
        xtimes = 224//xskip
        ytimes = 224//yskip
        nums = 30  #default number of
        output_j1 = torch.zeros(y.shape[0]).repeat(nums).view(y.shape[0],nums)
        output_i1 = torch.zeros(y.shape[0]).repeat(nums).view(y.shape[0],nums)
        matrix = torch.zeros([ytimes*xtimes]).repeat(1,y.shape[0]).view(y.shape[0],ytimes*xtimes)
        max_loss = torch.zeros(y.shape[0]).to(y.device)
        all_loss = torch.zeros(y.shape[0]).to(y.device)

        for i in range(xtimes):
            for j in range(ytimes):
                num = gradient[:,:,yskip*j:(yskip*j+height),xskip*i:(xskip*i+width)]
                loss = torch.sum(torch.sum(torch.sum(torch.mul(num,num),1),1),1)
                matrix[:,j*ytimes+i] = loss

        topk_values, topk_indices = torch.topk(matrix,nums)
        output_j1 = topk_indices//xtimes
        output_i1 = topk_indices %xtimes

        output_j = torch.zeros(y.shape[0]) + output_j1[:,0].float()
        output_i = torch.zeros(y.shape[0]) + output_i1[:,0].float()
        with torch.set_grad_enabled(False):
            for l in range(output_j1.size(1)):
                sticker = X + mean
                for m in range(output_j1.size(0)):
                    sticker[m,:,yskip*output_j1[m,l]:(yskip*output_j1[m,l]+height),xskip*output_i1[m,l]:(xskip*output_i1[m,l]+width)] = 255/2
                sticker1 = sticker.detach() - mean.detach()
                all_loss = nn.CrossEntropyLoss(reduction='none')(model(sticker1),y)
                padding_j = torch.zeros(y.shape[0]) + output_j1[:,l].float()
                padding_i = torch.zeros(y.shape[0]) + output_i1[:,l].float()
                output_j[all_loss > max_loss] = padding_j[all_loss > max_loss]
                output_i[all_loss > max_loss] = padding_i[all_loss > max_loss]
                max_loss = torch.max(max_loss, all_loss)
            #print(output_j,output_i)
        return self.cpgd(X,y,width, height, xskip, yskip, output_j, output_i ,mean)


    def cpgd(self,X,y,width, height, xskip, yskip, out_j, out_i,mean):
        model = self.base_classifier
        model.eval()
        alpha = self.alpha
        num_iter = self.iters
        sticker = torch.zeros(X.shape, requires_grad=True)
        for num,ii in enumerate(out_i):
            j = int(out_j[num].item())
            i = int(ii.item())
            with torch.no_grad(): #I added this
                sticker[num,:,yskip*j:(yskip*j+height),xskip*i:(xskip*i+width)] = 1
        sticker = sticker.to(y.device)


        delta = torch.zeros_like(X, requires_grad=True)+255/2
        #delta = torch.rand_like(X, requires_grad=True).to(y.device)
        #delta.data = delta.data * 255

        X1 = torch.rand_like(X, requires_grad=True).to(y.device)
        X1.data = X.detach()*(1-sticker)+((delta.detach()-mean)*sticker)


        for t in range(num_iter):
            loss = nn.CrossEntropyLoss()(model(X1), y)
            loss.backward()

            X1.data = (X1.detach() + alpha*X1.grad.detach().sign()*sticker)
            X1.data = ((X1.detach() + mean).clamp(0,255)-mean)
            X1.grad.zero_()
        return (X1).detach()



def sticker_train_model(model, criterion, optimizer, scheduler, alpha, iters, search, num_epochs=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs[:,[2,1,0],:,:] #rgb to bgr
                ROA_module = ROA(model,alpha,iters)
                with torch.set_grad_enabled(search==1):
                    if search == 0:
                        ROA_inputs = ROA_module.exhaustive_search(inputs, labels,args.width,args.height,args.stride,args.stride)
                        # if the number is wrong, run gradient_based_search, since this method can save time
                    else:
                        ROA_inputs = ROA_module.gradient_based_search(inputs, labels, args.width,args.height,args.stride,args.stride)
                        #ROA_inputs = ROA_module.gradient_based_search(inputs, labels,args.width,args.height,args.stride,args.stride)

                optimizer.zero_grad()
                if phase == 'train':
                    model.train()

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(ROA_inputs)
                    _, preds = torch.max(outputs, 1)
                    labels = labels.to(device)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model_ft.state_dict(), '../donemodel/new_sticker_model0'+str(args.out)+'.pt')
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("model", type=str, help="original(clean) model you want to do DOA ")
    parser.add_argument("-alpha", type=int, help="alpha leanrning rate")
    parser.add_argument("-iters", type=int, help="iterations of PGD ")
    parser.add_argument("-out", type=int, help="name of final model")
    parser.add_argument("-search", type=int, help="method of searching, \
        '0' is exhaustive_search, '1' is gradient_based_search")
    parser.add_argument("-epochs", type=int, help="epochs")
    parser.add_argument("--stride", type=int, default=10, help="the skip pixels when searching")
    parser.add_argument("--width", type=int, default= 70, help="width of the rectuagluar occlusion")
    parser.add_argument("--height", type=int, default=70, help="height of the rectuagluar occlusion")
    args = parser.parse_args()

    print(args)


    torch.manual_seed(123456)
    torch.cuda.empty_cache()
    print('output model will locate on ../donemodel/new_balaned_adv'+str(args.out)+'.pt')
    dataloaders,dataset_sizes =data_process(batch_size =32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = VGG_16()
    model_ft.load_state_dict(torch.load('../donemodel/'+args.model))
    #model_ft.load_weights()
    model_ft.to(device)

    # model_ft = nn.DataParallel(model,device_ids=[0,1])

    criterion = nn.CrossEntropyLoss()

    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

    model_ft = sticker_train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,args.alpha, args.iters,args.search, num_epochs=args.epochs)
    test(model_ft,dataloaders,dataset_sizes)

    torch.save(model_ft.state_dict(), '../donemodel/new_balaned_adv'+str(args.out)+'.pt')
