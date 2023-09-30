# This file is based on (pytorch version of) https://github.com/mahmoods01/accessorize-to-a-crime 
# Mahmood Sharif


'''
Eyeglass Frame Attack 
Type 'python glass_attack.py {}.pt'  to run 
{} is the name of your model want to attack. Note that you cannot attack randomized smoothing in this file, 
please use smooth_glassattack.py
itera   = [1 , 2  , 3  , 5  , 7  , 10 , 20 , 50 , 100 , 300 ] # this is default numbers we used in experiment, 
which is the iterations of attacks 

Note that the attack is in digit space (not involved rotation and scale) (fixed eyeglass frame mask),
and untargeted (maximize the loss of (f(x),y) )
'''
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from origin_train import data_process
#from origin_train2 import data_process,data_process2
import numpy as np
import argparse
import torchvision
#import cv2
from torchvision import datasets, models, transforms
import sys
import numpy
from new_vgg_face import VGG_16
import time
#from save_image import save_image 
#uncomment to see some images 
numpy.set_printoptions(threshold=sys.maxsize)

def md_loss_targeted(x, y_target):
    x_target=torch.unsqueeze(x[np.arange(x.shape[0]), y_target],1).expand(-1,x.shape[1])
    return torch.sum(nn.functional.relu(x+1e-15-x_target),dim=1)

def md_loss_max( x, y_targets):
    
    a=torch.arange(60)
    b=np.arange(60)
    y_nontargets=a[np.delete(b,y_targets.cpu().numpy())]
    x_targets=torch.index_select(x,1,y_targets.to('cuda'))
    x_nontargets=torch.index_select(x,1,y_nontargets.to('cuda'))
    x_target_max=torch.unsqueeze(torch.max(x_targets,1)[0],1).expand(-1,x_nontargets.shape[1])
    return torch.sum(nn.functional.relu(x_nontargets+1e-15-x_target_max),dim=1)




def choose_color(model,X,t,glass,mean):
    model.eval()
    potential_starting_color0 = [128,220,160,200,220]
    potential_starting_color1 = [128,130,105,175,210]
    potential_starting_color2 = [128,  0, 55, 30, 50]

    min_loss = (torch.ones(X.shape[0])*1000000).to(t.device)
    min_delta = torch.zeros_like(X)
     
    
    for i in range(len(potential_starting_color0)):
        delta1 = torch.zeros(X.size()).to(t.device)


        delta1[:,0,:,:] = glass[0,:,:]*potential_starting_color2[i]
        delta1[:,1,:,:] = glass[1,:,:]*potential_starting_color1[i]
        delta1[:,2,:,:] = glass[2,:,:]*potential_starting_color0[i]

        all_loss = md_loss_max(model(X+delta1-mean),t)
        min_delta[all_loss < min_loss] = delta1.detach()[all_loss < min_loss]
        min_loss = torch.min(min_loss, all_loss)

    return min_delta


def glass_attack(model, X, t, glass, alpha=1, num_iter=20,momentum=0.4):
    """ Construct glass frame adversarial examples on the examples X"""

    model.eval()
    mean = torch.Tensor(np.array([129.1863 , 104.7624,93.5940])).view(1, 3, 1, 1)
    de = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = mean.to(de)
    X1 = (X+mean)*(1-glass)
    
    with torch.set_grad_enabled(False):
        color_glass = choose_color(model,X1,t,glass,mean)
    with torch.set_grad_enabled(True):
        X1= X1+color_glass-mean
        
    
        for i in range(num_iter):
            ts=time.time()
            X1.requires_grad=True
            loss = md_loss_max(model(X1), t)
            #print (time.time()-ts)
            #print (loss[:10].cpu().detach())
            grad=torch.autograd.grad(loss.sum(), [X1])[0].detach()
            #print (time.time()-ts)
            X1.requires_grad=False 
            delta_change =  grad*glass
            max_val,indice = torch.max(torch.abs(delta_change.view(delta_change.shape[0], -1)),1)
            r = alpha * delta_change /max_val[:,None,None,None]
            r[r.isnan()]=0
            if i == 0:
                delta = r
            else:
                delta = momentum * delta.detach() + r

            delta[(-delta.detach() + X1.detach() + mean) >255] = 0 
            delta[(-delta.detach() + X1.detach() + mean) < 0 ] = 0 
            X2 = (X1.detach() - delta.detach())
            X1[loss>0]=X2[loss>0]
            X1 = torch.round(X1.detach()+mean) - mean
            #print (time.time()-ts)
            #X1.zero_grad()

        #return (X1).detach()
                    
         
    with torch.no_grad():
        Xadv = (X1).detach()
        model.eval()

        outputs = model(Xadv)

        _, predicted = torch.max(outputs.data, 1)
    return sum(predicted==j for j in t).bool().cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument("model", type=str, help="test_model")
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True 
    torch.cuda.random.manual_seed(12345)
    torch.random.manual_seed(12345) 
    torch.manual_seed(12345)


    alpha   = 20
    itera   = [300]#[1 , 2  , 3  , 5  , 7  , 10 , 20 , 50 , 100 , 300 ] # this is default numbers we used in experiment
    restart = 1
    Nclass=60
    
    with open('perm4','rb') as f:
        perm=pickle.load(f)

    
    glass=torchvision.io.read_image('silhouette.png').float()
    glass[glass>0.5]=1
    glass[glass<0.5]=0
    
    model = VGG_16() 
    model.load_state_dict(torch.load('../donemodel/'+args.model))

    model.eval()
    batch_size = 1#64
    dataloaders,dataset_sizes =data_process(batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    glass = glass.to(device)
    
    
        
    total = 0 
    print (len(dataloaders['test']))
    x=torch.zeros((744,3,224,224))
    y=torch.zeros(744)
    for data in dataloaders['test']:
        images, labels = data
        images = images[:,[2,1,0],:,:] #rgb to bgr
        bs=labels.size(0)
        x[total:total+bs]=images
        y[total:total+bs]=labels
        total+=bs
    print (total)
    dic={}
    """
    for i in range(10,60,10):
        for k in range(i,60,10):
    """

    ts=0
    c=0
    x=x[:batch_size]
    y=y[:batch_size]
    for i in [10,10,10,10,10,10,10,10,10,10]:
        for k in [10,10,10,10,10]:
            print (i,k)
            targets=torch.tensor(perm[k:60]).long().to(device)
            source=torch.tensor(perm[:i]).long()
            #select=torch.isin(y,source)
            select=torch.isin(y,y)

            x_in=x[select]

            
        
            total=x_in.size(0)
            print (total)
        
        
            record=np.zeros(total)
            for j in range(int(np.ceil(total/batch_size))):
                start=j*batch_size
                end=min(batch_size*(j+1),total)
                print (start,end)
                images = x_in[start:end].to(device) 
                st=time.time()
                record[start:end]=glass_attack(model, images, targets,glass, alpha ,300)
                ts+=time.time()-st
                c+=1
                print (ts/c)
            dic[(i,60-k)]=record
            with open('4MAX','wb') as f:
                pickle.dump(dic,f)
        
        
        
