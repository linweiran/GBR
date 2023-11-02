import torch
import torchvision
import numpy as np
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from load_dataset import load_gtsrb
import pickle

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--data_path',type=str, default='/data/GTSRB')

args = parser.parse_args()
print("ARGS: ", args)
MAIN_DIR=args.data_path

(x_train, y_train), (x_test, y_test), (x_val, y_val)=load_gtsrb(data_path=MAIN_DIR)
print ("data loaded sucessfullly!")


x_train=np.swapaxes(x_train, 1, 3)
x_test=np.swapaxes(x_test, 1, 3)
x_val=np.swapaxes(x_val, 1, 3)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


from basic_net import GTSRBNet2,GTSRBNet3,GTSRBNet4,GTSRBNet6,GTSRBNet7

from basic_operations import train,test
from attacks import pgd_attack,autopgd, APGD_caller,PGD,APGD_targeted
from defenses import FreeAdvTrain,NormalTrain




origins=[0,1,2,3,4,5,6,7,8]
pairs=[(0,14),(0,15),(0,17),(1,14),(1,15),(1,17),(2,0),(2,14),(2,15),(2,17),(3,0),(3,1),(3,14),(3,15),(3,17),(4,0),(4,1),(4,14),(4,15),(4,17),(5,0),(5,1),(5,14),(5,15),(5,17),(6,0),(6,1),(6,14),(6,15),(6,17),(7,0),(7,1),(7,2),(7,14),(7,15),(7,17),(8,0),(8,1),(8,2),(8,3),(8,14),(8,15),(8,17)]
targets=[0,1,2,3,14,15,17]

accl=list()
robustnesslAPGD=list()
robustnesslAPGDT=list()
acclS=list()
acclT=list()
robustnesslAPGDS=list()
robustnesslAPGDSt_best=list()
robustnesslAPGDSt_worst=list()
robustnesslAPGDSt_average=list()
robustnesslAPGDSt_max=list()
robustnesslAPGDSt_group=list()

for random_seed in range(100):
    torch.manual_seed(random_seed)
    torch.cuda.random.manual_seed(random_seed)
    np.random.seed(random_seed)
    #initialize model
    
    ADVnetwork = GTSRBNet3()
    ADVoptimizer=optim.Adam(ADVnetwork.parameters())

    
    eps=8./255
    device='cuda'
    batch_size=384
    nclass=43
    norm='Linf'
    model=FreeAdvTrain(ADVnetwork,ADVoptimizer, x_train, y_train,x_test,y_test,512,max_iters=50,eps=eps,m=8,device=device,dataset='GTSRB',nclass=nclass,x_val=x_val,y_val=y_val,origins=origins,pairs=pairs)
    model=NormalTrain(ADVnetwork,ADVoptimizer, x_train, y_train,x_test,y_test,batch_size,max_iters=50,eps=eps,m=6,device=device,dataset='GTSRB',nclass=nclass,x_val=x_val,y_val=y_val,origins=origins,pairs=pairs)


    
    
    robustnesslAPGDT.append(1-APGD_targeted(model,x_test,y_test,0,nclass,eps,norm,bs=256))
    accl.append(test(model,x_test,y_test,512,device=device))    
    robustnesslAPGD.append(autopgd(model,x_test,y_test,0,eps=eps,device='cuda',bs=512,norm=norm))
    test_select=np.isin(y_test,origins)
    x_t, y_t=x_test[test_select], y_test[test_select]
    acclS.append(test(model,x_t,y_t,batch_size,device='cuda'))
    robustnesslAPGDS.append(autopgd(model,x_t,y_t,0,eps=eps,device='cuda',bs=batch_size,norm=norm))
    apgd=PGD(model,nclass=nclass,eps=eps,loss='md_max',norm=norm)
    succ=APGD_caller(apgd,pairs,x_t, y_t,bs=batch_size,auto=True)
    robustnesslAPGDSt_max.append(1-succ)
    apgd=PGD(model,nclass=nclass,eps=eps,loss='md_group',norm=norm)
    succ=APGD_caller(apgd,pairs,x_t, y_t,bs=batch_size,auto=True)
    robustnesslAPGDSt_group.append(1-succ)
    apgd=PGD(model,nclass=nclass,eps=eps,loss='md_single',norm=norm)
    succ,best,worst=APGD_caller(apgd,pairs,x_t, y_t,bs=batch_size,auto=True)
    robustnesslAPGDSt_average.append(1-succ)
    robustnesslAPGDSt_best.append(1-best)
    robustnesslAPGDSt_worst.append(1-worst)   
    test_select=np.isin(y_test,targets)
    x_t, y_t=x_test[test_select], y_test[test_select]
    acclT.append(test(model,x_t,y_t,batch_size,device='cuda')) 
     

    performance={}
    with open('models/AGTSRB-performance3-L_inf-indivlg','wb') as f:       
            performance['Average accuracy']=accl
            performance['Accuracy on targeted classes']=acclT
            performance['Targeted Robustness, assessed by Auto-PGD']=robustnesslAPGDT
            performance['Untargeted Robustness, assessed by Auto-PGD']=robustnesslAPGD
            performance['Group-based Robustness, assessed by attacks with MDMAX loss']=robustnesslAPGDSt_max
            performance['Group-based Robustness, assessed by attacks with MDMUL loss']=robustnesslAPGDSt_group
            performance['Group-based Robustness, assessed by average guess attacks']=robustnesslAPGDSt_average
            performance['Group-based Robustness, assessed by best guess attacks']=robustnesslAPGDSt_best
            
            pickle.dump(performance,f)
    



