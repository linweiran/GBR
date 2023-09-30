import pickle
import numpy as np
import torch
import time
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.random.manual_seed(0)
torch.random.manual_seed(0)
torch.manual_seed(0)
np.random.seed(0)
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--data_path',type=str, default='/data/ILSVRC')
parser.add_argument('--distance',type=str, default='l2')
parser.add_argument('--permutation',type=int, default=0)

args = parser.parse_args()
print("ARGS: ", args)
MAIN_DIR=args.data_path
distance=args.distance
permutation=args.permutation
if distance != "l2":
        distance="linf"

with open("A"+str(permutation),'rb') as f:
    a=pickle.load(f)
print (a)
with open("Y",'rb') as f:
    y=pickle.load(f)
with open("X",'rb') as f:
    x=pickle.load(f)
print (y.shape)

from robustness.model_utils import make_and_restore_model


from robustness.datasets import ImageNet
ds=ImageNet(MAIN_DIR)
if distance != "l2":
	model, _ = make_and_restore_model(arch='resnet50', dataset=ds,resume_path='imagenet_linf_8.pt')
else:
	model, _ = make_and_restore_model(arch='resnet50', dataset=ds,resume_path='imagenet_l2_3_0.pt')

model.cuda().eval()

from attacks import PGD

if distance != "l2":
	pgd=PGD(model,nclass=1000,eps=8./255,loss='md_max')
else:
	pgd=PGD(model,nclass=1000,eps=3.0,norm='L2',loss='md_max')

dic={}

for i in range(10,60,10):
    for j in range(i,60,10):
        print (i,j)
        targets=a[j:60]
        source=a[:i]
        select=np.isin(y,source)

        st=time.time()
        robustness=pgd.perturb(x[select],y[select],y_targets=torch.tensor(targets),bs=10)
        dic[(i,60-j)]=robustness
        
        print(time.time()-st)
        
        with open("MAX"+str(permutation)+distance,'wb') as f:
            pickle.dump(dic,f)




