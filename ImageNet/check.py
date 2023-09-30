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

with open("A",'rb') as f:
    a=pickle.load(f)
print (a)
with open("Y",'rb') as f:
    y=pickle.load(f)
with open("X",'rb') as f:
    x=pickle.load(f)



from robustness.model_utils import make_and_restore_model


from robustness.datasets import ImageNet
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--data_path',type=str, default='/data/ILSVRC')
parser.add_argument('--distance',type=str, default='l2')

args = parser.parse_args()
print("ARGS: ", args)
MAIN_DIR=args.data_path
distance=args.distance
if distance != "l2":
	distance="linf"


ds=ImageNet('MAIN_DIR')
if distance != "l2":
	model, _ = make_and_restore_model(arch='resnet50', dataset=ds,resume_path='imagenet_linf_8.pt')
else:
	model, _ = make_and_restore_model(arch='resnet50', dataset=ds,resume_path='imagenet_l2_3_0.pt')

model.cuda().eval()

from attacks import PGD

record=np.zeros((60,x.shape[0]))
if distance != "l2":
	pgd=PGD(model,nclass=1000,eps=8./255,loss='md_single',rand_t=True)
else:
	pgd=PGD(model,nclass=1000,eps=3.0,norm='L2',loss='md_single',rand_t=True)
from basic_operations import test
test(model, x,y,10,device='cuda')
from attacks import autopgd
if distance == "l2":
	print(autopgd(model,x,y,0,eps=3.0,device='cuda',bs=10,norm='L2'))
else:
	print(autopgd(model,x,y,0,eps=8./255,device='cuda',bs=10))




for i in range(60):
    y_targets=torch.ones(x.shape[0])*a[i]

    robustness=pgd.perturb(x,y,y_targets=y_targets,bs=10)
    record[i]=robustness


    with open(distance,'wb') as f:
        pickle.dump(record,f)


