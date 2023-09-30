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

x=x[:1]
y=y[:1]


from robustness.model_utils import make_and_restore_model


from robustness.datasets import ImageNet
ds=ImageNet('../ILSVRC')
#model, _ = make_and_restore_model(arch='resnet50', dataset=ds,resume_path='imagenet_linf_8.pt')
model, _ = make_and_restore_model(arch='resnet50', dataset=ds,resume_path='imagenet_l2_3_0.pt')

model.cuda().eval()

from attacks import PGD

record=np.zeros((60,x.shape[0]))
pgd=PGD(model,nclass=1000,eps=8./255,loss='md_single',rand_t=True)
#pgd=PGD(model,nclass=1000,eps=3.0,norm='L2',loss='md_single',rand_t=True)
from basic_operations import test
#test(model, x,y,10,device='cuda')
from attacks import autopgd
#print(autopgd(model,x,y,0,eps=3.0,device='cuda',bs=10,norm='L2'))
#print(autopgd(model,x,y,0,eps=8./255,device='cuda',bs=10))
"""
from robustness.datasets import ImageNet
ds=ImageNet('../ILSVRC')
_, test_loader = ds.make_loaders(workers=0, batch_size=50,only_val=True)
acc=0
rob=0
count=0
for im, label in iter(test_loader):
    img=im.numpy()
    labeling=label.numpy()
    c=0
    x=[]
    y=[]
    for i in range(50):
        c+=1
        x.append(img[i])
        y.append(labeling[i])
    x=np.array(x)
    y=np.array(y)
    total=x.shape[0]
    per=np.random.permutation(total)
    x=x[per]
    y=y[per]
    
    acc+=test(model, x,y,10,device='cuda')
    rob+=autopgd(model,x,y,0,eps=8./255,device='cuda',bs=10)
    #rob+=autopgd(model,x,y,0,eps=3.0,device='cuda',bs=10,norm='L2')
    count+=1
    print(acc/count,rob/count)
"""

st=time.time()
print (time.time()-st)
for i in range(50):
    y_targets=torch.ones(x.shape[0])*a[i]

    #robustness=pgd.perturb(x,y,y_targets=y_targets,bs=10)
    robustness=pgd.perturb(x,y,y_targets=y_targets,bs=1)
    record[i]=robustness
    print (time.time()-st)
""""

    with open('l2-1','wb') as f:
        pickle.dump(record,f)
"""

