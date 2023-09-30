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


with open("A2",'rb') as f:
    a=pickle.load(f)
print (a)
with open("Y",'rb') as f:
    y=pickle.load(f)
with open("X",'rb') as f:
    x=pickle.load(f)
print (y.shape)

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
#pgd=PGD(model,nclass=1000,eps=8./255,loss='md_group')
pgd=PGD(model,nclass=1000,eps=3.0,norm='L2',loss='md_group')

dic={}
t=0
c=0
for i in [10,10,10,10,10,10,10,10,10,10]:#range(10,60,10):
    for j in [10,10,10,10,10]:#range(i,60,10):
        print (i,j)
        targets=a[j:60]
        source=a[:i]
        #select=np.isin(y,source)
        select=np.isin(y,y)
        st=time.time()
        robustness=pgd.perturb(x[select],y[select],y_targets=torch.tensor(targets),bs=1)
        t+=(time.time()-st)
        c+=1
        print (t/c)
        dic[(i,60-j)]=robustness
        #with open("GROUP2L2-log",'wb') as f:
        #    pickle.dump(dic,f)




