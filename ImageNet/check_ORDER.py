import pickle
import numpy as np
import torch
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
print (y.shape)
from robustness.model_utils import make_and_restore_model


from robustness.datasets import ImageNet
ds=ImageNet('../ILSVRC')
model, _ = make_and_restore_model(arch='resnet50', dataset=ds,resume_path='imagenet_linf_8.pt')
#model, _ = make_and_restore_model(arch='resnet50', dataset=ds,resume_path='imagenet_l2_3_0.pt')

model.cuda().eval()

from attacks import PGD

prev=np.zeros((60,x.shape[0]))
after=np.zeros((60,x.shape[0]))
pgd=PGD(model,nclass=1000,eps=8./255,loss='md_single',rand_t=True)
#pgd=PGD(model,nclass=1000,eps=3.0,norm='L2',loss='md_single',rand_t=True)



for i in range(20):
    y_targets=torch.ones(x.shape[0])*a[i]
    
    pre,aft=pgd.get_md(x,y,y_targets=y_targets,bs=10)
    prev[i]=pre
    after[i]=aft
    with open('prev-linf-1','wb') as f:
        pickle.dump(prev,f)
    with open('after-linf-1','wb') as f:
        pickle.dump(after,f)


