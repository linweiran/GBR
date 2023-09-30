
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
    aa=pickle.load(f)
print (aa)


with open("A",'rb') as f:
    a=pickle.load(f)
print (a)


indices=list()
for i in range(60):
    d=a[i]
    indices.append(aa.index(d))

print (indices)
with open("Y",'rb') as f:
    y=pickle.load(f)

with open("l2",'rb') as f:
    l2=pickle.load(f)
with open("linf",'rb') as f:
    linf=pickle.load(f)

data=linf

record={}
for i in range(10,60,10):
    for j in range(i,60,10):
        print (i,j)
        targets=a[j:60]
        source=a[:i]
        
        look=data[:,np.isin(y,source)]
        look=look[indices[j:60]]
        print (np.mean(look))
        print (np.mean(np.sum(look,axis=0)==60-j))
        print (np.mean(np.sum(look,axis=0)>0))
        record[(i,60-j)]=(np.mean(look),np.mean(np.sum(look,axis=0)==60-j),np.mean(np.sum(look,axis=0)>0))
with open('guessLinf','wb') as f:
    pickle.dump(record,f)
print (record)
