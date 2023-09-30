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

with open("l2",'rb') as f:
    data=pickle.load(f)


y=y[:1000]
data=data[:1000]

record={}
chance={}
for i in range(10,60,10):
    print (i)
        
    selects=list()
    for t in range(100000):
        rands=np.random.rand(i)
        selected=list()
        for k in range(i):
            select=(y==a[k])
            indices=np.arange(y.shape[0])[select]
            lucky=indices[int(np.floor(rands[k]*indices.shape[0]))]
            selected.append(lucky)
        selects.append(selected)
    for j in range(i,60,10):
        used={}
        for k in range(2,11):
            used[k]=list()
        for selected in selects:
            look=data[:,selected]
            look=look[j:60]
            counts=np.sum(np.sum(look,axis=1)>0)
            
            for k in range(2,11):
                if (counts>=k) and (np.sum(np.sum(look[60-j-5:],axis=1)>0)>0):#the last five are managers
                    used[k].append(selected)
        for k in range(2,11):
            chance[(i,j,k)]=len(used[k])/100000
            if len(used[k])>=1000:
                touse=used[k]
                touse=touse[:1000]
                record[(i,j,k)]=touse
            
with open('chancesl2M','wb') as f:
    pickle.dump(chance,f)
with open('indicesl2M','wb') as f:
    pickle.dump(record,f)

