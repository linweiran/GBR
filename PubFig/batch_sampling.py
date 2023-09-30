import pickle
import numpy as np
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.random.manual_seed(0)
torch.random.manual_seed(0)
torch.manual_seed(0)
np.random.seed(0)

with open("perm",'rb') as f:
    perm=pickle.load(f)
print (perm)
with open("labels-test",'rb') as f:
    labels=pickle.load(f)

with open("REFtest",'rb') as f:
    data=pickle.load(f)


record={}
chances={}
for i in range(10,60,10):
    print (i)
    selects=list()
    for t in range(100000):
        rands=np.random.rand(i)
        selected=list()
        for k in range(i):
            select=(labels==perm[k])
            indices=np.arange(labels.shape[0])[select]
            lucky=indices[int(np.floor(rands[k]*indices.shape[0]))]
            selected.append(lucky)
        selects.append(selected)
    
    
    for j in range(i,60,10):
        targets=perm[j:60]
        source=perm[:i]
        used={}
        for k in range(5,65-j,5):
            used[k]=list()
        
        for selected in selects:

            look=data[:,selected]
            look=look[targets]
            
            counts=np.sum(np.sum(look,axis=1)>0)
            for k in range(5,65-j,5):
                if (counts>=k):
                    used[k].append(selected)
        for k in range(5,65-j,5):
            chances[(i,j,k)]=len(used[k])/100000
            if len(used[k])>=1000:
                touse=used[k]
                touse=touse[:1000]
                record[(i,j,k)]=touse            
        
        
with open('indices','wb') as f:
    pickle.dump(record,f)
with open('chance','wb') as f:
    pickle.dump(chances,f)
