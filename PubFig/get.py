
import pickle
import numpy as np
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.random.manual_seed(0)
torch.random.manual_seed(0)
torch.manual_seed(0)
np.random.seed(0)

with open("perm3",'rb') as f:
    perm=pickle.load(f)
print (perm)
with open("labels-test",'rb') as f:
    labels=pickle.load(f)

with open("REFtest",'rb') as f:
    data=pickle.load(f)


record={}
for i in range(10,60,10):
    for j in range(i,60,10):
        print (i,60-j)
        targets=perm[j:60]
        source=perm[:i]
        #look=data[np.isin(np.arange(60),targets)]
        look=data[targets]
        look=look[:,np.isin(labels,source)]
        print (np.mean(look))
        print (np.mean(np.sum(look,axis=0)==60-j))
        print (np.mean(np.sum(look,axis=0)>0))
        record[(i,60-j)]=(np.mean(look),np.mean(np.sum(look,axis=0)==60-j),np.mean(np.sum(look,axis=0)>0))
with open('guess4','wb') as f:
    pickle.dump(record,f) 
