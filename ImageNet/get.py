
import pickle
import numpy as np
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.random.manual_seed(0)
torch.random.manual_seed(0)
torch.manual_seed(0)
np.random.seed(0)

import argparse
parser = argparse.ArgumentParser()


parser.add_argument('--distance',type=str, default='l2')
parser.add_argument('--permutation',type=int, default=0)

args = parser.parse_args()
print("ARGS: ", args)
distance=args.distance
permutation=args.permutation
if distance != "l2":
        distance="linf"

with open("A"+str(permutation),'rb') as f:
    a=pickle.load(f)
print (a)

with open("A",'rb') as f:
    aa=pickle.load(f)
print (aa)


indices=list()
for i in range(60):
    d=a[i]
    indices.append(aa.index(d))

print (indices)
with open("Y",'rb') as f:
    y=pickle.load(f)

with open(distance,'rb') as f:
    data=pickle.load(f)


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
with open('guess'+distance,'wb') as f:
    pickle.dump(record,f)

