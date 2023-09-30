import pickle
import numpy as np
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.random.manual_seed(0)
torch.random.manual_seed(0)
torch.manual_seed(0)
np.random.seed(0)
from k_search import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--method',type=str, default='baseline')
parser.add_argument('--distance',type=str, default='l2')


args = parser.parse_args()
print("ARGS: ", args)
method=args.method
distance=args.distance
if distance != "l2":
        distance="linf"

with open("A",'rb') as f:
    a=pickle.load(f)
print (a)
with open("Y",'rb') as f:
    y=pickle.load(f)[:1000]


if distance=="linf":
    with open("linf",'rb') as f:
        REF=pickle.load(f)[:1000]
    with open("indiceslinfK",'rb') as f:
        indices=pickle.load(f)
    with open("matrixlinf_valid",'rb') as f:
        matrix=pickle.load(f)
    matrix=1-matrix
    with open("prev-linf",'rb') as f:
        ORDERprev=pickle.load(f)
    with open("after-linf",'rb') as f:
        ORDERafter=pickle.load(f)
else:
    with open("l2",'rb') as f:
        REF=pickle.load(f)[:1000]
    with open("indicesl2K",'rb') as f:
        indices=pickle.load(f)
    with open("matrixl2_valid",'rb') as f:
        matrix=pickle.load(f)
    matrix=1-matrix
    with open("prev-l2",'rb') as f:
        ORDERprev=pickle.load(f)
    with open("after-l2",'rb') as f:
        ORDERafter=pickle.load(f)

record={}
for tag in indices:
    (i,j,g)=tag
    print (i,j,g)
    targets=a[j:60]
    source=a[:i]
    indice=indices[tag]
    counts=list()
    for images in indice:
        
        if method=="matrix":
            labels=y[images]
            l=np.in1d(a, labels).nonzero()[0]
            relevant=matrix[j:60][:,l]
        if method=="prev":
            relevant=ORDERprev[j:60][:,images]
        if method=="after":
            relevant=ORDERafter[j:60][:,images]
        if method=="baseline":
            relevant=np.random.permutation((60-j)*i).reshape((60-j,i))
        if method=="matrix-prev":
            labels=y[images]
            l=np.in1d(a, labels).nonzero()[0]
            relevant=matrix[j:60][:,l]
            relevant*=ORDERprev[j:60][:,images]
        if method=="matrix-after":
            labels=y[images]
            l=np.in1d(a, labels).nonzero()[0]
            relevant=matrix[j:60][:,l]
            relevant*=ORDERafter[j:60][:,images]

        succ=REF[j:60][:,images]
        
        
        ordering=np.argsort(relevant,axis=None)
        total=0
        hits=list()
        attempts=np.zeros(succ.shape)
        c=0
        while (k_search(attempts,g)==False):
            idx=ordering[c]
            c+=1
            row=idx // i
            col=idx % i
            
            attempts[row][col]=succ[row][col]
            total+=1
            
            
        print (total)
        counts.append(total)
    record[tag]=counts
with open("TIME-"+distance+"-"+method+"K",'wb') as f:
    pickle.dump(record,f)

