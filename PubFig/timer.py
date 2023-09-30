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
    labels=pickle.load(f).astype(int)

with open("indices",'rb') as f:
    indices=pickle.load(f)
with open("matrix",'rb') as f:
    matrix=pickle.load(f)
matrix=1-matrix
with open("REFtest",'rb') as f:
    REF=pickle.load(f)
with open("ORDERprev",'rb') as f:
    ORDERprev=pickle.load(f)
with open("ORDERafter",'rb') as f:
    ORDERafter=pickle.load(f)
#method="baseline"
#method="matrix"
#method="prev"
#method="matrix-prev"
method="matrix-after"
#method="after"


record={}
for tag in indices:
    (i,j,g)=tag
    print (i,j,g)
    targets=perm[j:60]
    source=perm[:i]
    indice=indices[tag]
    counts=list()
    for images in indice:
        if method=="matrix":
            label=labels[images]
                
            relevant=matrix[targets][:,label]
        if method=="prev":
            relevant=ORDERprev[targets][:,images]
        if method=="after":
            relevant=ORDERafter[targets][:,images]
        if method=="baseline":
            relevant=np.random.permutation((60-j)*i).reshape((60-j,i))
        
        if method=="matrix-prev":
            label=labels[images]
            relevant=matrix[targets][:,label]
            relevant*=ORDERprev[targets][:,images]
        if method=="matrix-after":
            label=labels[images]
            relevant=matrix[targets][:,label]
            relevant*=ORDERafter[targets][:,images]        
        
        succ=REF[targets][:,images]
        
        
        ordering=np.argsort(relevant,axis=None)
        total=0
        hits=list()
        c=0
        while (len(hits)<g):
            idx=ordering[c]
            c+=1
            row=idx // i
            col=idx % i
            
            if row in hits:
                continue
            
            if succ[row][col]==1:
                hits.append(row)
            total+=1
            
            
        print (total)
        counts.append(total)
    record[tag]=counts
with open("TIME-"+method,'wb') as f:
    pickle.dump(record,f)

