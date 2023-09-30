import pickle
import numpy as np
NUM=60
with open('REF','rb') as f:
    ref=pickle.load(f)
with open('labels','rb') as f:
    labels=pickle.load(f)
print (labels.shape)
print (ref.shape)
ref=ref[:NUM]
mat=np.zeros((NUM,60))
total=np.zeros((NUM,60))
for i in range(ref.shape[0]):
    for j in range(ref.shape[1]):
        mat[i][int(labels[j])]+=ref[i][j]
        total[i][int(labels[j])]+=1
mat/=total
for i in range(ref.shape[0]):
    mat[i][i]=0
with open('matrix','wb') as f:
    pickle.dump(mat,f)
