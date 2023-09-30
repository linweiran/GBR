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
a=np.array(a)
p1=np.random.permutation(60).astype(int)
p2=np.random.permutation(60).astype(int)
p3=np.random.permutation(60).astype(int)
p4=np.random.permutation(60).astype(int)


print (p1)
print (a[p1])
with open("A0",'wb') as f:
    pickle.dump(a.tolist(),f)

with open("A1",'wb') as f:
    pickle.dump(a[p1].tolist(),f)
with open("A2",'wb') as f:
    pickle.dump(a[p2].tolist(),f)

with open("A3",'wb') as f:
    pickle.dump(a[p3].tolist(),f)
with open("A4",'wb') as f:
    pickle.dump(a[p4].tolist(),f)

