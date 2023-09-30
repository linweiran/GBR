import numpy as np
import pickle
np.random.seed(4)
a=np.random.permutation(60)
print (a)
with open("perm4",'wb') as f:
    pickle.dump(a,f)


