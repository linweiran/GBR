import numpy as np
import pickle
for i in range(5):
	np.random.seed(4)
	a=np.random.permutation(60)

	with open("perm"+str(i),'wb') as f:
    		pickle.dump(a,f)


