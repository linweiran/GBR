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


args = parser.parse_args()
print("ARGS: ", args)
distance=args.distance
if distance != "l2":
        distance="linf"
with open("A",'rb') as f:
    a=pickle.load(f)
a=np.array(a)
with open("Y",'rb') as f:
    y=pickle.load(f)
with open("X",'rb') as f:
    x=pickle.load(f)

with open(distance,'rb') as f:
    succ=pickle.load(f)


print (y.shape)
print (a.shape)

y_test=y[:1000]
y_valid=y[1000:]
succ_test=succ[:,:1000]
succ_valid=succ[:,1000:]




matrix_test=np.zeros((60,60))
matrix_valid=np.zeros((60,60))
total_test=np.zeros((60,60))
total_valid=np.zeros((60,60))
for i in range(1000):
    for j in range(60):
        total_test[j][np.where(a==y_test[i])[0][0]]+=1
        matrix_test[j][np.where(a==y_test[i])[0][0]]+=succ_test[j][i]
for i in range(2000):
    for j in range(60):

        total_valid[j][np.where(a==y_valid[i])[0][0]]+=1
        matrix_valid[j][np.where(a==y_valid[i])[0][0]]+=succ_valid[j][i]

for i in range(60):
    matrix_test[i][i]=0
    matrix_valid[i][i]=0



print (np.sum(matrix_valid))
print (np.sum(matrix_test))

matrix_test/=total_test
matrix_valid/=total_valid

with open('matrix'+distance+'_test','wb') as f:
    pickle.dump(matrix_test,f)
with open('matrix'+distance+'_valid','wb') as f:
    pickle.dump(matrix_valid,f)
