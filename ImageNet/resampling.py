import numpy as np
np.random.seed(0)
a=np.random.choice(1000,60,replace=False).tolist()
print (a)
from robustness.datasets import ImageNet
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--data_path',type=str, default='/data/ILSVRC')

args = parser.parse_args()
print("ARGS: ", args)
MAIN_DIR=args.data_path

ds=ImageNet(MAIN_DIR)
_, test_loader = ds.make_loaders(workers=0, batch_size=50,only_val=True)
x=[]
y=[]
c=0
for im, label in iter(test_loader):
    img=im.numpy()
    labeling=label.numpy()
    for i in range(50):
        if labeling[i] in a:
            c+=1
            x.append(img[i])
            y.append(labeling[i])
    print (c)
x=np.array(x)
y=np.array(y)
total=x.shape[0]
print (total)
per=np.random.permutation(total)
x=x[per]
y=y[per]
import pickle
with open('X','wb') as f:
    pickle.dump(x,f)
with open('Y','wb') as f:
    pickle.dump(y,f)
with open('A','wb') as f:
    pickle.dump(a,f)

