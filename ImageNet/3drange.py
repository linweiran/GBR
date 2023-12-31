import numpy as np
import pickle
group={}
best={}
average={}
worst={}
maxs={}
import argparse
parser = argparse.ArgumentParser()


parser.add_argument('--distance',type=str, default='l2')

args = parser.parse_args()
print("ARGS: ", args)
distance=args.distance


for k in range(10,60,10):
    for i in range(10,70-k,10):
        glist=np.zeros(5)
        mlist=np.zeros(5)
        alist=np.zeros(5)
        wlist=np.zeros(5)
        blist=np.zeros(5)

        for j in range(5):
            st=str(j)
            Gname="GROUP"+st+distance
            Mname="MAX"+st+distance

            gu="guess"+st+distance
            with open(gu,'rb') as f:
                data=pickle.load(f)
                (alist[j],wlist[j],blist[j])=data[(k,i)]
            with open(Gname,'rb') as f:
                data=pickle.load(f)
                glist[j]=np.mean(data[(k,i)])
            with open(Mname,'rb') as f:
                data=pickle.load(f)
                mlist[j]=np.mean(data[(k,i)])
        group[(k,i)]=( glist.mean(), glist.max(), glist.min())
        best[(k,i)]=( blist.mean(), blist.max(),blist.min())
        average[(k,i)]=( alist.mean(), alist.max(),alist.min())
        worst[(k,i)]= ( wlist.mean(), wlist.max(),wlist.min())
        maxs[(k,i)]= ( mlist.mean(), mlist.max(),mlist.min())

with open('3drange'+distance,'wb') as f:
    pickle.dump((group,best,average,worst,maxs),f)



