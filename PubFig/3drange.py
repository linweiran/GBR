import numpy as np
import pickle
group={}
best={}
average={}
worst={}
maxs={}
maxl=[]
groupl=[]
averagel=[]
bestl=[]


for k in range(10,60,10):
    for i in range(10,70-k,10):
        glist=np.zeros(5)
        mlist=np.zeros(5)
        alist=np.zeros(5)
        wlist=np.zeros(5)
        blist=np.zeros(5)

        for j in range(5):
            if j==0:
                st=""
            else:
                st=str(j)
            Gname=st+"GROUP-log"
            Mname=st+"MAX"

            gu="guess"+st
            with open(gu,'rb') as f:
                data=pickle.load(f)
                (alist[j],wlist[j],blist[j])=data[(k,i)]
            with open(Gname,'rb') as f:
                data=pickle.load(f)
                glist[j]=np.mean(data[(k,i)])
            with open(Mname,'rb') as f:
                data=pickle.load(f)
                mlist[j]=np.mean(data[(k,i)])
        #group[(k,i)]=( glist.mean(), glist.max(), glist.min())
        #best[(k,i)]=( blist.mean(), blist.max(),blist.min())
        #average[(k,i)]=( alist.mean(), alist.max(),alist.min())
        #worst[(k,i)]= ( wlist.mean(), wlist.max(),wlist.min())
        #maxs[(k,i)]= ( mlist.mean(), mlist.max(),mlist.min())

        maxl+=mlist.tolist()
        groupl+=glist.tolist()
        averagel+=alist.tolist()
        bestl+=blist.tolist()


print ((np.array(maxl)/np.array(bestl)).min(),(np.array(maxl)/np.array(bestl)).max())
print ((np.array(maxl)/np.array(averagel)).min(),(np.array(maxl)/np.array(averagel)).max())
print ((np.array(groupl)/np.array(bestl)).min(),(np.array(groupl)/np.array(bestl)).max())
print ((np.array(groupl)/np.array(averagel)).min(),(np.array(groupl)/np.array(averagel)).max())



#with open('3drangeface','wb') as f:
#    pickle.dump((group,best,average,worst,maxs),f)



