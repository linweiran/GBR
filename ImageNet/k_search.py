import numpy as np
a=np.zeros((4,3))

a[0]=1
a[:,0]=1

print (a)



def k_search_helper(a,i,t,l):
    
    if len(set(l)) != len(l):
        return False
    if t==0:
        return True
    if t>a.shape[0]-i:
        return False
    if k_search_helper(a,i+1,t,l):
        return True
    js=np.where(a[i]>0)[0].tolist()
    for j in js:
        if k_search_helper(a,i+1,t-1,l+[j]):
            return True
    return False

def k_search(a,t):
    if (a.shape[0]>a.shape[1]):
        a=np.swapaxes(a,0,1)
    return k_search_helper(a,0,t,[])

print (k_search(a,0))
print (k_search(a,1))
print (k_search(a,2))
print (k_search(a,3))

