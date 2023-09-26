import torchvision
import numpy as np
import pickle as pkl
from os import path, listdir
def load_mnist():
    train_dataset=torchvision.datasets.MNIST('.', train=True, download=True)
    trainx=train_dataset.data.numpy()
    trainx=np.reshape(trainx,(60000,28,28,1))
    trainx=np.swapaxes(trainx, 1, 3).astype(float)
    trainx/=255
    trainy=train_dataset.targets.numpy()
    test_dataset=torchvision.datasets.MNIST('.', train=False, download=True)
    testx=test_dataset.data.numpy()
    testy=test_dataset.targets.numpy()
    testx=np.reshape(testx,(10000,28,28,1))
    testx=np.swapaxes(testx, 1, 3).astype(float)
    testx/=255
    return trainx,trainy,testx,testy

def load_gtsrb(data_path='GTSRB/', one_hot=False, shuffle=True):
    """
    load the GTSRB dataset
    """
    np.random.seed(0)
    x_train = pkl.load( open(path.join(data_path, 'x_train.pkl'), mode='rb') )
    y_train = pkl.load( open(path.join(data_path, 'y_train.pkl'), mode='rb') )
    x_test = pkl.load( open(path.join(data_path, 'x_test.pkl'), mode='rb') )
    y_test = pkl.load( open(path.join(data_path, 'y_test.pkl'), mode='rb') )
    x_val = pkl.load( open(path.join(data_path, 'x_val.pkl'), mode='rb') )
    y_val = pkl.load( open(path.join(data_path, 'y_val.pkl'), mode='rb') )
    """
    if one_hot:
        y_train = utils.to_categorical(y_train)
        y_test = utils.to_categorical(y_test)
        y_val = utils.to_categorical(y_val)
    """
    if shuffle:
        perm = np.random.permutation(x_train.shape[0])
        x_train = x_train[perm]
        y_train = y_train[perm]
        perm = np.random.permutation(x_val.shape[0])
        x_val = x_val[perm]
        y_val = y_val[perm]
        perm = np.random.permutation(x_test.shape[0])
        x_test = x_test[perm]
        y_test = y_test[perm]
    return (x_train, y_train), (x_test, y_test), (x_val, y_val)

def load_CelebA(data_path='CelebA/', shuffle=True, partition='test'):
    np.random.seed(0)
    x=np.load(data_path+'x_'+partition+'.npy')
    y=np.load(data_path+'y_'+partition+'.npy')
    if shuffle:
        perm = np.random.permutation(x.shape[0])
        x=x[perm]
        y=y[perm]
    return x,y


def data_select(x_train, y_train, x_test, y_test, x_val, y_val, source_select):
    #source_select is in list format
    #source=np.zeros(43)
    #source[[13,14,15,17]]=1
    #source_select=np.where(source==1)[0].tolist()
    train_select=np.isin(y_train,source_select)
    test_select=np.isin(y_test,source_select)
    val_select=np.isin(y_val,source_select)
    x_train_select=x_train[train_select]
    y_train_select=y_train[train_select]
    x_test_select=x_test[test_select]
    y_test_select=y_test[test_select]
    x_val_select=x_val[val_select]
    y_val_select=y_val[val_select]
    return x_train_select,y_train_select,x_test_select,y_test_select,x_val_select,y_val_select
    

def data_selects(x,y,source_select):
    select=np.isin(y,source_select)
    x=x[select]
    y=y[select]
    return x,y


