import torchvision
import numpy as np
import pickle as pkl
from os import path, listdir

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




