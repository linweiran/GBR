"""
preprocess the GTSRB data (train/val/test) and pickle the
numpy arrays.

Before running, remember to download the GTSRB dataset
and to place in the directory.
(link: http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads)
"""

from skimage import io, transform
import numpy as np
from os import path
from csv import DictReader
import pickle

# CONSTS
IM_SIZE = 48              # images would be resized to IM_SIZExIM_SIZE
N_CHANNELS = 3            # images are RGB
N_CLASSES = 43            # number of classes
FRAC_VAL = 0.05           # fraction of training data to use for validation
N_TRAIN = 39209           # number of training images (incl. validation)
N_TEST = 12630            # number of test images
SEED = 123                # seed for rand to make output consistent
MAIN_DIR = 'data/GTSRB/'  # main GTSRB dir

def preprocess_im(im, im_sz):
    """
    Preprocess image similarly to:
    Li and Wang, "Real-Time Traffic Sign Recognition Based on Efficient 
    CNNs in the Wild." IEEE Trans. on Intl. Trans. Sys., 2018.
    """
    # center image
    min_side = min(im.shape[:-1])
    centre = im.shape[0]//2, im.shape[1]//2
    im = im[centre[0]-min_side//2:centre[0]+min_side//2,
            centre[1]-min_side//2:centre[1]+min_side//2,
            :]
    
    # resize
    im = transform.resize(im, (im_sz, im_sz))
    
    return im

def read_gtsrb_im(im_path, row):
    """
    Read an image given a the annotation info from the CSV
    """
    im = io.imread(im_path)
    # im = im[ int(row['Roi.Y1']):int(row['Roi.Y2']), \
    #          int(row['Roi.X1']):int(row['Roi.X2']), \
    #          : ]
    return im

def store(np_arr, out_path):
    """
    pickle np_arr and store in in out_path
    """
    fout = open(out_path, mode='wb')
    pickle.dump(np_arr, fout)
    fout.close()

# seed rand
np.random.seed(SEED)

# load train images
x_train = np.zeros((N_TRAIN, IM_SIZE, IM_SIZE, N_CHANNELS))
y_train = np.zeros((N_TRAIN,))
j = 0
for i in range(N_CLASSES):
    if i<10:
        im_dir = path.join(MAIN_DIR, 'Final_Training/Images/0000%d/'%i)
        csv_path = path.join(im_dir, 'GT-0000%d.csv'%i)
    else:
        im_dir = path.join(MAIN_DIR, 'Final_Training/Images/000%d/'%i)
        csv_path = path.join(im_dir, 'GT-000%d.csv'%i)
    reader = DictReader(open(csv_path, mode='r'), delimiter=';')
    for row in reader:
        im_path = path.join(im_dir, row['Filename'])
        im = read_gtsrb_im(im_path, row)
        im = preprocess_im(im, IM_SIZE)
        x_train[j] = im
        y_train[j] = i
        j += 1

# split train to val and train
perm = np.random.permutation(N_TRAIN)
x_train = x_train[perm]
y_train = y_train[perm]
n_val = int(FRAC_VAL*N_TRAIN)
x_val = x_train[:n_val]
y_val = y_train[:n_val]
x_train = x_train[n_val:]
y_train = y_train[n_val:]

# load test images
x_test = np.zeros((N_TEST, IM_SIZE, IM_SIZE, N_CHANNELS))
y_test = np.zeros((N_TEST,))
j = 0
csv_path = path.join(MAIN_DIR, 'GT-final_test.csv')
reader = DictReader(open(csv_path, mode='r'),delimiter=';')
for row in reader:
    im_path = path.join( MAIN_DIR, \
                         'Final_Test/Images/', \
                         row['Filename'] )
    im = read_gtsrb_im(im_path, row)
    im = preprocess_im(im, IM_SIZE)
    x_test[j] = im
    y_test[j] = int(row['ClassId'])
    j += 1

# store
store(x_train, path.join(MAIN_DIR, 'x_train.pkl'))
store(y_train, path.join(MAIN_DIR, 'y_train.pkl'))
store(x_val, path.join(MAIN_DIR, 'x_val.pkl'))
store(y_val, path.join(MAIN_DIR, 'y_val.pkl'))
store(x_test, path.join(MAIN_DIR, 'x_test.pkl'))
store(y_test, path.join(MAIN_DIR, 'y_test.pkl'))
