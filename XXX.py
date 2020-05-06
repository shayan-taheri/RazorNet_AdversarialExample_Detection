"""Test ImageNet pretrained DenseNet"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import cv2
import numpy as np
import keras
from keras.optimizers import SGD
import keras.backend as K
import glob
import matplotlib.pyplot as plt
import sklearn.metrics as sklm
import itertools
from keras.utils import Sequence
import time

# We only test DenseNet-121 in this script for demo purpose
from densenet169 import DenseNet

classes=2

# Use pre-trained weights for Tensorflow backend
weights_path = '/home/shayan/Adversarial_Examples/imagenet_models/densenet169_weights_tf.h5'

print('Start Reading Data')

# "0" = Botnet Traffic Data and "1" = Normal Traffic Data
X_train = []
Y_train = []

ix = 0
for filename in glob.glob('/home/shayan/Drug_Images/Drug_Original_Train/' + '*.jpg'):
    if ix <= 7291:
        im=cv2.imread(filename)
        X_train.append(im)
        Y_train.append([0])
        ix = ix + 1

ix = 0
for filename in glob.glob('/home/shayan/Drug_Images/Drug_Attacked_Train/' + '*.png'):
    if ix <= 36455:
        im=cv2.imread(filename)
        X_train.append(im)
        Y_train.append([1])
        ix = ix + 1

ix = 0
for filename in glob.glob('/home/shayan/Drug_Images/Drug_Original_Test/' + '*.jpg'):
    if ix <= 800:
        im=cv2.imread(filename)
        X_train.append(im)
        Y_train.append([0])
        ix = ix + 1

ix = 0
for filename in glob.glob('/home/shayan/Drug_Images/Drug_Attacked_Test/' + '*.png'):
    if ix <= 4000:
        im=cv2.imread(filename)
        X_train.append(im)
        Y_train.append([1])
        ix = ix + 1

print('Finish Reading Data')

print('\nSpliting data')

vect = range(0, len(X_train))

from sklearn.utils import shuffle
vect = shuffle(vect, random_state=0)

X_train_BAK = X_train
Y_train_BAK = Y_train

for iy in range(0,len(vect)):
    iz = vect[iy]
    X_train[iy] = X_train_BAK[iz]
    Y_train[iy] = Y_train_BAK[iz]

VALIDATION_SPLIT = 0.1
n = int(len(X_train) * (1-VALIDATION_SPLIT))
X_valid = X_train[n:]
X_train = X_train[:n]
Y_valid = Y_train[n:]
Y_train = Y_train[:n]

VALIDATION_SPLIT = 0.2
n = int(len(X_train) * (1-VALIDATION_SPLIT))
X_test = X_train[n:]
X_train = X_train[:n]
Y_test = Y_train[n:]
Y_train = Y_train[:n]

ix = 0
for filename in glob.glob('/home/shayan/Drug_Images/Drug_Original_Train/' + '*.jpg'):
    if ix <= 7291:
        im=cv2.imread(filename)
        X_valid.append(im)
        Y_valid.append([0])
        ix = ix + 1

ix = 0
for filename in glob.glob('/home/shayan/Drug_Images/Drug_Attacked_Train/' + '*.png'):
    if ix <= 36455:
        im=cv2.imread(filename)
        X_valid.append(im)
        Y_valid.append([1])
        ix = ix + 1

ix = 0
for filename in glob.glob('/home/shayan/Drug_Images/Drug_Original_Test/' + '*.jpg'):
    if ix <= 800:
        im=cv2.imread(filename)
        X_valid.append(im)
        Y_valid.append([0])
        ix = ix + 1

ix = 0
for filename in glob.glob('/home/shayan/Drug_Images/Drug_Attacked_Test/' + '*.png'):
    if ix <= 4000:
        im=cv2.imread(filename)
        X_valid.append(im)
        Y_valid.append([1])
        ix = ix + 1

iuv = 0/home/shayan/Codes/Adversarial
for filename in glob.glob('/home/shayan/Drug_Images/Drug_Original_Test/' + '*.jpg'):
    if iuv <= 800:
        im = cv2.imread(filename)
        X_test.append(im)
        Y_test.append([1])
        iuv = iuv + 1

iuv = 0
for filename in glob.glob('/home/shayan/Drug_Images/Drug_Attacked_Test/' + '*.png'):
    if iuv <= 4000:
        im = cv2.imread(filename)
        X_test.append(im)
        Y_test.append([1])
        iuv = iuv + 1

from keras.utils import to_categorical

Y_train = np.array(Y_train)
Y_train = to_categorical(Y_train)

Y_valid = np.array(Y_valid)
Y_valid = to_categorical(Y_valid)

********************************************************************************** HIDDEN


    confusion.append(sklm.confusion_matrix(targ.flatten(), predict.flatten()))

    f.write(str(['Area Under ROC Curve (AUC): ', auc]))
    f.write('\n')
    f.write('Confusion: ')
    f.write('\n')
    f.write(str(np.array(confusion)))
    f.write('\n')
    f.write(str(['Precision: ', precision]))
    f.write('\n')
    f.write(str(['Recall: ', recall]))
    f.write('\n')
    f.write(str(['F-1 Score: ', f1s]))
    f.write('\n')
    f.write(str(['Kappa: ', kappa]))
    f.write('\n')
    f.write(str(['Training Time: ', train_time]))
    f.write('\n')
    f.write(str(['Testing Time: ', test_time]))
    f.close()
