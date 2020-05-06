"""
Code to visualize noise of all adversarial algorithm.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import numpy as np
import keras
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from timeit import default_timer
import tensorflow as tf
import cv2
import glob
import noise
from scipy.misc import imsave

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print('\nLoading CIFAR')

from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = np.concatenate((X_train,X_test))

print(X_train.shape)

for ix in range(0,len(X_train)):

    im = X_train[ix]
    im = cv2.resize(im, (32, 32))

    imsave('/home/USER_NAME/Original_Dataset/' + str(ix) + '.png', im)
