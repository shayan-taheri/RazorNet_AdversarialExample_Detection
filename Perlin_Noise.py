"""
Code to visualize noise of all adversarial algorithm.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
import numpy as np
from scipy.misc import toimage, imsave

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

shape = (32, 32)
scale = 10.0
octaves = 10
persistence = .5
lacunarity = 2.5


print('\nLoading CIFAR')

from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

### ***************** LOADING DATASETS *******************

for ix in range(0,len(X_train)):

    im = X_train[ix]
    im = cv2.resize(im, (32, 32))

    noisy_data = np.zeros(im.shape)

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            noisy_data[i][j] = im[i][j] + noise.pnoise2(i / scale,
                                                        j / scale,
                                                        octaves=octaves,
                                                        persistence=persistence,
                                                        lacunarity=lacunarity,
                                                        repeatx=32,
                                                        repeaty=32,
                                                        base=0)

    imsave('/home/shayan/Perlin_Dataset/Cifar_' + str(ix) + '.png', noisy_data)
