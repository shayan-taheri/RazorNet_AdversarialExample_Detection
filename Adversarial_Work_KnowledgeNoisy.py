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

classes=3

# Use pre-trained weights for Tensorflow backend
weights_path = '/home/shayan/Adversarial_Examples/imagenet_models/densenet169_weights_tf.h5'

print('Start Reading Data')

********************************************************************************** HIDDEN

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

iuv = 0
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

icv = 0
for filename in glob.glob('/home/shayan/Noisy_Bangla_AWGN/*.png'):
    if icv <= 197889:
        im = cv2.imread(filename)
        X_train.append(im)
        Y_train.append([2])
        icv = icv + 1

ity = 0
for filename in glob.glob('/home/shayan/Noisy_Bangla_AWGN/*.png'):
    if ity <= 197889:
        im = cv2.imread(filename)
        X_valid.append(im)
        Y_valid.append([2])
        ity = ity + 1

icv = 0
for filename in glob.glob('/home/shayan/Noisy_Bangla_Contrast/*.png'):
    if icv <= 197889:
        im = cv2.imread(filename)
        X_train.append(im)
        Y_train.append([2])
        icv = icv + 1

ity = 0
for filename in glob.glob('/home/shayan/Noisy_Bangla_Contrast/*.png'):
    if ity <= 197889:
        im = cv2.imread(filename)
        X_valid.append(im)
        Y_valid.append([2])
        ity = ity + 1

icv = 0
for filename in glob.glob('/home/shayan/Noisy_Bangla_MotionBlur/*.png'):
    if icv <= 197889:
        im = cv2.imread(filename)
        X_train.append(im)
        Y_train.append([2])
        icv = icv + 1

ity = 0
for filename in glob.glob('/home/shayan/Noisy_Bangla_MotionBlur/*.png'):
    if ity <= 197889:
        im = cv2.imread(filename)
        X_valid.append(im)
        Y_valid.append([2])
        ity = ity + 1

icv = 0
for filename in glob.glob('/home/shayan/Perlin_Dataset/*.png'):
    if icv <= 100000:
        im = cv2.imread(filename)
        X_train.append(im)
        Y_train.append([2])
        icv = icv + 1

ity = 0
for filename in glob.glob('/home/shayan/Perlin_Dataset/*.png'):
    if ity <= 100000:
        im = cv2.imread(filename)
        X_valid.append(im)
        Y_valid.append([2])
        ity = ity + 1

from keras.utils import to_categorical

Y_train = np.array(Y_train)
Y_train = to_categorical(Y_train)

Y_valid = np.array(Y_valid)
Y_valid = to_categorical(Y_valid)

Y_test = np.array(Y_test)
Y_test = to_categorical(Y_test)

class MY_Generator(Sequence):

    def __init__(self, image_data, labels, batch_size):
        self.image_data, self.labels = image_data, labels
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(len(self.image_data) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.image_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        arr_batch_x = np.array([cv2.resize(ity, (224, 224)).astype(np.float32) for ity in batch_x])

        for ix in range(0, arr_batch_x.shape[0]):
            im = arr_batch_x[ix]
            im[:, :, 0] = (im[:, :, 0] - 103.94) * 0.017
            im[:, :, 1] = (im[:, :, 1] - 116.78) * 0.017
            im[:, :, 2] = (im[:, :, 2] - 123.68) * 0.017
            arr_batch_x[ix] = im

        return arr_batch_x, batch_y

batch_size = 1

my_training_batch_generator = MY_Generator(X_train, Y_train, batch_size)
my_validation_batch_generator = MY_Generator(X_valid, Y_valid, batch_size)

del X_train
del Y_train
del X_valid
del Y_valid

NumNonTrainable = [0,10,75,200,525]

X_testNP = np.zeros([len(X_test),224,224,3])

for id in range(0,len(X_testNP)):
    im = cv2.resize(X_test[id], (224, 224))
    X_testNP[id] = im

for ib in range(0,len(NumNonTrainable)):

    # Test pretrained model
    model = DenseNet(reduction=0.5, classes=classes, weights_path=weights_path, NumNonTrainable=NumNonTrainable[ib])

    # Learning rate is changed to 1e-3
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    start = time.time()
    model.fit_generator(generator=my_training_batch_generator,
                          epochs=1,
                          verbose=1,
                          shuffle=True,
                          validation_data=my_validation_batch_generator)

    end = time.time()

    train_time = end - start

    start = time.time()

    score = model.evaluate(X_testNP, Y_test, verbose=0)

    print(score[0])
    print(score[1])

    end = time.time()

    test_time = end - start

    f = open("/home/shayan/Codes/DenseNet-Keras-master/Stat_KnowledgeNoisy_" + str(ib) + ".txt", "w")

    f.write(str(['Test loss: ', score[0]]))
    f.write('\n')

    f.write(str(['Test accuracy: ', score[1]]))
    f.write('\n')

    confusion = []
    precision = []
    recall = []
    f1s = []
    kappa = []
    auc = []
    roc = []

    scores = np.array([np.argmax(t) for t in np.asarray(model.predict(X_testNP))])
    predict = np.array([np.argmax(t) for t in np.round(np.asarray(model.predict(X_testNP)))])
    targ = np.array([np.argmax(t) for t in Y_test])

    confusion.append(sklm.confusion_matrix(targ.flatten(), predict.flatten()))
    precision.append(sklm.precision_score(targ.flatten(), predict.flatten(), average = 'macro'))
    recall.append(sklm.recall_score(targ.flatten(), predict.flatten(), average = 'macro'))
    f1s.append(sklm.f1_score(targ.flatten(), predict.flatten(), average = 'macro'))
    kappa.append(sklm.cohen_kappa_score(targ.flatten(), predict.flatten()))

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
