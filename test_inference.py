"""Test ImageNet pretrained DenseNet"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import cv2
import numpy as np
from keras.optimizers import SGD
import keras.backend as K

# We only test DenseNet-121 in this script for demo purpose
from densenet169 import DenseNet

im = cv2.resize(cv2.imread('resources/cat.jpg'), (224, 224)).astype(np.float32)
#im = cv2.resize(cv2.imread('resources/shark.jpg'), (224, 224)).astype(np.float32)

# Subtract mean pixel and multiple by scaling constant
# Reference: https://github.com/shicai/DenseNet-Caffe
im[:,:,0] = (im[:,:,0] - 103.94) * 0.017
im[:,:,1] = (im[:,:,1] - 116.78) * 0.017
im[:,:,2] = (im[:,:,2] - 123.68) * 0.017

if K.image_dim_ordering() == 'th':
  # Transpose image dimensions (Theano uses the channels as the 1st dimension)
  im = im.transpose((2,0,1))

  # Use pre-trained weights for Theano backend
  weights_path = 'imagenet_models/densenet169_weights_th.h5'
else:
  # Use pre-trained weights for Tensorflow backend
  weights_path = 'imagenet_models/densenet169_weights_tf.h5'

# Insert a new dimension for the batch_size
im = np.expand_dims(im, axis=0)

# Test pretrained model
model = DenseNet(reduction=0.5, classes=10, weights_path=weights_path)

# Learning rate is changed to 1e-3
sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

out = model.predict(im)

# Load ImageNet classes file
classes = []
with open('resources/classes.txt', 'r') as list_:
    for line in list_:
        classes.append(line.rstrip('\n'))

print 'Prediction: '+str(classes[np.argmax(out)])

from keras.datasets import cifar10
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

img_size = 32
img_chan = 3
n_classes = 10

X_train = np.reshape(X_train, [-1, img_size, img_size, img_chan])
X_train = X_train.astype(np.float32) / 255
X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
X_test = X_test.astype(np.float32) / 255

from keras.utils import to_categorical
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

print('\nSpliting data')

ind = np.random.permutation(X_train.shape[0])
X_train, Y_train = X_train[ind], Y_train[ind]

VALIDATION_SPLIT = 0.1
n = int(X_train.shape[0] * (1-VALIDATION_SPLIT))
X_valid = X_train[n:]
X_train = X_train[:n]
Y_valid = Y_train[n:]
Y_train = Y_train[:n]

X_Train = np.empty([500, 224, 224, 3])

for i1 in range(500):

    X_Train[i1] = cv2.resize(X_train[i1], (224, 224)).astype(np.float32)

    # Subtract mean pixel and multiple by scaling constant
    # Reference: https://github.com/shicai/DenseNet-Caffe
    X_Train[i1][:,:,0] = (X_Train[i1][:,:,0] - 103.94) * 0.017
    X_Train[i1][:,:,1] = (X_Train[i1][:,:,1] - 116.78) * 0.017
    X_Train[i1][:,:,2] = (X_Train[i1][:,:,2] - 123.68) * 0.017

# Insert a new dimension for the batch_size
 X_Train = np.expand_dims(X_Train, axis=0)

X_Valid = np.empty([500, 224, 224, 3])

for i2 in range(500):
    X_Valid[i2] = cv2.resize(X_valid[i2], (224, 224)).astype(np.float32)

    # Subtract mean pixel and multiple by scaling constant
    # Reference: https://github.com/shicai/DenseNet-Caffe
    X_Valid[i2][:, :, 0] = (X_Valid[i2][:, :, 0] - 103.94) * 0.017
    X_Valid[i2][:, :, 1] = (X_Valid[i2][:, :, 1] - 116.78) * 0.017
    X_Valid[i2][:, :, 2] = (X_Valid[i2][:, :, 2] - 123.68) * 0.017

    # Insert a new dimension for the batch_size
    X_Valid[i2] = np.expand_dims(X_Valid[i2], axis=0)

batch_size = 16
epochs = 10

print(X_Train.shape)
print(X_Valid.shape)

print(Y_train[0:500,:].shape)
print(Y_valid[0:500,:].shape)

# Start Fine-tuning
model.fit(X_Train, Y_train[0:500,:],
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True,
          verbose=1,
          validation_data=(X_valid, Y_valid[0:500,:])
)

# Make predictions
predictions_valid = model.predict(X_Valid[0])

# Cross-entropy loss score
score = log_loss(Y_valid[0], predictions_valid)


