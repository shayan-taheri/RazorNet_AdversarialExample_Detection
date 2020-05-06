"""
Code to visualize noise of all adversarial algorithm.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np

import matplotlib
matplotlib.use('Agg')           # noqa: E402
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf

from ConvX import MaxMinconv2d, MaxMinConvolution2D, Convolution2D, conv2d

from attacks import fgm, jsma, deepfool

from ConvX import activations

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

img_size = 28
img_chan = 1
n_classes = 10

print('\nLoading MNIST')

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.reshape(X_train, [-1, img_size, img_size, img_chan])
X_train = X_train.astype(np.float32) / 255
X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
X_test = X_test.astype(np.float32) / 255

to_categorical = tf.keras.utils.to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('\nSpliting data')

ind = np.random.permutation(X_train.shape[0])
X_train, y_train = X_train[ind], y_train[ind]

VALIDATION_SPLIT = 0.1
n = int(X_train.shape[0] * (1-VALIDATION_SPLIT))
X_valid = X_train[n:]
X_train = X_train[:n]
y_valid = y_train[n:]
y_train = y_train[:n]

print('\nConstruction graph')

def model(x, logits=False, training=False):

    with tf.variable_scope('conv0'):

        z = MaxMinconv2d(x, nb_filter=32, nb_row=3, nb_col=3, activation=tf.nn.relu)

    with tf.variable_scope('conv1'):

        z = MaxMinconv2d(x, nb_filter=32, nb_row=3, nb_col=3, activation=tf.nn.relu)

    z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv2'):

        z = MaxMinconv2d(x, nb_filter=64, nb_row=3, nb_col=3, activation=tf.nn.relu)

    with tf.variable_scope('conv3'):

        z = MaxMinconv2d(x, nb_filter=64, nb_row=3, nb_col=3, activation=tf.nn.relu)

    z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])

    with tf.variable_scope("fc1"):
        z = tf.contrib.layers.fully_connected(z, num_outputs=200, activation_fn=tf.nn.relu)

    with tf.variable_scope("fc2"):
        z = tf.contrib.layers.fully_connected(z, num_outputs=200, activation_fn=tf.nn.relu)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


class Dummy:
    pass


env = Dummy()


with tf.variable_scope('model'):
    env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x')
    env.y = tf.placeholder(tf.float32, (None, n_classes), name='y')
    env.training = tf.placeholder_with_default(False, (), name='mode')

    env.ybar, logits = model(env.x, logits=True, training=env.training)

    with tf.variable_scope('acc'):
        count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
        env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

    with tf.variable_scope('loss'):
        xent1 = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                               logits=logits)

        xent2 = tf.losses.hinge_loss(labels=env.y,
                                     logits=logits)

        env.loss = tf.reduce_sum(xent1 + xent2, name='loss')

    with tf.variable_scope('train_op'):
        optimizer = tf.train.AdamOptimizer()
        env.train_op = optimizer.minimize(env.loss)

    env.saver = tf.train.Saver()

with tf.variable_scope('model', reuse=True):
    env.adv_eps = tf.placeholder(tf.float32, (), name='adv_eps')
    env.adv_D = tf.placeholder(tf.float32, (), name='adv_D')
    env.adv_epochs = tf.placeholder(tf.int32, (), name='adv_epochs')
    env.adv_y = tf.placeholder(tf.int32, (), name='adv_y')

    env.x_fgsm = fgm(model, env.x, epochs=env.adv_epochs, eps=env.adv_eps)
    env.x_deepfool = deepfool(model, env.x, epochs=env.adv_epochs, batch=True, noise=True, D=env.adv_D)
    env.x_jsma = jsma(model, env.x, env.adv_y, eps=env.adv_eps,
                      epochs=env.adv_epochs)


print('\nInitializing graph')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

def evaluate(sess, env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
    #    print(' batch {0}/{1}'.format(batch + 1, n_batch))
    #    print('\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


def train(sess, env, X_data, y_data, X_valid=None, y_valid=None, epochs=1,
          load=False, shuffle=True, batch_size=128, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        if not hasattr(env, 'saver'):
            print('\nError: cannot find saver op')
            return
        print('\nLoading saved model')
        return env.saver.restore(sess, 'model/{}'.format(name))

    print('\nTrain model')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))

        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
        #    print(' batch {0}/{1}'.format(batch + 1, n_batch))
        #    print('\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                              env.y: y_data[start:end],
                                              env.training: True})
        if X_valid is not None:
            evaluate(sess, env, X_valid, y_valid)

    if hasattr(env, 'saver'):
        print('\n Saving model')
        env.saver.save(sess, 'model/{}'.format(name))


def predict(sess, env, X_data, batch_size=128):
    """
    Do inference by running env.ybar.
    """
    print('\nPredicting')
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
    #    print(' batch {0}/{1}'.format(batch + 1, n_batch))
    #    print('\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval


def make_fgsm(sess, env, X_data, epochs=1, eps=0.01, batch_size=128):
    print('\nMaking adversarials via FGSM')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch))
        print('\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        feed_dict = {env.x: X_data[start:end], env.adv_eps: eps,
                     env.adv_epochs: epochs}
        adv = sess.run(env.x_fgsm, feed_dict=feed_dict)
        X_adv[start:end] = adv
    print()

    return X_adv


def make_jsma(sess, env, X_data, epochs=0.2, eps=1.0, batch_size=128):
    print('\nMaking adversarials via JSMA')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch))
        print('\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        feed_dict = {
            env.x: X_data[start:end],
            env.adv_y: np.random.choice(n_classes),
            env.adv_epochs: epochs,
            env.adv_eps: eps}
        adv = sess.run(env.x_jsma, feed_dict=feed_dict)
        X_adv[start:end] = adv
    print()

    return X_adv


def make_deepfool(sess, env, X_data, epochs=1, batch_size=1, noise=True, D=0.1, batch=True):
    print('\nMaking adversarials via DeepFool')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch))
        print('\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        feed_dict = {env.x: X_data[start:end], env.adv_epochs: epochs, env.adv_D: D}
        adv = sess.run(env.x_deepfool, feed_dict=feed_dict)
        X_adv[start:end] = adv
    print()

    return X_adv


print('\nTraining')

train(sess, env, X_train, y_train, X_valid, y_valid, load=False, epochs=1,
      name='mnist')

print('\nEvaluating on clean data')
evaluate(sess, env, X_test, y_test)

X_adv_fgsm = X_test
X_adv_jsma = X_test
X_adv_deepfool = X_test

for i in range(len(X_test)):

    xorg, y0 = X_test[i], y_test[i]

    xorg = np.expand_dims(xorg, axis=0)

    xadvs = [make_fgsm(sess, env, xorg, eps=0, epochs=1),
    make_jsma(sess, env, xorg, eps=0, epochs=1),
    make_deepfool(sess, env, xorg, D=0.1, noise=True, epochs=1, batch=True)]

    print('\nEvaluating on Single FGSM adversarial data')

    evaluate(sess, env, xadvs[0], y_test)

    print('\nEvaluating on Single JSMA adversarial data')

    evaluate(sess, env, xadvs[1], y_test)

    print('\nEvaluating on Single DeepFool adversarial data')

    evaluate(sess, env, xadvs[2], y_test)

    X_adv_fgsm[i] = xadvs[0]
    X_adv_jsma[i] = xadvs[1]
    X_adv_deepfool[i] = xadvs[2]

    xorg = np.squeeze(xorg)
  #  xadvs = [xorg] + xadvs
    xadvs = [np.squeeze(e) for e in xadvs]

    print('\nSaving figure')

    fig = plt.figure()
    plt.imshow(xorg)
    plt.savefig('/home/USER_NAME/PycharmProjects/attack_compact/mnist/original_mnist_' + str(i) + '.png')
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(xadvs[0])
    plt.savefig('/home/USER_NAME/PycharmProjects/attack_compact/mnist/xadvs_fgsm_mnist_' + str(i) + '.png')
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(xadvs[1])
    plt.savefig('/home/USER_NAME/PycharmProjects/attack_compact/mnist/xadvs_jsma_mnist_' + str(i) + '.png')
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(xadvs[2])
    plt.savefig('/home/USER_NAME/PycharmProjects/attack_compact/mnist/xadvs_deepfool_mnist_' + str(i) + '.png')
    plt.close(fig)

print('\nEvaluating on FGSM adversarial data')

evaluate(sess, env, X_adv_fgsm, y_test)

print('\nEvaluating on JSMA adversarial data')

evaluate(sess, env, X_adv_jsma, y_test)

print('\nEvaluating on DeepFool adversarial data')

evaluate(sess, env, X_adv_deepfool, y_test)