#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import gensim
from text_cnn import TextCNN
from tensorflow.contrib import learn
from tensorflow.python.client import device_lib

# Data Preparation
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{

def merge(ob1, ob2):
    """
    an object's __dict__ contains all its
    attributes, methods, docstrings, etc.
    """
    ob1.__dict__.update(ob2.__dict__)
    return ob1

normal_path = '/home/shayan/Datasets/HTTPs/Normal_Train'
anomalous_path = '/home/shayan/Datasets/HTTPs/Anomalous_Train'

normal_dirs = os.listdir(normal_path)
anomalous_dirs = os.listdir(anomalous_path)

sort_normal_dirs = sorted(normal_dirs)
sort_anomalous_dirs = sorted(anomalous_dirs)

ix = 0

x_train_size = []
y_train_size = []

x_dev_size = []
y_dev_size = []

while (ix < len(sort_normal_dirs)):

    normal_training_file = str(normal_path + '/' + sort_normal_dirs[ix])

    anomalous_training_file = str(anomalous_path + '/' + sort_anomalous_dirs[ix])

    # Load data
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(normal_training_file, anomalous_training_file)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vars()['vocab_processor', str(ix)] = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vars()['vocab_processor', str(ix)].fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))

    vars()['x_train', str(ix)] = x_shuffled[:dev_sample_index]
    vars()['y_train', str(ix)] = y_shuffled[:dev_sample_index]

    vars()['x_dev', str(ix)] = x_shuffled[dev_sample_index:]
    vars()['y_dev', str(ix)] = y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vars()['vocab_processor', str(ix)].vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(vars()['y_train', str(ix)]), len(vars()['y_dev', str(ix)])))

    x_train_size.append(np.size(vars()['x_train', str(ix)], 1))
    y_train_size.append(np.size(vars()['y_train', str(ix)], 1))

    x_dev_size.append(np.size(vars()['x_train', str(ix)], 1))
    y_dev_size.append(np.size(vars()['y_train', str(ix)], 1))

    if (ix == 0):
        vocab_processor_out = vars()['vocab_processor', str(ix)]

    if (ix >= 1):

        vocab_processor_out = merge(vocab_processor_out, vars()['vocab_processor', str(ix)])

    ix = ix + 1

vars()['x_train_update', str(0)] = np.ndarray(shape=(np.size(vars()['x_train', str(0)], 0), max(x_train_size)), dtype=int)
x_train_all = vars()['x_train_update', str(0)]
iy = 1
while (iy < len(sort_normal_dirs)):

    vars()['x_train_update', str(iy)] = np.ndarray(shape=(np.size(vars()['x_train', str(iy)], 0), max(x_train_size)), dtype=int)
    for iz in range(np.size(vars()['x_train', str(iy)],0)):
        vars()['x_train_update', str(iy)][iz] = np.append(vars()['x_train', str(iy)][iz], np.zeros((1, abs(max(x_train_size) - np.size(vars()['x_train', str(iy)], 1))), dtype=int))
    x_train_all = np.vstack((x_train_all, vars()['x_train_update', str(iy)]))
    iy = iy + 1

vars()['y_train_update', str(0)] = np.ndarray(shape=(np.size(vars()['y_train', str(0)], 0), max(y_train_size)), dtype=int)
y_train_all = vars()['y_train_update', str(0)]
ir = 1
while (ir < len(sort_normal_dirs)):
    vars()['y_train_update', str(ir)] = np.ndarray(shape=(np.size(vars()['y_train', str(ir)], 0), max(y_train_size)), dtype=int)
    for iz in range(np.size(vars()['y_train', str(ir)],0)):
        vars()['y_train_update', str(ir)][iz] = np.append(vars()['y_train', str(ir)][iz], np.zeros((1, abs(max(y_train_size) - np.size(vars()['y_train', str(ir)], 1))), dtype=int))
    y_train_all = np.vstack((y_train_all, vars()['y_train_update', str(ir)]))
    ir = ir + 1

vars()['x_dev_update', str(0)] = np.ndarray(shape=(np.size(vars()['x_dev', str(0)], 0), max(x_dev_size)), dtype=int)
x_dev_all = vars()['x_dev_update', str(0)]
it = 1
while (it < len(sort_normal_dirs)):
    vars()['x_dev_update', str(it)] = np.ndarray(shape=(np.size(vars()['x_dev', str(it)], 0), max(x_dev_size)), dtype=int)
    for iz in range(np.size(vars()['x_dev', str(it)],0)):
        vars()['x_dev_update', str(it)][iz] = np.append(vars()['x_dev', str(it)][iz], np.zeros((1, abs(max(x_dev_size) - np.size(vars()['x_dev', str(it)], 1))), dtype=int))
    x_dev_all = np.vstack((x_dev_all, vars()['x_dev_update', str(it)]))
    it = it + 1

vars()['y_dev_update', str(0)] = np.ndarray(shape=(np.size(vars()['y_dev', str(0)], 0), max(y_dev_size)), dtype=int)
y_dev_all = vars()['y_dev_update', str(0)]
ih = 1
while (ih < len(sort_normal_dirs)):
    vars()['y_dev_update', str(ih)] = np.ndarray(shape=(np.size(vars()['y_dev', str(ih)], 0), max(y_dev_size)), dtype=int)
    for iz in range(np.size(vars()['y_dev', str(ih)],0)):
        vars()['y_dev_update', str(ih)][iz] = np.append(vars()['y_dev', str(ih)][iz], np.zeros((1, abs(max(y_dev_size) - np.size(vars()['y_dev', str(ih)], 1))), dtype=int))
    y_dev_all = np.vstack((y_dev_all, vars()['y_dev_update', str(ih)]))
    ih = ih + 1


print(x_train_all)
print(y_train_all)
print(x_dev_all)
print(y_dev_all)
print(vocab_processor_out)
print(type(vocab_processor_out))

