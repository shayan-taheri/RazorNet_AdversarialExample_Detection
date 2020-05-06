#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import time
import datetime
import data_helpers
import gensim
from text_cnn import TextCNN
from tensorflow.contrib import learn
from tensorflow.python.client import device_lib
import os


# Parameters
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
tf.flags.DEFINE_integer("batch_size", 2, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", True, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

def merge(ob1, ob2):
    """
    an object's __dict__ contains all its
    attributes, methods, docstrings, etc.
    """
    ob1.__dict__.update(ob2.__dict__)
    return ob1

def preprocess():

    # Data Preparation
    # ==================================================

    normal_path = '/home/shayan/HTTPs/Normal_Train'
    anomalous_path = '/home/shayan/HTTPs/Anomalous_Train'

    normal_dirs = os.listdir(normal_path)
    anomalous_dirs = os.listdir(anomalous_path)

    sort_normal_dirs = sorted(normal_dirs)
    sort_anomalous_dirs = sorted(anomalous_dirs)
    
#    files_numbers = len(sort_normal_dirs)

    files_numbers = 1000


    ix = 0
    x_train_size = []
    y_train_size = []
    x_dev_size = []
    y_dev_size = []

    while (ix < files_numbers):

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

    vars()['x_train_update', str(0)] = np.ndarray(shape=(np.size(vars()['x_train', str(0)], 0), max(x_train_size)),
                                                  dtype=int)
    x_train_all = vars()['x_train_update', str(0)]
    iy = 1
    while (iy < files_numbers):
        vars()['x_train_update', str(iy)] = np.ndarray(
            shape=(np.size(vars()['x_train', str(iy)], 0), max(x_train_size)), dtype=int)
        for iz in range(np.size(vars()['x_train', str(iy)], 0)):
            vars()['x_train_update', str(iy)][iz] = np.append(vars()['x_train', str(iy)][iz], np.zeros(
                (1, abs(max(x_train_size) - np.size(vars()['x_train', str(iy)], 1))), dtype=int))
        x_train_all = np.vstack((x_train_all, vars()['x_train_update', str(iy)]))
        iy = iy + 1
    vars()['y_train_update', str(0)] = np.ndarray(shape=(np.size(vars()['y_train', str(0)], 0), max(y_train_size)),
                                                  dtype=int)

    y_train_all = vars()['y_train_update', str(0)]
    ir = 1
    while (ir < files_numbers):
        vars()['y_train_update', str(ir)] = np.ndarray(
            shape=(np.size(vars()['y_train', str(ir)], 0), max(y_train_size)), dtype=int)
        for iz in range(np.size(vars()['y_train', str(ir)], 0)):
            vars()['y_train_update', str(ir)][iz] = np.append(vars()['y_train', str(ir)][iz], np.zeros(
                (1, abs(max(y_train_size) - np.size(vars()['y_train', str(ir)], 1))), dtype=int))
        y_train_all = np.vstack((y_train_all, vars()['y_train_update', str(ir)]))
        ir = ir + 1
    vars()['x_dev_update', str(0)] = np.ndarray(shape=(np.size(vars()['x_dev', str(0)], 0), max(x_dev_size)), dtype=int)

    x_dev_all = vars()['x_dev_update', str(0)]
    it = 1
    while (it < files_numbers):
        vars()['x_dev_update', str(it)] = np.ndarray(shape=(np.size(vars()['x_dev', str(it)], 0), max(x_dev_size)),
                                                     dtype=int)
        for iz in range(np.size(vars()['x_dev', str(it)], 0)):
            vars()['x_dev_update', str(it)][iz] = np.append(vars()['x_dev', str(it)][iz], np.zeros(
                (1, abs(max(x_dev_size) - np.size(vars()['x_dev', str(it)], 1))), dtype=int))
        x_dev_all = np.vstack((x_dev_all, vars()['x_dev_update', str(it)]))
        it = it + 1
    vars()['y_dev_update', str(0)] = np.ndarray(shape=(np.size(vars()['y_dev', str(0)], 0), max(y_dev_size)), dtype=int)

    y_dev_all = vars()['y_dev_update', str(0)]
    ih = 1
    while (ih < files_numbers):
        vars()['y_dev_update', str(ih)] = np.ndarray(shape=(np.size(vars()['y_dev', str(ih)], 0), max(y_dev_size)),
                                                     dtype=int)
        for iz in range(np.size(vars()['y_dev', str(ih)], 0)):
            vars()['y_dev_update', str(ih)][iz] = np.append(vars()['y_dev', str(ih)][iz], np.zeros(
                (1, abs(max(y_dev_size) - np.size(vars()['y_dev', str(ih)], 1))), dtype=int))
        y_dev_all = np.vstack((y_dev_all, vars()['y_dev_update', str(ih)]))
        ih = ih + 1

    return x_train_all, y_train_all, vocab_processor_out, x_dev_all, y_dev_all


def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    # Training
    # ==================================================
    with tf.Graph().as_default():

        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
        )
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        #out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Development (Validation) summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Specify the path to the checkpoint file
        checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.chk")

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("Training Process: {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print(
                "Validation/Development Process: {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss,
                                                                                          accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

    return


def main(argv=None):

    x_train_all, y_train_all, vocab_processor_out, x_dev_all, y_dev_all = preprocess()

    print(x_train_all)
    print(len(x_train_all))
    print(y_train_all)
    print(x_dev_all)
    print(y_dev_all)
    print(vocab_processor_out)
    print(type(vocab_processor_out))

    train(x_train_all, y_train_all, vocab_processor_out, x_dev_all, y_dev_all)

    print(device_lib.list_local_devices())


if __name__ == '__main__':
    tf.app.run()
