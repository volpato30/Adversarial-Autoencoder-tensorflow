from __future__ import absolute_import, division, print_function

import math
import os

import numpy as np
import scipy.misc
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from aae import AAE

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("updates_per_epoch", 300, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 5000, "max epoch")
flags.DEFINE_float("learning_rate", 1e-1, "learning rate")
flags.DEFINE_string("working_directory", "./", "wd")
flags.DEFINE_string("log_directory", "./log", "wd")
flags.DEFINE_integer("hidden_size", 2, "size of the hidden unit")

FLAGS = flags.FLAGS

if __name__ == "__main__":
    data_directory = os.path.join(FLAGS.working_directory, "MNIST")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    mnist = input_data.read_data_sets(data_directory, one_hot=False)
    model = AAE(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate,
            FLAGS.log_directory)
    for epoch in range(FLAGS.max_epoch):
        recons_loss_train, classify_loss_train = 0., 0.

        for i in range(FLAGS.updates_per_epoch):
            images, _ = mnist.train.next_batch(FLAGS.batch_size)
            recons_loss_value, classify_loss_value, summary = \
                    model.update_params(images)
            recons_loss_train += recons_loss_value
            classify_loss_train += classify_loss_value
        # write summary
        if epoch % 10 == 9:
            model.train_writer.add_summary(summary, epoch)
        recons_loss_train = recons_loss_train / \
            (FLAGS.updates_per_epoch * FLAGS.batch_size)
        classify_loss_train = classify_loss_train / \
            (FLAGS.updates_per_epoch * FLAGS.batch_size)
        global_step = tf.contrib.framework.get_or_create_global_step()

        print("step: {},\tlearn rate: {:6f},\treconstruction loss {:.6f},\t\
            class loss: {:.6f}".format(global_step.eval(model.sess),
            model.learn_rate.eval(session=model.sess), recons_loss_train,
            classify_loss_train))

        model.generate_and_save_images(FLAGS.working_directory)

        if epoch % 100 == 99:
            NUM_POINTS = 50000
            imgs = mnist.train.images[:NUM_POINTS,]
            labels = np.asarray(mnist.train.labels[:NUM_POINTS], np.int32)
            X_encode = np.zeros((NUM_POINTS, FLAGS.hidden_size), dtype=np.float32)
            for i in range(NUM_POINTS//500):
                encoded_feature = model.encode_features(imgs[i*500:(i+1)*500,:])
                X_encode[i*500:(i+1)*500,:] = encoded_feature
            if FLAGS.hidden_size > 2:
                print('Use first 2 pcs for visualization')
                X_train_embedded = PCA().fit_transform(X_encode)
                plt.figure(figsize=(20,10))
                plt.scatter(X_train_embedded[:, 0], X_train_embedded[:, 1], c=labels,
                    marker='.', cmap=cm.Vega10)
                plt.savefig('latent_space.png')
            elif FLAGS.hidden_size == 2:
                plt.figure(figsize=(20,10))
                plt.scatter(X_encode[:, 0], X_encode[:, 1], c=labels,
                    marker='.', cmap=cm.Vega10)
                plt.savefig('latent_space.png')
