'''TensorFlow implementation of http://arxiv.org/pdf/1511.06434.pdf'''

from __future__ import absolute_import, division, print_function

import math
import os

import numpy as np
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope
import tensorflow as tf
from scipy.misc import imsave

from utils import discriminator, decoder, encoder


def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0,
        stddev=std, dtype=tf.float32)
    return input_layer + noise

class AAE(object):

    def __init__(self, hidden_size, batch_size, learning_rate, log_dir):
        self.input_tensor = tf.placeholder(tf.float32, [None, 28 * 28])
        # add gaussian noise to the input
        input_with_noise = gaussian_noise_layer(self.input_tensor, 0.3)

        with arg_scope([layers.fully_connected], activation_fn=tf.nn.relu):
            with tf.variable_scope("encoder"):
                self.latent_representation = encoder(input_with_noise,
                        hidden_size)
                encoder_params_num = len(tf.trainable_variables())
            with tf.variable_scope('encoder', reuse=True):
                self.true_latent_representation = encoder(self.input_tensor,
                        hidden_size)
            with tf.variable_scope('decoder'):
                self.recons = decoder(self.latent_representation)
                autoencoder_params_num = len(tf.trainable_variables())
            with tf.variable_scope('decoder', reuse=True):
                self.sampled_imgs = decoder(tf.random_normal([batch_size,
                        hidden_size]))

            pos_samples = tf.random_normal([batch_size, hidden_size],
                stddev=5.)
            neg_samples = self.latent_representation
            with tf.variable_scope('discriminator'):
                pos_samples_pred = discriminator(pos_samples)
            with tf.variable_scope('discriminator', reuse=True):
                neg_samples_pred = discriminator(neg_samples)
            #define losses
            reconstruction_loss = tf.reduce_mean(tf.square(self.recons -
                    self.input_tensor)) #* 28 * 28 scale recons loss
            classification_loss = tf.losses.sigmoid_cross_entropy(\
                    tf.ones(tf.shape(pos_samples_pred)), pos_samples_pred) +\
                    tf.losses.sigmoid_cross_entropy(tf.zeros(
                    tf.shape(neg_samples_pred)), neg_samples_pred)
            tf.summary.scalar('reconstruction_loss', reconstruction_loss)
            tf.summary.scalar('classification_loss', classification_loss)
            # define references to params
            params = tf.trainable_variables()
            encoder_params = params[:encoder_params_num]
            decoder_params = params[encoder_params_num:autoencoder_params_num]
            autoencoder_params = encoder_params + decoder_params
            discriminator_params = params[autoencoder_params_num:]
            # record true positive rate and true negative rate
            correct_pred_pos = tf.equal(tf.cast(pos_samples_pred>0, tf.float32),
                tf.ones(tf.shape(pos_samples_pred)))
            self.true_pos_rate = tf.reduce_mean(tf.cast(correct_pred_pos,
                tf.float32))
            correct_pred_neg = tf.equal(tf.cast(neg_samples_pred<0, tf.float32),
                tf.ones(tf.shape(pos_samples_pred)))
            self.true_neg_rate = tf.reduce_mean(tf.cast(correct_pred_neg,
                tf.float32))
            tf.summary.scalar('true_pos_rate', self.true_pos_rate)
            tf.summary.scalar('true_neg_rate', self.true_neg_rate)
            global_step = tf.contrib.framework.get_or_create_global_step()
            self.learn_rate = self._get_learn_rate(global_step, learning_rate)
            self.train_autoencoder = layers.optimize_loss(reconstruction_loss,
                    global_step, self.learn_rate/10, optimizer=lambda lr: \
                    tf.train.MomentumOptimizer(lr, momentum=0.9), variables=
                    autoencoder_params, update_ops=[])
            self.train_discriminator = layers.optimize_loss(classification_loss,
                    global_step, self.learn_rate, optimizer=lambda lr: \
                    tf.train.MomentumOptimizer(lr, momentum=0.1), variables=
                    discriminator_params, update_ops=[])
            self.train_encoder = layers.optimize_loss(-classification_loss,
                    global_step, self.learn_rate/10, optimizer=lambda lr: \
                    tf.train.MomentumOptimizer(lr, momentum=0.1), variables=
                    encoder_params, update_ops=[])
            self.sess = tf.Session()
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(log_dir,
                                      self.sess.graph)
            self.sess.run(tf.global_variables_initializer())
    #use the learn rate schedule introduced by the original paper
    def _get_learn_rate(self, global_step, learning_rate):
        boundaries = [np.int64(50*900), np.int64(1000*900)]
        values = [learning_rate, learning_rate/10, learning_rate/100]
        return tf.train.piecewise_constant(global_step, boundaries, values)

    def update_params(self, inputs):
        recons_loss_value = self.sess.run(self.train_autoencoder,
                {self.input_tensor: inputs})
        classify_loss_value = self.sess.run(self.train_discriminator,
                {self.input_tensor: inputs})
        summary, _ = self.sess.run([self.merged, self.train_encoder], {self.input_tensor: inputs})
        return recons_loss_value, classify_loss_value, summary

    def encode_features(self, inputs):
        return self.sess.run(self.true_latent_representation,
            {self.input_tensor: inputs})

    def generate_and_save_images(self, directory):
        '''Generates the images using the model and saves them in the directory

        Args:
            num_samples: number of samples to generate
            directory: a directory to save the images
        '''
        imgs = self.sess.run(self.sampled_imgs)
        for k in range(imgs.shape[0]):
            imgs_folder = os.path.join(directory, 'imgs')
            if not os.path.exists(imgs_folder):
                os.makedirs(imgs_folder)

            imsave(os.path.join(imgs_folder, '%d.png') % k,
                   imgs[k].reshape(28, 28))
