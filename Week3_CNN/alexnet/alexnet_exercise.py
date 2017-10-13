from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import time

import numpy as np
import tensorflow as tf

def print_features(t):
  print(t.op.name, ' ', t.get_shape().as_list())

class AlexNet(object):

  def __init__(self, x, num_classes, is_training):
    self.X = x
    self.is_training = is_training
    self.NUM_CLASSES = num_classes

    self.create()

  def create(self):
    """Create the network graph.
    We will use tf.layers.conv2d/max_pooling2d/dense/dropout, tf.nn.lrn, etc.
    """
    # input image size = 224x224x3

    ##################################
    # 1-1. conv1 = tf.layers.conv2d...
    # print_features(conv1)
    # tensor shape: 55(56)x55x96
    #
    # 1-2. Batch Normalization
    #
    # 1-3. Activation
    #
    # 1-4. pool1
    # tensor shape: 27(26)x27x256
    ###################################

    #print(self.X.shape)

    conv1 = tf.layers.conv2d(self.X, 96, [11,11],
                             strides=[4,4],
                             padding='SAME',
                             activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             bias_initializer=tf.zeros_initializer(),
                             name='conv1')

    conv1 = tf.layers.batch_normalization(conv1, training=self.is_training)
    conv1 = tf.nn.relu(conv1, name='relu1')
    conv1 = tf.layers.max_pooling2d(conv1, [3,3],
                                    strides=[2,2],
                                    padding='VALID',
                                    name='pool1')

    print_features(conv1)
    #sys.exit()

    ###################################
    # 2-1. conv2
    # tensor shape: 27(26)x27x256
    #
    # 2-2. Batch Normalization
    #
    # 2-3. Activation
    #
    # 2-4. pool2 
    # tensor shape: 13x13x256
    ###################################

    conv2 = tf.layers.conv2d(conv1, 256, [5, 5],
                             strides=[1, 1],
                             padding='SAME',
                             activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             bias_initializer=tf.ones_initializer(),#bias_initializer=tf.zeros_initializer(),
                             name='conv2')

    conv2 = tf.layers.batch_normalization(conv2, training=self.is_training)
    conv2 = tf.nn.relu(conv2, name='relu2')
    conv2 = tf.layers.max_pooling2d(conv2, [3, 3],
                                    strides=[2, 2],
                                    padding='VALID',
                                    name='pool2')

    print_features(conv2)
    #sys.exit()

    ###################################
    # 3. conv3 (with BN, Activation)
    # tensor shape: 13x13x256
    ###################################

    conv3 = tf.layers.conv2d(conv2, 384, [3, 3],
                             strides=[1, 1],
                             padding='SAME',
                             activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             bias_initializer=tf.zeros_initializer(),
                             name='conv3')

    conv3 = tf.layers.batch_normalization(conv3, training=self.is_training)
    conv3 = tf.nn.relu(conv3, name='relu3')

    print_features(conv3)
    #sys.exit()

    ###################################
    # 4. conv4 (with BN, Activation)
    # tensor shape: 13x13x256
    ###################################

    conv4 = tf.layers.conv2d(conv3, 384, [3, 3],
                             strides=[1, 1],
                             padding='SAME',
                             activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             bias_initializer=tf.ones_initializer(),#bias_initializer=tf.zeros_initializer(),
                             name='conv4')

    conv4 = tf.layers.batch_normalization(conv4, training=self.is_training)
    conv4 = tf.nn.relu(conv4, name='relu4')

    print_features(conv4)
    #sys.exit()

    ###################################
    # 5-1. conv5 (with BN, Activation)
    # tensor shape: 13x13x256
    #
    # 5-2. pool5
    # tensor shape: 6x6x256
    #
    # 5-3. reshape
    # tensor shape: 9216
    ###################################

    conv5 = tf.layers.conv2d(conv4, 256, [3, 3],
                             strides=[1, 1],
                             padding='SAME',
                             activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             bias_initializer=tf.ones_initializer(),#bias_initializer=tf.zeros_initializer(),
                             name='conv5')

    conv5 = tf.layers.batch_normalization(conv5, training=self.is_training)
    conv5 = tf.nn.relu(conv5, name='relu5')
    conv5 = tf.layers.max_pooling2d(conv5, [3, 3],
                                    strides=[2, 2],
                                    padding='VALID',
                                    name='pool5')
    conv5 = tf.reshape(conv5, shape=[-1,6*6*256], name='reshape')

    print_features(conv5)
    #sys.exit()

    ###################################
    # 6-1. fc6
    #
    # 6-2. drop6
    # tensor shape: 512
    ###################################

    fc6 = tf.layers.dense(conv5, 512,
                          activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                          bias_initializer=tf.ones_initializer(),#bias_initializer=tf.constant_initializer(0.5),
                          name='fc6')

    fc6 = tf.layers.dropout(fc6, rate=0.5, training=self.is_training, name='dropout6')

    print(fc6)
    #sys.exit()

    ###################################
    # 7-1. fc7
    #
    # 7-2. drop7
    # tensor shape: 128
    ###################################

    fc7 = tf.layers.dense(fc6, 128,
                          activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                          bias_initializer=tf.ones_initializer(),#bias_initializer=tf.constant_initializer(0.5),
                          name='fc7')

    #fc7 = tf.layers.dropout(fc7, rate=0.5, training=self.is_training, name='dropout7')

    print(fc7)
    #sys.exit()

    ###################################
    # 8. self.logits(fc8)
    # tensor shape: 10
    ###################################

    fc8 = tf.layers.dense(fc7, 10,
                          activation=None,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                          bias_initializer=tf.ones_initializer(),#bias_initializer=tf.constant_initializer(0.5),
                          name='fc8')

    #fc8 = tf.layers.dropout(fc8, rate=0.5, training=self.is_training, name='dropout8')

    self.logits = fc8
    print(self.logits)
    #sys.exit()