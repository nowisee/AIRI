from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import time

import numpy as np
import tensorflow as tf

# original inception module
def inception_block_1(net, is_training=False):
  """35x35 resnet block"""
  with tf.variable_scope("branch_0"):
    br0 = tf.layers.conv2d(net, 24, [1, 1], padding='SAME',
                           activation=None,
                           use_bias=True, name='conv_1x1',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    br0 = tf.layers.batch_normalization(br0, training=is_training)
    br0 = tf.nn.relu(br0)
  with tf.variable_scope("branch_1"):
    br1 = tf.layers.conv2d(net, 24, [1, 1], padding='SAME',
                           activation=None,
                           use_bias=True, name='conv_1x1',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    br1 = tf.nn.relu(tf.layers.batch_normalization(br1))
    br1 = tf.layers.conv2d(br1, 24, [5, 5], padding='SAME',
                           activation=None,
                           use_bias=True, name='conv_3x3',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    br1 = tf.layers.batch_normalization(br1, training=is_training)
    br1 = tf.nn.relu(br1)
  with tf.variable_scope("branch_2"):
    br2 = tf.layers.conv2d(net, 24, [1, 1], padding='SAME',
                           activation=None,
                           use_bias=True, name='conv_1x1',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    br2 = tf.nn.relu(tf.layers.batch_normalization(br2))
    br2 = tf.layers.conv2d(br2, 24, [5, 5], padding='SAME',
                           activation=None,
                           use_bias=True, name='conv_3x3',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    br2 = tf.layers.batch_normalization(br2, training=is_training)
    br2 = tf.nn.relu(br2)
  with tf.variable_scope("branch_3"):
    # BN?
    br3 = tf.layers.max_pooling2d(net, [3, 3], [1, 1], padding='SAME',
                                  name='pool_3x3')
    br3 = tf.layers.conv2d(br3, 24, [1, 1], padding='SAME',
                           activation=None,
                           use_bias=True, name='conv_1x1',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    br3 = tf.layers.batch_normalization(br3, training=is_training)
    br3 = tf.nn.relu(br3)

  concatenated = tf.concat([br0, br1, br2, br3], 3)
  return concatenated

# original inception module
def inception_block_2(net, is_training=False):
  """35x35 resnet block"""
  with tf.variable_scope("branch2_0"):
    br20 = tf.layers.conv2d(net, 96, [1, 1], padding='SAME',
                           activation=None,
                           use_bias=True, name='conv_1x1',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    br20 = tf.layers.batch_normalization(br20, training=is_training)
    br20 = tf.nn.relu(br20)
  with tf.variable_scope("branch2_1"):
    br21 = tf.layers.conv2d(net, 96, [1, 1], padding='SAME',
                           activation=None,
                           use_bias=True, name='conv_1x1',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    br21 = tf.nn.relu(tf.layers.batch_normalization(br21))
    br21 = tf.layers.conv2d(br21, 96, [3, 3], padding='SAME',
                           activation=None,
                           use_bias=True, name='conv_3x3',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    br21 = tf.layers.batch_normalization(br21, training=is_training)
    br21 = tf.nn.relu(br21)
  with tf.variable_scope("branch2_2"):
    br22 = tf.layers.conv2d(net, 96, [1, 1], padding='SAME',
                           activation=None,
                           use_bias=True, name='conv_1x1',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    br22 = tf.nn.relu(tf.layers.batch_normalization(br22))
    br22 = tf.layers.conv2d(br22, 96, [5, 5], padding='SAME',
                           activation=None,
                           use_bias=True, name='conv_3x3',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    br22 = tf.layers.batch_normalization(br22, training=is_training)
    br22 = tf.nn.relu(br22)
  with tf.variable_scope("branch2_3"):
    # BN?
    br23 = tf.layers.max_pooling2d(net, [3, 3], [1, 1], padding='SAME',
                                  name='pool_3x3')
    br23 = tf.layers.conv2d(br23, 96, [1, 1], padding='SAME',
                           activation=None,
                           use_bias=True, name='conv_1x1',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    br23 = tf.layers.batch_normalization(br23, training=is_training)
    br23 = tf.nn.relu(br23)

  concatenated = tf.concat([br20, br21, br22, br23], 3)
  return concatenated

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

    conv1 = tf.layers.conv2d(self.X, 96, [7,7],
                             strides=[2,2],
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

    conv2 = tf.layers.conv2d(conv1, 96, [3,3],
                             strides=[2,2],
                             padding='SAME',
                             activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             bias_initializer=tf.ones_initializer(),#bias_initializer=tf.zeros_initializer(),
                             name='conv2')

    conv2 = tf.layers.batch_normalization(conv2, training=self.is_training)
    conv2 = tf.nn.relu(conv2, name='relu2')
    conv2 = tf.layers.max_pooling2d(conv2, [3,3],
                                    strides=[2,2],
                                    padding='VALID',
                                    name='pool2')

    print_features(conv2)

    icpt1 = inception_block_1(conv2, self.is_training)
    print_features(icpt1)

    icpt2 = inception_block_2(icpt1, self.is_training)
    print_features(icpt2)

    print(icpt2.get_shape()[1:3])
    ave_pool = tf.layers.average_pooling2d(icpt2, icpt2.get_shape()[1:3], strides=[1,1], padding='VALID', name='pool3')
    print_features(ave_pool)

    ave_pool_reshape = tf.reshape(ave_pool, [-1, 1*1*384])
    print(ave_pool_reshape.get_shape())

    ###################################
    # 6-1. fc6
    #
    # 6-2. drop6
    # tensor shape: 512
    ###################################

    fc = tf.layers.dense(ave_pool_reshape, 10,
                          activation=None,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                          bias_initializer=tf.ones_initializer(),#bias_initializer=tf.constant_initializer(0.5),
                          name='fc')

    self.logits = fc
    print(self.logits)
    #sys.exit()