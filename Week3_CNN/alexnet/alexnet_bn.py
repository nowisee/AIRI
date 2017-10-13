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
    self.NUM_CLASSES = num_classes
    self.is_training = is_training

    self.create()

  def create(self):
    """Create the network graph.
    We will use tf.layers.conv2d/max_pooling2d, dense, dropout,
    batch_normalization, etc.
    """
    # input image size = 224x224x3

    conv1 = tf.layers.conv2d(self.X, 96, [11, 11], strides=[4, 4], 
	#padding='SAME', activation=tf.nn.relu, use_bias=True, name='conv1',
	padding='SAME', activation=None, use_bias=True, name='conv1',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    conv1 = tf.layers.batch_normalization(conv1, training=self.is_training)
    lrn1 = tf.nn.relu(conv1)

    # 55(56)x55x96
    print_features(conv1)

    #lrn1 = self.lrn(conv1, 5, 2, 1e-4, 0.75, 'lrn1')
    pool1 = tf.layers.max_pooling2d(lrn1, [3, 3], strides=[2, 2], 
	padding='VALID', name='pool1')

    print_features(pool1)

    conv2 = tf.layers.conv2d(pool1, 128*2, [5, 5], padding='SAME', 
	#activation=tf.nn.relu, use_bias=True, name='conv2',
	activation=None, use_bias=True, name='conv2',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        bias_initializer=tf.ones_initializer())
    conv2 = tf.layers.batch_normalization(conv2, training=self.is_training)
    lrn2 = tf.nn.relu(conv2)

    # 27(26)x27x256
    print_features(conv2)

    #lrn2 = self.lrn(conv2, 5, 2, 1e-4, 0.75, 'lrn2')
    pool2 = tf.layers.max_pooling2d(lrn2, [3, 3], strides=[2, 2], 
	padding='VALID', name='pool2')

    print_features(pool2)

    conv3 = tf.layers.conv2d(pool2, 192*2, [3, 3], padding='SAME',
	#activation=tf.nn.relu, use_bias=True, name='conv3',
	activation=None, use_bias=True, name='conv3',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    conv3 = tf.layers.batch_normalization(conv3, training=self.is_training)
    conv3 = tf.nn.relu(conv3)
    
    # 13x13x384
    print_features(conv3)

    conv4 = tf.layers.conv2d(conv3, 192*2, [3, 3], padding='SAME',
	#activation=tf.nn.relu, use_bias=True, name='conv4',
	activation=None, use_bias=True, name='conv4',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        bias_initializer=tf.ones_initializer())
    conv4 = tf.layers.batch_normalization(conv4, training=self.is_training)
    conv4 = tf.nn.relu(conv4)

    # 13x13x384
    print_features(conv4)

    conv5 = tf.layers.conv2d(conv4, 128*2, [3, 3], padding='SAME',
	#activation=tf.nn.relu, use_bias=True, name='conv5',
	activation=None, use_bias=True, name='conv5',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        bias_initializer=tf.ones_initializer())
    conv5 = tf.layers.batch_normalization(conv5, training=self.is_training)
    conv5 = tf.nn.relu(conv5)

    print_features(conv5)

    pool5 = tf.layers.max_pooling2d(conv5, [3, 3], strides=[2, 2], 
	padding='VALID', name='pool5')

    # 6x6x256
    print_features(pool5)

    flattened = tf.reshape(pool5, [-1,6*6*256])
    #fc6 = tf.layers.dense(flattened, 4096, activation=tf.nn.relu,
    fc6 = tf.layers.dense(flattened, 512, activation=tf.nn.relu,
	use_bias=True, name='fc6',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        bias_initializer=tf.ones_initializer())
    #drop6 = tf.layers.dropout(fc6, rate=0.5, training=self.is_training, name='drop6')

    #print_features(drop6)

    #fc7 = tf.layers.dense(drop6, 4096, activation=tf.nn.relu,
    #fc7 = tf.layers.dense(drop6, 128, activation=tf.nn.relu,
    fc7 = tf.layers.dense(fc6, 128, activation=tf.nn.relu,
	use_bias=True, name='fc7',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        bias_initializer=tf.ones_initializer())
    #drop7 = tf.layers.dropout(fc7, rate=0.5, training=self.is_training, name='drop7')

    #print_features(drop7)

    #self.logits = tf.layers.dense(drop7, self.NUM_CLASSES, activation=None,
    self.logits = tf.layers.dense(fc7, self.NUM_CLASSES, activation=None,
	use_bias=True, name='logits',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        bias_initializer=tf.ones_initializer())

    print_features(self.logits)

  #def load_weights(self):
    
  def lrn(self, x, radius, bias, alpha, beta, name):
    """Create a local response normalization layer."""
    with tf.name_scope('LRN'):
      return tf.nn.local_response_normalization(x, depth_radius=radius,
                                                bias=bias, alpha=alpha,
                                                beta=beta, name=name)
