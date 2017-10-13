#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function, division, absolute_import

import sys
import random
import numpy as np
import tensorflow as tf


NUM_OUTPUTS         = 48
ID_SIZE             = 8


def new_cell_base(args):
    if args.rnn_type == 'lstm':
        cell = tf.contrib.rnn.BasicLSTMCell(
            args.rnn_size,
            forget_bias=5.0,
            state_is_tuple=True)
        return cell
    if args.rnn_type == 'lnlstm':  # layernormbasiclstmcell
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
            args.rnn_size,
            dropout_keep_prob=args.keep_prob,
            activation=tf.nn.relu,
            norm_gain=0.9,
            norm_shift=0.1,
            forget_bias=5.0
        )
        return cell
    raise Exception("model type not supported: {}".format(args.rnn_type))


def new_cell(args,input_size,infer):
    cell = new_cell_base(args)
    # training mode
    if args.rnn_type != 'lnlstm' and (not infer and args.keep_prob < 1):
        cell = tf.contrib.rnn.DropoutWrapper(
            cell,
            state_keep_prob=args.keep_prob,
            variational_rnn=True,
            input_size=input_size,
            dtype=tf.float32)
    return cell


class Model():
    def __init__(self, args, infer=False):

        device     = args.device
        seq_length = args.seq_length
        keep_prob  = args.keep_prob
        rnn_type   = args.rnn_type
        rnn_size   = args.rnn_size
        num_layers = args.num_layers
        grad_clip  = args.grad_clip
        learning_rate = args.learning_rate

        if infer:
            seq_length = 1
            keep_prob = 1.0



        with tf.device('/cpu:0'):
            global_step        = tf.Variable(0,dtype=tf.int64,trainable=False,name='global_step')
            epoch              = tf.Variable(1,dtype=tf.int64,trainable=False,name='epoch')
            step_update        = tf.placeholder(dtype=tf.int64,shape=None,name='step_update')
            epoch_update       = tf.placeholder(dtype=tf.int64,shape=None,name='epoch_update')
            self.global_step   = global_step
            self.epoch         = epoch
            self.step_update   = step_update
            self.epoch_update  = epoch_update
            with tf.control_dependencies(
                [tf.assign(global_step, step_update),
                 tf.assign(epoch, epoch_update),]):
                self.update_op = tf.constant(1)



        with tf.device(device):
            input_data = tf.placeholder(
                dtype=tf.float32,
                shape=[None, seq_length, NUM_OUTPUTS],
                name='input_data')
            target_data = tf.placeholder(
                dtype=tf.float32,
                shape=[None, seq_length, NUM_OUTPUTS],
                name='target_data')
            seq_length  = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='seq_length')
            motion_id   = tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, ID_SIZE],
                    name='mot_id')
            batch_size = tf.shape(seq_length)[0]

        with tf.device(device):
            cell_list = \
                [new_cell(args, NUM_OUTPUTS, infer)] + \
                [new_cell(args, rnn_size, infer)] * (num_layers-1)
            cell = tf.contrib.rnn.MultiRNNCell(cell_list)

            # initial_state = cell.zero_state(
            #    batch_size=batch_size, dtype=tf.float32)

            motion_state = tf.layers.dense(motion_id,
                                rnn_size * 2 * num_layers)  # ==> (-1, rnn_size * 2 * num_layers)
            initial_state = tuple([
                tf.contrib.rnn.LSTMStateTuple(*tf.split(x, 2, axis=1))
                for x in tf.split(motion_state, num_layers, axis=1)])

            outputs, last_state = tf.nn.dynamic_rnn(
                cell,
                input_data,
                sequence_length=seq_length,
                initial_state = initial_state,
                dtype=tf.float32
            )
            output = tf.layers.dense(outputs, NUM_OUTPUTS)

            # loss A MSE on relative coords
            # loss = tf.losses.mean_squared_error(
            #     target_data[:,:,:-1],
            #     output[:,:,:-1])
            loss = tf.losses.mean_squared_error(
                        target_data,
                        output)

            tvars        = tf.trainable_variables()
            grads, _     = tf.clip_by_global_norm(
                tf.gradients(loss, tvars), grad_clip)
            optimizer    = tf.train.AdamOptimizer(learning_rate)
            train_op     = optimizer.apply_gradients(zip(grads, tvars))

        self.input_data = input_data
        self.target_data = target_data
        self.seq_length = seq_length
        self.motion_id  = motion_id
        self.cell = cell
        self.initial_state = initial_state
        self.last_state = last_state
        self.output = output
        self.loss = loss
        self.train_op = train_op
