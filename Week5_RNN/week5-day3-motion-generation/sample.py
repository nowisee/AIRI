#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import argparse
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from model import Model, NUM_OUTPUTS, ID_SIZE

# main code (not in a main function since I want to run this script in IPython as well).

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--motion_id_list', type=str, required=True)
    parser.add_argument('--sample_length', type=int, default=1000)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--device', default='/cpu:0')
    sample_args = parser.parse_args()
    return sample_args

def sample(model, sess, motion_id, num):
    """
    motion_id: one-hot encoded motion_id
    """
    #prev_state   = sess.run(model.cell.zero_state(1, tf.float32))
    prev_state   = sess.run(model.initial_state,{model.motion_id:[motion_id]})
    prev_vec     = np.zeros((1, 1, NUM_OUTPUTS), dtype=np.float32)
    strokes      = np.zeros((num, NUM_OUTPUTS), dtype=np.float32)
    output       = np.zeros(NUM_OUTPUTS, dtype=np.float32)

    for i in tqdm(range(num)):
        feed                = {
            model.input_data:    prev_vec,
            model.initial_state: prev_state,
            model.seq_length:    [1],
        }
        o_rest, next_state  = sess.run(
            [model.output, model.last_state],
            feed)
        strokes[i, :]       = o_rest[0]
        prev_vec[0, 0, :]   = o_rest[0, 0, :]
        prev_state          = next_state

    return strokes


if __name__ == '__main__':
    sample_args = parse_args()

    output_file      = sample_args.output_file
    sample_length    = sample_args.sample_length
    save_dir         = sample_args.save_dir
    checkpoint       = sample_args.checkpoint

    try: os.makedirs(os.path.dirname(output_file))
    except: pass

    config_file = save_dir + '/' + 'config.pkl'
    with open(config_file, 'rb') as f:
        saved_args = pickle.load(f)

    # override --device
    saved_args.device = sample_args.device

    # workaround
    try: saved_args.rnn_type = saved_args.model
    except: pass

    model = Model(saved_args, True)

    tf_config = tf.ConfigProto(allow_soft_placement=True,
                               gpu_options={'allow_growth': True})
    sess = tf.InteractiveSession(config=tf_config)
    saver = tf.train.Saver()
    if not checkpoint: checkpoint = tf.train.latest_checkpoint(save_dir)
    print("loading model: ", checkpoint, file=sys.stderr)

    saver.restore(sess, checkpoint)

    motion_id = np.zeros([ID_SIZE],dtype=np.float32)
    args_motion_id = np.array([float(x) for x in sample_args.motion_id_list.split(',')])
    motion_id[:len(args_motion_id)] = args_motion_id[:]

    data = sample(model, sess, motion_id, sample_length)
    np.savetxt(output_file, data, fmt='%f', delimiter=',')

