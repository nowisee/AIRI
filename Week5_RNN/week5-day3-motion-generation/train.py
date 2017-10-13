#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function, division, absolute_import

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import pickle
import time

import numpy as np
import tensorflow as tf

from dataloader import DataLoader
from model import Model, ID_SIZE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--rnn_type', required=True, help='rnn, gru, lstm, or lnlstm')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--seq_length', type=int, default=100) # 300)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--grad_clip', type=float, default=10.)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--keep_prob', type=float, default=1.0)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--augment_data', type=int, default=1)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--device', default='/gpu:0')
    args = parser.parse_args()
    return args


def one_hot(arr,num_classes):
    """
    arr = 1-dim array of int (N >= 1)
    """
    arr_1hot = np.zeros([len(arr),num_classes],dtype=np.float32)
    arr_1hot[np.arange(len(arr)),arr] = 1.0
    return arr_1hot


def train(args):

    batch_size = args.batch_size
    seq_length = args.seq_length
    num_epochs = args.num_epochs
    save_every = args.save_every
    save_dir = args.save_dir
    data_dir = args.data_dir
    augment_data = args.augment_data
    checkpoint = args.checkpoint

    try: os.makedirs(data_dir)
    except: pass

    data_loader = DataLoader(data_dir=data_dir,
                             augment_data=augment_data)

    model = Model(args)

    writer = tf.summary.FileWriter(
        save_dir,
        graph=tf.get_default_graph())

    config = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options={'allow_growth': True})

    with tf.Session(config=config) as sess:

        saver = tf.train.Saver()
        if not checkpoint:
            checkpoint = tf.train.latest_checkpoint(save_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print(('=== graph restored ===', checkpoint), file=sys.stderr)
        else:
            tf.global_variables_initializer().run()

        start, global_step = sess.run([model.epoch,model.global_step])

        for e in range(start, num_epochs + 1):

            num_batches, train_batch = data_loader.batch_data(batch_size, seq_length)

            for x, y, ids in train_batch:

                t_start = time.time()

                feed = {
                    model.input_data:   x,
                    model.target_data:  y,
                    model.motion_id:    one_hot(ids,ID_SIZE),
                    model.seq_length:   [len(t) for t in x],
                }
                _, loss, = sess.run(
                    [
                        model.train_op,
                        model.loss,
                    ],
                    feed)

                global_step += 1

                if global_step % 200 == 0:
                    summary = tf.Summary(
                        value=[
                            tf.Summary.Value(
                                tag='day3/loss',
                                simple_value=loss
                            ),
                        ]
                    )
                    writer.add_summary(summary, global_step=global_step)

                    t_elapsed = time.time() - t_start
                    print(
                        "epoch {}, step {}, loss = {:.5f}, elapsed = {:.3f}"
                        .format(e,
                                global_step,
                                loss,
                                t_elapsed))
                    t_start = time.time()

                if global_step % save_every == 0 and (global_step > 0):
                    checkpoint_path = save_dir + '/' + 'model.ckpt'
                    cp = saver.save(sess, checkpoint_path, global_step=global_step)
                    with open(save_dir + '/' + 'config.pkl', 'wb') as f:
                        pickle.dump(args, f, protocol=2)
                    print("model saved to {}".format(cp))

                sess.run(   [model.update_op],
                            {model.epoch_update:e,
                             model.step_update:global_step})

        checkpoint_path = save_dir + '/' + 'model.ckpt'
        cp = saver.save(sess, checkpoint_path, global_step=global_step)
        with open(save_dir + '/' + 'config.pkl', 'wb') as f:
            pickle.dump(args, f, protocol=2)
        print("model saved to {}".format(cp))


if __name__ == '__main__':
    args = parse_args()
    train(args)
