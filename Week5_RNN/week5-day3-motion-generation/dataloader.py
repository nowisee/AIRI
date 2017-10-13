# coding: utf-8
from __future__ import print_function
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import os
import sys

from os.path import join
import pickle
import numpy as np
import random


class DataLoader():
    def __init__(self, data_dir='.', augment_data=1):
        self.data = []
        self.data_id = []

        logger.info(('checking:',data_dir))
        for dirname, _, filelist in os.walk(data_dir):
            for filename in filelist:
                filepath = join(dirname,filename)
                if filename.endswith('.csv'):
                    logger.info(('loadtxt',filepath))

                    data = np.loadtxt(filepath, delimiter=',')
                    data_id = int(filename[:-4])

                    if augment_data > 1:
                        # mirror augment * xN augment
                        data = np.vstack([data, data[::-1]] * augment_data)
                        logger.info(('augment > 1','shape',data.shape))

                    self.data.append(data)
                    self.data_id.append(data_id)

        self.data_len  = [len(d) for d in self.data]
        self.data_prob = np.array(self.data_len, dtype=np.float32) / \
            np.sum(self.data_len, dtype=np.float32)


    def batch_data(self, batch_size, seq_length):

        data_batches = [
            len(d) - (seq_length + 1) + 1
            for d in self.data]
        num_batches = np.sum(data_batches, dtype=np.int32)

        logger.info((
            'num_batches:',num_batches,
            'batch_size:',batch_size,
            'seq_length:',seq_length))

        def next_batch():
            for _ in range(num_batches):
                x_batch, y_batch, id_batch = \
                    [], [], []
                    
                for _ in range(batch_size):
                    data_choice = np.random.choice(range(len(self.data)),p=self.data_prob)
                    data        = self.data[data_choice]
                    index       = np.random.randint(0,len(data)-(seq_length+1))
                    x           = data[index:index+seq_length,:]
                    y           = data[index+1:index+1+seq_length,:]
                    x_batch.append(x)
                    y_batch.append(y)

                    data_id     = self.data_id[data_choice]
                    id_batch.append(data_id)
                    
                yield x_batch, y_batch, id_batch

        return num_batches, next_batch()

