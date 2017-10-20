"""
    make_tfrec.py

        2017.09.15
"""
import os
import sys
import math
import random
import numpy as np
import tensorflow as tf

def get_tfrecord_filename(name, dataset_dir, split_name, shard_id, num_shards):
    output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (name, split_name, shard_id, num_shards)
    return os.path.join(dataset_dir, output_filename)

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(image_data):
    return tf.train.Example(features=tf.train.Features(
            feature= {
                'image/data': bytes_feature(image_data),
            }
    ))

def convert_dataset(name, dataset_dir, split_name, filenames, num_shards):
    num_per_shard = int(math.ceil(len(filenames)/float(num_shards)))
    print("file len:", len(filenames), "num per shard:", num_per_shard)

    for shard_id in range(num_shards):
        output_filepath = get_tfrecord_filename(name, dataset_dir, split_name, shard_id, num_shards)

        with tf.python_io.TFRecordWriter(output_filepath) as tfrec_writer:
            start = shard_id * num_per_shard
            end = min((shard_id + 1) * num_per_shard, len(filenames))
            for i in range(start, end):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1, len(filenames), shard_id))
                sys.stdout.flush()
                image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                example = image_to_tfexample(image_data)
                tfrec_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()

def collect_filenames(path, filters=[]):
    filenames = []
    count = 0
    for root, dirs, files in os.walk(path):
        for name in files:
            fname = os.path.join(root, name)
            if len(filters) > 0:
                for s in filters:
                    if fname.find(s) >= 0:
                        filenames.append(fname)
                        count += 1
                        break
    print("%d files collected" % len(filenames))
    return np.array(filenames)

if __name__ == "__main__":
    filenames = collect_filenames('./MsCeleb', ['.jpg'])

    random.shuffle(filenames)

    filenames = filenames[:350000]

    convert_dataset('celeb', './tfrec', 'train', filenames, 5)
