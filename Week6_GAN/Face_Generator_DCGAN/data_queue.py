"""
    data_queue.py

"""
import tensorflow as tf

def read_and_decode(filename_q):
    with tf.name_scope("read_and_decode"):
        reader = tf.TFRecordReader()
        _, example = reader.read(filename_q)

        features = tf.parse_single_example(example, features={
            'image/data': tf.FixedLenFeature([], tf.string),
        })

        image = tf.image.decode_jpeg(features['image/data'])

        return image

def preprocess(image, width=64, height=64):
    with tf.name_scope("preprocess"):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, [height, width])
        image = tf.reshape(image, [height, width, 3])
        image = image * 2.0 - 1.0
        return image

def make_data_pipeline(shard_filenames, num_epochs, batch_size):
    filename_q = tf.train.string_input_producer(shard_filenames, num_epochs)

    image = read_and_decode(filename_q)
    image = preprocess(image)

    images = tf.train.shuffle_batch([image],
        batch_size, num_threads=1, capacity=10000, allow_smaller_final_batch=False, min_after_dequeue=1000)

    return images
