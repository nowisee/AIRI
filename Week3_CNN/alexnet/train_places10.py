import os
import sys
import numpy as np
import tensorflow as tf

from alexnet_bn import AlexNet
from tensorflow.python.lib.io import tf_record

import time
import h5py
import random

# train, val data file name
FILE_PREFIX = '/data1/UbuntuData/dataset/Places10/'
TRAIN_FILENAME = FILE_PREFIX + 'places365_challenge_256x256_train.tfrecords.gz'
VAL_FILENAME = FILE_PREFIX + 'places365_challenge_256x256_val.tfrecords.gz'
#TRAIN_FILENAME = FILE_PREFIX + 'places365_challenge_256x256_train.hdf5.gz'
#VAL_FILENAME = FILE_PREFIX + 'places365_challenge_256x256_val.hdf5.gz'

# class_number-class_name file name
CATEGORY_FILENAME = ''
SELECTED_CLASS = {0:'baseball_field', 1:'beach', 2:'canyon', 3:'forest_path',
                  4:'industrial_area', 5:'lake-natural', 6:'swamp',
                  7:'temple-asia',8:'train_station-platform', 9:'waterfall'}

# Learning params
LEARNING_RATE = 0.01
#MOMENTUM = 0.9
#WEIGHT_DECAY = 0.0005
DECAY_RATE = 0.95
#DECAY_RATE = 0.9
NUM_EPOCHS = 90
#NUM_EPOCHS = 30
BATCH_SIZE = 100
#BATCH_SIZE = 40
TRAINSET_SIZE = 320000
VALSET_SIZE = 80000

# How often we want to write the tf.summary data to disk
DISPLAY_STEP = 100
#DISPLAY_STEP = 250

# Network params
#NUM_CLASSES = 365
NUM_CLASSES = 10

# Path for tf.summary.FileWriter and to store model checkpoints
FILEWRITER_PATH = './tensorboard'
CHECKPOINT_PATH = './tensorboard/checkpoints'
# Recover all weight variables of the last checkpoint
RECOVER_CKPT = False

# Create parent path if it doesn't exist
if not os.path.isdir(FILEWRITER_PATH):
  os.makedirs(FILEWRITER_PATH)

if not os.path.isdir(CHECKPOINT_PATH):
  os.makedirs(CHECKPOINT_PATH)

def print_features(t):
  print(t.op.name, ' ', t.get_shape().as_list())

# Make string_input_producer, TFRecordsReader, and Queue and Shuffle batches.
def input_pipeline(mode, batch_size=BATCH_SIZE,
                   num_epochs=NUM_EPOCHS):
  with tf.name_scope('img_pipeline'):
    if mode == 'train':
      filenames = [TRAIN_FILENAME]
      image_feature = 'train/image'
      label_feature = 'train/label'
    else:
      filenames = [VAL_FILENAME]
      image_feature = 'val/image'
      label_feature = 'val/label'

    feature = {image_feature: tf.FixedLenFeature([], tf.string),
               label_feature: tf.FixedLenFeature([], tf.int64)}

    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer(filenames,
                                                    num_epochs=NUM_EPOCHS+1)
    # Define a reader and read the next record
    options = tf_record.TFRecordOptions(compression_type=tf_record
                                            .TFRecordCompressionType.GZIP)
    reader = tf.TFRecordReader(options=options)
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features[image_feature], tf.uint8)
      
    # Cast label data into one_hot encoded
    label = tf.cast(features[label_feature], tf.int32)
    label = tf.one_hot(label, NUM_CLASSES)
    # Reshape image data into the original shape
    image = tf.reshape(image, [256,256,3])

    # Any preprocessing here ...
    # 1. random cropping 224x224
    # 2. random LR-flipping
    image = tf.random_crop(image, [224,224,3])
    image = tf.image.random_flip_left_right(image)

    #print_features(image)
      
    # Creates batches by randomly shuffling tensors
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 100
    num_threads = 6
    capacity = min_after_dequeue + (num_threads + 2) * BATCH_SIZE
    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=BATCH_SIZE,
                                            capacity=capacity,
                                            num_threads=num_threads,
                                            min_after_dequeue=min_after_dequeue)

    #print("input_pipeline will return now.")
    return images, labels

def myGenerator(mode, batch_size):
  if mode == 'train':
    data_filename = TRAIN_FILENAME
    total_size = TRAINSET_SIZE
    pt = 0#pt_train
  else:
    data_filename = VAL_FILENAME
    total_size = VALSET_SIZE
    pt = 0#pt_val

  with h5py.File(data_filename, 'r') as f:
    while 1:
      # Do we have to set pt again?
      # We don't use left over.
      if pt+batch_size > f['label'].shape[0]:
        pt = 0

      images = f['image'][pt:pt+batch_size]
      labels = f['label'][pt:pt+batch_size]
      pt += batch_size

      image = []
      label = []
      # Cast label data into one_hot encoded
      eye_array = np.eye(NUM_CLASSES)
      for x in labels:
        label.append(eye_array[int(x)])
      # Now, it is a numpy array
      labels = np.array(label)

      # Any preprocessing here ...
      # 1. random cropping 224x224
      # 2. random LR-flipping
      for im in images:
        rand_x = random.randint(0,256-224-1)
        rand_y = random.randint(0,256-224-1)
        image.append(im[rand_x:rand_x+224,rand_y:rand_y+224,:])

        if random.randint(0,1) == 0:
          image[-1] = image[-1][:,::-1,...]

      # Now, it is a numpy array
      images = np.array(image)    

      yield (pt,images,labels)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [BATCH_SIZE, 224, 224, 3])
y = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES])
is_training = tf.placeholder(tf.bool)

# Initialize model
model = AlexNet(x, NUM_CLASSES, is_training)

# Get model output
logits = model.logits

# List of all trainable variables and save them to the summary
var_list = [v for v in tf.trainable_variables()]
#print([v.name for v in var_list])

for var in var_list:
  tf.summary.histogram(var.name, var)

#l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in var_list
#                     if 'kernel' in v.name ]) * WEIGHT_DECAY

# Op for calculating the loss
with tf.name_scope('cross_ent'):
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=y))
  #loss += l2_loss
  # Add the loss to summary
  tf.summary.scalar('cross_entropy', loss)


batch = tf.Variable(0, trainable=False, dtype=tf.float32, name='global_step')

# Train op
with tf.name_scope('optimizer'):
  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
           learning_rate = LEARNING_RATE,   # Base learning rate.
           global_step = batch * BATCH_SIZE,  # Current index into the dataset.
           decay_steps = TRAINSET_SIZE,      # Decay step.
           decay_rate = DECAY_RATE,     # Decay rate.
           staircase=True)

  # Get gradients of all trainable variables
  gradients = tf.gradients(loss, var_list)
  gradients = list(zip(gradients, var_list))

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    #optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM).minimize(loss,global_step=batch)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=batch)

# Add gradients to summary
for gradient, var in gradients:
  tf.summary.histogram(var.name + '/gradient', gradient)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
  correct_pred = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  # Add the accuracy to the summary
  tf.summary.scalar('train_accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Start Tensorflow session
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                      gpu_options={'allow_growth':True})) as sess:
                                      #gpu_options={'allow_growth':True},
                                      #log_device_placement=True)) as sess:

  print("sleep a while,")
  sys.stdout.flush()
  time.sleep(3)
  print("and wake up.")
  sys.stdout.flush()

  # Declare train/validation input pipelines.
  images, labels = input_pipeline('train')
  val_images, val_labels = input_pipeline('val')

  # Declare train/validation input generators.
  #gen_train = myGenerator('train', BATCH_SIZE)
  #gen_val = myGenerator('val', BATCH_SIZE)

  # Initialize the FileWriter
  writer = tf.summary.FileWriter(FILEWRITER_PATH)

  # Initialize an saver for store model checkpoints
  saver = tf.train.Saver()

  # Get the number of training/validation steps per epoch
  train_batches_per_epoch = int(TRAINSET_SIZE // BATCH_SIZE)
  val_batches_per_epoch = int(VALSET_SIZE // BATCH_SIZE)

  # Initialize all global and local variables
  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
  sess.run(init_op)
  print("Initialized global and local variables.")

  if RECOVER_CKPT:
    latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_PATH)
    print("Loading the last checkpoint: " + latest_ckpt)
    saver.restore(sess, latest_ckpt)
    last_epoch = int(latest_ckpt.replace('_','*').replace('-','*').split('*')[3])
  else:
    last_epoch = 0

  # Add the model graph to TensorBoard
  writer.add_graph(sess.graph)

  # Finalize default graph
  tf.get_default_graph().finalize()

  print("Start training...")
  print("Open Tensorboard at --logdir {}".format(FILEWRITER_PATH))

  # Create a coordinator and run all QueueRunner objects
  coord = tf.train.Coordinator()
  # Start queueing threads.
  threads = tf.train.start_queue_runners(coord=coord)

  # Loop over number of epochs
  for epoch in range(last_epoch, NUM_EPOCHS):
    print("Epoch number: {}".format(epoch+1))
    start_time = time.time()

    for step in range(train_batches_per_epoch):

      # load a train batch from train input_pipeline
      image_batch, label_batch = sess.run([images, labels])
      #print("%dms, batch loaded." % (1000 * (time.time()-start_time)))
      #sys.stdout.flush()
      #start_time = time.time()

      # load a train batch from trainset generator
      #pt, image_batch, label_batch = gen_train.__next__()
      #print(pt, image_batch.shape, label_batch.shape)

      _, l, lr, pred, summaries = sess.run(
                                [optimizer, 
                                 loss, 
                                 learning_rate,
                                 accuracy, 
                                 merged_summary],
                                feed_dict={x: image_batch,
                                           y: label_batch,
                                           is_training: True})
      #print("%dms, training finished." % (1000 * (time.time()-start_time)))
      #sys.stdout.flush()
      #start_time = time.time()

      if step % DISPLAY_STEP == DISPLAY_STEP-1:
        writer.add_summary(summaries, epoch * train_batches_per_epoch + step)
        print("%dms, Epoch %d (%.1f%%), Minibatch loss: %.3f, lr: %.6f, acc: %.1f%%" 
            % (1000 * (time.time()-start_time), epoch+1,
               100 * (step+1) * BATCH_SIZE / TRAINSET_SIZE,
               l, lr, 100 * pred))
        sys.stdout.flush()
        start_time = time.time()

    # Validate the model on the entire validation set
    print("Start validation...")
    start_time = time.time()
    
    test_acc = 0
    test_count = 0
    for _ in range(val_batches_per_epoch):
      val_image_batch, val_label_batch = sess.run([val_images, val_labels])

      #val_pt, val_image_batch, val_label_batch = gen_val.__next__()
      #print(val_pt, val_image_batch.shape, val_label_batch.shape)

      val_pred = sess.run([accuracy], feed_dict={x: val_image_batch,
                                                 y: val_label_batch,
                                                 is_training: False})[0]
      test_acc += val_pred
      test_count += 1

    test_acc /= test_count

    valacc_summary = tf.Summary()
    valacc_summary.value.add(tag='val_accuracy', simple_value=test_acc)
    writer.add_summary(valacc_summary, (epoch+1) * train_batches_per_epoch)
    del valacc_summary

    print("%dms, Epoch: %d Validation accuracy = %.1f%%" % (
              1000 * (time.time()-start_time),
              epoch+1,
              100 * test_acc))
    print("Saving checkpoint of model...")
    # save checkpoint of the model
    checkpoint_name = os.path.join(CHECKPOINT_PATH,
                       'places10-alexnet_ep-'+str(epoch+1)+'_step')
    save_path = saver.save(sess, checkpoint_name, global_step=batch)
    print("Epoch: %d, Model checkpoint saved at %s" % (epoch+1, save_path))
    sys.stdout.flush()

    # Reset graph to default
    #tf.reset_default_graph()

  # Stop the threads
  coord.request_stop()

  # Wait for threads to stop
  coord.join(threads)
  sess.close()
  print("Done!")
