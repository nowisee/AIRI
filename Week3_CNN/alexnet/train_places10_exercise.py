import os
import sys
import numpy as np
import tensorflow as tf

from alexnet_exercise import AlexNet
from tensorflow.python.lib.io import tf_record

import time

# train, val data file name
FILE_PREFIX = '/data/Places10/'
TRAIN_FILENAME = FILE_PREFIX + 'places365_challenge_256x256_train.tfrecords.gz'
VAL_FILENAME = FILE_PREFIX + 'places365_challenge_256x256_val.tfrecords.gz'

# class_number-class_name dict
SELECTED_CLASS = {0:'baseball_field', 1:'beach', 2:'canyon', 3:'forest_path',
                  4:'industrial_area', 5:'lake-natural', 6:'swamp',
                  7:'temple-asia',8:'train_station-platform', 9:'waterfall'}

# Learning params
LEARNING_RATE = 0.01
DECAY_RATE = 0.95
#WEIGHT_DECAY = 0.0005

# total epochs
NUM_EPOCHS = 15

# mini-batch size
BATCH_SIZE = 100

# total train and validation dataset size
TRAINSET_SIZE = 320000
VALSET_SIZE = 80000

# How often we want to write the tf.summary data to disk (by # of mini-batches)
DISPLAY_STEP = 100

# Network params
NUM_CLASSES = 10

# Path for tf.summary.FileWriter and to store model checkpoints
FILEWRITER_PATH = './tensorboard'
CHECKPOINT_PATH = './tensorboard/checkpoints'
# Recover all weight variables from the last checkpoint
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
    # select dataset (train/validation)
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
    # Get serialized examples
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
    # - random cropping 224x224
    # - random LR-flipping
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

#######################################################################################
# 1. Declare TF placeholders for graph input and output and dropout control parameter.
#######################################################################################
x = tf.placeholder(shape=[None,224,224,3], dtype=tf.float32, name='X')
y = tf.placeholder(shape=[None,10], dtype=tf.int64, name='Y')
is_training = tf.placeholder(shape=None, dtype=tf.bool, name='TRAIN')

# Initialize model
model = AlexNet(x, NUM_CLASSES, is_training)

# Get model output
logits = model.logits
pred = tf.nn.softmax(logits)

# List of all trainable variables and save them to the summary
var_list = [v for v in tf.trainable_variables()]
for var in var_list:
  tf.summary.histogram(var.name, var)

#l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in var_list
#                     if 'kernel' in v.name ]) * WEIGHT_DECAY

###########################################################################
# 2. Op for calculating the loss (tf.nn.softmax_cross_entropy_with_logits)
###########################################################################
with tf.name_scope('cross_ent'):
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

  #loss += l2_loss

  # Add the loss to summary
  tf.summary.scalar('cross_entropy', loss)

######################################################################
# 3. Train optimizer
# Optimizer: set up a variable that's incremented once per batch and
# controls the learning rate decay.
# (tf.train.exponential_decay, tf.train.GradientDescentOptimizer)
# - declare a variable batch to manage global step.
# - use an exponential schedule starting at LEARNING_RATE = 0.01,
#   decays with DECAY_RATE = 0.95.
# - use gradient descent optimizer.
######################################################################
batch = tf.Variable(0, trainable=False, dtype=tf.float32, name='global_step')
with tf.name_scope('optimizer'):
  learning_rate = tf.train.exponential_decay(  learning_rate=0.01,
                                               global_step=batch * BATCH_SIZE,
                                               decay_steps=TRAINSET_SIZE,
                                               decay_rate=0.95,
                                               staircase=True)

  # Get gradients of all trainable variables
  gradients = tf.gradients(loss, var_list)
  gradients = list(zip(gradients, var_list))

  # The below two lines MUST be included, when you are using Batch Normalization (tensorflow-1.3)
  # https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=batch)

# Add gradients to summary
for gradient, var in gradients:
  tf.summary.histogram(var.name + '/gradient', gradient)

######################################################################
# 4. Evaluation op: Accuracy of the model
# (tf.equal, tf.argmax, tf.nn.softmax, tf.reduce_mean)
######################################################################
with tf.name_scope("accuracy"):
  correct_pred = tf.equal(tf.argmax(tf.nn.softmax(logits), axis=1), tf.argmax(y, axis=1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

  # Add the accuracy to the summary
  tf.summary.scalar('train_accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

######################################################################
# 5. Start Tensorflow session
# (tf.Session, tf.ConfigProto)
######################################################################
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options={'allow_growth':True})) as sess:

  # Declare train/validation input pipelines.
  images, labels = input_pipeline('train')
  val_images, val_labels = input_pipeline('val')

  # Initialize the FileWriter
  writer = tf.summary.FileWriter(FILEWRITER_PATH)

  # Initialize an saver for store model checkpoints
  saver = tf.train.Saver()

  ###############################################
  # 6. Initialize all global and local variables
  ###############################################
  init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
  sess.run(init_op)

  # (optional) load model weights
  if RECOVER_CKPT:
    latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_PATH)
    print("Loading the last checkpoint: " + str(latest_ckpt))
    saver.restore(sess, latest_ckpt)
    last_epoch = int(latest_ckpt.replace('_','*').replace('-','*').split('*')[3])
  else:
    last_epoch = 0

  # Add the model graph to TensorBoard
  writer.add_graph(sess.graph)

  # Finalize default graph
  tf.get_default_graph().finalize()

  ### run input_pipeline threads
  # Create a coordinator and run all QueueRunner objects
  coord = tf.train.Coordinator()
  # Start queueing threads.
  threads = tf.train.start_queue_runners(coord=coord)

  ##############################################################
  # 7. Get the number of training/validation steps per epoch
  ##############################################################
  train_batches_per_epoch = TRAINSET_SIZE // BATCH_SIZE
  val_batches_per_epoch = VALSET_SIZE // BATCH_SIZE

  # Loop over number of epochs
  print("Start training...")
  for epoch in range(last_epoch, NUM_EPOCHS):
    print("Epoch number: {}".format(epoch+1))

    for step in range(train_batches_per_epoch):
      # load a train batch from train input_pipeline
      image_batch, label_batch = sess.run([images, labels])

      ##################################
      # 8. train model with this batch.
      ##################################
      _, l, pred_, summaries = sess.run(  [optimizer, loss, accuracy, merged_summary],
                                         feed_dict={x:image_batch, y:label_batch, is_training:True}  )

      if step % DISPLAY_STEP == DISPLAY_STEP-1:
        writer.add_summary(summaries, epoch * train_batches_per_epoch + step)
        print("step: ", step, "loss: ", l)

    # Validate the model on the entire validation set
    print("Start validation...")
    test_acc = 0
    test_count = 0
    for _ in range(val_batches_per_epoch):
      # load a validation batch from validation input_pipeline
      val_image_batch, val_label_batch = sess.run([val_images, val_labels])

      ###################################################################
      # 9. get validation predictions and calculate validation accuracy 
      ###################################################################
      val_pred = sess.run(accuracy, feed_dict={x:val_image_batch, y:val_label_batch, is_training:False})

    # Calculate test accuracy from predicted results.
      test_acc += val_pred
      test_count += 1
    test_acc /= test_count

    # write validation accuracy to summary
    valacc_summary = tf.Summary()
    valacc_summary.value.add(tag='val_accuracy', simple_value=test_acc)
    writer.add_summary(valacc_summary, (epoch+1) * train_batches_per_epoch)

    # save checkpoint of the model at each epoch
    print("Saving checkpoint of model...")
    checkpoint_name = os.path.join(CHECKPOINT_PATH,
                       'places10-alexnet_ep-'+str(epoch+1)+'_step')
    save_path = saver.save(sess, checkpoint_name, global_step=batch)
    print("Epoch: %d, Model checkpoint saved at %s" % (epoch+1,
                                             checkpoint_name+'-(#global_step)'))
    sys.stdout.flush()

  # Stop input_pipeline threads
  coord.request_stop()
  # Wait for threads to stop
  coord.join(threads)
  sess.close()
print("Done!")
