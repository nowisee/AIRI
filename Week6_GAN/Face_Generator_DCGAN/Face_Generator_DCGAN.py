import tensorflow as tf
import numpy as np
import scipy.misc
import sys
import os
from random import uniform
import data_queue

from tensorflow.python.lib.io import tf_record

image_row = 64
image_col = 64

noise_size = 100

BATCH_SIZE = 400
TRAINSET_SIZE = 350000

NUM_EPOCHS = 50

DISPLAY_STEP = 10


# Path for tf.summary.FileWriter and to store model checkpoints
FILEWRITER_PATH = './tensorboard'
CHECKPOINT_PATH = './tensorboard/checkpoint'
# Recover all weight variables from the last checkpoint
RECOVER_CKPT = True

data_file_names = [
    './tensorboard/celeb_tfrec/celeb_train_00000-of-00005.tfrecord',
    './tensorboard/celeb_tfrec/celeb_train_00001-of-00005.tfrecord',
    './tensorboard/celeb_tfrec/celeb_train_00002-of-00005.tfrecord',
    './tensorboard/celeb_tfrec/celeb_train_00003-of-00005.tfrecord',
    './tensorboard/celeb_tfrec/celeb_train_00004-of-00005.tfrecord'
]



# Create parent path if it doesn't exist
if not os.path.isdir(FILEWRITER_PATH):
  os.makedirs(FILEWRITER_PATH)

if not os.path.isdir(CHECKPOINT_PATH):
  os.makedirs(CHECKPOINT_PATH)



def print_features(t):
    #print()
    print(t.op.name, ' ', t.get_shape().as_list())

def load_image(path):
    return scipy.misc.imread(path)

def save_image(path, image):
    scipy.misc.imsave(path, image)

def image_merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if len(images.shape) == 4:
        img = np.zeros((h * size[0], w * size[1], images.shape[3]))
    else:
        img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        if len(images.shape) == 4:
            img[j*h:j*h+h, i*w:i*w+w, :] = image
        else:
            img[j*h:j*h+h, i*w:i*w+w] = image
    return img

def save_images(path, images, size):
    merged_image = image_merge(images, size)
    save_image(path, merged_image)

def resize_image(image, size):
    return scipy.misc.imresize(image, size)




##### Real Image Input
#fileobj = open('train-images.idx3-ubyte')
#real_images = np.fromfile(file=fileobj, dtype=np.uint8)
#real_images = real_images[16:].reshape([realImageNum, image_row*image_col]).astype(np.float)

#fileobj = open('train-labels.idx1-ubyte')
#real_labels = np.fromfile(file=fileobj, dtype=np.uint8)
#real_labels = real_labels[8:].reshape([realImageNum]).astype(np.int)


##### Input Pipeline
real_images = data_queue.make_data_pipeline(data_file_names, NUM_EPOCHS, BATCH_SIZE)

print(real_images.shape)



##### Placeholder
g_input_noise = tf.placeholder(shape=[None, noise_size], dtype=tf.float32, name='g_input_noise')

#d_input_image = tf.placeholder(shape=[None, image_row, image_col, 3], dtype=tf.float32, name='d_input_image')

input_label = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='d_input_label')

is_real = tf.placeholder(shape=None, dtype=tf.bool, name='is_real')


##### Model
## Generator Model
with tf.variable_scope("Generator"):
    print_features(g_input_noise)

    g_fc1 = tf.layers.dense(g_input_noise, 4*4*1024, activation=None, name='g_fc1',
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    print_features(g_fc1)

    g_reshape = tf.reshape(g_fc1, shape=[-1, 4, 4, 1024], name='g_reshape')
    print_features(g_reshape)

    conv_trans2 = tf.layers.conv2d_transpose(g_reshape, 512, [5,5], [2,2], padding='SAME',
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                             name='g_conv_trans2')
    conv_trans2 = tf.layers.batch_normalization(conv_trans2, name='g_batch_normalization2')
    conv_trans2 = tf.nn.relu(conv_trans2, name='g_ReLU2')
    print_features(conv_trans2)

    conv_trans3 = tf.layers.conv2d_transpose(conv_trans2, 256, [5,5], [2,2], padding='SAME',
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                             name='g_conv_trans3')
    conv_trans3 = tf.layers.batch_normalization(conv_trans3, name='g_batch_normalization3')
    conv_trans3 = tf.nn.relu(conv_trans3, name='g_ReLU3')

    print_features(conv_trans3)

    conv_trans4 = tf.layers.conv2d_transpose(conv_trans3, 128, [5, 5], [2, 2], padding='SAME',
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                             name='g_conv_trans4')
    conv_trans4 = tf.layers.batch_normalization(conv_trans4, name='g_batch_normalization4')
    conv_trans4 = tf.nn.relu(conv_trans4, name='g_ReLU4')
    print_features(conv_trans4)

    conv_trans5 = tf.layers.conv2d_transpose(conv_trans4, 3, [5, 5], [2, 2], padding='SAME',
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                             name='g_conv_trans5')
    conv_trans5 = tf.nn.tanh(conv_trans5, name='g_tanh5')
    print_features(conv_trans5)

    #generated_images = tf.multiply(tf.add(conv_trans5, 1), 255/2)
    generated_images = conv_trans5


## Discriminator Model
with tf.variable_scope("Discriminator"):
    #d_input = tf.where(is_real, real_images, generated_images)
    d_input = tf.cond(is_real, lambda: real_images, lambda: generated_images)

    conv1 = tf.layers.conv2d(d_input, 128, [5, 5], strides=[2, 2], padding='SAME', name='d_conv1',
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    conv1 = tf.layers.batch_normalization(conv1, name='d_batch_normalization1')
    conv1 = tf.maximum(conv1, 0.2*conv1, name='d_LeakyReLU1')   # Leaky ReLU
    print_features(conv1)

    conv2 = tf.layers.conv2d(conv1, 256, [5, 5], strides=[2, 2], padding='SAME', name='d_conv2',
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    conv2 = tf.layers.batch_normalization(conv2, name='d_batch_normalization2')
    conv2 = tf.maximum(conv2, 0.2 * conv2, name='d_LeakyReLU2')  # Leaky ReLU
    print_features(conv2)

    conv3 = tf.layers.conv2d(conv2, 512, [5, 5], strides=[2, 2], padding='SAME', name='d_conv3',
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    conv3 = tf.layers.batch_normalization(conv3, name='d_batch_normalization3')
    conv3 = tf.maximum(conv3, 0.2 * conv3, name='d_LeakyReLU3')  # Leaky ReLU
    print_features(conv3)

    conv4 = tf.layers.conv2d(conv3, 1024, [5, 5], strides=[2, 2], padding='SAME', name='d_conv4',
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    conv4 = tf.layers.batch_normalization(conv4, name='d_batch_normalization4')
    conv4 = tf.maximum(conv4, 0.2 * conv4, name='d_LeakyReLU4')  # Leaky ReLU
    print_features(conv4)

    d_reshape = tf.reshape(conv4, shape=[-1, 4*4*1024], name='d_reshape')
    print_features(d_reshape)

    logits = tf.layers.dense(d_reshape, 1, name='d_fc5')
    #logits = tf.nn.sigmoid(logits, name='d_sigmoid3')  ## use sigmoid_cross_entropy()
    print_features(logits)

    #sys.exit(1)


##### Loss
with tf.name_scope('sigmoid_ent'):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=input_label, logits=logits))

    # Add the loss to summary
    tf.summary.scalar('sigmoid_cross_entropy', loss)


##### Optimizer
with tf.name_scope('optimizer'):
    t_vars = tf.trainable_variables()
    #print(t_vars)

    ## Discriminator Optimizer
    d_vars = [var for var in t_vars if 'Discriminator' in var.name]
    #print(d_vars)
    d_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss, var_list=d_vars)


    ## Generator Optimizer
    g_vars = [var for var in t_vars if 'Generator' in var.name]
    #print(g_vars)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss, var_list=g_vars)



##### Accuracy
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(tf.nn.softmax(logits), axis=1), tf.argmax(input_label, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Add the accuracy to the summary
    tf.summary.scalar('train_accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options={'allow_growth':True})) as sess:
    # Initialize the FileWriter
    writer = tf.summary.FileWriter(FILEWRITER_PATH)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    # Initialize variables
    init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
    sess.run(init_op)

    # (optional) load model weights
    if RECOVER_CKPT:
        latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_PATH)
        print("Loading the last checkpoint: " + str(latest_ckpt))
        saver.restore(sess, latest_ckpt)
        last_epoch = int(latest_ckpt.replace('_','*').replace('-','*').split('*')[4])
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

    # get the number of training/validation steps per epoch
    train_batches_per_epoch = TRAINSET_SIZE // BATCH_SIZE


    # Loop over number of epochs
    print("Start training...")
    for epoch in range(last_epoch, NUM_EPOCHS):
        print("Epoch number: {}".format(epoch+1))

        total_d_loss1 = 0
        total_d_loss2 = 0
        total_g_loss1 = 0
        total_g_loss2 = 0

        for step in range(train_batches_per_epoch):
            label_real = np.ones([BATCH_SIZE, 1])#[[1] for _ in range(BATCH_SIZE)]
            label_real = label_real - 0.1

            label_fake = np.zeros([BATCH_SIZE, 1])#[[0] for _ in range(BATCH_SIZE)]
            label_fake = label_fake + 0.1
            #print(label_real.shape)

            batchIndex = step * BATCH_SIZE
            #image_batch = np.reshape(real_images[batchIndex:batchIndex + BATCH_SIZE], [BATCH_SIZE, image_row, image_col, 1])
            #image_batch = image_batch/127.5 - 1


            #print([uniform(0.1, 0.9)])
            #z = [[uniform(0.1, 0.9) for _ in range(noise_size)] for _ in range(BATCH_SIZE)]
            z = np.random.uniform(-1.0, 1.0, [BATCH_SIZE, noise_size])
            #print(z)
            #z = np.asarray(z)
            #print(z.shape)

            # train model with this batch.
            _, d_loss1, summaries1 = sess.run([d_optimizer, loss, merged_summary], feed_dict={g_input_noise: z, input_label: label_fake, is_real: False})
            _, d_loss2, summaries2 = sess.run([d_optimizer, loss, merged_summary], feed_dict={g_input_noise: z, input_label: label_real, is_real: True})

            _, g_loss1, summaries3 = sess.run([g_optimizer, loss, merged_summary], feed_dict={g_input_noise: z, input_label: label_real, is_real: False})
            _, g_loss2, summaries4 = sess.run([g_optimizer, loss, merged_summary], feed_dict={g_input_noise: z, input_label: label_real, is_real: False})

            total_d_loss1 += d_loss1
            total_d_loss2 += d_loss2
            total_g_loss1 += g_loss1
            total_g_loss2 += g_loss2

            #summaries = sess.run(merged_summary)

            if step % DISPLAY_STEP == DISPLAY_STEP-1:
                writer.add_summary(summaries1, epoch * train_batches_per_epoch + step)
                print("step: ", step+1)
                print("d_loss1: ", d_loss1, "d_loss2: ", d_loss2)
                print("g_loss1: ", g_loss1, "g_loss2: ", g_loss2)

                # save image
                output_image = sess.run([generated_images], feed_dict={g_input_noise: z})[0]
                output_image = np.int32((output_image+1)*255/2)

                #print(output_image.shape)

                output_image = np.reshape(output_image, [BATCH_SIZE, 64, 64, 3])
                #print(output_image.shape)


                save_images('./Result/output_image_' + str(epoch) + '.jpg', output_image[0:64], [8,8])


        # save checkpoint of the model at each epoch
        print("Saving checkpoint of model...")
        checkpoint_name = os.path.join(CHECKPOINT_PATH,
                           'DCGAN_FaceG_ep-'+str(epoch+1)+'_step')
        save_path = saver.save(sess, checkpoint_name, global_step=BATCH_SIZE)
        print("Epoch: %d, Model checkpoint saved at %s" % (epoch+1,
                                                 checkpoint_name+'-(#global_step)'))
        sys.stdout.flush()

    # Stop input_pipeline threads
    coord.request_stop()
    # Wait for threads to stop
    coord.join(threads)
    sess.close()

print("Done!")
