
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import scipy.misc
import sys

import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')


# In[6]:


##
imageNum = 60000
testImageNum = 10000
typesOfNumber = 10
row = 28
col = 28
batchSize = 256
batchCount = imageNum // batchSize

learningRate = 0.05
iteration = 100

##
X = tf.placeholder(shape=[None, row*col], dtype=tf.float32, name='X')
Y = tf.placeholder(shape=[None], dtype=tf.int64, name='Y')
Y_onehot = tf.one_hot(Y, typesOfNumber, axis=1)

W = tf.Variable(tf.zeros([row*col, typesOfNumber]))
B = tf.Variable(tf.zeros([typesOfNumber]))
weight = tf.zeros([row*col, typesOfNumber])

##
Y_pred = tf.matmul(X, W) + B
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred, labels=Y_onehot))
Y_pred_softmax = tf.nn.softmax(Y_pred)

trainer = tf.train.GradientDescentOptimizer(learningRate)
optimizer = trainer.minimize(loss)

correct = tf.equal(tf.argmax(Y_pred_softmax, 1), tf.argmax(Y_onehot, 1))
acc = tf.reduce_mean(tf.cast(correct, tf.float32))    

##
saver = tf.train.Saver()
path = './models/simple_nn'

# In[16]:


if sys.argv[1] == 'train':
    ##
    fileobj = open('train-images.idx3-ubyte')
    images = np.fromfile(file=fileobj, dtype=np.uint8)
    images = images[16:].reshape([imageNum, row*col]).astype(np.float)

    fileobj = open('train-labels.idx1-ubyte')
    labels = np.fromfile(file=fileobj, dtype=np.uint8)
    labels = labels[8:].reshape([imageNum]).astype(np.int)
    
    ##
    images = images / 255.0
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(iteration):
            total_loss = 0  

            for i in range(batchCount):
                batchIndex = i * batchSize
                img = np.reshape(images[batchIndex:batchIndex+batchSize], [batchSize, row*col])
                label = labels[batchIndex:batchIndex+batchSize]
                _, loss_v = sess.run([optimizer, loss], feed_dict={X:img, Y:label})
                total_loss += loss_v
            
	    saver.save(sess, path)
            print 'epoch_%04d: %.6f' %(epoch+1, total_loss), ', acc = ', acc.eval(session=sess, feed_dict={X:images, Y:labels})
            
        #print'Test image acc = ', acc.eval(session=sess, feed_dict={X:test_images, Y:test_labels})

        trained_weight = sess.run(W)
    
    ##
    trained_weight = np.transpose(trained_weight)
    weight = (trained_weight).reshape([typesOfNumber, row, col]).astype(np.float32)

    for i in range(typesOfNumber):
        maxVal = np.max(weight[i])
        minVal = np.min(weight[i])
        weight[i] = 255 * (weight[i] - minVal) / (maxVal-minVal)
        name = 'weight' + str(i) + '.jpg'
        scipy.misc.imsave(name, weight[i])


# In[73]:


if sys.argv[1] == 'test':
    if sys.argv[2] == 'MNIST_data':
        fileobj = open('t10k-images.idx3-ubyte')
        test_images = np.fromfile(file=fileobj, dtype=np.uint8)
        test_images = test_images[16:].reshape([testImageNum, row*col]).astype(np.float)

        fileobj = open('t10k-labels.idx1-ubyte')
        test_labels = np.fromfile(file=fileobj, dtype=np.uint8)
        test_labels = test_labels[8:].reshape([testImageNum]).astype(np.int)
        
        test_images = test_images / 255.0

        with tf.Session() as sess:
            saver.restore(sess, path)
            print'Test image acc = ', acc.eval(session=sess, feed_dict={X:test_images, Y:test_labels})
        
    else:
        test_image = scipy.misc.imread(sys.argv[2])
        test_image = test_image.reshape([1, row*col])
        test_image = test_image / 255.0
        with tf.Session() as sess:
            saver.restore(sess, path)
            np_Y_pred_softmax = sess.run(Y_pred_softmax, feed_dict={X:test_image})
            
            for i in range(typesOfNumber):
                if np_Y_pred_softmax[0][i] == np_Y_pred_softmax.max():
                    print i
    

