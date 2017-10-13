
# coding: utf-8

# """ simple linear regression """

# * import

# In[193]:


import tensorflow as tf
from fire_theft import *


# In[194]:


# xy_data = np.array(fire_theft_data)
# x_data = xy_data[:, 0]
# y_data = xy_data[:, 1]

learning_rate = 0.001
iteration = 500
trained_weight = 0
trained_bias = 0

input_value = 15.5


# * Placeholder

# In[195]:


X = tf.placeholder(tf.float32, name='x')
Y = tf.placeholder(tf.float32, name='y')


# * Variable

# In[196]:


W = tf.Variable(0.0, name='w')
b = tf.Variable(0.0, name='b')


# * Linear function

# In[197]:


Y_pred = W * X + b


# * Loss function

# In[198]:


loss = tf.square(Y_pred - Y)


# * Optimizer

# In[199]:


trainer = tf.train.GradientDescentOptimizer(learning_rate)
updateModel = trainer.minimize(loss)


# * Training

# In[200]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(iteration):
        total_loss = 0.0
        for x, y in fire_theft_data:
            _, err = sess.run([updateModel, loss], feed_dict={X:x, Y:y})
            total_loss += err
        print('loss: ', total_loss)
    trained_weight, trained_bias = sess.run([W, b])


# In[206]:


print('trained value: w = ',trained_weight , ' b = ', trained_bias)


# In[207]:


y_pred = trained_weight * input_value + trained_bias
print('input : ', input_value, 'predicted value : ', y_pred)

