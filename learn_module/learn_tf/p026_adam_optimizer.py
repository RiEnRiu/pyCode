#-*-coding:utf-8-*-
# book: <Neural Network Programming with Tensorflow>
# authors: Manpreet Singh Ghotra, Rajdeep Dua


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

GPU_CONFIG = tf.ConfigProto()
GPU_CONFIG.gpu_options.allow_growth=True
sess = tf.Session(config=GPU_CONFIG)

# input data set
xData = np.arange(100,step=0.1) # 0:0.1:100
yData = xData + 20 * np.sin(xData/10)

# scatter plot for input data
# plt.figure('tf')
plt.scatter(xData, yData)

# defining data size and batch size
nSample = 1000
batchSize = 100

# reshape
xData = np.reshape(xData, (nSample,1))
yData = np.reshape(yData, (nSample,1))

# input placeholders
x = tf.placeholder(tf.float32,shape=(batchSize,1))
y = tf.placeholder(tf.float32,shape=(batchSize,1))

# init weight and bias
with tf.variable_scope('linearRegression'):
    W = tf.get_variable('weights',(1,1),initializer=tf.random_normal_initializer())
    b = tf.get_variable('bias',(1,),initializer=tf.constant_initializer(0.0))

y_pred = tf.matmul(x,W) + b
loss = tf.reduce_sum((y-y_pred)**2/nSample)

# optimizer
opt = tf.train.AdamOptimizer().minimize(loss)
# with tf.Session(config=GPU_CONFIG) as sess:
sess.run(tf.global_variables_initializer())
    
# gradient descent loop for 500 steps
for _ in range(500):
    # random minibatch
    indices = np.random.choice(nSample, batchSize)
    X_batch, y_batch = xData[indices], yData[indices]

    # gradient descent step
    _, loss_val = sess.run([opt, loss], feed_dict={x:X_batch,y:y_batch})
    print('loss = {0}'.format(loss_val))



    trained_W, trained_b = sess.run([W, b])
    trained_W, trained_b = trained_W[0][0], trained_b[0]
    print('line:   y = {0} * x + {1}'.format(trained_W, trained_b))

_xData = xData.reshape((-1))

plt.plot(_xData, _xData*trained_W+trained_b, label='Fitted line')
plt.show()
sess.close()









