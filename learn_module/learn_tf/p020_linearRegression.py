#-*-coding:utf-8-*-
# book: <Neural Network Programming with Tensorflow>
# authors: Manpreet Singh Ghotra, Rajdeep Dua

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

GPU_CONFIG = tf.ConfigProto()
GPU_CONFIG.gpu_options.allow_growth=True
sess = tf.Session(config=GPU_CONFIG)

# config parameters
learningRate = 0.01
trainingEpochs = 1000
displayStep = 50

# create the training data
trainX = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,\
                   7.042,10.791,5.313,7.997,5.654,9.27,3.12])
trainY = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,\
                   2.827,3.465,1.65,2.904,2.42,2.94,1.34])
nTrainSamples = trainX.shape[0]

# create the testing data 
testX = np.array([6.83,4.668,8.9,7.91,5.7,8.7,3.1,2.1])
testY = np.array([1.84,2.273,3.2,2.831,2.92,3.24,1.35,1.03])
nTestSamples = testX.shape[0]

# tf input
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# initialize weights and bias
W = tf.Variable(np.random.randn(),name='weight')
b = tf.Variable(np.random.randn(),name='bias')

# linear model
linearModel = tf.add(tf.multiply(X,W),b)

# 0.5 mean squared error
loss = tf.reduce_sum(0.5*tf.pow(linearModel-Y,2)/nTrainSamples)
cost = tf.reduce_sum(0.5*tf.pow(linearModel-Y,2)/nTestSamples)

# Gradient descent
opt = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

# initializing variables
init = tf.global_variables_initializer()

# run
# with tf.Session(config=GPU_CONFIG) as sess:
sess.run(init)
# fitting the training data
for epoch in range(trainingEpochs):
    for x,y in zip(trainX,trainY):
        sess.run(opt,feed_dict={X:x,Y:y})
    # print logs
    if epoch%displayStep==0:
        c = sess.run(loss, feed_dict={X:trainX,Y:trainY})
        print('Epoch = {0:04d}    loss = {1:.9f}    W = {2:.9f}    b = {3:.9f}'.format(\
            epoch,\
            c,\
            sess.run(W),\
            sess.run(b)))


print('optimization done...')
WTrained = sess.run(W)
bTrained = sess.run(b)
trainingLoss = sess.run(loss, feed_dict={X:trainX,Y:trainY})
testingCost = sess.run(cost, feed_dict={X:testX,Y:testY})
print('Training loss = {0:.9f}, W = {1:.9f}, b = {2:.9f}'.format(\
        trainingLoss,\
        WTrained,\
        bTrained)) 
print('Testing cost = {0:.9f}'.format(testingCost)) 
print('Absolute mean square loss difference = {0:0.9f}'.format(abs(trainingLoss-testingCost)))

# display the plot
plt.ion()
plt.plot(trainX, trainY,'ro', label='Training data')
plt.plot(trainX, WTrained*trainX+bTrained, label='Fitted line')
plt.legend()
plt.show()

plt.ion()
plt.plot(testX, testY,'bo', label='Testing data')
plt.plot(testX, WTrained*testX+bTrained, label='Fitted line')
plt.legend()
plt.show()

input()
sess.close()


