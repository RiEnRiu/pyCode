#-*-coding:utf-8-*-
# book: <Neural Network Programming with Tensorflow>
# authors: Manpreet Singh Ghotra, Rajdeep Dua

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split

GPU_CONFIG = tf.ConfigProto()
GPU_CONFIG.gpu_options.allow_growth=True

# IRIS data set: There are 150 samples
# containing 3 classes of flowers "Iris setosa" "Variegated Iris" "Virginia iris"
# with feature about "sepal length" "sepal width" "petal length" "petal width"

RANDOMSEED = 40
tf.set_random_seed(RANDOMSEED)

def load_iris_date():
    data = np.genfromtxt('IRIS_data\iris.csv', delimiter=',')
    target = np.genfromtxt('IRIS_data\target.csv', delimiter=',').astype(np.int32) # 0 or 1 or 2
    # prepend the column of 1s for bias
    L, W = data.shape
    all_X = np.ones((L,W+1))
    all_X[:,1:] = data
    num_labels = len(np.unique(target)) # np.unique(): remove repetitive elements and sort
    all_y = np.eye(num_labels)[target]
    return train_test_split(all_X,all_y,test_size=0.33,random_state=RANDOMSEED)
    

# make network 
def initialize_weights(shape, stddev):
    weights = tf.random_normal(shape, stddev=stddev)
    return tf.Variable(weights)

def forward_propagation(X, weights_1, weights_2):
    sigmoid = tf.nn.sigmoid(tf.matmul(X,weights_1))
    y = tf.matmul(sigmoid, weights_2)
    return y


def run(h_size, stddev, sgd_steps):
    train_x,test_x,train_y,test_y = load_iris_date()

    # size of layers
    x_size = train_x.shape[1] # 4 features and 1 bias
    h_size = 256
    y_size = train_y.shape[1] # 3 classes

    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_size,log))
    test_accs = []
    train_accs = []
    time_taken_summary = []
    for sgd_step in sgd_steps:
        start_time = time.time()
        updates_sgd = tf.train.GradientDescentOptimizer(sgd_step).minimize(cost)
        sess = tf.Session(config=GPU_CONFIG)
        init = tf.initialize_all_variables()
        steps = 50
        sess.run(init)
        x = np.arange(steps)
        print('Step, train accuracy, test accuracy')

        for step in range(steps):
            # Train with each example
            for i in range(len(train_x)):
                sess.run(updates_sgd, feed_dict={X:train_x[i:i+1],y:train_y[i:i+1]})
                train_accuracy = np.mean(np.argmax(train_y,axis=1)==sess.run(predict,feed_dict={X:train_x,y:test_y})
                print('{0d}, 1.2f, 2.2f'.format(step+1.100*train_accuracy,100*test_accuracy))
                # x.append(step)
                

