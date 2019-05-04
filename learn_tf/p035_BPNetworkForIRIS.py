#-*-coding:utf-8-*-
# book: <Neural Network Programming with Tensorflow>
# authors: Manpreet Singh Ghotra, Rajdeep Dua

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

GPU_CONFIG = tf.ConfigProto()
GPU_CONFIG.gpu_options.allow_growth=True

# IRIS data set: There are 150 samples
# containing 3 classes of flowers "Iris setosa" "Variegated Iris" "Virginia iris"
# with feature about "sepal length" "sepal width" "petal length" "petal width"

def run(h_size, stddev, sgd_setp):
    pass

def load_iris_date():
    data = np.genfromtxt('IRIS_data\iris.csv', delimiter=',')
    target = np.genfromtxt('IRIS_data\target.csv', delimiter=',').astype(np.int32) # 0 or 1 or 2
    # prepend the column of 1s for bias
    L, W = data.shape
    all_X = np.ones((L,W+1))
    all_X[:,1:] = data
    num_labels = len(np.unique(target)) # np.unique(): remove repetitive elements and sort
    all_Y = np.eye(num_labels)[target]
    
    

def run(h_size, stddev, sgd_steps):
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
                

