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
sess = tf.Session(config=GPU_CONFIG)

def draw_plot(x,y, title, label):

    colours = ['blue', 'crimson', 'orange', 'red', 'black', 'green']
    i = 0
    # Defines a marker for each line
    markers = ['o', 'v', '*', '+', 'o', "x"]

    num_of_lines = len(y)

    for i in range(num_of_lines):
    # Scatter plot with point_size^2 = 75, and with respective colors
      plt.scatter(x, y[i], marker=markers[i], s=25, c=colours[i])
    # Connects points with lines, and with respective colours
      line1, = plt.plot(x, y[i], c=colours[i], label = label[i])
      import matplotlib.patches as mpatches
      from matplotlib.legend_handler import HandlerLine2D
      plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)}, loc =4)
      plt.title(title,y=1.05)


    plt.grid()
    plt.show()

# IRIS data set: There are 150 samples
# containing 3 classes of flowers "Iris setosa" "Variegated Iris" "Virginia iris"
# with feature about "sepal length" "sepal width" "petal length" "petal width"

RANDOMSEED = 40
tf.set_random_seed(RANDOMSEED)

def load_iris_date():
    data = np.genfromtxt(r'IRIS_data\iris.csv', delimiter=',')
    target = np.genfromtxt(r'IRIS_data\target.csv', delimiter=',').astype(np.int32) # 0 or 1 or 2
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
    # h_size = 256
    y_size = train_y.shape[1] # 3 classes

    # variables
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    stddev = stddev
    weights_1 = initialize_weights((x_size, h_size), stddev)
    weights_2 = initialize_weights((h_size, y_size), stddev)

    y_pred = forward_propagation(X, weights_1, weights_2)
    predict = tf.argmax(y_pred, dimension=1)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))    
    test_accs = []
    train_accs = []
    time_taken_summary = []
    for sgd_step in sgd_steps:
        start_time = time.time()
        updates_sgd = tf.train.GradientDescentOptimizer(sgd_step).minimize(cost)
        # sess = tf.Session(config=GPU_CONFIG)
        # init
        init = tf.initialize_all_variables()
        steps = 50
        sess.run(init)

        test_acc = []
        train_acc = []
        print('Step, train accuracy, test accuracy')
        for step in range(steps):
            # Train with each example
            for i in range(len(train_x)):
                sess.run(updates_sgd, feed_dict={X:train_x[i:i+1],y:train_y[i:i+1]})
            # accuracy for each step
            train_accuracy = np.mean(np.argmax(train_y,axis=1) == sess.run(predict,feed_dict={X:train_x,y:test_y}))
            test_accuracy = np.mean(np.argmax(test_y, axis=1) == sess.run(predict, feed_dict={X: test_x, y: test_y}))
            print('{0:d}, {1:.2f}, {2:.2f}'.format(step+1,100*train_accuracy, 100*test_accuracy))
            test_acc.append(100 * test_accuracy)
            train_acc.append(100 * train_accuracy)
        end_time = time.time()
        diff = end_time -start_time
        time_taken_summary.append((sgd_step,diff))
        #t = [np.array(test_acc)]
        #t.append(train_acc)
        train_accs.append(train_acc)

    # draw
    title = "Steps vs Training Accuracy-" + " sgd steps: 0.01,0.02, 0.03"
    label = ['SGD Step 0.01', 'SGD Step 0.02','SGD Step 0.03']
    print("Time Taken Summary :" + str(time_taken_summary))
    x = np.arange(steps)
    draw_plot(x, train_accs, title, label)

        


sgd_steps = [0.01, 0.02, 0.03]
run(128,0.1,sgd_steps)
sess.close()     