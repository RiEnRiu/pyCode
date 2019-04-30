#-*-coding:utf-8-*-
# book: <Neural Network Programming with Tensorflow>
# authors: Manpreet Singh Ghotra, Rajdeep Dua

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

GPU_CONFIG = tf.ConfigProto()
GPU_CONFIG.gpu_options.allow_growth=True

# IRIS data set: There are 150 samples
# containing 3 classes of flowers "Iris setosa" "Variegated Iris" "Virginia iris"
# with feature about "sepal length" "sepal width" "petal length" "petal width"

def run(h_size, stddev, sgd_setp):
    pass

def load_iris_date():
    data = np.genfromtxt('.\iris.csv', delimiter=',')
    target = np.genfromtxt('.\target.csv', delimiter=',').astype(np.int32) # 0 or 1 or 2
    # prepend the column of 1s for bias
    L, W = data.shape
    all_X = np.ones((L,W+1))
    all_X[:,1:] = data
    num_labels = len(np.unique(target)) # np.unique(): remove repetitive elements and sort
    all_Y = np.eye(num_labels)[target]
    


