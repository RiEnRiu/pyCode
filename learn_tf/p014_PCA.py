#-*-coding:utf-8-*-
# book: <Neural Network Programming with Tensorflow>
# authors: Manpreet Singh Ghotra, Rajdeep Dua

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plts

GPU_CONFIG = tf.ConfigProto()
GPU_CONFIG.gpu_options.allow_growth=True
sess = tf.Session(config=GPU_CONFIG)

# PCA by SVD
text = ['I','like','enjoy','deep','learning','NLP','flying','.']

xMatrix = np.array([[0,2,1,0,0,0,0,0],
                    [2,0,0,1,0,1,0,0],
                    [1,0,0,0,0,0,1,0],
                    [0,1,0,0,1,0,0,0],
                    [0,0,0,1,0,0,0,1],
                    [0,1,0,0,0,0,0,1],
                    [0,0,1,0,0,0,0,1],
                    [0,0,0,0,1,1,1,0]],dtype=np.float32)

def pca(mat):
    mat = tf.constant(mat,dtype=tf.float32)
    mean = tf.reduce_mean(mat,0)
    less = mat-mean
    s,u,v = tf.svd(less,full_matrices=True,compute_uv=True)
    s2 = s**2
    variance_ratio = s2/tf.reduce_sum(s2)
    # with tf.Session(config=GPU_CONFIG) as sess:
    run = sess.run(variance_ratio)
    return run

print(pca(xMatrix))
sess.close()
