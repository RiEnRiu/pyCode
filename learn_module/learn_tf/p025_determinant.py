#-*-coding:utf-8-*-
# book: <Neural Network Programming with Tensorflow>
# authors: Manpreet Singh Ghotra, Rajdeep Dua


import tensorflow as tf
import numpy as np

GPU_CONFIG = tf.ConfigProto()
GPU_CONFIG.gpu_options.allow_growth=True
sess = tf.Session(config=GPU_CONFIG)

x = np.array([[10.0,15.0,20.0],[0.0,1.0,5.0],[3.0,5.0,7.0]],dtype=np.float32)

det = tf.matrix_determinant(x)

# with tf.Session(config=GPU_CONFIG) as sess:
print(sess.run(det))
sess.close()