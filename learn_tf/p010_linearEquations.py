#-*-coding:utf-8-*-
# book: <Neural Network Programming with Tensorflow>
# authors: Manpreet Singh Ghotra, Rajdeep Dua

import tensorflow as tf
import numpy as np

GPU_CONFIG = tf.ConfigProto()
GPU_CONFIG.gpu_options.allow_growth=True
sess = tf.Session(config=GPU_CONFIG)

# linear equations
# canonical circle qeuation: x**2+y**2+d*x+e*y+f=0 -> d*x+e*y+f=-(x**2+y**2)
# XA=B -> find A=[[d],[e],[f]]
points = tf.constant([[2,1],[0,5],[-1,2]],dtype=tf.float32)
X = tf.constant([[2,1,1],[0,5,1],[-1,2,1]],dtype=tf.float32) # x,y,1
B = -tf.constant([[5],[25],[5]],dtype=tf.float32)            # -(x**2+y**2)
A = tf.matrix_solve(X,B)
# with tf.Session(config=GPU_CONFIG) as sess:
result = sess.run(A)
d,e,f = result.flatten()
print('Circle is: x**2 + y**2 + {0}*x + {1}*y + {2}=0'.format(d,e,f))
print('*********************************************************')

sess.close()









