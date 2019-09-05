#-*-coding:utf-8-*-
# book: <Neural Network Programming with Tensorflow>
# authors: Manpreet Singh Ghotra, Rajdeep Dua

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

GPU_CONFIG = tf.ConfigProto()
GPU_CONFIG.gpu_options.allow_growth=True
sess = tf.Session(config=GPU_CONFIG)

# SVD: A(m,n) = U(m,r) * S(r,r) * int(V(r,n))  ???
path = ''
text = ['I','like','enjoy','deep','learning','NLP','flying','.']

xMatrix = np.array([[0,2,1,0,0,0,0,0],
                    [2,0,0,1,0,1,0,0],
                    [1,0,0,0,0,0,1,0],
                    [0,1,0,0,1,0,0,0],
                    [0,0,0,1,0,0,0,1],
                    [0,1,0,0,0,0,0,1],
                    [0,0,1,0,0,0,0,1],
                    [0,0,0,0,1,1,1,0]],dtype=np.float32)

X_tensor = tf.convert_to_tensor(xMatrix,dtype=tf.float32)

# tf SVD
# with tf.Session(config=GPU_CONFIG) as sess:
tf_S, tf_U, tf_Vh = sess.run(tf.svd(X_tensor, full_matrices=False))


plt.figure('tf')
plt.ion()
for i in range(len(text)):
    plt.text(tf_U[i,0],tf_U[i,1],text[i])
plt.ylim(-0.8,0.8)
plt.xlim(-0.8,2.0)
plt.show()

# np SVD
np_U,np_S,np_Vh = np.linalg.svd(xMatrix,full_matrices=False)

plt.figure('np')
plt.ion()
for i in range(len(text)):
    plt.text(np_U[i,0],np_U[i,1],text[i])
plt.ylim(-0.8,0.8)
plt.xlim(-0.8,2.0)
plt.show()
input()


sess.close()







