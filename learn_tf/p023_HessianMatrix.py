#-*-coding:utf-8-*-
# book: <Neural Network Programming with Tensorflow>
# authors: Manpreet Singh Ghotra, Rajdeep Dua


import tensorflow as tf
import numpy as np

GPU_CONFIG = tf.ConfigProto()
GPU_CONFIG.gpu_options.allow_growth=True

X = tf.Variable(np.random.random_sample(),dtype=tf.float32)
y = tf.Variable(np.random.random_sample(),dtype=tf.float32)

def createCons(x):
    return tf.constant(x,dtype=tf.float32)

# q([X,y]) = X**2 + 2*X*y + 3*y**2 + 4*X + 5*y + 6 
function = tf.pow(X,createCons(2)) + createCons(2)*X*y + \
            createCons(3)*tf.pow(y,createCons(2)) + createCons(4)*X + \
            createCons(5)*y + createCons(6)

# compute hessian
def hessian(func, varbles):
    matrix = []
    for v_1 in varbles:
        tmp = []
        for v_2 in varbles:
            # calculate derivative twice, first w.r.t v2 and then w.r.t v1
            tmp.append(tf.gradients(tf.gradients(func,v_2)[0],v_1)[0])
        tmp = [createCons(0) if t==None else t for t in tmp]
        tmp = tf.stack(tmp)
        matrix.append(tmp)
    matrix = tf.stack(matrix)
    return matrix

hessian = hessian(function, [X, y])

sess = tf.Session(config=GPU_CONFIG)
sess.run(tf.initialize_all_variables())
print(sess.run(hessian))

