#-*-coding:utf-8-*-
# book: <Neural Network Programming with Tensorflow>
# authors: Manpreet Singh Ghotra, Rajdeep Dua

import tensorflow as tf
import numpy as np

GPU_CONFIG = tf.ConfigProto()
GPU_CONFIG.gpu_options.allow_growth=True
sess = tf.Session(config=GPU_CONFIG)

# page = 4
# norm
vertor = tf.constant([4,5,6],dtype=tf.float32)
eucNorm = tf.norm(vertor,ord='euclidean')
# with tf.Session(config=GPU_CONFIG) as sess:
print(sess.run(eucNorm))

# page = 5
# matrices to tensor object
matrix1 = [[1.0,2.0],[3.0,40]]
matrix2 = np.array([[1.0,2.0],[3.0,40]],dtype=np.float32)
matrix3 = tf.constant([[1.0,2.0],[3.0,40]],dtype=tf.float32)

print(type(matrix1))
print(type(matrix2))
print(type(matrix3))

tensorForM1 = tf.convert_to_tensor(matrix1,dtype=tf.float32)
tensorForM2 = tf.convert_to_tensor(matrix2,dtype=tf.float32)
tensorForM3 = tf.convert_to_tensor(matrix3,dtype=tf.float32)

print(tensorForM1)
print(tensorForM2)
print(tensorForM3)

# page = 6
# mul
mat1 = tf.constant([[4,5,6],[3,2,1]])
mat2 = tf.constant([[7,8,9],[10,11,12]])
mult = tf.multiply(mat1,mat2) # element wise
dotprod = tf.matmul(mat1, tf.transpose(mat2)) # mat mul, (mat1)*(mat2)'
# with tf.Session(config=GPU_CONFIG) as sess:
print(sess.run(mult))
print(sess.run(dotprod))
print('*********************************************************')

# page = 6
# trace
mat = tf.constant([[0,1,2],[3,4,5],[6,7,8]],dtype=tf.float32)
mat = tf.trace(mat) # sum of diagonal elements
# with tf.Session(config=GPU_CONFIG) as sess:
print(sess.run(mat))
print('*********************************************************')

# page = 7
# transpose
x = [[1,2,3],[4,5,6]]
x  =tf.convert_to_tensor(x)
xtrans = tf.transpose(x)

y = [[[1,2,3],[6,5,4]],[[4,5,6],[3,6,3]]]
y = tf.convert_to_tensor(y)
ytrans = tf.transpose(y,perm=[0,2,1]) # ytrans[x,y,z] = y[x,z,y], change order of index

# with tf.Session(config=GPU_CONFIG) as sess:
print(sess.run(xtrans))
print(sess.run(ytrans))
print('*********************************************************')

# page = 8
# diag
mat = tf.constant([[0,1,2],[3,4,5],[6,7,8]],dtype=tf.float32)
diag_mat = tf.diag_part(mat) # to a vector
mat = tf.diag([1,2,3,4])     # to a 2-d mat, all = 0 except diag
# with tf.Session(config=GPU_CONFIG) as sess:
print(sess.run(diag_mat))
print(sess.run(mat))
print('*********************************************************')

# page = 8
# I mat
identity = tf.eye(3,3)
# with tf.Session(config=GPU_CONFIG) as sess:
print(sess.run(identity))
print('*********************************************************')

# page = 9
# inverse
mat = tf.constant([[1,0,0],[0,1,0],[0,0,1]],dtype=tf.float32)
inv_mat = tf.matrix_inverse(mat)
# with tf.Session(config=GPU_CONFIG) as sess:
print(sess.run(inv_mat)) # raise bug if mat can't be inversed
print('*********************************************************')

sess.close()

# page = 10
# linear equations
# canonical circle qeuation: x**2+y**2+d*x+e*y+f=0 -> d*x+e*y+f=-(x**2+y**2)
# XA=B -> find A=[[d],[e],[f]]
# points = tf.constant([[2,1],[0,5],[-1,2]],dtype=tf.float32)
# X = tf.constant([[2,1,1],[0,5,1],[-1,2,1]],dtype=tf.float32)
# B = tf.constant([[5],[25],[5]],dtype=tf.float32)
# A = tf.matrix_solve(X,B)
# with tf.Session(config=GPU_CONFIG) as sess:
#     result = sess.run(A)
#     d,e,f = result.flatten()
#     print('Circle is: x**2 + y**2 + {d}*x + {e}*y + {f}=0'.format(**locals()))












