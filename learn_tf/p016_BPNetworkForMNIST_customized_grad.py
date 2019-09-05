#-*-coding:utf-8-*-
# book: <Neural Network Programming with Tensorflow>
# authors: Manpreet Singh Ghotra, Rajdeep Dua

import tensorflow as tf

GPU_CONFIG = tf.ConfigProto()
GPU_CONFIG.gpu_options.allow_growth=True
sess = tf.Session(config=GPU_CONFIG)

# get mnist dataset, 60k training samples, 10k testing samples
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('MNIST_data/',one_hot=True)

# x represents images iwth 784 values as columns (28,28)
# y represents output digit
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

# initialize weights and biases [w1,b1][w2,b2]
numNeuronsInDeepLayer = 30
w1 = tf.Variable(tf.truncated_normal([784,numNeuronsInDeepLayer]))
b1 = tf.Variable(tf.truncated_normal([1,numNeuronsInDeepLayer]))
w2 = tf.Variable(tf.truncated_normal([numNeuronsInDeepLayer,10]))
b2 = tf.Variable(tf.truncated_normal([1,10]))

# non-linear sigmoid function at each neuron
# f(x) = 1/(1+exp(-x))
def sigmoid(x):
    sigma = 1/(1+tf.exp(-x))
    # sigma = tf.div(tf.constant(1.0),
    #                tf.add(tf.constant(1.0),tf.exp(tf.negative(x))))
    return sigma

# starting from first layer with wx+b, then apply sigmoid to add non-linearity
z1 = tf.add(tf.matmul(x,w1),b1)
a1 = sigmoid(z1)
z2 = tf.add(tf.matmul(a1,w2),b2)
a2 = sigmoid(z2)

# calculate the loss (delta)
loss = tf.subtract(a2,y)

# derivative of the sigmoid function
# f'(x) = f(x)*(1-f(x))
def sigmaprime(x):
    return tf.multiply(sigmoid(x),tf.subtract(tf.constant(1.0),sigmoid(x)))

# backward propagation
dz2 = tf.multiply(loss,sigmaprime(z2))
db2 = dz2
dw2 = tf.matmul(tf.transpose(a1),dz2)

da1 = tf.matmul(dz2,tf.transpose(w2))
dz1 = tf.multiply(da1,sigmaprime(z1))
db1 = dz1
dw1 = tf.matmul(tf.transpose(x),dz1)

# finally update the network
eta = tf.constant(0.5)
step = [tf.assign(w1,tf.subtract(w1,tf.multiply(eta,dw1))),\
        tf.assign(b1,tf.subtract(b1,tf.multiply(eta, tf.reduce_mean(db1,axis=[0])))),\
        tf.assign(w2,tf.subtract(w2,tf.multiply(eta,dw2))),\
        tf.assign(b2,tf.subtract(b2,tf.multiply(eta, tf.reduce_mean(db2,axis=[0]))))]

acct_mat = tf.equal(tf.arg_max(a2,1),tf.arg_max(y,1))
acct_ret = tf.reduce_sum(tf.cast(acct_mat,tf.float32))


# sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(100000):
    batch_xs, batch_ys = data.train.next_batch(10)
    sess.run(step,feed_dict={x:batch_xs,y:batch_ys})
    if i%1000==0:
        res = sess.run(acct_ret,feed_dict={
              x:data.test.images,\
              y:data.test.labels})
        print(res)

sess.close()

'''
s'(x) = s(x) * (1 - s(x))

z1 = x * w1 + b1
a1 = s(z1)
z2 = a1 * w2 + b2
a2 = s(z2)

dloss = a2 - y
loss = 1/2 * dloss **2

dz2 = d(loss)/d(z2) 
    = dloss * d(a2)/d(z2)
    = dloss * d(s(z2))/d(z2)
    = dloss * s'(z2)
db2 = d(loss)/d(b2) 
    = d(loss)/d(z2) * 1
    = dz2
dw2 = d(loss)/d(w2) 
    = d(loss)/d(z2) * a1
    = dz2 * a1
da1 = d(loss)/d(a1) 
    = d(loss)/d(z2) * w2
    = dz2 * w2
dz1 = d(loss)/d(z1) 
    = d(loss)/d(a1) * s'(z1)
    = da1 * s'(z1)
db1 = d(loss)/d(b1) 
    = d(loss)/d(z1) * 1
    = dz1
dw1 = d(loss)/d(w1)
    = d(loss)/d(z1) * x
    = dz1 * x
'''

