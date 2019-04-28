#-*-coding:utf-8-*-
# book: <Neural Network Programming with Tensorflow>
# authors: Manpreet Singh Ghotra, Rajdeep Dua


raise ImportError('Read and learn the cost function as an example but not run it.')

import tensorflow as tf

# the format of cost function and training function
OPERATION_NAME = tf.nn.sigmoid_cross_entropy_with_logits
ACTUAL_VALUE = None
PREDICTED_VALUE = None
cost = tf.reduce_mean(OPERATION_NAME(labels=ACTUAL_VALUE,logits=PREDICTED_VALUE))
updates_sgd = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

# some cost functions

'''
1. sigmoid cross entropy 
a] S(x) = sigmoid(x) = 1/(1+exp(-x))
b] x = logits, z = labels
c] H(x,z) = z*log(S(x)**-1) + (1-z)*log((1-S(x))**-1)
          = z*log(1+exp(-x)) + (1-z)*(log(1+exp(-x))-log(exp(-x)))
          = z*log(1+exp(-x)) + log(1+exp(-x)) - log(exp(-x)) - z*(log(1+exp(-x)) - z*log(exp(-x))
          = log(1+exp(-x)) - log(exp(-x)) - z*log(exp(-x))
          = log(1+exp(-x)) + x(1-z)    <==>    log(1+exp(x)) - x*z
d] H(x,z) = max(x,0) - x*z + log(1+exp(-abs(x)))  # avoid overflows
'''











