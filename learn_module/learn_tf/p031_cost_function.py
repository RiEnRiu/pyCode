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


'''
1. sigmoid cross entropy 
a] S(x) = sigmoid(x) = 1/(1+exp(-x))
b] x = logits, z = labels
c] H(x,z) = z*log(S(x)**-1) + (1-z)*log((1-S(x))**-1)
          = z*log(1+exp(-x)) + (1-z)*(log(1+exp(-x))-log(exp(-x)))
          = z*log(1+exp(-x)) + log(1+exp(-x)) - log(exp(-x)) - z*(log(1+exp(-x)) + z*log(exp(-x))
          = log(1+exp(-x)) - log(exp(-x)) - z*log(exp(-x))
          = log(1+exp(-x)) + x - x*z    <==>    log(1+exp(x)) - x*z
                  x>0                                    x<0
d] H(x,z) = max(x,0) - x*z + log(1+exp(-abs(x)))  # avoid overflows
'''
# _sentinel: placehold parameter, do not set it
# labels: z = labels, labels.shape == logits.shape， labels.dtype==tf.float32 or labels.dtype==tf.float64 
# logits: x = logits, logits.shape == labels.shape， logits.dtype==tf.float32 or logits.dtype==tf.float64 
tf.nn.sigmoid_cross_entropy_with_logits(_sentinel=None, labels=None,logits=None,name=None)



'''
1. sigmoid cross entropy with weight
a] S(x) = sigmoid(x) = 1/(1+exp(-x))
b] x = logits, z = labels
c] H(x,z) = w*z*log(S(x)**-1) + (1-z)*log((1-S(x))**-1)
          = w*z*log(1+exp(-x)) + (1-z)*(log(1+exp(-x))-log(exp(-x)))
          = w*z*log(1+exp(-x)) + log(1+exp(-x)) - log(exp(-x)) - z*(log(1+exp(-x)) + z*log(exp(-x))
          = (w*z+1-z)*log(1+exp(-x)) + x - z*x
          = ((w-1)*z+1)*log(1+exp(-x)) + x - x*z        <==>    ((w-1)*z+1)*log(1+exp(x)) - w*x*z
          =           a*log(1+exp(-x)) + a*x - w*x*z    <==>              a*log(1+exp(x)) - w*x*z
d] H(x,z) = max(x,0)*a - w*x*z + a*log(1+exp(-abs(x)))  # avoid overflows
'''
# _sentinel: placehold parameter, do not set it
# labels: z = labels, labels.shape == logits.shape， labels.dtype==tf.float32 or labels.dtype==tf.float64 
# logits: x = logits, logits.shape == labels.shape， logits.dtype==tf.float32 or logits.dtype==tf.float64 
# logits: probability about all classes. For example 10 classes: [0.8,0.1,0.1,0,0,0,0,0,0,0], one-hot type's vector
# pos_weight: weight of positive sample, such as 2.0 or 3.0 or 0.5
tf.nn.weighted_cross_entropy_with_logits(_sentinel=None, labels=None,logits=None,dim=-1,name=None)



'''
1. softmax cross entropy 
a] S(x) = softmax(x) = exp(xi)/reduce_sum(exp(x)), S'(x) = S(x)*(1-S(x))
b] x = logits, z = labels
c] H(x,z)  = z*log(S(x)**-1) + (1-z)*log((1-S(x))**-1)
d] dH(x,z)   
     dx    = -z*S'(x)/S(x) + (z-1)*(1-S'(x))/(1-S(x))
           = -z*(1-S(x)) + (1-z)*S(x)
           = S(x) - z
'''

# _sentinel: placehold parameter, do not set it
# labels: z = labels, labels.shape == logits.shape， labels.dtype==tf.float32 or labels.dtype==tf.float64 
# logits: x = logits, logits.shape == labels.shape， logits.dtype==tf.float32 or logits.dtype==tf.float64 
# logits: probability about all classes. For example 10 classes: [0.8,0.1,0.1,0,0,0,0,0,0,0], one-hot type's vector
# dim: which dimension does softmax do?
tf.nn.softmax_cross_entropy_with_logits_v2(_sentinel=None, labels=None,logits=None,dim=-1,name=None)

# _sentinel: placehold parameter, do not set it
# labels: z = labels, labels.shape == logits.shape， labels.dtype==tf.float32 or labels.dtype==tf.float64 
# logits: x = logits, logits.shape == labels.shape， logits.dtype==tf.int32 or logits.dtype==tf.int64 
# logits: probability about 1 class. For example [4] means the class with index of 4 is 100% while others are 0%
# dim: which dimension does softmax do?
tf.nn.sparse_softmax_cross_entropy_with_logits(_sentinel=None, labels=None,logits=None,dim=-1,name=None)








