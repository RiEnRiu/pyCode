#-*-coding:utf-8-*-
# book: <Neural Network Programming with Tensorflow>
# authors: Manpreet Singh Ghotra, Rajdeep Dua

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np




GPU_CONFIG = tf.ConfigProto()
GPU_CONFIG.gpu_options.allow_growth=True
sess = tf.Session(config=GPU_CONFIG)


# conv2d: whether rotate kernel? 
def test_whether_rotate_kernel():
    i = tf.constant([[1.0, 1.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 1.0, 1.0],
                     [0.0, 0.0, 1.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0, 0.0]], dtype=tf.float32)
    k = tf.constant([[1.0, 0.0, 1.0],
                     [0.0, 1.0, 0.0],
                     [1.0, 0.0, 0.0]], tf.float32)
    # kernel: 
    kernel = tf.reshape(k, [3,3,1,1], name='kernel')

    # image: NHWC
    image = tf.reshape(i, [1,4,5,1], name='image')

    _strides = [1,1,1,1]

    res = tf.nn.conv2d(image, kernel, strides=_strides, padding='VALID')

    # with tf.Session(config=GPU_CONFIG) as sess:
    ri,rk,rc = sess.run([image,kernel,res])
    print('image shape: {0}'.format(ri.shape))
    print(np.squeeze(ri))
    print('kernel shape: {0}'.format(rk.shape))
    print(np.squeeze(rk))
    print('ans shape: {0}'.format(rc.shape))
    print(np.squeeze(rc))
    print('conv2d: not rotate kernel.')


def test_strides_and_padding():
    i = tf.constant([
                     [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                     [0.1, 1.1, 2.1, 3.1, 4.1, 5.1],
                     [0.2, 1.2, 2.2, 3.2, 4.2, 5.2],
                     [0.3, 1.3, 2.3, 3.3, 4.3, 5.3],
                     [0.4, 1.4, 2.4, 3.4, 4.4, 5.4],
                     [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]], dtype=tf.float32)
    k = tf.constant([
                     [0.0, 0.5, 0.0],
                     [0.0, 0.5, 0.0],
                     [0.0, 0.5, 0.0]], tf.float32)
    # kernel: HWCN
    kernel = tf.reshape(k, [3,3,1,1], name='kernel')

    # image: NHWC
    image = tf.reshape(i, [1,6,6,1], name='image')

    _strides = [1,3,3,1]

    res_valid = tf.nn.conv2d(image, kernel, strides=_strides, padding='VALID')
    res_same = tf.nn.conv2d(image, kernel, strides=_strides, padding='SAME')



    with tf.Session(config=GPU_CONFIG) as sess:
        ri,rk,rcs,rcv = sess.run([image,kernel,res_same,res_valid])
        print('image shape: {0}'.format(ri.shape))
        print(np.squeeze(ri))
        print('kernel shape: {0}'.format(rk.shape))
        print(np.squeeze(rk))
        print('ans(padding=SAME) shape: {0}'.format(rcs.shape))
        print(np.squeeze(rcs))
        print('ans(padding=VALID) shape: {0}'.format(rcv.shape))
        print(np.squeeze(rcv))
        print('conv2d(padding=SAME):')
        print('    1. fill the image with 0 so that it can conv2d into the same shape')
        print('    2. run conv2d')
        print('    3. find the value according to the strides')
        print('conv2d(padding=VALID):')
        print('    1. run conv2d')
        print('    2. find the value according to the strides')
  

def test_train_mnist():
    batch_size = 50
    learning_rate = 0.01
    image = tf.placeholder(tf.float32,[batch_size,28,28,1])
    label = tf.placeholder(tf.float32,[batch_size,10])
    ker1 = tf.random_uniform([5, 5, 1, 12], minval=0.1, maxval=0.5, dtype=tf.float32)
    ker1 = tf.Variable(ker1)
    bias1 = tf.Variable(tf.zeros((1,1,1,12),tf.float32))
    conv1 = tf.nn.conv2d(image, ker1, strides=[1,1,1,1], padding='SAME')
    relu1 = tf.nn.relu(conv1+bias1)
    pool1 = tf.nn.max_pool(relu1,(1,2,2,1),strides=(1,2,2,1),padding='SAME')
    ker2 = tf.random_uniform([5, 5, 12, 20], minval=0.1, maxval=0.5, dtype=tf.float32)
    ker2 = tf.Variable(tf.Variable(ker2))
    bias2 = tf.zeros((1,1,1,20),tf.float32)
    conv2 = tf.nn.conv2d(pool1, ker2, strides=[1,1,1,1], padding='SAME')
    relu2 = tf.nn.relu(conv2+bias2)
    pool2 = tf.nn.max_pool(relu2,(1,2,2,1),strides=(1,2,2,1),padding='SAME')
    pool2 = tf.reshape(pool2,(batch_size,-1))
    wfc1 = tf.random_uniform((980,100), minval=0.1, maxval=0.5, dtype=tf.float32)
    wfc1 = tf.Variable(wfc1)
    bfc1 = tf.Variable(tf.zeros((batch_size,100),tf.float32))
    fc1 = tf.matmul(pool2,wfc1)+bfc1
    relu3 = tf.nn.relu(fc1)
    wfc2 = tf.random_uniform((100,10), minval=0.1, maxval=0.5, dtype=tf.float32)
    wfc2 = tf.Variable(wfc2)
    bfc2 = tf.Variable(tf.zeros((batch_size,10),tf.float32))
    fc2 = tf.matmul(relu3,wfc2)+bfc2

    softmaxloss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label,logits=fc2)
    step = tf.train.GradientDescentOptimizer(learning_rate).minimize(softmaxloss)

    # sess = tf.InteractiveSession()
    acct_mat = tf.equal(tf.arg_max(softmaxloss,1),tf.arg_max(label,1))
    acct_ret = tf.reduce_sum(tf.cast(acct_mat,tf.float32))
    sess.run(tf.global_variables_initializer())

    data = input_data.read_data_sets('MNIST_data/',one_hot=True)

    for i in range(1000):
        batch_xs, batch_ys = data.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape((batch_size,28,28,1))
        sess.run(step, feed_dict={image:batch_xs,label:batch_ys})
        if i%10000==0:
            res = sess.run(acct_ret,feed_dict={
                  image:data.test.images.reshape((-1,28,28,1))[:50],\
                  label:data.test.labels[:50]})
            print(res)
    




#test_whether_rotate_kernel()
#print('')
#test_strides_and_padding()
#print('')
test_train_mnist()
print('')

sess.close()     

