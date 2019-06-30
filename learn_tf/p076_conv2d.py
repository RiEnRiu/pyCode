#-*-coding:utf-8-*-
# book: <Neural Network Programming with Tensorflow>
# authors: Manpreet Singh Ghotra, Rajdeep Dua

import tensorflow as tf
import numpy as np

GPU_CONFIG = tf.ConfigProto()
GPU_CONFIG.gpu_options.allow_growth=True



# conv2d: whether rotate kernel? 
def test_whether_rotate_kernel():
    i = tf.constant([
                     [1.0, 1.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 1.0, 1.0],
                     [0.0, 0.0, 1.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0, 0.0]], dtype=tf.float32)
    k = tf.constant([
                     [1.0, 0.0, 1.0],
                     [0.0, 1.0, 0.0],
                     [1.0, 0.0, 0.0]], tf.float32)
    # kernel: 
    kernel = tf.reshape(k, [3,3,1,1], name='kernel')

    # image: NHWC
    image = tf.reshape(i, [1,4,5,1], name='image')

    _strides = [1,1,1,1]

    res = tf.nn.conv2d(image, kernel, strides=_strides, padding='VALID')

    with tf.Session(config=GPU_CONFIG) as sess:
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
  





if __name__=='__main__':
    test_whether_rotate_kernel()
    print('')
    test_strides_and_padding()


