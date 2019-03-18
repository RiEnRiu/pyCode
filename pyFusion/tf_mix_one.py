import tensorflow as tf
import numpy as np
import cv2



class mix_one_tf_tool:

    def __init__(self,gpu_index=0,gpu_menory = 1.0):
        with tf.device('/gpu:'+str(gpu_index)):
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = gpu_menory
            self.sess = tf.Session(config=config)

            self.bg = tf.placeholder(dtype=tf.float32, shape=[1,None,None,3])
            self.obj = tf.placeholder(dtype=tf.float32, shape=[1,None,None,3])
            self.alpha = tf.placeholder(dtype=tf.float32, shape=[1,None,None,1])
            self.thresh = tf.placeholder(dtype=tf.float32, shape=[1,1,1,1])
            self.max_itr = tf.placeholder(dtype=tf.float32, shape=[1,1,1,1])

            self.Kx = tf.constant(value=   [0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,1,1,1,0,0,0,0,0,0,0,0,0],dtype=tf.float32,shape=[3,3,3,1])
            self.Ky = tf.constant(value=   [0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,0,0,0,0,0,0,1,1,1,0,0,0],dtype=tf.float32,shape=[3,3,3,1])
            self.Klapx = tf.constant(value=[0,0,0,0,0,0,0,0,0,1,1,1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0],dtype=tf.float32,shape=[3,3,3,1])
            self.Klapy = tf.constant(value=[0,0,0,1,1,1,0,0,0,0,0,0,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0],dtype=tf.float32,shape=[3,3,3,1])
            self.Kconv = tf.constant(value=[0,0,0,1,1,1,0,0,0,1,1,1, 0, 0, 0,1,1,1,0,0,0,1,1,1,0,0,0],dtype=tf.float32,shape=[3,3,3,1])

            self.alpha_c3 = tf.concat([self.alpha,self.alpha,self.alpha],3)

            self.mask01_c3 = tf.round(self.alpha_c3)
            self.invmask01_c3 = 1.0-self.mask01_c3

            self.pasted_bg = self.alpha_c3*self.obj+(1.0-self.alpha_c3)*self.bg

            self.bg_grad_x = tf.nn.depthwise_conv2d(input=self.bg, filter=self.Kx, strides=[1,1,1,1], padding='SAME')
            self.bg_grad_y = tf.nn.depthwise_conv2d(input=self.bg, filter=self.Ky, strides=[1,1,1,1], padding='SAME')

            self.obj_grad_x = tf.nn.depthwise_conv2d(input=self.obj, filter=self.Kx, strides=[1,1,1,1], padding='SAME')
            self.obj_grad_y = tf.nn.depthwise_conv2d(input=self.obj, filter=self.Ky, strides=[1,1,1,1], padding='SAME')

            self.grad_x = self.obj_grad_x* self.mask01_c3 + self.bg_grad_x * self.invmask01_c3
            self.grad_y = self.obj_grad_y* self.mask01_c3 + self.bg_grad_y * self.invmask01_c3

            self.lapx = tf.nn.depthwise_conv2d(input=self.grad_x, filter=self.Klapx, strides=[1,1,1,1], padding='SAME')
            self.lapy = tf.nn.depthwise_conv2d(input=self.grad_y, filter=self.Klapy, strides=[1,1,1,1], padding='SAME')

            self.lap = self.lapx+self.lapy

            self.fin_i, self.fin_thresh, self.fin_max_diff, self.fin_mix = \
                tf.while_loop(self.cond, self.body, (0.0, 1.0, 1000.0, self.pasted_bg))
            self.out_mix = self.fin_mix
        self.sess.run(tf.global_variables_initializer())
        return

    def cond(self,i,value, pre_max_diff,pre_mix):
        return tf.reduce_all([i<self.max_itr, value>self.thresh])

    def body(self,i,value ,pre_max_diff,pre_mix):
        conv2d_result = tf.nn.depthwise_conv2d(input=pre_mix,filter=self.Kconv,strides=[1,1,1,1],padding='SAME')
        this_mix_out255_out0 = (conv2d_result+self.lap)*0.25 * self.mask01_c3 +pre_mix * self.invmask01_c3
        #this_mix = this_mix_out255>255?255:this_mix_out255
        this_mix_out0 = tf.minimum(this_mix_out255_out0,255)
        this_mix = tf.maximum(this_mix_out0,0)
        #this_mix = this_mix_out255
        max_diff = tf.reduce_max(tf.abs(this_mix-pre_mix))
        thresh_result = tf.abs(max_diff-pre_max_diff)/pre_max_diff
        i = i + 1
        return i,thresh_result, max_diff, this_mix

    def tf_acc(self, cv_bg,cv_obj,cv_alpha,cv_thresh, cv_max_itr):

        cv_bg = cv_bg.reshape([1,cv_bg.shape[0],cv_bg.shape[1],cv_bg.shape[2]])
        cv_obj = cv_obj.reshape([1,cv_obj.shape[0],cv_obj.shape[1],cv_obj.shape[2]])
        cv_alpha = cv2.rectangle(cv_alpha,(0,0),(cv_alpha.shape[1]-1,cv_alpha.shape[0]-1),0,1)
        cv_alpha = cv_alpha.reshape([1,cv_alpha.shape[0],cv_alpha.shape[1],1])
        tf_thresh = [[[[cv_thresh]]]]
        tf_max_itr = [[[[cv_max_itr]]]]

        cv_out_mix = self.sess.run(self.out_mix, \
                                   feed_dict={self.bg:cv_bg, \
                                              self.obj:cv_obj, \
                                              self.alpha:cv_alpha, \
                                              self.thresh:tf_thresh, \
                                              self.max_itr:tf_max_itr})

        #mmm_index = cv_out_mix.argmax()
        #print(mmm_index)
        #print(str(cv_out_mix.flatten()[mmm_index])+'    '+str(cv_out_mix.astype(np.uint8).flatten()[mmm_index]))
        #cv2.imshow('org',cv_out_mix.astype(np.uint8)[0])
        #cv_out_mix[cv_out_mix>255]=255
        #cv2.imshow('max255',cv_out_mix.astype(np.uint8)[0])
        #cv2.waitKey(0)


        return cv_out_mix.astype(np.uint8)[0]


    def close_sess(self):
        self.sess.close()




