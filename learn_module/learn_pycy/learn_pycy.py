import numpy as np
import cv2
import time

# python setup.py build_ext --inplace
from cy_nms import cy_nms

def np_nms(boxes, scores, thresh = 0.45):
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    # areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # w = np.maximum(0.0, xx2 - xx1 + 1)
        # h = np.maximum(0.0, yy2 - yy1 + 1)
        w = np.maximum(0.0, xx2 - xx1) 
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def random_boxes_scores(nboxes):
    random_scores = np.random.rand(nboxes).astype(np.float32)
    tmp = np.random.rand(nboxes,4).astype(np.float32)
    random_boxes = np.zeros(tmp.shape, tmp.dtype)
    random_boxes[:,0] = np.fmin(tmp[:,0],tmp[:,2])
    random_boxes[:,1] = np.fmin(tmp[:,1],tmp[:,3])
    random_boxes[:,2] = np.fmax(tmp[:,0],tmp[:,2])
    random_boxes[:,3] = np.fmax(tmp[:,1],tmp[:,3])
    return random_boxes, random_scores


if __name__=='__main__':
    # jit tfgpu tfcpu np cy cpp 
    # nboxes = 100000
    # thresh = 0.9

    # jit np cpp cy tfgpu tfcpu 
    # nboxes = 100000
    # thresh = 0.5

    # np cpp cy tfgpu tgcpu jit
    # nboxes = 100000
    # thresh = 0.1

    # tfgpu tfcpu cy np jit cpp  
    # nboxes = 10000
    # thresh = 0.9

    # tfgpu tfcpu cy np cpp jit  
    nboxes = 10000
    thresh = 0.5

    # np cy cpp tfgpu tfcpu jit 
    # nboxes = 10000
    # thresh = 0.1

    # tfgpu tfcpu cy cpp np jit  
    # nboxes = 1000
    # thresh = 0.9

    # jit np cpp cy tfgpu tfcpu 
    # nboxes = 1000
    # thresh = 0.5

    # cy tfgpu tfcpu cpp np jit 
    # nboxes = 1000
    # thresh = 0.1

    boxes, scores = random_boxes_scores(nboxes)
    np_nms_keep = []
    jit_nms_keep = []
    cy_nms_keep = []
    cpp_nms_keep = []
    tfcpu_nms_keep = []
    tfgpu_nms_keep = []
 
    t0 = time.time()
    np_nms_keep = np_nms(boxes, scores, thresh)
    t1 = time.time()
    # jit_nms_keep = jit_nms(boxes, scores, thresh)
    t2 = time.time()
    cy_nms_keep = cy_nms(boxes, scores, thresh)
    t3 = time.time()
    # cpp_nms_keep = cpp_nms(boxes, scores, thresh)
    t4 = time.time()
    # tfcpu_nms_keep = sess.run(cpu_result, feed_dict={boxes_cpu_ph:boxes,scores_cpu_ph:scores})
    t5 = time.time()
    # tfgpu_nms_keep = sess.run(gpu_result, feed_dict={boxes_gpu_ph:boxes,scores_gpu_ph:scores})
    t6 = time.time()

    print('   np_nms({0}) time = {1:>.7} s / {2} boxes'.format(len(np_nms_keep),t1-t0,nboxes))
    print('  jit_nms({0}) time = {1:>.7} s / {2} boxes'.format(len(jit_nms_keep),t2-t1,nboxes))
    print('   cy_nms({0}) time = {1:>.7} s / {2} boxes'.format(len(cy_nms_keep),t3-t2,nboxes))
    print('  cpp_nms({0}) time = {1:>.7} s / {2} boxes'.format(len(cpp_nms_keep),t4-t3,nboxes))
    print('tfcpu_nms({0}) time = {1:>.7} s / {2} boxes'.format(len(tfcpu_nms_keep),t5-t4,nboxes))
    print('tfgpu_nms({0}) time = {1:>.7} s / {2} boxes'.format(len(tfgpu_nms_keep),t6-t5,nboxes))

