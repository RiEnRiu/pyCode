#-*-coding:utf-8-*-
import cv2
import numpy as np
import math
import threading
import random
import time

#Rotate
def __rotate_one(img, angle):
    diagonal = int(math.sqrt(img.shape[0]*img.shape[0]+img.shape[1]*img.shape[1])) + 1
    bg_shape = list(img.shape)
    bg_shape[0] = diagonal
    bg_shape[1] = diagonal
    bg = np.zeros(bg_shape,img.dtype)
    bg_roi_x1 = int((diagonal-img.shape[1])/2)
    bg_roi_y1 = int((diagonal-img.shape[0])/2)
    bg_roi_x2 = bg_roi_x1+img.shape[1]
    bg_roi_y2 = bg_roi_y1+img.shape[0]
    bg[bg_roi_y1:bg_roi_y2,bg_roi_x1:bg_roi_x2] = img
    r = cv2.getRotationMatrix2D((diagonal/2,diagonal/2),angle,1.0)
    rot_output = cv2.warpAffine(bg,r,(bg.shape[1],bg.shape[0]))
    c = math.fabs(math.cos(angle/180*math.pi))
    s = math.fabs(math.sin(angle/180*math.pi))
    true_w = int(img.shape[1] * c + img.shape[0]*s)
    true_h = int(img.shape[0] * c + img.shape[1]*s)
    rot_output_roi_x1 = int((diagonal-true_w)/2)
    rot_output_roi_y1 = int((diagonal-true_h)/2)
    rot_output_roi_x2 = rot_output_roi_x1+true_w
    rot_output_roi_y2 = rot_output_roi_y1+true_h
    return rot_output[rot_output_roi_y1:rot_output_roi_y2,rot_output_roi_x1:rot_output_roi_x2]

def rotate(img,*angles):
    len_angles = len(angles)
    if len_angles==0:
        raise ValueError('There must be at least 1 in \"angles\"')
    if len_angles == 1:
        return __rotate_one(img,angles[0])
    else:
        return [__rotate_one(img,a) for a in angles]
           
#Flip
def flipUD(img):
    return cv2.flip(img,0)
def flipLR(img):
    return cv2.flip(img,1)
#def flip(img):
#    return cv2.flip(img,random.randint(0,1))
    
#Crop
def __crop_one(img,rate):
    img_shape = img.shape
    is_1c = len(img.shape) == 2
    if rate>1:
        raise ValueError('rate > 1')
    elif rate<=0:
        raise ValueError('rate <= 0')
    ih,iw = img_shape[0], img_shape[1]
    h,w = ih*(1.0-rate)-1.0, iw*(1.0-rate)-1.0
    if w<=0 or h<=0:
        raise ValueError('crop image({0},{1}) into {2}% will be None'.format(img.shape[1],img.shape[0],rate*100))
    x1 = random.randint(0,int(w))
    y1 = random.randint(0,int(h))
    x2 = x1+ int(iw*rate)
    y2 = y1+ int(ih*rate)
    return img[y1:y2,x1:x2].copy()

def crop(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
    if len_rates == 1:
        return __crop_one(img,rates[0])
    else:
        return [__crop_one(img,r) for r in rates]

#Affine
def __affineX_one(img, rate):
    ih,iw = img.shape[0], img.shape[1]
    src = np.array([[0.0,0.0],[iw,0.0],[0.0,ih]],np.float32)
    dist = src.copy()
    offset_x = abs(rate * iw)
    if rate>0:
        dist[0,0] += offset_x
        dist[1,0] += offset_x
    else:
        dist[2,0] += offset_x
    r = cv2.getAffineTransform(src,dist)
    return cv2.warpAffine(img,r,(iw+int(offset_x),ih))


def __affineY_one(img, rate):
    ih,iw = img.shape[0], img.shape[1]
    src = np.array([[0.0,0.0],[iw,0.0],[0.0,ih]],np.float32)
    offset_y = abs(rate * ih)
    dist = src.copy()
    if rate>0:
        dist[1,1] += offset_y
    else:
        dist[0,1] += offset_y
        dist[2,1] += offset_y
    r = cv2.getAffineTransform(src,dist)
    return cv2.warpAffine(img,r,(iw,ih+int(offset_y)))
      
def affineX(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
    if len_rates == 1:
        return __affineX_one(img,rates[0])
    else:
        return [__affineX_one(img,r) for r in rates]

        
def affineY(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
    if len_rates == 1:
        return __affineY_one(img,rates[0])
    else:
        return [__affineY_one(img,r) for r in rates]

#Noise
def __add_noise_one(img,rate):
    noise = np.random.randint(-255,255,img.shape)
    added_noise = img.astype(np.float32)
    added_noise[:] += noise*rate
    added_noise[added_noise>255]=255
    added_noise[added_noise<0]=0
    added_noise = added_noise.astype(img.dtype)
    return added_noise

def add_noise(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
    if len_rates == 1 and len(img.shape)==3 and img.shape[0]==4:
        rimg = np.empty(img.shape, dtype=img.dtype)
        rimg[:,:,0:3] = __add_noise_one(img[:,:,0:3],rates[0])
        rimg[:,:,3] = img[:,:3]
        return rimg
    elif len_rates != 1 and len(img.shape)==3 and img.shape[0]==4:
        rimgs = [np.empty(img.shape, dtype=img.dtype) for r in rates]
        for rimg,rate in zip(rimgs,rates):
            rimg[:,:,0:3] = __add_noise_one(img[:,:,0:3],rate)
            rimg[:,:,3] = img[:,:3]
        return rimgs
    elif len_rates == 1 and (len(img.shape)!=3 or (img.shape[0]!=4)):
        return __add_noise_one(img,rates[0])
    else:
        return [__add_noise_one(img,r) for r in rates]

#Hue
def adjust_hue(img,*anlges):
    len_anlges = len(anlges)
    if len_anlges==0:
        raise ValueError('There must be at least 1 in \"anlges\"')
    bgr = img[:,:,0:3]
    hls = cv2.cvtColor(bgr,cv2.COLOR_BGR2HLS)
    tmp_hls = hls.copy()
    bgrs = []
    for anlge in anlges:
        tmp_h = hls[:,:,0]*2.0+anlge
        tmp_h -= tmp_h//360*360
        tmp_hls[:,:,0] = tmp_h/2
        tmp_bgr = cv2.cvtColor(tmp_hls,cv2.COLOR_HLS2BGR)
        bgrs.append(tmp_bgr)
    if len_anlges==1 and img.shape[2] >= 4:
        return np.concatenate((bgrs[0],img[:,:,3:]),axis=2)
    elif len_anlges!=1 and img.shape[2] >= 4:
        a = img[:,:,3:4]
        return [np.concatenate(x,a,axis=2) for x in bgrs]
    elif len_anlges==1 and img.shape[2] < 4:
        return bgrs[0]
    else:
        return bgrs
  
#Lightness
def adjust_lightness(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
    bgr = img[:,:,0:3]
    hls = cv2.cvtColor(bgr,cv2.COLOR_BGR2HLS)
    tmp_hls = hls.copy()
    bgrs = []
    for rate in rates:
        tmp_l = hls[:,:,1]/255.0+rate
        tmp_l = np.maximum(tmp_l,0.0)
        tmp_l = np.minimum(tmp_l,1.0)
        tmp_hls[:,:,1] = tmp_l*255.0
        tmp_bgr = cv2.cvtColor(tmp_hls,cv2.COLOR_HLS2BGR)
        bgrs.append(tmp_bgr)
    if len_rates==1 and img.shape[2] >= 4:
        return np.concatenate((bgrs[0],img[:,:,3:]),axis=2)
    elif len_rates!=1 and img.shape[2] >= 4:
        a = img[:,:,3:4]
        return [np.concatenate(x,a,axis=2) for x in bgrs]
    elif len_rates==1 and img.shape[2] < 4:
        return bgrs[0]
    else:
        return bgrs

#Saturation
def adjust_saturation(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
    bgr = img[:,:,0:3]
    hls = cv2.cvtColor(bgr,cv2.COLOR_BGR2HLS)
    tmp_hls = hls.copy()
    bgrs = []
    for rate in rates:
        tmp_s = hls[:,:,2]/255.0+rate
        tmp_s = np.maximum(tmp_s,0.0)
        tmp_s = np.minimum(tmp_s,1.0)
        tmp_hls[:,:,2] = tmp_s*255.0
        tmp_bgr = cv2.cvtColor(tmp_hls,cv2.COLOR_HLS2BGR)
        bgrs.append(tmp_bgr)
    if len_rates==1 and img.shape[2] >= 4:
        return np.concatenate((bgrs[0],img[:,:,3:]),axis=2)
    elif len_rates!=1 and img.shape[2] >= 4:
        a = img[:,:,3:4]
        return [np.concatenate(x,a,axis=2) for x in bgrs]
    elif len_rates==1 and img.shape[2] < 4:
        return bgrs[0]
    else:
        return bgrs

#Perspective
def __perspective_one(img,type,rate):
    img_cols = img.shape[1]
    img_rows = img.shape[0]
    src = np.array([[0.0,0.0],[img_cols,0.0],[img_cols,img_rows],[0.0,img_rows]],np.float32)
    dist = src.copy()
    offset_x = rate*img_cols
    offset_y = rate*img_rows
    if type == 'U':
        dist[0,0] += offset_x
        dist[1,0] -= offset_x
    elif type == 'UR':
        dist[1,0] -= offset_x
        dist[1,1] += offset_y
    elif type == 'R':
        dist[1,1] += offset_y
        dist[2,1] -= offset_y
    elif type == 'DR':
        dist[2,0] -= offset_x
        dist[2,1] -= offset_y
    elif type == 'D':
        dist[2,0] -= offset_x
        dist[3,0] += offset_x
    elif type == 'DL':
        dist[3,0] += offset_x 
        dist[3,1] -= offset_y
    elif type == 'L':
        dist[0,1] += offset_y
        dist[3,1] -= offset_y
    elif type == 'UL':
        dist[0,0] += offset_x 
        dist[0,1] += offset_y
    r = cv2.getPerspectiveTransform(src,dist)
    return cv2.warpPerspective(img,r,(img_cols, img_rows))

#def perspective(img,*rates):
#    len_rates = len(rates)
#    if len_rates==0:
#        raise ValueError('There must be at least 1 in \"rates\"')
#        return None
#    output = []
#    rand_type = random.choices(('U','UR','R','DR','D','DL','L','UL'),k=len(rates))
#    for t,rate in zip(rand_type,rates):
#        output.append(__perspective_one(img,t,rate))
#    if len_rates==1:
#        return output[0]
#    else:
#        return output

def perspectiveU(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
    if len_rates == 1:
        return __perspective_one(img,'U',rates[0])
    else:
        return [__perspective_one(img,'U',r) for r in rates]

def perspectiveUR(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
    if len_rates == 1:
        return __perspective_one(img,'UR',rates[0])
    else:
        return [__perspective_one(img,'UR',r) for r in rates]

def perspectiveR(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
    if len_rates == 1:
        return __perspective_one(img,'R',rates[0])
    else:
        return [__perspective_one(img,'R',r) for r in rates]

def perspectiveDR(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
    if len_rates == 1:
        return __perspective_one(img,'DR',rates[0])
    else:
        return [__perspective_one(img,'DR',r) for r in rates]

def perspectiveD(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
        return None
    if len_rates == 1:
        return __perspective_one(img,'D',rates[0])
    else:
        return [__perspective_one(img,'D',r) for r in rates]

def perspectiveDL(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
    if len_rates == 1:
        return __perspective_one(img,'DL',rates[0])
    else:
        return [__perspective_one(img,'DL',r) for r in rates]

def perspectiveL(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
    if len_rates == 1:
        return __perspective_one(img,'L',rates[0])
    else:
        return [__perspective_one(img,'L',r) for r in rates]

def perspectiveUL(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
    if len_rates == 1:
        return __perspective_one(img,'UL',rates[0])
    else:
        return [__perspective_one(img,'UL',r) for r in rates]


#class RESIZE_TYPE:
IMRESIZE_STRETCH = 0
IMRESIZE_ROUNDUP = 1
IMRESIZE_ROUNDUP_CROP = 2
IMRESIZE_ROUNDDOWN = 3
IMRESIZE_ROUNDDOWN_FILL_BLACK = 4
IMRESIZE_ROUNDDOWN_FILL_SELF = 5 

def _cv2_resize(src,dsize,interpolation=None):
    if interpolation is None:
        return cv2.resize(src,dsize)
    else:
        return cv2.resize(src,dsize,interpolation=interpolation)

def imResize(src,dsize,fx=None,fy=None,interpolation=None, rtype=None):
    if fx is not None and fy is not None:
        dsize = int(fx*dsize[0]),int(fy*dsize[1])
    elif (fx is None and fy is not None) or (fx is not None and fy is None):
        raise ValueError('invalid input, fx={0} and fy={1}'.format(fx,fy))
    if rtype is None:
        return _cv2_resize(src,dsize,interpolation=interpolation)
    elif rtype==IMRESIZE_STRETCH:
        return _cv2_resize(src,dsize,interpolation=interpolation)
    elif rtype==IMRESIZE_ROUNDUP:
        dw,dh = dsize
        ih,iw = src.shape[0],src.shape[1]
        fx = dw/iw
        fy = dh/ih
        if fy > fx:
            _dsize = (int(iw*fy), dh)
        else:
            _dsize = (dw, int(ih*fx))
        return _cv2_resize(src,_dsize,interpolation=interpolation)
    elif rtype==IMRESIZE_ROUNDUP_CROP:
        dw,dh = dsize
        ih,iw = src.shape[0],src.shape[1]
        fx = dw/iw
        fy = dh/ih
        if fy > fx:
            _dw = int(iw*fy)
            img = _cv2_resize(src,(_dw,dh),interpolation=interpolation)
            x0 =  int((_dw-dw)/2)
            return img[:,x0:x0+dw]
        else:
            _dh = int(ih*fx)
            img = _cv2_resize(src,(dw,_dh),interpolation=interpolation)
            y0 =  int((_dh-dh)/2)
            return img[y0:y0+dh,:]
    elif rtype==IMRESIZE_ROUNDDOWN:
        dw,dh = dsize
        ih,iw = src.shape[0],src.shape[1]
        fx = dw/iw
        fy = dh/ih
        if fx > fy:
            _dsize = (int(iw*fy), dh)
        else:
            _dsize = (dw, int(ih*fx))
        return _cv2_resize(src,_dsize,interpolation=interpolation)
    elif rtype==IMRESIZE_ROUNDDOWN_FILL_BLACK:
        dw,dh = dsize
        ih,iw = src.shape[0],src.shape[1]
        fx = dw/iw
        fy = dh/ih
        rimg_shape = list(src.shape)
        rimg_shape[0],rimg_shape[1] = dh,dw
        rimg_shape = tuple(rimg_shape)
        rimg = np.zeros(rimg_shape,src.dtype)
        if fx > fy:
            _dw = int(iw*fy)
            img = _cv2_resize(src,(_dw,dh),interpolation=interpolation)
            x0 =  int((dw-_dw)/2)
            rimg[:,x0:x0+_dw] = img
            return rimg
        else:
            _dh = int(ih*fx)
            img = _cv2_resize(src,(dw,_dh),interpolation=interpolation)
            y0 =  int((dh-_dh)/2)
            rimg[y0:y0+_dh,:] = img
            return rimg
    elif rtype==IMRESIZE_ROUNDDOWN_FILL_SELF:
        dw,dh = dsize
        ih,iw = src.shape[0],src.shape[1]
        fx = dw/iw
        fy = dh/ih
        rimg_shape = list(src.shape)
        rimg_shape[0],rimg_shape[1] = dh,dw
        rimg_shape = tuple(rimg_shape)
        rimg = np.empty(rimg_shape,src.dtype)
        if fx > fy:
            _dw = int(iw*fy)
            img = _cv2_resize(src,(_dw,dh),interpolation=interpolation)
            x0 =  int((dw-_dw)/2)
            x1 = x0+_dw
            rimg[:,x0:x1] = img
            if ih!=0 and iw!=0:
                if x0>0:
                    rimg[:,:x0] = img[:,0:1]
                if x1<dw:
                    rimg[:,x1:] = img[:,-2:-1]
            return rimg

        else:
            _dh = int(ih*fx)
            img = _cv2_resize(src,(dw,_dh),interpolation=interpolation)
            y0 =  int((dh-_dh)/2)
            y1 = y0+_dh
            rimg[y0:y1,:] = img
            if ih!=0 and iw!=0:
                if y0>0:
                    rimg[:y0] = img[0:1]
                if y1<dh:
                    rimg[y1:] = img[-2:-1]
            return rimg
    else:
        raise ValueError('invaild rtype={0}'.format(rtype))
    
def pntResize(pnt_src,img_src_shape,dsize,fx=None,fy=None,rtype=None):
    x,y = pnt_src
    dw,dh = dsize
    ih,iw = img_src_shape[0],img_src_shape[1]
    if fx is not None and fy is not None:
        dsize = int(fx*iw),int(fy*ih)
    elif (fx is None and fy is not None) or (fx is not None and fy is None):
        raise ValueError('invalid input, fx={0} and fy={1}'.format(fx,fy))
    else:
        fx,fy = dw/iw,dh/ih
    dsize = (dw,dh)
    if rtype is None:
        return (int(x*fx),int(y*fy))
    elif rtype==IMRESIZE_STRETCH:
        return (int(x*fx),int(y*fy))
    elif rtype==IMRESIZE_ROUNDUP:
        f = max(fx,fy)
        return (int(x*f),int(y*f))
    elif rtype==IMRESIZE_ROUNDUP_CROP:
        if fy > fx:
            _dw = int(iw*fy)
            x0 =  int((_dw-dw)/2)
            return (int(x*fy-x0),int(y*fy))
        else:
            _dh = int(ih*fx)
            y0 =  int((_dh-dh)/2)
            return (int(x*fx),int(y*fx-y0))
    elif rtype==IMRESIZE_ROUNDDOWN:
        f = min(fx,fy)
        return (int(x*f),int(y*f))
    elif rtype==IMRESIZE_ROUNDDOWN_FILL_BLACK:
        if fx > fy:
            _dw = int(iw*fy)
            x0 =  int((_dw-dw)/2)
            return (int(x*fy-x0),int(y*fy))
        else:
            _dh = int(ih*fx)
            y0 =  int((_dh-dh)/2)
            return (int(x*fx),int(y*fx-y0))
    elif rtype==IMRESIZE_ROUNDDOWN_FILL_SELF:
        if fx > fy:
            _dw = int(iw*fy)
            x0 =  int((_dw-dw)/2)
            return (int(x*fy-x0),int(y*fy))
        else:
            _dh = int(ih*fx)
            y0 =  int((_dh-dh)/2)
            return (int(x*fx),int(y*fx-y0))
    else:
        raise ValueError('invaild rtype={0}'.format(rtype))

def pntRecover(pnt_dst,img_src_shape,dsize,fx=None,fy=None,rtype=None):
    rx,ry = pnt_dst
    dw,dh = dsize
    ih,iw = img_src_shape[0],img_src_shape[1]
    if fx is None:
        fx = dsize[0]/iw
    else:
        dw = fx*iw
    if fy is None:
        fy = dsize[1]/ih
    else:
        dh = fy*ih
    dsize = (dw,dh)
    if rtype is None:
        return (int(rx/fx),int(ry/fy))
    elif rtype==IMRESIZE_STRETCH:
        return (int(rx/fx),int(ry/fy))
    elif rtype==IMRESIZE_ROUNDUP:
        f = max(fx,fy)
        return (int(rx/f),int(ry/f))
    elif rtype==IMRESIZE_ROUNDUP_CROP:
        if fy > fx:
            _dw = int(iw*fy)
            x0 =  int((_dw-dw)/2)
            return (int((rx+x0)/fy),int(ry/fy))
        else:
            _dh = int(ih*fx)
            y0 =  int((_dh-dh)/2)
            return (int(rx/fx),int((ry+y0)/fx))
    elif rtype==IMRESIZE_ROUNDDOWN:
        f = min(fx,fy)
        return (int(rx/f),int(ry/f))
    elif rtype==IMRESIZE_ROUNDDOWN_FILL_BLACK:
        if fx > fy:
            _dw = int(iw*fy)
            x0 =  int((_dw-dw)/2)
            return (int((rx+x0)/fy),int(ry/fy))
        else:
            _dh = int(ih*fx)
            y0 =  int((_dh-dh)/2)
            return (int(rx/fx),int((ry+y0)/fx))
    elif rtype==IMRESIZE_ROUNDDOWN_FILL_SELF:
        if fx > fy:
            _dw = int(iw*fy)
            x0 =  int((_dw-dw)/2)
            return (int((rx+x0)/fy),int(ry/fy))
        else:
            _dh = int(ih*fx)
            y0 =  int((_dh-dh)/2)
            return (int(rx/fx),int((ry+y0)/fx))
    else:
        raise ValueError('invaild rtype={0}'.format(rtype))





#class imResizer:

#    def __init__(self, resize_type, dsize, interpolation=cv2.INTER_LINEAR):
#        self._set(resize_type, dsize, interpolation)
#        return

#    def _set(self, resize_type, dsize, interpolation):
#        if resize_type != IMRESIZE_STRETCH and \
#           resize_type != IMRESIZE_ROUNDUP and \
#           resize_type != IMRESIZE_ROUNDUP_CROP and \
#           resize_type != IMRESIZE_ROUNDDOWN and \
#           resize_type != IMRESIZE_ROUNDDOWN_FILL_BLACK and \
#           resize_type != IMRESIZE_ROUNDDOWN_FILL_SELF:
#            raise ValueError('invaild RESIZE_TYPE with value={0}'.format(self._resize_type))
#        self._resize_type = resize_type
#        self._dsize = dsize
#        self._dw,self._dh = dsize
#        self._interpolation = interpolation
#        #(int(iw), int(ih)): (cv_w, cv_h, save_w, save_h, fx, bx, fy, by)
#        self._param = {}
#        return
    
#    def _transParam(self, iw, ih):
#        # output = (cv_w,cv_h), (bx,sw,by,sh), (fx,bx,fy,by)
#        found_param = self._param.get((iw,ih))
#        if found_param is not None:
#            return found_param
#        dw,dh = self._dw,self._dh
#        fx = dw/iw
#        fy = dh/ih
#        if self._resize_type==IMRESIZE_STRETCH:
#            output = (dw,dh), (0,dw,0,dh), (fx,0,fy,0)
#        elif self._resize_type==IMRESIZE_ROUNDUP:
#            if fy > fx:
#                f = fy
#                cv_w,cv_h = (int(iw*f), dh)
#            else:
#                f = fx
#                cv_w,cv_h = (dw, int(ih*f))
#            cv_w,cv_h = (int(iw*f),int(ih*f))
#            output = (cv_w,cv_h), (0,cv_w,0,cv_h), (f,0,f,0)
#        elif self._resize_type==IMRESIZE_ROUNDUP_CROP:
#            if fy > fx:
#                f = fy
#                cv_w,cv_h = (int(iw*f), dh)
#            else:
#                f = fx
#                cv_w,cv_h = (dw, int(ih*f))
#            output = (cv_w,cv_h), ((cv_w-dw)//2,dw,(cv_h-dh)//2,dh), (f,(dw-cv_w)//2,f,(dh-cv_h)//2)
#        elif self._resize_type==IMRESIZE_ROUNDDOWN:
#            f = min(fx, fy)
#            cv_w,cv_h = (int(iw*f),int(ih*f))
#            output = (cv_w,cv_h), (0,cv_w,0,cv_h), (f,0,f,0)
#        elif self._resize_type==IMRESIZE_ROUNDDOWN_FILL_BLACK or self._resize_type==IMRESIZE_ROUNDDOWN_FILL_SELF:
#            if fy > fx:
#                f = fx
#                cv_w,cv_h = (dw, int(ih*f))
#            else:
#                f = fy
#                cv_w,cv_h = (int(iw*f), dh)

#            output = (cv_w,cv_h), ((dw-cv_w)//2,dw,(dh-cv_h)//2,dh), (f,(dw-cv_w)//2,f,(dh-cv_h)//2)
#        else:
#            output = (dw,dh), (0,dw,0,dh), (fx,0,fy,0)
#        self._param[iw, ih] = output
#        return output

#    # TODO
#    def _transParam(self, iw, ih):
#        # output = (cv_w, cv_h, save_w, save_h, fx, bx, fy, by)
#        found_param = self._param.get((iw,ih))
#        if found_param is not None:
#            return found_param
#        dw,dh = self._dw,self._dh
#        fx = dw/iw
#        fy = dh/ih
#        if self._resize_type==IMRESIZE_STRETCH:
#            output = (dw, dh, dw, dh, fx, 0, fy, 0)
#        elif self._resize_type==IMRESIZE_ROUNDUP:
#            f = fy if fy > fx else fx
#            cv_w,cv_h = (int(iw*f),int(ih*f))
#            output = (cv_w, cv_h, cv_w, cv_h, f, 0, f, 0)
#        elif self._resize_type==IMRESIZE_ROUNDUP_CROP:
#            if fy > fx:
#                f = fy
#                cv_w,cv_h = (int(iw*f), dh)
#            else:
#                f = fx
#                cv_w,cv_h = (dw, int(ih*f))
#            output = (cv_w, cv_h, dw, dh, f, (dw-cv_w)/2, f, (dh-cv_h)/2)
#        elif self._resize_type==IMRESIZE_ROUNDDOWN:
#            f = fx if fy > fx else fy
#            cv_w,cv_h = (int(iw*f),int(ih*f))
#            output = (cv_w, cv_h, cv_w, cv_h, f, 0, f, 0)
#        elif self._resize_type==IMRESIZE_ROUNDDOWN_FILL_BLACK or self._resize_type==IMRESIZE_ROUNDDOWN_FILL_SELF:
#            if fy > fx:
#                f = fx
#                cv_w,cv_h = (dw, int(ih*f))
#            else:
#                f = fy
#                cv_w,cv_h = (int(iw*f), dh)

#            output = (cv_w, cv_h, iw, ih, f, (dw-cv_w)/2, f,(dh-cv_h)/2)
#        else:
#            output = (dw, dh, dw, dh, fx, 0,fy, 0)
#        self._param[iw, ih] = output
#        return output

#    def imResize(self,src):
#        # param = (cv_w, cv_h, save_w, save_h, fx, bx, fy, by)
#        ih,iw = int(src.shape[0]),int(src.shape[1])
#        cv_w, cv_h, save_w, save_h, _, bx, _, by = self._transParam(iw,ih)
#        if self._resize_type==IMRESIZE_STRETCH \
#            or self._resize_type==IMRESIZE_ROUNDUP \
#            or self._resize_type==IMRESIZE_ROUNDDOWN:
#            return _cv2_resize(src, dsize = (cv_w, cv_h), interpolation = self._interpolation)
#        elif self._resize_type==IMRESIZE_ROUNDUP_CROP:
#            cvrimg = _cv2_resize(src, dsize = (cv_w, cv_h), interpolation = self._interpolation)
#            if cv_w == save_w:
#                ymin = int(-by)
#                return cvrimg[:,ymin:ymin+save_h]
#            else:
#                xmin = int(-bx)
#                return cvrimg[xmin:xmin+save_w]
#        elif self._resize_type==IMRESIZE_ROUNDDOWN_FILL_BLACK:
#            cvrimg = _cv2_resize(src, dsize = (cv_w, cv_h), interpolation = self._interpolation)
#            img_to_save_shape = list(src.shape)
#            img_to_save_shape[0], img_to_save_shape[1] = save_h, save_w
#            img_to_save = np.zeros(img_to_save_shape, src.dtype)
#            if cv_w == save_w:
#                ymin = int(by)
#                img_to_save[:,ymin:ymin+save_h] = cvrimg
#                return img_to_save
#            else:
#                xmin = int(bx)
#                img_to_save[xmin:xmin+save_w] = cvrimg
#                return img_to_save
#        elif self._resize_type==IMRESIZE_ROUNDDOWN_FILL_SELF:
#            cvrimg = _cv2_resize(src, dsize = (cv_w, cv_h), interpolation = self._interpolation)
#            img_to_save_shape = list(src.shape)
#            img_to_save_shape[0], img_to_save_shape[1] = save_h, save_w
#            img_to_save = np.empty(img_to_save_shape, src.dtype)
#            if cv_w == save_w:
#                ymin = int(by)
#                ymax_add_1 = ymin+save_h
#                img_to_save[:,ymin:ymax_add_1] = cvrimg
#                img_to_save[:,:ymin] = img_to_save[:,ymin]
#                img_to_save[:,ymax_add_1:] = img_to_save[:,ymax_add_1]
#                return img_to_save
#            else:
#                xmin = int(bx)
#                xmax_add_1 = xmin+save_w
#                img_to_save[xmin:xmax_add_1] = cvrimg
#                img_to_save[:xmin] = img_to_save[xmin]
#                img_to_save[xmax_add_1:] = img_to_save[xmax_add_1]
#                return img_to_save
#        else:
#            return _cv2_resize(src, dsize = (cv_w, cv_h), interpolation = self._interpolation)       

#    def pntResize(self,pnt_src,img_src_shape):
#        _, _, _, _, fx, bx, fy, by = self._transParam(int(img_src_shape[1]),int(img_src_shape[0]))
#        return (int(pnt_src[0]*fx+bx),int(pnt_src[1]*fy+by))

#    def pntRecover(self,pnt_dst,img_src_shape):
#        _, _, _, _, fx, bx, fy, by = self._transParam(int(img_src_shape[1]),int(img_src_shape[0]))
#        return (int((pnt_dst[0]-bx)/fx),int((pnt_dst[1]-by)/fy))

class cvRect():
    def __init__(self, _x, _y, _w, _h):
        if _w<0 or _h<0:
            raise ValueError('init cvRect Error : width < 0 or height < 0')
        self.x = int(_x)
        self.y = int(_y)
        self.width = int(_w)
        self.height = int(_h)

    def area(self):
        return self.width * self.height

    def br(self):
        return (self.x+self.width,self.y+self.height)

    def contains(self,pt):
        return self.x <= pt[0] and\
                pt[0] < self.x + self.width and\
                self.y <= pt[1] and\
                pt[1] < self.y + self.height

    def empty(self):
        return self.width<=0 or self.height<=0
    
    def tl(self):
        return (self.x,self.y) 

    def roi(self,img):
        return img[self.y:self.height,self.x:self.width]

    def tolist(self):
        return [self.x,self.y,self.width,self.height]

    def __getitem__(self,i):
        if i==0 or i==-4:
            return self.x
        elif i==1 or i==-3:
            return self.y
        elif i==2 or i==-2:
            return self.width
        elif i==3 or i==-1:
            return self.height
        else:
            raise IndexError('cvRect index out of range')

    def tobndbox(self):
        import voc as pbvoc
        return pbvoc.bndbox(self.x,self.y,self.x+self.width-1,self.y+self.height-1)

__imshow_img_dict = dict()
__imshow_destroy_set = set()
__imshow_p_show_thread = None
__imshow_waitKey_delay = 10
__imshow_key_value = -1


# once it runs, all images can only show in this thread function
def show_thread_fun():
    global __imshow_img_dict
    global __imshow_destroy_set
    global __imshow_waitKey_delay
    global __imshow_key_value
    while 1:
        num_win_destroy = 0
        win_destroy_keys = list(__imshow_destroy_set)
        for win in win_destroy_keys:
            if(__imshow_img_dict.get(win) is not None):
                __imshow_img_dict.pop(win)
            try:
                cv2.destroyWindow(win)
                num_win_destroy += 1
            except Exception as e:
                # print(e)
                __imshow_destroy_set.remove(win)
        # show
        num_win_showing = 0
        win_show_keys = list(__imshow_img_dict.keys())
        for win in win_show_keys:
            img = __imshow_img_dict[win]
            try:
                cv2.imshow(win,img)
                num_win_showing += 1
            except Exception as e:
                print(e)
                __imshow_destroy_set.add(win)
                continue
        if(num_win_showing==0 and num_win_destroy==0):
            time.sleep(0.01)
        else:
            __imshow_key_value = cv2.waitKey(__imshow_waitKey_delay)


def imshow(winname,mat):
    global __imshow_img_dict
    global __imshow_p_show_thread
    if __imshow_p_show_thread is None or __imshow_p_show_thread.is_alive()==False:
        __imshow_brk_show_thread = False
        __imshow_p_show_thread = threading.Thread(target=show_thread_fun,daemon=True)
        __imshow_p_show_thread.start()
    # low donw the CPU usage
    # this_img_to_show = np.zeros(mat.shape,mat.dtype)
    # this_img_to_show = this_img_to_show + mat
    this_img_to_show = mat.copy()
    __imshow_img_dict[winname] = this_img_to_show
    return

def destroyWindow(winname):
    global __imshow_destroy_set
    __imshow_destroy_set.add(winname)
    return

def destroyAllWindows():
    global __imshow_img_dict
    global __imshow_destroy_set
    __imshow_destroy_set.update(__imshow_img_dict.keys())
    return

def waitKey(delay=None):
    global __imshow_waitKey_delay
    global __imshow_key_value
    __imshow_key_value = -1
    __imshow_waitKey_delay = delay
    if delay is None or delay==0:
        key = -1
        while key == -1:
            key = __imshow_key_value
            time.sleep(0.001)
        return key
    else:
        begin = time.time()
        end = time.time()
        key = -1
        while (end-begin)*1000<delay:
            key = __imshow_key_value
            if key !=-1:
                break
            else:
                end = time.time()
        return key
        

if __name__=='__main__':
    ##################################################################
    #test module
    ##################################################################
    import sys
    sys.path.append('../')
    print(sys.path)
    input()
    import pyBoost as pb

    def test_imResize():
        img = pb.read_color_ring()
        pnt_src = (random.randint(0,img.shape[1]-1),random.randint(0,img.shape[0]-1))
        cv2.destroyWindow('img')
        cv2.imshow('img',cv2.circle(img.copy(),pnt_src,3,(0,0,0),3))
        cv2.waitKey(0)
        dsize = (300,400)
        color = (0,0,0)
        interpolation = cv2.INTER_CUBIC
        None_img = imResize(img,dsize,interpolation=interpolation)
        None_pnt = pntResize(pnt_src,img.shape,dsize)
        cv2.destroyWindow('None_img_{0}'.format(str(dsize)))
        cv2.imshow('None_img_{0}'.format(str(dsize)),cv2.circle(None_img,None_pnt,3,color,3))
        cv2.waitKey(0)
        str_img = imResize(img,dsize,interpolation=interpolation,rtype=IMRESIZE_STRETCH)
        str_pnt = pntResize(pnt_src,img.shape,dsize,rtype=IMRESIZE_STRETCH)
        cv2.destroyWindow('str_img_{0}'.format(str(dsize)))
        cv2.imshow('str_img_{0}'.format(str(dsize)),cv2.circle(str_img,str_pnt,3,color,3))
        cv2.waitKey(0)
        up_img = imResize(img,dsize,interpolation=interpolation,rtype=IMRESIZE_ROUNDUP)
        up_pnt = pntResize(pnt_src,img.shape,dsize,rtype=IMRESIZE_ROUNDUP)
        cv2.destroyWindow('up_img_{0}'.format(str(dsize)))
        cv2.imshow('up_img_{0}'.format(str(dsize)),cv2.circle(up_img,up_pnt,3,color,3))
        cv2.waitKey(0)
        up_crop_img = imResize(img,dsize,interpolation=interpolation,rtype=IMRESIZE_ROUNDUP_CROP)
        up_crop_pnt = pntResize(pnt_src,img.shape,dsize,rtype=IMRESIZE_ROUNDUP_CROP)
        cv2.destroyWindow('up_crop_img_{0}'.format(str(dsize)))
        cv2.imshow('up_crop_img_{0}'.format(str(dsize)),cv2.circle(up_crop_img,up_crop_pnt,3,color,3))
        cv2.waitKey(0)
        down_img = imResize(img,dsize,interpolation=interpolation,rtype=IMRESIZE_ROUNDDOWN)
        down_pnt = pntResize(pnt_src,img.shape,dsize,rtype=IMRESIZE_ROUNDDOWN)
        cv2.destroyWindow('down_img_{0}'.format(str(dsize)))
        cv2.imshow('down_img_{0}'.format(str(dsize)),cv2.circle(down_img,down_pnt,3,color,3))
        cv2.waitKey(0)
        down_black_img = imResize(img,dsize,interpolation=interpolation,rtype=IMRESIZE_ROUNDDOWN_FILL_BLACK)
        down_blac_pnt = pntResize(pnt_src,img.shape,dsize,rtype=IMRESIZE_ROUNDDOWN_FILL_BLACK)
        cv2.destroyWindow('down_black_img_{0}'.format(str(dsize)))
        cv2.imshow('down_black_img_{0}'.format(str(dsize)),cv2.circle(down_black_img,down_blac_pnt,3,color,3))
        cv2.waitKey(0)
        down_self_img = imResize(img,dsize,interpolation=interpolation,rtype=IMRESIZE_ROUNDDOWN_FILL_SELF)
        down_self_pnt = pntResize(pnt_src,img.shape,dsize,rtype=IMRESIZE_ROUNDDOWN_FILL_SELF)
        cv2.destroyWindow('down_self_img_{0}'.format(str(dsize)))
        cv2.imshow('down_self_img_{0}'.format(str(dsize)),cv2.circle(down_self_img,down_self_pnt,3,color,3))
        cv2.waitKey(0)
        dsize = (400,300)
        interpolation = None
        None_img = imResize(img,dsize,interpolation=interpolation)
        None_pnt = pntResize(pnt_src,img.shape,dsize)
        cv2.destroyWindow('None_img_{0}'.format(str(dsize)))
        cv2.imshow('None_img_{0}'.format(str(dsize)),cv2.circle(None_img,None_pnt,3,color,3))
        cv2.waitKey(0)
        str_img = imResize(img,dsize,interpolation=interpolation,rtype=IMRESIZE_STRETCH)
        str_pnt = pntResize(pnt_src,img.shape,dsize,rtype=IMRESIZE_STRETCH)
        cv2.destroyWindow('str_img_{0}'.format(str(dsize)))
        cv2.imshow('str_img_{0}'.format(str(dsize)),cv2.circle(str_img,str_pnt,3,color,3))
        cv2.waitKey(0)
        up_img = imResize(img,dsize,interpolation=interpolation,rtype=IMRESIZE_ROUNDUP)
        up_pnt = pntResize(pnt_src,img.shape,dsize,rtype=IMRESIZE_ROUNDUP)
        cv2.destroyWindow('up_img_{0}'.format(str(dsize)))
        cv2.imshow('up_img_{0}'.format(str(dsize)),cv2.circle(up_img,up_pnt,3,color,3))
        cv2.waitKey(0)
        up_crop_img = imResize(img,dsize,interpolation=interpolation,rtype=IMRESIZE_ROUNDUP_CROP)
        up_crop_pnt = pntResize(pnt_src,img.shape,dsize,rtype=IMRESIZE_ROUNDUP_CROP)
        cv2.destroyWindow('up_crop_img_{0}'.format(str(dsize)))
        cv2.imshow('up_crop_img_{0}'.format(str(dsize)),cv2.circle(up_crop_img,up_crop_pnt,3,color,3))
        cv2.waitKey(0)
        down_img = imResize(img,dsize,interpolation=interpolation,rtype=IMRESIZE_ROUNDDOWN)
        down_pnt = pntResize(pnt_src,img.shape,dsize,rtype=IMRESIZE_ROUNDDOWN)
        cv2.destroyWindow('down_img_{0}'.format(str(dsize)))
        cv2.imshow('down_img_{0}'.format(str(dsize)),cv2.circle(down_img,down_pnt,3,color,3))
        cv2.waitKey(0)
        down_black_img = imResize(img,dsize,interpolation=interpolation,rtype=IMRESIZE_ROUNDDOWN_FILL_BLACK)
        down_black_pnt = pntResize(pnt_src,img.shape,dsize,rtype=IMRESIZE_ROUNDDOWN_FILL_BLACK)
        cv2.destroyWindow('down_black_img_{0}'.format(str(dsize)))
        cv2.imshow('down_black_img_{0}'.format(str(dsize)),cv2.circle(down_black_img,down_black_pnt,3,color,3))
        cv2.waitKey(0)
        down_self_img = imResize(img,dsize,interpolation=interpolation,rtype=IMRESIZE_ROUNDDOWN_FILL_SELF)
        down_self_pnt = pntResize(pnt_src,img.shape,dsize,rtype=IMRESIZE_ROUNDDOWN_FILL_SELF)
        cv2.destroyWindow('down_self_img_{0}'.format(str(dsize)))
        cv2.imshow('down_self_img_{0}'.format(str(dsize)),cv2.circle(down_self_img,down_self_pnt,3,color,3))
        cv2.waitKey(0)


    #####################################################################

    def test_imshow():
        import cv2
        import numpy as np
        img = cv2.imread(r'C:\Users\admin\Desktop\chess.jpg')
        pb.img.imshow('img',img)
        print(3)
        time.sleep(1)
        print(2)
        time.sleep(1)
        print(1)
        key = 0
        while key!=27:
            pb.img.imshow('img',img)
            key = pb.img.waitKey(10)
            print(key)
        key = 0
        while key!=27:
            pb.img.imshow('img1',img)
            key = pb.img.waitKey(0)
            print(key)

    #####################################################################
    def test_augment():
        import cv2
        import numpy as np 
        import argparse
        import os

        global img
        global opt_bar
        global opt_max_number
        global param_bar
        global param
        global aug_mat

        user_img_view_size = (750,450)
        # user_img_view_size = (1080,720)
        img = np.zeros([450,750,3],np.uint8)
        opt_bar = 0
        opt_max_number = 18
        param_bar = 10
        param = 10
        aug_mat = img.copy()

        def on_trackbar_share_opt_bar(value):
            global img
            global opt_bar
            global opt_max_number
            global param_bar
            global param
            global aug_mat

            opt_bar = value

            aug_text = 'AUG_CONFIG = '
            fanwei = 'Range = '
            shuoming = 'Explanation: '
            cv2.imshow('original', img)
            if(opt_bar == 0):
                #Rotate
                param = 1.0 / 100.0*360.0 * param_bar
                aug_mat = pb.img.rotate(img, param)
                cv2.putText(aug_mat, aug_text+'Rotate('+str(param)+')', (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "(---,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming+"Rotate by one angle (degree)", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 1):
                #FlipUD
                aug_mat = pb.img.flipUD(img)
                cv2.putText(aug_mat, aug_text + "FlipUD", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "(NULL)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Flip up down", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 2):
                #FlipLR
                aug_mat = pb.img.flipLR(img)
                cv2.putText(aug_mat, aug_text + "FlipLR", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "(NULL)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Flip left right", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 3):
                #Crop
                param = 1.0 / 100.0* param_bar
                if(param<0.01):
                     param = 0.01
                aug_mat = pb.img.crop(img, param)
                cv2.putText(aug_mat, aug_text + "Crop(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "(0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Random crop image by area percentage", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 4):
                #AffineX
                param = 1.0 / 50.0* (param_bar - 50)
                aug_mat = pb.img.affineX(img, param)
                cv2.putText(aug_mat, aug_text + "AffineX(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[-1,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "X axis affine", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 5):
                #AffineY
                param = 1.0 / 50.0* (param_bar - 50)
                aug_mat = pb.img.affineY(img, param)
                cv2.putText(aug_mat, aug_text + "AffineY(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[-1,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Y axis affine", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 6):
                #Noise
                param = 1.0 / 100.0* (param_bar)
                aug_mat = pb.img.add_noise(img, param)
                cv2.putText(aug_mat, aug_text + "Noise(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Add gauss noise", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 7):
                #Hue
                param = 1.0 / 100.0*360.0 * param_bar
                aug_mat = pb.img.adjust_hue(img, param)
                cv2.putText(aug_mat, aug_text + "Hue(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "(---,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Hue angle (degree)", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 8):
                #Saturation
                param = 1.0 / 50.0* (param_bar - 50)
                aug_mat = pb.img.adjust_saturation(img, param)
                cv2.putText(aug_mat, aug_text + "Saturation(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Saturation", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 9):
                #Lightness
                param = 1.0 / 50.0* (param_bar - 50)
                aug_mat = pb.img.adjust_lightness(img, param)
                cv2.putText(aug_mat, aug_text + "Lightness(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Lightness", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 10):
                #PerspectiveUL
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveUL(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveUL(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "UL Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 11):
                #PerspectiveU
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveU(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveU(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "U  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 12):
                #PerspectiveUR
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveUR(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveUR(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "UR Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 13):
                #PerspectiveL
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveL(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveL(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "L  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 14):
                #PerspectiveR
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveR(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveR(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "R  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 15):
                #PerspectiveDL
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveDL(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveDL(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "DL Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 16):
                #PerspectiveD
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveD(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveD(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "D  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 17):
                #PerspectiveDR
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveDR(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveDR(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "DR Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            #elif(opt_bar == 18):
            #    #Distort
            #    param = 1.0 / 20.0* param_bar
            #    pb.img.imDistort(img, aug_mat, param)
            #    cv2.putText(aug_mat, aug_text + "Distort(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
            #    cv2.putText(aug_mat, fanwei + "[0,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
            #    cv2.putText(aug_mat, shuoming + "Add distort", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
            #    cv2.imshow("augmented", aug_mat)
            return

        def on_trackbar_share_param_bar(value):
            global img
            global opt_bar
            global opt_max_number
            global param_bar
            global param
            global aug_mat

            param_bar = value

            aug_text = 'AUG_CONFIG = '
            fanwei = 'Range = '
            shuoming = 'Explanation: '
            cv2.imshow('original', img)
            if(opt_bar == 0):
                #Rotate
                param = 1.0 / 100.0*360.0 * param_bar
                aug_mat = pb.img.rotate(img, param)
                cv2.putText(aug_mat, aug_text+'Rotate('+str(param)+')', (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "(---,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming+"Rotate by one angle (degree)", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 1):
                #FlipUD
                aug_mat = pb.img.flipUD(img)
                cv2.putText(aug_mat, aug_text + "FlipUD", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "(NULL)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Flip up down", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 2):
                #FlipLR
                aug_mat = pb.img.flipLR(img)
                cv2.putText(aug_mat, aug_text + "FlipLR", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "(NULL)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Flip left right", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 3):
                #Crop
                param = 1.0 / 100.0* param_bar
                if(param<0.01):
                     param = 0.01
                aug_mat = pb.img.crop(img, param)
                cv2.putText(aug_mat, aug_text + "Crop(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "(0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Random crop image by area percentage", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 4):
                #AffineX
                param = 1.0 / 50.0* (param_bar - 50)
                aug_mat = pb.img.affineX(img, param)
                cv2.putText(aug_mat, aug_text + "AffineX(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[-1,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "X axis affine", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 5):
                #AffineY
                param = 1.0 / 50.0* (param_bar - 50)
                aug_mat = pb.img.affineY(img, param)
                cv2.putText(aug_mat, aug_text + "AffineY(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[-1,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Y axis affine", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 6):
                #Noise
                param = 1.0 / 100.0* (param_bar)
                aug_mat = pb.img.add_noise(img, param)
                cv2.putText(aug_mat, aug_text + "Noise(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Add gauss noise", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 7):
                #Hue
                param = 1.0 / 100.0*360.0 * param_bar
                aug_mat = pb.img.adjust_hue(img, param)
                cv2.putText(aug_mat, aug_text + "Hue(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "(---,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Hue angle (degree)", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 8):
                #Saturation
                param = 1.0 / 50.0* (param_bar - 50)
                aug_mat = pb.img.adjust_saturation(img, param)
                cv2.putText(aug_mat, aug_text + "Saturation(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Saturation", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 9):
                #Lightness
                param = 1.0 / 50.0* (param_bar - 50)
                aug_mat = pb.img.adjust_lightness(img, param)
                cv2.putText(aug_mat, aug_text + "Lightness(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Lightness", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 10):
                #PerspectiveUL
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveUL(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveUL(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "UL Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 11):
                #PerspectiveU
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveU(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveU(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "U  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 12):
                #PerspectiveUR
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveUR(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveUR(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "UR Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 13):
                #PerspectiveL
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveL(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveL(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "L  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 14):
                #PerspectiveR
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveR(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveR(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "R  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 15):
                #PerspectiveDL
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveDL(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveDL(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "DL Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 16):
                #PerspectiveD
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveD(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveD(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "D  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 17):
                #PerspectiveDR
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveDR(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveDR(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "DR Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            #elif(opt_bar == 18):
            #    #Distort
            #    param = 1.0 / 20.0* param_bar
            #    pb.img.imDistort(img, aug_mat, param)
            #    cv2.putText(aug_mat, aug_text + "Distort(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
            #    cv2.putText(aug_mat, fanwei + "[0,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
            #    cv2.putText(aug_mat, shuoming + "Add distort", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
            #    cv2.imshow("augmented", aug_mat)
            return
        
        #argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--img',type=str, default='', help='Image you want to view')
        opt = parser.parse_args()

        # load image to show
        if(opt.img == ''):
            org_img = pb.read_color_ring()
        else:
            img_path = opt.img
            org_img = cv2.imread(img_path)
            if org_img is None or os.path.isfile(img_path)==False:
                sys.exit('Can\'t find file: {0}'.format(img_path))
        img = pb.img.imResize(org_img, user_img_view_size, rtype=pb.img.IMRESIZE_ROUNDUP)

        # show and run
        key_value = 0
        while(1):
            cv2.namedWindow('original',cv2.WINDOW_AUTOSIZE|cv2.WINDOW_KEEPRATIO|cv2.WINDOW_GUI_EXPANDED)
            cv2.createTrackbar('option','original',opt_bar,opt_max_number-1,on_trackbar_share_opt_bar)
            cv2.createTrackbar('param','original',param_bar,100,on_trackbar_share_param_bar)
            on_trackbar_share_opt_bar(opt_bar)
            on_trackbar_share_param_bar(param_bar)

            key_value = cv2.waitKey(0)
        
            if (key_value == ord('d') or key_value == ord('D')):
                opt_bar +=1
                if (opt_bar >= opt_max_number):
                    opt_bar = 0
            elif (key_value == ord('a') or key_value == ord('A')):
                opt_bar -=1
                if (opt_bar < 0):
                    opt_bar = opt_max_number - 1
            elif(key_value == ord('s') or key_value == ord('S')):
                param_bar +=10
                if(param_bar>100):
                    param_bar-=100
            elif(key_value == ord('W') or key_value == ord('w')):
                param_bar -=10
                if(param_bar<0):
                    param_bar+=100

            if (key_value == 27 or key_value == ord('\r')):
                break
        cv2.destroyAllWindows()


    #####################################################################
    save_folder_name = 'pyBoost_test_output'
    test_imResize()
    # test_imshow()
    # test_augment() 



