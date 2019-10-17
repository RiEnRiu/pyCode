#-*-coding:utf-8-*-
import cv2
import numpy as np
import math
import threading
import random
import time

#Rotate
def __rotate_one(img,angle):
    is_1c = len(img.shape) == 2
    diagonal = int(math.sqrt(img.shape[0]*img.shape[0]+img.shape[1]*img.shape[1])) + 1
    bg_shape = list(img.shape)
    bg_shape[0] = diagonal
    bg_shape[1] = diagonal
    bg = np.zeros(bg_shape,img.dtype)
    bg_roi_x1 = int((diagonal-img.shape[1])/2)
    bg_roi_y1 = int((diagonal-img.shape[0])/2)
    bg_roi_x2 = bg_roi_x1+img.shape[1]
    bg_roi_y2 = bg_roi_y1+img.shape[0]
    if is_1c:
        bg[bg_roi_y1:bg_roi_y2,bg_roi_x1:bg_roi_x2] = img
    else:
        bg[bg_roi_y1:bg_roi_y2,bg_roi_x1:bg_roi_x2,:] = img
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

    if is_1c:
        return rot_output[rot_output_roi_y1:rot_output_roi_y2,rot_output_roi_x1:rot_output_roi_x2]
    else:
        return rot_output[rot_output_roi_y1:rot_output_roi_y2,rot_output_roi_x1:rot_output_roi_x2,:]

def rotate(img,*angles):
    len_angles = len(angles)
    if len_angles==0:
        raise ValueError('There must be at least 1 in \"angles\"')
        return None
    output = []
    for angle in angles:
        output.append(__rotate_one(img,angle))
    if len_angles == 1:
        return output[0]
    else:
        return output
           
#Flip
def flipUD(img):
    return cv2.flip(img,0)
def flipLR(img):
    return cv2.flip(img,1)
def flip(img):
    return cv2.flip(img,random.randint(0,1))
    
#Crop
def __crop_one(img,rate):
    img_shape = img.shape
    is_1c = len(img.shape) == 2
    if rate>1:
        rate = 1.0
    if rate<0:
        rate = 0.0
        
    max_shape = [img_shape[0] * (1.0-rate)-1.0,img_shape[1] * (1.0-rate)-1.0]
    if max_shape[0]<=0 or max_shape[1]<=0:
        return img.copy()
    x1 = random.randint(0,int(max_shape[1]))
    y1 = random.randint(0,int(max_shape[0]))
    x2 = x1+ int(img_shape[1]*rate)
    y2 = y1+ int(img_shape[0]*rate)
    if is_1c:
        return img[y1:y2,x1:x2].copy()
    else:
        return img[y1:y2,x1:x2,:].copy()

def crop(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
        return None
    output = []
    for rate in rates:
        output.append(__crop_one(img,rate))
    if len_rates == 1:
        return output[0]
    else:
        return output

#Affine
def __affine_one(img,is_X,rate):
    img_cols = img.shape[1]
    img_rows = img.shape[0]
    src = np.array([[0.0,0.0],[img_cols,0.0],[0.0,img_rows]],np.float32)
    if is_X:
        if rate<0:
            offset_x = int(-rate * img_cols)
            dist = np.array([[src[0][0]+offset_x,src[0][1]],\
            [src[1][0]+offset_x,src[1][1]],[src[2][0],src[2][1]]],\
            np.float32)
            r = cv2.getAffineTransform(src,dist)
            output = cv2.warpAffine(img,r,(img_cols+offset_x,img_rows))
        else:
            offset_x = int(rate * img_cols)
            dist = np.array([[src[0][0],src[0][1]],\
            [src[1][0],src[1][1]],[src[2][0]+offset_x,src[2][1]]],\
            np.float32)
            r = cv2.getAffineTransform(src,dist)
            output = cv2.warpAffine(img,r,(img_cols+offset_x,img_rows))
    else:
        if rate<0:
            offset_y = int(-rate * img_rows)
            dist = np.array([[src[0][0],src[0][1]+offset_y],\
            [src[1][0],src[1][1]],[src[2][0],src[2][1]+offset_y]],\
            np.float32)
            r = cv2.getAffineTransform(src,dist)
            output = cv2.warpAffine(img,r,(img_cols,img_rows+offset_y))
        else:
            offset_y = int(rate * img_rows)
            dist = np.array([[src[0][0],src[0][1]],\
            [src[1][0],src[1][1]+offset_y],[src[2][0],src[2][1]]],\
            np.float32)
            r = cv2.getAffineTransform(src,dist)
            output = cv2.warpAffine(img,r,(img_cols,img_rows+offset_y))
    return output

def affine(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
        return None
    output = []
    for rate in rates:
        output.append(__affine_one(img,random.randint(0,1),rate))
    if len_rates == 1:
        return output[0]
    else:
        return output
      
def affineX(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
        return None
    output = []
    for rate in rates:
        output.append(__affine_one(img,True,rate))
    if len_rates == 1:
        return output[0]
    else:
        return output

        
def affineY(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
        return None
    output = []
    for rate in rates:
        output.append(__affine_one(img,False,rate))
    if len_rates == 1:
        return output[0]
    else:
        return output

#Noise
def __add_noise_one(img,rate,is_4c):
    if is_4c is not True:
        noise = np.random.randint(-255,255,img.size)
        f_noise = noise.reshape(img.shape).astype(np.float32)
        added_noise = (f_noise*rate+img)
        added_noise[added_noise>255]=255
        added_noise[added_noise<0]=0
        return added_noise.astype(img.dtype)
    else:
        noise = np.random.randint(-255,255,(img.shape[0],img.shape[1],3))
        f_noise = noise.reshape((img.shape[0],img.shape[1],3)).astype(np.float32)
        added_noise = img.copy()
        added_noise[:,:,0:3] = (f_noise*rate+img[:,:,0:3]).astype(np.uint8)
        added_noise[added_noise>255]=255
        added_noise[added_noise<0]=0
        return added_noise

def add_noise(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
        return None
    is_4c = len(img.shape) == 3
    if is_4c:
        is_4c = img.shape[2] == 4
    output = []
    for rate in rates:
        output.append(__add_noise_one(img,rate,is_4c))
    if len_rates == 1:
        return output[0]
    else:
        return output

#Hue
def adjust_hue(img,*anlges):
    len_anlges = len(anlges)
    if len_anlges==0:
        raise ValueError('There must be at least 1 in \"anlges\"')
        return None

    is_4c = img.shape[2] == 4
    if is_4c:
        bgr = img[:,:,0:3]
        one_output = np.zeros(img.shape,img.dtype)
        one_output[:,:,3] = img[:,:,3]
    else:
        bgr = img
        one_output = np.zeros(img.shape,img.dtype)
        
    hls = cv2.cvtColor(bgr,cv2.COLOR_BGR2HLS)
        
    output = []
    for anlge in anlges:
        temp_hls = hls.copy()
        f_temp_h = temp_hls[:,:,0]*2.0+anlge
        f_temp_h[f_temp_h>360] = f_temp_h[f_temp_h>360]-360
        f_temp_h[f_temp_h<0] = f_temp_h[f_temp_h<0]+360
        f_temp_h /=2.0
        temp_hls[:,:,0] = f_temp_h
        one_output[:,:,0:3] = cv2.cvtColor(temp_hls,cv2.COLOR_HLS2BGR)
        output.append(one_output)

    if len_anlges==1:
        return output[0]
    else:
        return output

#Lightness
def adjust_lightness(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
        return None
    is_4c = img.shape[2] == 4
    if is_4c:
        bgr = img[:,:,0:3]
        one_output = np.zeros(img.shape,img.dtype)
        one_output[:,:,3] = img[:,:,3]
    else:
        bgr = img
        one_output = np.zeros(img.shape,img.dtype)
        
    hls = cv2.cvtColor(bgr,cv2.COLOR_BGR2HLS)
    output = []
    for rate in rates:
        temp_hls = hls.copy()
        f_temp_l = temp_hls[:,:,1]/255.0+rate
        f_temp_l[f_temp_l>1] = 1.0
        f_temp_l[f_temp_l<0] = 0.0
        temp_hls[:,:,1] = f_temp_l*255.0
        one_output[:,:,0:3] = cv2.cvtColor(temp_hls,cv2.COLOR_HLS2BGR)
        output.append(one_output)

    if len_rates==1:
        return output[0]
    else:
        return output

#Saturation
def adjust_saturation(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
        return None
    is_4c = img.shape[2] == 4
    if is_4c:
        bgr = img[:,:,0:3]
        one_output = np.zeros(img.shape,img.dtype)
        one_output[:,:,3] = img[:,:,3]
    else:
        bgr = img
        one_output = np.zeros(img.shape,img.dtype)
        
    hls = cv2.cvtColor(bgr,cv2.COLOR_BGR2HLS)
        
    output = []
    for rate in rates:
        temp_hls = hls.copy()
        f_temp_s = temp_hls[:,:,2]/255.0+rate
        f_temp_s[f_temp_s>1] = 1.0
        f_temp_s[f_temp_s<0] = 0.0
        temp_hls[:,:,2] = f_temp_s*255.0
        one_output[:,:,0:3] = cv2.cvtColor(temp_hls,cv2.COLOR_HLS2BGR)
        output.append(one_output.copy())

    if len_rates==1:
        return output[0]
    else:
        return output

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

def perspective(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
        return None
    output = []
    rand_type = random.choices(('U','UR','R','DR','D','DL','L','UL'),k=len(rates))
    for t,rate in zip(rand_type,rates):
        output.append(__perspective_one(img,t,rate))
    if len_rates==1:
        return output[0]
    else:
        return output

def perspectiveU(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
        return None
    output = []
    for rate in rates:
        output.append(__perspective_one(img,'U',rate))
    if len_rates==1:
        return output[0]
    else:
        return output

def perspectiveUR(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
        return None
    output = []
    for rate in rates:
        output.append(__perspective_one(img,'UR',rate))
    if len_rates==1:
        return output[0]
    else:
        return output

def perspectiveR(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
        return None
    output = []
    for rate in rates:
        output.append(__perspective_one(img,'R',rate))
    if len_rates==1:
        return output[0]
    else:
        return output

def perspectiveDR(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
        return None
    output = []
    for rate in rates:
        output.append(__perspective_one(img,'DR',rate))
    if len_rates==1:
        return output[0]
    else:
        return output

def perspectiveD(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
        return None
    output = []
    for rate in rates:
        output.append(__perspective_one(img,'D',rate))
    if len_rates==1:
        return output[0]
    else:
        return output

def perspectiveDL(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
        return None
    output = []
    for rate in rates:
        output.append(__perspective_one(img,'DL',rate))
    if len_rates==1:
        return output[0]
    else:
        return output

def perspectiveL(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
        return None
    output = []
    for rate in rates:
        output.append(__perspective_one(img,'L',rate))
    if len_rates==1:
        return output[0]
    else:
        return output

def perspectiveUL(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
        return None
    output = []
    for rate in rates:
        output.append(__perspective_one(img,'UL',rate))
    if len_rates==1:
        return output[0]
    else:
        return output


#class RESIZE_TYPE:
IMRESIZE_STRETCH = 0
IMRESIZE_ROUNDUP = 1
IMRESIZE_ROUNDUP_CROP = 2
IMRESIZE_ROUNDDOWN = 3
IMRESIZE_ROUNDDOWN_FILL_BLACK = 4
IMRESIZE_ROUNDDOWN_FILL_SELF = 5 

class imResizer:

    def __init__(self, resize_type, dsize, interpolation=cv2.INTER_LINEAR):
        self.set(resize_type, dsize, interpolation)
        return

    def set(self, resize_type, dsize ,interpolation):
        self._resize_type = resize_type
        self._dsize = dsize
        self._interpolation = interpolation

        #key: ((cv_w, cv_h), (save_w,save_h), (fx, bx), (fy, by))
        self._param = {}

        return
    
    def _transParam(self, this_img_width, this_img_height):
        # output = ((cv_w, cv_h), (save_w,save_h), (fx, bx), (fy, by))
        src_size = (int(this_img_width), int(this_img_height))

        found_param = self._param.get(src_size)
        if found_param is not None:
            return found_param

        if self._resize_type==IMRESIZE_STRETCH:
            output = (self._dsize, self._dsize, (self._dsize[0]/src_size[0], 0),(self._dsize[1]/src_size[1], 0))

        elif self._resize_type==IMRESIZE_ROUNDUP:
            _fx,_fy = self._dsize[0]/src_size[0], self._dsize[1]/src_size[1]
            _f = _fy if _fy > _fx else _fx
            _cv_size = (int(src_size[0]*_f),int(src_size[1]*_f))
            output = (_cv_size, _cv_size, (_f, 0),(_f, 0))

        elif self._resize_type==IMRESIZE_ROUNDUP_CROP:
            _fx, _fy = self._dsize[0]/src_w, self._dsize[1]/src_h
            if _fy > _fx:
                _f = _fy
                _cv_size = (int(src_size[0]*_f), self._dsize[1])
            else:
                _f = _fx
                _cv_size = (self._dsize[0], int(src_size[1]*_f))
            output = (_cv_size, self._dsize, (_f, (self._dsize[0]-_cv_size[0])/2),\
                    (_f,(self._dsize[1]-_cv_size[1])/2))

        elif self._resize_type==IMRESIZE_ROUNDDOWN:
            _fx,_fy = self._dsize[0]/src_size[0], self._dsize[1]/src_size[1]
            _f = _fx if _fy > _fx else _fy
            _cv_size = (int(src_size[0]*_f),int(src_size[1]*_f))
            output = (_cv_size, _cv_size, (_f, 0),(_f, 0))

        elif self._resize_type==IMRESIZE_ROUNDDOWN_FILL_BLACK or self._resize_type==IMRESIZE_ROUNDDOWN_FILL_SELF:
            _fx,_fy = self._dsize[0]/src_size[0], self._dsize[1]/src_size[1]
            if _fy > _fx:
                _f = _fx
                _cv_size = (self._dsize[0], int(src_size[1]*_f))
            else:
                _f = _fy
                _cv_size = (int(src_size[0]*_f), self._dsize[1])

            output = (_cv_size, self._dsize, (_f, (self._dsize[0]-_cv_size[0])/2),\
                    (_f,(self._dsize[1]-_cv_size[1])/2))

        else:
            print(self._dsize)
            print((self._dsize[0]/src_size[0], 0),(self._dsize[1]/src_size[1], 0))
            output = (self._dsize, self._dsize, (self._dsize[0]/src_size[0], 0),(self._dsize[1]/src_size[1], 0))

        self._param[src_size] = output
        return output

    # TODO: error in IMRESIZE_ROUNDUP_CROP
    def imResize(self,src):
        # param = ((cv_w, cv_h), (save_w,save_h), (fx, bx), (fy, by))
        param = self._transParam(src.shape[1],src.shape[0])

        if self._resize_type==IMRESIZE_STRETCH \
            or self._resize_type==IMRESIZE_ROUNDUP \
            or self._resize_type==IMRESIZE_ROUNDDOWN:
            return cv2.resize(src, dsize = param[0], interpolation = self._interpolation)

        elif self._resize_type==IMRESIZE_ROUNDUP_CROP:
            cvrimg = cv2.resize(src, dsize = param[0], interpolation = self._interpolation)
            if param[0][0] == param[1][0]:
                ymin = int(-param[3][1])
                return cvrimg[:,ymin:ymin+param[1][1]]
            else:
                xmin = int(-param[3][0])
                return cvrimg[xmin:xmin+param[1][0]]

        elif self._resize_type==IMRESIZE_ROUNDDOWN_FILL_BLACK:
            cvrimg = cv2.resize(src, dsize = param[0], interpolation = self._interpolation)
            img_to_save_shape = list(src.shape)
            img_to_save_shape[0] = param[1][1]
            img_to_save_shape[1] = param[1][0]
            img_to_save = np.zeros(img_to_save_shape, src.dtype)
            if param[0][0] == param[1][0]:
                ymin = int(param[3][1])
                img_to_save[:,ymin:ymin+param[1][1]] = cvrimg
                return img_to_save
            else:
                xmin = int(param[3][0])
                img_to_save[xmin:xmin+param[1][0]] = cvrimg
                return img_to_save

        elif self._resize_type==IMRESIZE_ROUNDDOWN_FILL_SELF:
            cvrimg = cv2.resize(src, dsize = param[0], interpolation = self._interpolation)
            img_to_save_shape = list(src.shape)
            img_to_save_shape[0] = param[1][1]
            img_to_save_shape[1] = param[1][0]
            img_to_save = np.zeros(img_to_save_shape, src.dtype)
            if param[0][0] == param[1][0]:
                ymin = int(param[3][1])
                ymax_add_1 = ymin+param[1][1]
                img_to_save[:,ymin:ymax_add_1] = cvrimg
                img_to_save[:,:ymin] = img_to_save[:,ymin]
                img_to_save[:,ymax_add_1:] = img_to_save[:,ymax_add_1]
                return img_to_save
            else:
                xmin = int(param[3][0])
                xmax_add_1 = xmin+param[1][0]
                img_to_save[xmin:xmax_add_1] = cvrimg
                img_to_save[:xmin] = img_to_save[xmin]
                img_to_save[xmax_add_1:] = img_to_save[xmax_add_1]
                return img_to_save

        else:
            return cv2.resize(src, dsize = param[0], interpolation = self._interpolation)       

    def pntResize(self,pnt_src,img_src_shape):
        # param = ((cv_w, cv_h), (save_w,save_h), (fx, bx), (fy, by))
        param = self._transParam(img_src_shape[1],img_src_shape[0])
        return (int(pnt_src[0]*param[2][0]+param[2][1]),int(pnt_src[1]*param[3][0]+param[3][1]))

    def pntRecover(self,pnt_dst,img_src_shape):
        # param = ((cv_w, cv_h), (save_w,save_h), (fx, bx), (fy, by))
        param = self._transParam(img_src_shape[1],img_src_shape[0])
        return (int((pnt_dst[0]-param[2][1])/param[2][0]),int((pnt_dst[1]-param[3][1])/param[3][0]))

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
    import pyBoost as pb

    def test_imResizer(save_folder_name):
        import os
        import time
        import tqdm
        img_path = r'G:\obj_mask_10'
        img_list = pb.deep_scan_file(img_path,'.jpg.png.jpeg',False)
        imrszr = pb.img.imResizer(pb.img.IMRESIZE_ROUNDDOWN, (400,400), cv2.INTER_CUBIC)
        save_path = os.path.join(save_folder_name,'imResizer')
        begin_time_point = time.time()
        for i,one_img_name in tqdm.tqdm(enumerate(img_list)):
            frone_path,name = os.path.split(one_img_name)
            pb.makedirs(os.path.join(save_path,frone_path))
            img = cv2.imread(os.path.join(img_path,one_img_name),cv2.IMREAD_UNCHANGED)
            img_to_save = imrszr.imResize(img)
            cv2.imwrite(os.path.join(save_path,frone_path,name),img_to_save)
        end_time_point = time.time()
        print((end_time_point-begin_time_point))

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
        from color_ring import read_color_ring

        global img
        global opt_bar
        global opt_max_number
        global param_bar
        global param
        global aug_mat

        user_img_view_size = (750,450)
        img = np.zeros([450,750,3],np.uint8)
        opt_bar = 0
        opt_max_number = 21
        param_bar = 10
        param = 10
        aug_mat = img.copy()

        def makeDefaultImg():
            global img
            global opt_bar
            global opt_max_number
            global param_bar
            global param
            global aug_mat

            img[:,:,:]=255
            j=0
            while(j<450/50+1):
                cv2.line(img,(0,int(j*50)),(750,int(j*50)),(0,0,0),3)
                j+=1
            j=0
            while(j<750/50+1):
                cv2.line(img,(int(j*50),0),(int(j*50),450),(0,0,0),3)
                j+=1

            _box_size = 50-5
            img[(3*50+3):((3*50+3)+_box_size),(6*50+3):((6*50+3)+_box_size),0] = 255 
            img[(3*50+3):((3*50+3)+_box_size),(6*50+3):((6*50+3)+_box_size),1] = 0
            img[(3*50+3):((3*50+3)+_box_size),(6*50+3):((6*50+3)+_box_size),2] = 255

            img[(3*50+3):((3*50+3)+_box_size),(7*50+3):((7*50+3)+_box_size),0] = 0
            img[(3*50+3):((3*50+3)+_box_size),(7*50+3):((7*50+3)+_box_size),1] = 0
            img[(3*50+3):((3*50+3)+_box_size),(7*50+3):((7*50+3)+_box_size),2] = 255

            img[(3*50+3):((3*50+3)+_box_size),(8*50+3):((8*50+3)+_box_size),0] = 0 
            img[(3*50+3):((3*50+3)+_box_size),(8*50+3):((8*50+3)+_box_size),1] = 255
            img[(3*50+3):((3*50+3)+_box_size),(8*50+3):((8*50+3)+_box_size),2] = 255

            img[(4*50+3):((4*50+3)+_box_size),(6*50+3):((6*50+3)+_box_size),0] = 0 
            img[(4*50+3):((4*50+3)+_box_size),(6*50+3):((6*50+3)+_box_size),1] = 0
            img[(4*50+3):((4*50+3)+_box_size),(6*50+3):((6*50+3)+_box_size),2] = 0

            img[(4*50+3):((4*50+3)+_box_size),(7*50+3):((7*50+3)+_box_size),0] = 255 
            img[(4*50+3):((4*50+3)+_box_size),(7*50+3):((7*50+3)+_box_size),1] = 255
            img[(4*50+3):((4*50+3)+_box_size),(7*50+3):((7*50+3)+_box_size),2] = 255

            img[(4*50+3):((4*50+3)+_box_size),(8*50+3):((8*50+3)+_box_size),0] = 255/2 
            img[(4*50+3):((4*50+3)+_box_size),(8*50+3):((8*50+3)+_box_size),1] = 255/2
            img[(4*50+3):((4*50+3)+_box_size),(8*50+3):((8*50+3)+_box_size),2] = 255/2

            img[(5*50+3):((5*50+3)+_box_size),(6*50+3):((6*50+3)+_box_size),0] = 255 
            img[(5*50+3):((5*50+3)+_box_size),(6*50+3):((6*50+3)+_box_size),1] = 0
            img[(5*50+3):((5*50+3)+_box_size),(6*50+3):((6*50+3)+_box_size),2] = 0

            img[(5*50+3):((5*50+3)+_box_size),(7*50+3):((7*50+3)+_box_size),0] = 255 
            img[(5*50+3):((5*50+3)+_box_size),(7*50+3):((7*50+3)+_box_size),1] = 255
            img[(5*50+3):((5*50+3)+_box_size),(7*50+3):((7*50+3)+_box_size),2] = 0

            img[(5*50+3):((5*50+3)+_box_size),(8*50+3):((8*50+3)+_box_size),0] = 0 
            img[(5*50+3):((5*50+3)+_box_size),(8*50+3):((8*50+3)+_box_size),1] = 255
            img[(5*50+3):((5*50+3)+_box_size),(8*50+3):((8*50+3)+_box_size),2] = 0

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
                #Flip
                aug_mat = pb.img.flip(img)
                cv2.putText(aug_mat, aug_text + "Flip", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "(NULL)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Random flip", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 4):
                #Crop
                param = 1.0 / 100.0* param_bar
                if(param<0.01):
                     param = 0.01
                aug_mat = pb.img.crop(img, param)
                cv2.putText(aug_mat, aug_text + "Crop(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "(0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Random crop image by area percentage", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 5):
                #AffineX
                param = 1.0 / 50.0* (param_bar - 50)
                aug_mat = pb.img.affineX(img, param)
                cv2.putText(aug_mat, aug_text + "AffineX(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[-1,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "X axis affine", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 6):
                #AffineY
                param = 1.0 / 50.0* (param_bar - 50)
                aug_mat = pb.img.affineY(img, param)
                cv2.putText(aug_mat, aug_text + "AffineY(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[-1,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Y axis affine", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 7):
                #AffineY
                param = 1.0 / 50.0* (param_bar - 50)
                aug_mat = pb.img.affine(img, param)
                cv2.putText(aug_mat, aug_text + "Affine(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[-1,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "X or Y axis affine", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 8):
                #Noise
                param = 1.0 / 100.0* (param_bar)
                aug_mat = pb.img.add_noise(img, param)
                cv2.putText(aug_mat, aug_text + "Noise(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Add gauss noise", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 9):
                #Hue
                param = 1.0 / 100.0*360.0 * param_bar
                aug_mat = pb.img.adjust_hue(img, param)
                cv2.putText(aug_mat, aug_text + "Hue(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "(---,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Hue angle (degree)", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 10):
                #Saturation
                param = 1.0 / 50.0* (param_bar - 50)
                aug_mat = pb.img.adjust_saturation(img, param)
                cv2.putText(aug_mat, aug_text + "Saturation(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Saturation", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 11):
                #Lightness
                param = 1.0 / 50.0* (param_bar - 50)
                aug_mat = pb.img.adjust_lightness(img, param)
                cv2.putText(aug_mat, aug_text + "Lightness(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Lightness", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 12):
                #PerspectiveUL
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveUL(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveUL(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "UL Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 13):
                #PerspectiveU
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveU(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveU(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "U  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 14):
                #PerspectiveUR
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveUR(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveUR(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "UR Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 15):
                #PerspectiveL
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveUL(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveL(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "L  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 16):
                #Perspective
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspective(img, param)
                cv2.putText(aug_mat, aug_text + "Perspective(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Random perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 17):
                #PerspectiveR
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveR(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveR(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "R  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 18):
                #PerspectiveDL
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveDL(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveDL(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "DL Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 19):
                #PerspectiveD
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveD(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveD(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "D  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 20):
                #PerspectiveDR
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveDR(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveDR(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "DR Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            #elif(opt_bar == 21):
            #    #Distort
            #    param = 1.0 / 20.0* param_bar
            #    pb.img.imDistort(img, aug_mat, param)
            #    cv2.putText(aug_mat, aug_text + "Distort(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
            #    cv2.putText(aug_mat, fanwei + "[0,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
            #    cv2.putText(aug_mat, shuoming + "Add distort", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
            #    cv2.imshow("augmented", aug_mat)
            #elif(opt_bar == 21):
            #    #Pyramid
            #    aug_mat.fill(255 / 2)
            #    cv2.putText(aug_mat, aug_text + "Pyramid(donw,up)", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
            #    cv2.putText(aug_mat, fanwei + "(0,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
            #    cv2.putText(aug_mat, shuoming + "Build image pyramid", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
            #    cv2.imshow("augmented", aug_mat)
            #elif(opt_bar == 23):#NotBackup
            #    aug_mat.setTo(Scalar(255 / 2, 255 / 2, 255 / 2))
            #    cv2.putText(aug_mat, aug_text + "NotBackup", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
            #    cv2.putText(aug_mat, fanwei + "(NULL)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
            #    cv2.putText(aug_mat, shuoming + "A flag means do not keep prestep images", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
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
                #Flip
                aug_mat = pb.img.flip(img)
                cv2.putText(aug_mat, aug_text + "Flip", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "(NULL)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Random flip", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 4):
                #Crop
                param = 1.0 / 100.0* param_bar
                if(param<0.01):
                     param = 0.01
                aug_mat = pb.img.crop(img, param)
                cv2.putText(aug_mat, aug_text + "Crop(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "(0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Random crop image by area percentage", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 5):
                #AffineX
                param = 1.0 / 50.0* (param_bar - 50)
                aug_mat = pb.img.affineX(img, param)
                cv2.putText(aug_mat, aug_text + "AffineX(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[-1,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "X axis affine", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 6):
                #AffineY
                param = 1.0 / 50.0* (param_bar - 50)
                aug_mat = pb.img.affineY(img, param)
                cv2.putText(aug_mat, aug_text + "AffineY(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[-1,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Y axis affine", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 7):
                #AffineY
                param = 1.0 / 50.0* (param_bar - 50)
                aug_mat = pb.img.affine(img, param)
                cv2.putText(aug_mat, aug_text + "Affine(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[-1,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "X or Y axis affine", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 8):
                #Noise
                param = 1.0 / 100.0* (param_bar)
                aug_mat = pb.img.add_noise(img, param)
                cv2.putText(aug_mat, aug_text + "Noise(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Add gauss noise", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 9):
                #Hue
                param = 1.0 / 100.0*360.0 * param_bar
                aug_mat = pb.img.adjust_hue(img, param)
                cv2.putText(aug_mat, aug_text + "Hue(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "(---,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Hue angle (degree)", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 10):
                #Saturation
                param = 1.0 / 50.0* (param_bar - 50)
                aug_mat = pb.img.adjust_saturation(img, param)
                cv2.putText(aug_mat, aug_text + "Saturation(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Saturation", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 11):
                #Lightness
                param = 1.0 / 50.0* (param_bar - 50)
                aug_mat = pb.img.adjust_lightness(img, param)
                cv2.putText(aug_mat, aug_text + "Lightness(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Lightness", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 12):
                #PerspectiveUL
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveUL(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveUL(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "UL Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 13):
                #PerspectiveU
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveU(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveU(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "U  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 14):
                #PerspectiveUR
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveUR(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveUR(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "UR Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 15):
                #PerspectiveL
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveL(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveL(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "L  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 16):
                #Perspective
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspective(img, param)
                cv2.putText(aug_mat, aug_text + "Perspective(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "Random perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 17):
                #PerspectiveR
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveR(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveR(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "R  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 18):
                #PerspectiveDL
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveDL(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveDL(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "DL Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 19):
                #PerspectiveD
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveD(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveD(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "D  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            elif(opt_bar == 20):
                #PerspectiveDR
                param = 1.0 / 100.0* param_bar
                aug_mat = pb.img.perspectiveDR(img, param)
                cv2.putText(aug_mat, aug_text + "PerspectiveDR(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.putText(aug_mat, shuoming + "DR Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
                cv2.imshow("augmented", aug_mat)
            #elif(opt_bar == 21):
            #    #Distort
            #    param = 1.0 / 20.0* param_bar
            #    pb.img.imDistort(img, aug_mat, param)
            #    cv2.putText(aug_mat, aug_text + "Distort(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
            #    cv2.putText(aug_mat, fanwei + "[0,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
            #    cv2.putText(aug_mat, shuoming + "Add distort", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
            #    cv2.imshow("augmented", aug_mat)
            #elif(opt_bar == 21):
            #    #Pyramid
            #    aug_mat.fill(255 / 2)
            #    cv2.putText(aug_mat, aug_text + "Pyramid(donw,up)", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
            #    cv2.putText(aug_mat, fanwei + "(0,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
            #    cv2.putText(aug_mat, shuoming + "Build image pyramid", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
            #    cv2.imshow("augmented", aug_mat)
            #elif(opt_bar == 23):#NotBackup
            #    aug_mat.setTo(Scalar(255 / 2, 255 / 2, 255 / 2))
            #    cv2.putText(aug_mat, aug_text + "NotBackup", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
            #    cv2.putText(aug_mat, fanwei + "(NULL)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
            #    cv2.putText(aug_mat, shuoming + "A flag means do not keep prestep images", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
            #    cv2.imshow("augmented", aug_mat)
            return
        
        #argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--img',type=str, default='', help='Image you want to view')
        opt = parser.parse_args()

        # load image to show
        img_resizer = pb.img.imResizer(pb.img.IMRESIZE_ROUNDUP,user_img_view_size)
        if(opt.img == ''):
            org_img = read_color_ring()
        else:
            img_path = opt.img
            org_img = cv2.imread(img_path)
            if org_img is None or os.path.isfile(img_path)==False:
                sys.exit('Can\'t find file: {0}'.format(img_path))
        img = img_resizer.imResize(org_img) 

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
    # test_imResizer(save_folder_name)
    # test_imshow()
    test_augment() 



