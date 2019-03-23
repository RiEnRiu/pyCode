#-*-coding:utf-8-*-
import cv2
import numpy as np
import math
import threading


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
    elif len_angles==1:
        return __rotate_one(img,angles)
    else:
        output = []
        for angle in angles:
            output.append(__rotate_one(img,angles))
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
    elif len_rates==1:
        return __crop_one(img,rates)
    else:
        output = []
        for rate in rates:
            output.append(__crop_one(img,rate))
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
    elif len_rates==1:
        return __affine_one(img,random.randint(0,1),rates)
    else:
        output = []
        for rate in rates:
            output.append(__affine_one(img,random.randint(0,1),rate))
        return output
      
def affineX(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
        return None
    elif len_rates==1:
        return __affine_one(img,True,rates)
    else:
        output = []
        for rate in rates:
            output.append(__affine_one(img,True,rate))
        return output
        
def affineY(img,*rates):
    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
        return None
    elif len_rates==1:
        return __affine_one(img,False,rates)
    else:
        output = []
        for rate in rates:
            output.append(__affine_one(img,False,rate))
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
    output = []
    is_4c = len(img.shape) == 3
    if is_4c:
        is_4c = img.shape[2] == 4

    len_rates = len(rates)
    if len_rates==0:
        raise ValueError('There must be at least 1 in \"rates\"')
        return None
    elif len_rates==1:
        return __add_noise_one(img,rates,is_4c)
    else:
        output = []
        for rate in rates:
            output.append(__add_noise_one(img,rate,is_4c))
        return output

#Hue
def adjust_hue(img,anlges):
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
    return output

#Lightness
def adjust_lightness(img,rates):
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
    return output

#Saturation
def adjust_saturation(img,rates):
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

def perspective(img,rates):
    type_erum = ('U','UR','R','DR','D','DL','L','UL')
    output = []
    type_np = np.random.randint(0,7,len(rates),np.int)
    #print(type_erum[type_np[0]])
    for i in range(len(rates)):
        output.append(__perspective_one(img,type_erum[type_np[i]],rates[i]))
    return output

def perspectiveU(img,rates):
    output = []
    for rate in rates:
        output.append(__perspective_one(img,'U',rate))
    return output

def perspectiveUR(img,rates):
    output = []
    for rate in rates:
        output.append(__perspective_one(img,'UR',rate))
    return output

def perspectiveR(img,rates):
    output = []
    for rate in rates:
        output.append(__perspective_one(img,'R',rate))
    return output

def perspectiveDR(img,rates):
    output = []
    for rate in rates:
        output.append(__perspective_one(img,'DR',rate))
    return output

def perspectiveD(img,rates):
    output = []
    for rate in rates:
        output.append(__perspective_one(img,'D',rate))
    return output

def perspectiveDL(img,rates):
    output = []
    for rate in rates:
        output.append(__perspective_one(img,'DL',rate))
    return output

def perspectiveL(img,rates):
    output = []
    for rate in rates:
        output.append(__perspective_one(img,'L',rate))
    return output

def perspectiveUL(img,rates):
    output = []
    for rate in rates:
        output.append(__perspective_one(img,'UL',rate))
    return output


#class RESIZE_TYPE:
IMRESIZE_STRETCH = 0
IMRESIZE_ROUNDUP = 1
IMRESIZE_ROUNDUP_CROP = 2
IMRESIZE_ROUNDDOWN = 3
IMRESIZE_ROUNDDOWN_FILL_BLACK = 4
IMRESIZE_ROUNDDOWN_FILL_SELF = 5 

class imResizer:

    def __init__(self, resize_type, dsize, interpolation):
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

        if self._resize_type==RESIZE_STRETCH:
            output = (self._dsize, self._dsize, (self._dsize[0]/src_size[0], 0),(self._dsize[1]/src_size[1], 0))

        elif self._resize_type==RESIZE_ROUNDUP:
            _fx,_fy = self._dsize[0]/src_size[0], self._dsize[1]/src_size[1]
            _f = _fy if _fy > _fx else _fx
            _cv_size = (int(src_size[0]*_f),int(src_size[1]*_f))
            output = (_cv_size, _cv_size, (_f, 0),(_f, 0))

        elif self._resize_type==RESIZE_ROUNDUP_CROP:
            _fx, _fy = self._dsize[0]/src_w, self._dsize[1]/src_h
            if _fy > _fx:
                _f = _fy
                _cv_size = (int(src_size[0]*_f), self._dsize[1])
            else:
                _f = _fx
                _cv_size = (self._dsize[0], int(src_size[1]*_f))
            output = (_cv_size, self._dsize, (_f, (self._dsize[0]-_cv_size[0])/2),\
                    (_f,(self._dsize[1]-_cv_size[1])/2))

        elif self._resize_type==RESIZE_ROUNDDOWN:
            _fx,_fy = self._dsize[0]/src_size[0], self._dsize[1]/src_size[1]
            _f = _fx if _fy > _fx else _fy
            _cv_size = (int(src_size[0]*_f),int(src_size[1]*_f))
            output = (_cv_size, _cv_size, (_f, 0),(_f, 0))

        elif self._resize_type==RESIZE_ROUNDDOWN_FILL_BLACK or self._resize_type==RESIZE_ROUNDDOWN_FILL_SELF:
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
            output = (self._dsize, self._dsize, (self._dsize[0]/src_size[0], 0),(self._dsize[1]/src_size[1], 0))

        self._param[src_size] = output
        return output

    def imResize(self,src):
        # param = ((cv_w, cv_h), (save_w,save_h), (fx, bx), (fy, by))
        param = self._transParam(src.shape[1],src.shape[0])

        if self._resize_type==RESIZE_STRETCH \
            or self._resize_type==RESIZE_ROUNDUP \
            or self._resize_type==RESIZE_ROUNDDOWN:
            return cv2.resize(src, dsize = param[0], interpolation = self._interpolation)

        elif self._resize_type==RESIZE_ROUNDUP_CROP:
            cvrimg = cv2.resize(src, dsize = param[0], interpolation = self._interpolation)
            if param[0][0] == param[1][0]:
                ymin = int(-param[3][1])
                return cvrimg[:,ymin:ymin+param[1][1]]
            else:
                xmin = int(-param[3][0])
                return cvrimg[xmin:xmin+param[1][0]]

        elif self._resize_type==RESIZE_ROUNDDOWN_FILL_BLACK:
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

        elif self._resize_type==RESIZE_ROUNDDOWN_FILL_SELF:
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

if __name__=='__main__':
    ##################################################################
    #test module
    ##################################################################
    import sys
    sys.path.append('../')
    import pyBoost as pb
    print(pb.img.imResizer)

    def test_imResizer(save_folder_name):
        import os
        import time
        import tqdm
        img_path = r'G:\obj_mask_10'
        img_list = pb.scan_file_r(img_path,'.jpg.png.jpeg',False)
        imrszr = pb.img.imResizer(pb.img.RESIZE_ROUNDDOWN, (400,400), cv2.INTER_CUBIC)
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
    save_folder_name = 'pyBoost_test_output'
    test_imResizer(save_folder_name)



