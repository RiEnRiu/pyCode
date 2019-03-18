import cv2
import numpy as np
import random
import math

import pyBoost as pb
import time
#from pyBoost import _IOThread,FPS


#out a list of augmented images, even if it output only one images
#input image of CV_8UC1,CV_8UC3,CV_8UC4
class cvAugmentor:
    #Rotate
    def __imRotate_one(img,angle):
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
        if(is_1c):
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

        if(is_1c):
            return rot_output[rot_output_roi_y1:rot_output_roi_y2,rot_output_roi_x1:rot_output_roi_x2]
        else:
            return rot_output[rot_output_roi_y1:rot_output_roi_y2,rot_output_roi_x1:rot_output_roi_x2,:]

    def imRotate(img,angles):
        output = []
        for angle in angles:
            output.append(cvAugmentor.__imRotate_one(img,angle))
        return output
           
    #Flip
    def imFlipUD(img):
        return [cv2.flip(img,0)]
    def imFlipLR(img):
        return [cv2.flip(img,1)]
    def imFlip(img):
        return [cv2.flip(img,random.randint(0,1))]
    
    #Crop
    def __imCrop_one(img,rate):
        imgshape = img.shape
        is_1c = len(img.shape) == 2
        if(rate>1):
            rate = 1.0
        if(rate<0):
            rate = 0.0
        
        maxShape = [imgshape[0] * (1.0-rate)-1.0,imgshape[1] * (1.0-rate)-1.0]
        if(maxShape[0]<=0 or maxShape[1]<=0):
            return img.copy()
        x1 = random.randint(0,int(maxShape[1]))
        y1 = random.randint(0,int(maxShape[0]))
        x2 = x1+ int(imgshape[1]*rate)
        y2 = y1+ int(imgshape[0]*rate)
        if(is_1c):
            return img[y1:y2,x1:x2].copy()
        else:
            return img[y1:y2,x1:x2,:].copy()

    def imCrop(img,rates):
        output = []
        for rate in rates:
            output.append(cvAugmentor.__imCrop_one(img,rate))
        return output

    #Affine
    def __imAffine_one(img,is_X,rate):
        img_cols = img.shape[1]
        img_rows = img.shape[0]
        src = np.array([[0.0,0.0],[img_cols,0.0],[0.0,img_rows]],np.float32)
        if(is_X):
            if(rate<0):
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
            if(rate<0):
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

    def imAffine(img,rates):
        output = []
        for rate in rates:
            output.append(cvAugmentor.__imAffine_one(img,random.randint(0,1),rate))
        return output
      
    def imAffineX(img,rates):
        output = []
        for rate in rates:
            output.append(cvAugmentor.__imAffine_one(img,True,rate))
        return output
        
    def imAffineY(img,rates):
        output = []
        for rate in rates:
            output.append(cvAugmentor.__imAffine_one(img,False,rate))
        return output

    #Noise
    def imNoise(img,rates):
        output = []
        is_4c = len(img.shape) == 3
        if(is_4c):
            is_4c = img.shape[2] == 4
        if(is_4c is not True):
            for rate in rates:
                noise = np.random.randint(-255,255,img.size)
                f_noise = noise.reshape(img.shape).astype(np.float32)
                added_noise = (f_noise*rate+img)
                added_noise[added_noise>255]=255
                added_noise[added_noise<0]=0
                output.append(added_noise.astype(img.dtype))
        else:
            for rate in rates:
                noise = np.random.randint(-255,255,(img.shape[0],img.shape[1],3))
                f_noise = noise.reshape((img.shape[0],img.shape[1],3)).astype(np.float32)
                added_noise = img.copy()
                added_noise[:,:,0:3] = (f_noise*rate+img[:,:,0:3]).astype(np.uint8)
                added_noise[added_noise>255]=255
                added_noise[added_noise<0]=0
                output.append(added_noise)
        return output

    #Hue
    def imHue(img,anlges):
        is_4c = img.shape[2] == 4
        if(is_4c):
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
    def imLightness(img,rates):
        is_4c = img.shape[2] == 4
        if(is_4c):
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
    def imSaturation(img,rates):
        is_4c = img.shape[2] == 4
        if(is_4c):
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
    def __imPerspective_one(img,type,rate):
        img_cols = img.shape[1]
        img_rows = img.shape[0]
        src = np.array([[0.0,0.0],[img_cols,0.0],[img_cols,img_rows],[0.0,img_rows]],np.float32)
        dist = src.copy()
        
        offset_x = rate*img_cols
        offset_y = rate*img_rows

        if(type == 'U'):
            dist[0,0] += offset_x
            dist[1,0] -= offset_x
        elif(type == 'UR'):
            dist[1,0] -= offset_x
            dist[1,1] += offset_y
        elif(type == 'R'):
            dist[1,1] += offset_y
            dist[2,1] -= offset_y
        elif(type == 'DR'):
            dist[2,0] -= offset_x
            dist[2,1] -= offset_y
        elif(type == 'D'):
            dist[2,0] -= offset_x
            dist[3,0] += offset_x
        elif(type == 'DL'):
            dist[3,0] += offset_x 
            dist[3,1] -= offset_y
        elif(type == 'L'):
            dist[0,1] += offset_y
            dist[3,1] -= offset_y
        elif(type == 'UL'):
            dist[0,0] += offset_x 
            dist[0,1] += offset_y
        r = cv2.getPerspectiveTransform(src,dist)
        return cv2.warpPerspective(img,r,(img_cols, img_rows))

    def imPerspective(img,rates):
        type_erum = ('U','UR','R','DR','D','DL','L','UL')
        output = []
        type_np = np.random.randint(0,7,len(rates),np.int)
        #print(type_erum[type_np[0]])
        for i in range(len(rates)):
            output.append(cvAugmentor.__imPerspective_one(img,type_erum[type_np[i]],rates[i]))
        return output

    def imPerspectiveU(img,rates):
        output = []
        for rate in rates:
            output.append(cvAugmentor.__imPerspective_one(img,'U',rate))
        return output

    def imPerspectiveUR(img,rates):
        output = []
        for rate in rates:
            output.append(cvAugmentor.__imPerspective_one(img,'UR',rate))
        return output

    def imPerspectiveR(img,rates):
        output = []
        for rate in rates:
            output.append(cvAugmentor.__imPerspective_one(img,'R',rate))
        return output

    def imPerspectiveDR(img,rates):
        output = []
        for rate in rates:
            output.append(cvAugmentor.__imPerspective_one(img,'DR',rate))
        return output

    def imPerspectiveD(img,rates):
        output = []
        for rate in rates:
            output.append(cvAugmentor.__imPerspective_one(img,'D',rate))
        return output

    def imPerspectiveDL(img,rates):
        output = []
        for rate in rates:
            output.append(cvAugmentor.__imPerspective_one(img,'DL',rate))
        return output

    def imPerspectiveL(img,rates):
        output = []
        for rate in rates:
            output.append(cvAugmentor.__imPerspective_one(img,'L',rate))
        return output

    def imPerspectiveUL(img,rates):
        output = []
        for rate in rates:
            output.append(cvAugmentor.__imPerspective_one(img,'UL',rate))
        return output

    def imPerstep(img):
        return [img.copy()]

    #Distort
    #def __imDistort_one(img,rate):
    #    pass     

    #def imDistort(img,rates):
    #    pass    

    #def imPyramid(img,down_number,up_number):
    #    if(down_number<0):
    #        down_number = 0
    #    if(up_number<0):
    #        up_number = 0
    #    n = down_number + up_number + 1
    #    output = [[]]*n
    #    output[down_number] = img.copy()
    #    for i in range(down_number):
    #        true_i = down_number-1-i
    #        output[true_i] = cv2.pyrDown(output[true_i+1])
    #    for i in range(down_number+1,n):
    #        output[i] = cv2.pyrUp(output[i-1])
    #    return output

class RESIZE_TYPE:
    STRETCH = 0
    ROUNDUP = 1
    ROUNDUP_CROP = 2
    ROUNDDOWN = 3
    ROUNDDOWN_FILL_BLACK = 4
    ROUNDDOWN_FILL_SELF = 5 

#Not saft while using multi-thread
class imResizer:

    def __init__(self, _dsize, _resize_type, _interpolation):
        self.set(_dsize, _resize_type, _interpolation)
        self.pre_img_size = np.array([-1, -1], np.int)
        self.rate = np.array([1.0, 1.0], np.float)
        self.bias = np.array([0, 0], np.int)
        self.img_save_size = np.array([0, 0], np.int)
        return

    def set(self, _dsize, _resize_type ,_interpolation):
        self.dsize = np.array(_dsize,np.int)
        self.resize_type = _resize_type
        self.interpolation = _interpolation
        return

    def _resetTransParam(self, this_img_width, this_img_height):
        if(this_img_width == self.pre_img_size[0] and this_img_height == self.pre_img_size[1]):
            return 
        else:
            this_img_size = np.array([this_img_width, this_img_height],np.int)
        if(self.resize_type==RESIZE_TYPE.ROUNDUP):
            self.rate = self.dsize / this_img_size
            if (self.rate[1] > self.rate[0]):
                self.rate[0] = self.rate[1]
            else:
                self.rate[1] = self.rate[0]
            self.bias =  np.array([0, 0], np.int)
            self.img_save_size = (self.rate * this_img_size).astype(np.int)
        elif(self.resize_type==RESIZE_TYPE.ROUNDUP_CROP):
            self.rate = self.dsize / this_img_size
            if (self.rate[1] > self.rate[0]):
                self.rate[0] = self.rate[1]
                self.bias[0] = np.int((self.rate[0] * this_img_size[0] - self.dsize[0]) / 2)
                self.bias[1] = 0
            else:
                self.rate[1] = self.rate[0]
                self.bias[0] = 0
                self.bias[1] = np.int((self.rate[1] * this_img_size[1] - self.dsize[1]) / 2)
            self.img_save_size = self.dsize
        elif(self.resize_type==RESIZE_TYPE.ROUNDDOWN):
            self.rate = self.dsize / this_img_size
            if (self.rate[1] > self.rate[0]):
                self.rate[1] = self.rate[0]
            else:
                self.rate[0] = self.rate[1]
            self.bias =  np.array([0, 0], np.int)
            self.img_save_size = (self.rate * this_img_size).astype(np.int)
        elif(self.resize_type==RESIZE_TYPE.ROUNDDOWN_FILL_BLACK or self.resize_type==RESIZE_TYPE.ROUNDDOWN_FILL_SELF):
            self.rate = self.dsize / this_img_size
            if (self.rate[1] > self.rate[0]):
                self.rate[1] = self.rate[0]
                self.bias[0] = 0
                self.bias[1] = np.int((self.dsize[1] - self.rate[1] * this_img_size[1]) / 2)                
            else:
                self.rate[0] = self.rate[1]
                self.bias[0] = np.int((self.dsize[0] - self.rate[0] * this_img_size[0]) / 2);
                self.bias[1] = 0;
            self.img_save_size = self.dsize
        else:
            self.rate = self.dsize / this_img_size
            self.bias = np.array([0,0],np.int)
            self.img_save_size = self.dsize
        self.pre_img_size = this_img_size;
        return   
    
    def imResize(self, src):
        src_cols = src.shape[1];
        src_rows = src.shape[0];
        self._resetTransParam(src_cols,src_rows);
        img_size = np.array([src_cols,src_rows],np.int)
        pre_img_size = img_size
        if(self.resize_type==RESIZE_TYPE.ROUNDUP_CROP):
            t_dst = cv2.resize(src, tuple((self.rate * img_size).astype(np.int).tolist()),self.interpolation)
            dst = t_dst[self.bias[1]:(self.bias[1] + self.img_save_size[1]),self.bias[0]:(self.bias[0] + self.img_save_size[0])].copy()
        elif(self.resize_type==RESIZE_TYPE.ROUNDDOWN_FILL_BLACK):
            t_dst_shape = list(src.shape)
            t_dst_shape[0] = self.img_save_size[1]
            t_dst_shape[1] = self.img_save_size[0]
            t_dst = np.zeros(t_dst_shape,src.dtype)
            img_valid_size = (self.rate * img_size).astype(np.int)
            resized_roi = cv2.resize(src, tuple(img_valid_size.tolist()), 0, 0,self.interpolation)
            t_dst[self.bias[1]:(self.bias[1] + img_valid_size[1]), self.bias[0]:(self.bias[0] + img_valid_size[0])] = resized_roi
            dst = t_dst.copy()
        elif(self.resize_type==RESIZE_TYPE.ROUNDDOWN_FILL_SELF):
            t_dst_shape = list(src.shape)
            t_dst_shape[0] = self.img_save_size[1]
            t_dst_shape[1] = self.img_save_size[0]
            t_dst = np.zeros(t_dst_shape,src.dtype)
            img_valid_size = (self.rate * img_size).astype(np.int)
            resized_roi = cv2.resize(src, tuple(img_valid_size.tolist()), 0, 0,self.interpolation)
            t_dst[self.bias[1]:(self.bias[1] + img_valid_size[1]), self.bias[0]:(self.bias[0] + img_valid_size[0])] = resized_roi
            for i in range(self.bias[1]):
                t_dst[i] = t_dst[self.bias[1]].copy()
            for i in range((self.bias[1] + img_valid_size[1]),self.img_save_size[1]):
                t_dst[i] = t_dst[self.bias[1] + img_valid_size[1] - 1].copy()
            for j in range(self.bias[0]):
                t_dst[:,j] = t_dst[:,self.bias[0]]
            for j in range((self.bias[0]+img_valid_size[0]),self.img_save_size[0]):
                t_dst[:,j] = t_dst[:,self.bias[0]+img_valid_size[0]-1].copy()
            dst = t_dst.copy()
        else:#RESIZE_STRETCH  RESIZE_ROUNDUP  RESIZE_ROUNDDOWN
            dst = cv2.resize(src, tuple(self.img_save_size.tolist()), 0, 0, self.interpolation)
        return dst;

    def pntResize(self,pnt_src,im_src_shape):
        self._resetTransParam(im_src_shape[1],im_src_shape[0])
        pnt_dst_x = int(pnt_src[0] * self.rate[0]) + int(self.bias[0])
        pnt_dst_y = int(pnt_src[1] * self.rate[1]) + int(self.bias[1])
        return (pnt_dst_x, pnt_dst_y)

    def pntRecover(self,pnt_dst,im_src_shape):
        self._resetTransParam(im_src_shape[1],im_src_shape[0])
        pnt_src_x = int((pnt_dst[0] - self.bias[0])/self.rate[0])
        pnt_src_y = int((pnt_dst[1] - self.bias[1])/self.rate[1])
        return (pnt_src_x,pnt_src_y)


class imgIO(pb._IOThread):
    def __init__(self, read_param=None,use_write = True, cache=10):
        if(read_param is None):
            pb._IOThread.__init__(self, [], use_write, cache, imgIO.read_thread_fun, imgIO.write_thread_fun)
        else:
            pb._IOThread.__init__(self, read_param, use_write, cache, imgIO.read_thread_fun, imgIO.write_thread_fun)
        #self._read_param = read_param
        #self._use_write = use_write
        #self._cache = cache
        #self._read_thread_target = imgIO.read_thread_fun
        #self._write_thread_target = imgIO.write_thread_fun
        #self._reinit(self._read_thread_target, self._write_thread_target)

    def read_thread_fun(self):
        for one_read_param in self._read_param:
            if(self._read_thread_brk):
                break
            if(self._q_read.full()):
                time.sleep(max(min(self._cache/self._p_fps_read.get_only()/2, 0.0001),1))
                pass
            if(type(one_read_param)==str):
                one_img = cv2.imread(one_read_param)
                #print('1----'+ str(one_read_param))
                #print(one_img.dtype)
                #print(one_img.shape)
            elif(len(one_read_param)==2):
                one_img = cv2.imread(one_read_param[0],one_read_param[1])
                #print('2----'+str(one_read_param))
                #print(one_img.dtype)
                #print(one_img.shape)
            else:
                one_img = cv2.imread(one_read_param[0])
                #print('3----'+str(one_read_param))
                #print(one_img.dtype)
                #print(one_img.shape)
            if(one_img is None):
                print('Can not read image with param: '+str(one_read_param))
                raise IOError
            self._q_read.put(one_img)
            self._i_read_param += 1
        return 
            
    def write_thread_fun(self):
        while(self._write_thread_brk is not True or self._q_write.empty() is not True):
            if(self._q_write.empty()):
                time.sleep(max(min(self._cache/self._p_fps_read.get_only()/2, 0.0001),1))
                continue
            one_write_param = self._q_write.get()
            if(len(one_write_param)==3):
                ret = cv2.imwrite(one_write_param[0],one_write_param[1],one_write_param[2])
            else:
                ret = cv2.imwrite(one_write_param[0],one_write_param[1])
            if(ret==False):
                err_str = 'image write fail: '+one_write_param[0]
                self._err.append(err_str)
                raise IOError
        return 



##################################################################
#test module
##################################################################
def __test_imgIO(save_folder_name):
    from tqdm import tqdm
    #param
    read_path = r'G:\obj_mask_10'
    sleep_second = 0.05
    test_size = 0

    no_thread_save_path = os.path.join('.',save_folder_name,'imgIO','no_thread')
    thread_save_path = os.path.join('.',save_folder_name,'imgIO','thread')

    #begin
    img_list = scanner.file_r(read_path,'.jpg.png.jpeg',False)

    if(test_size!=0 and test_size<len(img_list)):
        img_list = img_list[0:test_size]
    random.shuffle(img_list)

    img_save = cv2.imread(os.path.join(read_path,img_list[0]),cv2.IMREAD_UNCHANGED)

    begin_time_point = time.time()
    
    for one_img in tqdm(img_list,ncols=0):
    #for one_img in img_list:
        img = cv2.imread(os.path.join(read_path,one_img),cv2.IMREAD_UNCHANGED)
        frone_path,name = os.path.split(one_img)
        if(os.path.isdir(os.path.join(no_thread_save_path,frone_path)) is not True):
            os.makedirs(os.path.join(no_thread_save_path,frone_path))
        cv2.imwrite(os.path.join(no_thread_save_path,frone_path,name),img)
        cv2.circle(img,(300,300),100,(0,0,255),30)
        if(sleep_second!=0):
            time.sleep(sleep_second)
    end_time_point = time.time()
    print((end_time_point-begin_time_point))

    lll = [[os.path.join(read_path,x),cv2.IMREAD_UNCHANGED] for x in img_list]
    #lll[100][0] = 'qowirya,fbkadsjbfk'#error filename.

    begin_time_point = time.time()
    imio = imgIO(lll)
    for one_img in tqdm(img_list,ncols=0):
    #for one_img in img_list:
        img = imio.read()
        if(img is not None):
            frone_path,name = os.path.split(one_img)
            if(os.path.isdir(os.path.join(thread_save_path,frone_path)) is not True):
                os.makedirs(os.path.join(thread_save_path,frone_path))
            imio.write(os.path.join(thread_save_path,frone_path,name),img.copy())
            cv2.circle(img,(300,300),100,(0,0,255),30)
        if(sleep_second!=0):
            sleep(sleep_second)
    imio.waitEnd()
    print(imio.error())

    end_time_point = time.time()
    print((end_time_point-begin_time_point))
  
    begin_time_point = time.time()

    for i in tqdm(range(len(img_list)),ncols=0):
        if(i%5==4):
            for j in range(5):
                one_img = img_list[i-j]
                img = cv2.imread(os.path.join(read_path,one_img),cv2.IMREAD_UNCHANGED)
                frone_path,name = os.path.split(one_img)
                if(os.path.isdir(os.path.join(no_thread_save_path,frone_path)) is not True):
                    os.makedirs(os.path.join(no_thread_save_path,frone_path))
                cv2.imwrite(os.path.join(no_thread_save_path,frone_path,name),img)
                cv2.circle(img,(300,300),100,(0,0,255),30)
            if(sleep_second!=0):
                sleep(sleep_second)
    end_time_point = time.time()
    print((end_time_point-begin_time_point))

    lll = [[os.path.join(read_path,x),cv2.IMREAD_UNCHANGED] for x in img_list]
    #lll[100][0] = 'qowirya,fbkadsjbfk'#error filename.

    begin_time_point = time.time()
    imio = imgIO(lll)
    for i in tqdm(range(len(img_list)),ncols=0):
        if(i%5==4):
            for j in range(5):
                one_img = img_list[i+j-4]
                img = imio.read()
                if(img is not None):
                    frone_path,name = os.path.split(one_img)
                    if(os.path.isdir(os.path.join(thread_save_path,frone_path)) is not True):
                        os.makedirs(os.path.join(thread_save_path,frone_path))
                    imio.write(os.path.join(thread_save_path,frone_path,name),img.copy())
                    cv2.circle(img,(300,300),100,(0,0,255),30)
            if(sleep_second!=0):
                sleep(sleep_second)
    imio.waitEnd()
    print(imio.error())

    end_time_point = time.time()
    print((end_time_point-begin_time_point))


def __test_imResizer(save_folder_name):
    from pyBoost import scanner
    img_path = r'G:\obj_mask_10'
    img_list = scanner.file_r(img_path,'.jpg.png.jpeg',False)
    imrszr = imResizer((400,400),RESIZE_TYPE.ROUNDUP,cv2.INTER_CUBIC)
    p_imgio = imgIO([[os.path.join(img_path,x),cv2.IMREAD_UNCHANGED] for x in img_list],True,60)
    save_path = os.path.join(save_folder_name,'imResizer')
    begin_time_point = time.time()
    for i,one_img_name in tqdm(enumerate(img_list)):
        frone_path,name = os.path.split(one_img_name)
        if(os.path.isdir(os.path.join(save_path,frone_path)) is not True):
            os.makedirs(os.path.join(save_path,frone_path))
        img = p_imgio.read()
        img_to_save = imrszr.imResize(img)
        p_imgio.write(os.path.join(save_path,frone_path,name),img_to_save.copy())
    print(p_imgio.error())
    end_time_point = time.time()
    print((end_time_point-begin_time_point))

if __name__=='__main__':


    save_folder_name = 'imBoost_test_output'
    __test_imResizer(save_folder_name)
    #__test_imgIO(save_folder_name)
