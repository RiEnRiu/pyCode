import cv2
import numpy as np
import os
import sys

import threading
import time
import queue



'''
there is some bugs
'''
#class _GetchUnix:
#    def __init__(self):
#        import tty, sys

#    def __call__(self):
#        import sys, tty, termios
#        fd = sys.stdin.fileno()
#        old_settings = termios.tcgetattr(fd)
#        try:
#            tty.setraw(sys.stdin.fileno())
#            ch = sys.stdin.read(1)
#        finally:
#            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
#        return ch

#class _GetchWindows:
#    def __init__(self):
#        import msvcrt

#    def __call__(self):
#        import msvcrt
#        return msvcrt.getch()

#class _Getch:
#    """Gets a single character from standard input.  Does not echo to the
#screen."""
#    def __init__(self):
#        try:
#            self.impl = _GetchWindows()
#        except ImportError:
#            self.impl = _GetchUnix()

#    def __call__(self): return self.impl()

#def getch():
#    out = ord(_Getch()())
#    if(out == 3 or out ==26):#Ctrl + C or Ctrl + Z
#        sys.exit('KeyboardInterrupt')
#    return out

#def _waitKey_thread_fun(key):
#    key[0] = ord(_Getch()())
#    print('key = {}'.format(key[0]))
#    return

#def waitKey(delay_ms):
#    if(delay_ms==0):
#        return getch()
#    else:
#        begin = time.time()
#        delay = delay_ms/1000
#        key_list = [-1]
#        thread_head = threading.Thread(target=_waitKey_thread_fun,args = (key_list,),daemon=True)
#        thread_head.start()
#        end = time.time()
#        while(end-begin < delay):
#            k = key_list[0]
#            if(k == 3 or k == 26):#Ctrl + C or Ctrl + Z
#                del thread_head
#                sys.exit('KeyboardInterrupt')
#            elif(k!=-1):
#                thread_head.join()
#                return k
#            end = time.time()
#        del thread_head
#        return -1


#class RESIZE_TYPE:
#    STRETCH = 0
#    ROUNDUP = 1
#    ROUNDUP_CROP = 2
#    ROUNDDOWN = 3
#    ROUNDDOWN_FILL_BLACK = 4
#    ROUNDDOWN_FILL_SELF = 5 

##Not saft while using multi-thread
#class imResizer:

#    def __init__(self, _dsize, _resize_type, _interpolation):
#        self.set(_dsize, _resize_type, _interpolation)
#        self.pre_img_size = np.array([-1, -1], np.int)
#        self.rate = np.array([1.0, 1.0], np.float)
#        self.bias = np.array([0, 0], np.int)
#        self.img_save_size = np.array([0, 0], np.int)
#        return

#    def set(self, _dsize, _resize_type ,_interpolation):
#        self.dsize = np.array(_dsize,np.int)
#        self.resize_type = _resize_type
#        self.interpolation = _interpolation
#        return

#    def _resetTransParam(self, this_img_width, this_img_height):
#        if(this_img_width == self.pre_img_size[0] and this_img_height == self.pre_img_size[1]):
#            return 
#        else:
#            this_img_size = np.array([this_img_width, this_img_height],np.int)
#        if(self.resize_type==RESIZE_TYPE.ROUNDUP):
#            self.rate = self.dsize / this_img_size
#            if (self.rate[1] > self.rate[0]):
#                self.rate[0] = self.rate[1]
#            else:
#                self.rate[1] = self.rate[0]
#            self.bias =  np.array([0, 0], np.int)
#            self.img_save_size = (self.rate * this_img_size).astype(np.int)
#        elif(self.resize_type==RESIZE_TYPE.ROUNDUP_CROP):
#            self.rate = self.dsize / this_img_size
#            if (self.rate[1] > self.rate[0]):
#                self.rate[0] = self.rate[1]
#                self.bias[0] = np.int((self.rate[0] * this_img_size[0] - self.dsize[0]) / 2)
#                self.bias[1] = 0
#            else:
#                self.rate[1] = self.rate[0]
#                self.bias[0] = 0
#                self.bias[1] = np.int((self.rate[1] * this_img_size[1] - self.dsize[1]) / 2)
#            self.img_save_size = self.dsize
#        elif(self.resize_type==RESIZE_TYPE.ROUNDDOWN):
#            self.rate = self.dsize / this_img_size
#            if (self.rate[1] > self.rate[0]):
#                self.rate[1] = self.rate[0]
#            else:
#                self.rate[0] = self.rate[1]
#            self.bias =  np.array([0, 0], np.int)
#            self.img_save_size = (self.rate * this_img_size).astype(np.int)
#        elif(self.resize_type==RESIZE_TYPE.ROUNDDOWN_FILL_BLACK or self.resize_type==RESIZE_TYPE.ROUNDDOWN_FILL_SELF):
#            self.rate = self.dsize / this_img_size
#            if (self.rate[1] > self.rate[0]):
#                self.rate[1] = self.rate[0]
#                self.bias[0] = 0
#                self.bias[1] = np.int((self.dsize[1] - self.rate[1] * this_img_size[1]) / 2)                
#            else:
#                self.rate[0] = self.rate[1]
#                self.bias[0] = np.int((self.dsize[0] - self.rate[0] * this_img_size[0]) / 2);
#                self.bias[1] = 0;
#            self.img_save_size = self.dsize
#        else:
#            self.rate = self.dsize / this_img_size
#            self.bias = np.array([0,0],np.int)
#            self.img_save_size = self.dsize
#        self.pre_img_size = this_img_size;
#        return   
    
#    def imResize(self, src):
#        src_cols = src.shape[1];
#        src_rows = src.shape[0];
#        self._resetTransParam(src_cols,src_rows);
#        img_size = np.array([src_cols,src_rows],np.int)
#        pre_img_size = img_size
#        if(self.resize_type==RESIZE_TYPE.ROUNDUP_CROP):
#            t_dst = cv2.resize(src, tuple((self.rate * img_size).astype(np.int).tolist()),self.interpolation)
#            dst = t_dst[self.bias[1]:(self.bias[1] + self.img_save_size[1]),self.bias[0]:(self.bias[0] + self.img_save_size[0])].copy()
#        elif(self.resize_type==RESIZE_TYPE.ROUNDDOWN_FILL_BLACK):
#            t_dst_shape = list(src.shape)
#            t_dst_shape[0] = self.img_save_size[1]
#            t_dst_shape[1] = self.img_save_size[0]
#            t_dst = np.zeros(t_dst_shape,src.dtype)
#            img_valid_size = (self.rate * img_size).astype(np.int)
#            resized_roi = cv2.resize(src, tuple(img_valid_size.tolist()), 0, 0,self.interpolation)
#            t_dst[self.bias[1]:(self.bias[1] + img_valid_size[1]), self.bias[0]:(self.bias[0] + img_valid_size[0])] = resized_roi
#            dst = t_dst.copy()
#        elif(self.resize_type==RESIZE_TYPE.ROUNDDOWN_FILL_SELF):
#            t_dst_shape = list(src.shape)
#            t_dst_shape[0] = self.img_save_size[1]
#            t_dst_shape[1] = self.img_save_size[0]
#            t_dst = np.zeros(t_dst_shape,src.dtype)
#            img_valid_size = (self.rate * img_size).astype(np.int)
#            resized_roi = cv2.resize(src, tuple(img_valid_size.tolist()), 0, 0,self.interpolation)
#            t_dst[self.bias[1]:(self.bias[1] + img_valid_size[1]), self.bias[0]:(self.bias[0] + img_valid_size[0])] = resized_roi
#            for i in range(self.bias[1]):
#                t_dst[i] = t_dst[self.bias[1]].copy()
#            for i in range((self.bias[1] + img_valid_size[1]),self.img_save_size[1]):
#                t_dst[i] = t_dst[self.bias[1] + img_valid_size[1] - 1].copy()
#            for j in range(self.bias[0]):
#                t_dst[:,j] = t_dst[:,self.bias[0]]
#            for j in range((self.bias[0]+img_valid_size[0]),self.img_save_size[0]):
#                t_dst[:,j] = t_dst[:,self.bias[0]+img_valid_size[0]-1].copy()
#            dst = t_dst.copy()
#        else:#RESIZE_STRETCH  RESIZE_ROUNDUP  RESIZE_ROUNDDOWN
#            dst = cv2.resize(src, tuple(self.img_save_size.tolist()), 0, 0, self.interpolation)
#        return dst;

#    def pntResize(self,pnt_src,im_src_shape):
#        self._resetTransParam(im_src_shape[1],im_src_shape[0])
#        pnt_dst_x = int(pnt_src[0] * self.rate[0]) + int(self.bias[0])
#        pnt_dst_y = int(pnt_src[1] * self.rate[1]) + int(self.bias[1])
#        return (pnt_dst_x, pnt_dst_y)

#    def pntRecover(self,pnt_dst,im_src_shape):
#        self._resetTransParam(im_src_shape[1],im_src_shape[0])
#        pnt_src_x = int((pnt_dst[0] - self.bias[0])/self.rate[0])
#        pnt_src_y = int((pnt_dst[1] - self.bias[1])/self.rate[1])
#        return (pnt_src_x,pnt_src_y)

class scanner_new_ver:
    def folder(path):
        if(os.path.isdir(path) is not True):
            print('It is not valid path: '+path)
            raise ValueError()
            return []
        return [x for x in os.listdir(path) if(os.path.isdir(os.path.join(path,x)))]

    # os.path.splitext('.txt') -> ['.txt', ''] , hard to use
    def _splitext(fileFullName):
        dot = fileFullName.rfind('.')
        if(dot==-1):
            return fileFullName,''
        else:
            return fileFullName[:dot],fileFullName[dot:]

    def _extsToLowerSet(exts):
        if(exts is None):
            return None
        elif(type(exts)==str):
            return {exts.lower()}
        else:
            return {x.lower() in set(exts)}

    def _pathTrim(path,fileFullName,with_root,with_ext):
        if(with_ext):
            if(with_root):
                return [os.path.join(path,x) for x in fileFullName],None
            else: #elif(with_root is not True):
                return fileFullName,None
        else:
            all_front,all_ext = [],[]
            for x in fileFullName:
                _front,_ext = scanner_new_ver._splitext(x)
                all_front.append(_front)
                all_ext.append(_ext)
            if(with_root):
                return [os.path.join(path,_front) for f in all_front],all_ext
            else: # elif(with_root is not True):
                return all_front, all_ext

    def file(path,exts=None,with_root=True,with_ext=True):
        if(os.path.isdir(path) is not True):
            print('It is not valid path: '+path)
            raise ValueError()
            return []
        dirs = os.listdir(path)
        all_file_full_name = [x for x in dirs if(os.path.isfile(os.path.join(path,x)))]
        exts_set = scanner_new_ver._extsToLowerSet(exts)
        if(exts_set is None):
            return scanner_new_ver._pathTrim(all_file_full_name)[0]
        else:
            out,all_exts = scanner_new_ver._pathTrim(all_file_full_name)
            if(all_exts is None):
                all_exts = [scanner_new_ver._splitext(x)[1] for x in out]
            return [x\
                    for i,x in enumerate(out)
                        if(all_exts[i] in exts_set)]
            

class scanner:
    def folder(path):
        if(os.path.isdir(path) is not True):
            print('It is not valid path: '+path)
            raise ValueError()
            return []
        return [x for x in os.listdir(path) if(os.path.isdir(os.path.join(path,x)))]
        
    def __scan_file(path,exts=None):
        if(os.path.isdir(path) is not True):
            print('It is not valid path: '+path)
            raise ValueError()
            return []
        dirs = os.listdir(path)
        if(exts is None):#all file
            return [x for x in dirs if(os.path.isfile(os.path.join(path,x)))]
        else:
            ext_set = set('.'+x.lower() for x in exts.split('.') if x!='')
            if('..' in exts or exts ==''):
                ext_set.add('')
            return [x \
                    for x in dirs \
                        if(os.path.isfile(os.path.join(path,x)) and \
                            (os.path.splitext(x)[1]).lower() in ext_set)]

    def __do_file_r(path,exts,father_folder):
        with_root_path = os.path.join(path,father_folder)
        files = scanner.__scan_file(with_root_path,exts)
        files_with_father = [os.path.join(father_folder,name) for name in file_list]
        for folder in scanner.folder(with_root_path):
            files_with_father += scanner.__do_file_r(path,exts,os.path.join(father_folder,folder))
        return files_with_father

    def file(path,exts=None,with_root=True,with_ext=True):
        file_name_list = scanner.__scan_file(path,exts)
        if(with_root and with_ext):
            return [os.path.join(path,name) for name in file_name_list]
        elif(with_ext):
            return file_name_list
        elif(with_root):
            return [os.path.join(path,os.path.splitext(name)[0]) for name in file_name_list]
        else:
            return [os.path.splitext(name)[0] for name in file_name_list]

    def file_r(path,exts=None,with_root=False,with_ext=True):
        file_name_with_relative_dir_list = scanner.__do_file_r(path,exts,'')
        if(with_root and with_ext):
            return [os.path.join(path,f) for f in file_name_with_relative_dir_list]
        elif(with_ext):
            return file_name_with_relative_dir_list
        elif(with_root):
            return [os.path.join(path,os.path.splitext(f)[0]) for f in file_name_with_relative_dir_list]
        else:
            return [os.path.splitext(f)[0] for f in file_name_with_relative_dir_list]

    def pair(path1,path2,exts1,exts2,with_root=True,with_ext=True):
        if(exts1 is None or exts2 is None):
            print('\"ext1\" and \"ext2\" in funsion \"scanner.pair()\" must be set.')
            raise ValueError()
            return [],[],[]
        file1 = scanner.file(path1,exts1,False) 
        file2 = scanner.file(path2,exts2,False) 
        front1 = {os.path.splitext(f1)[0]:f1 for f1 in file1}
        front2 = {os.path.splitext(f2)[0]:f2 for f2 in file2}
        out_pairs = [[front1[k],front2[k]] for k in (set(front1.keys()) & set(front2.keys()))]
        out_other1 = [front1[k] for k in (set(front1.keys()) - set(front2.keys()))]
        out_other2 = [front2[k] for k in (set(front2.keys()) - set(front1.keys()))]
        if(with_root and with_ext):
            return [[os.path.join(path1,x),os.path.join(path2,y)] for [x,y] in out_pairs],\
                    [os.path.join(path1,x) for x in out_other1],\
                    [os.path.join(path2,y) for y in out_other2]
        elif(with_ext):
            return out_pairs,out_other1,out_other2
        elif(with_root):
            return [[os.path.join(path1,os.path.splitext(x)[0]),os.path.join(path2,os.path.splitext(y)[0])] \
                        for [x,y] in out_pairs],\
                    [os.path.join(path1,os.path.splitext(x)[0]) for x in out_other1],\
                    [os.path.join(path2,os.path.splitext(y)[0]) for y in out_other2]
        else:
            return [[os.path.splitext(x)[0],path2,os.path.splitext(y)[0]] \
                        for [x,y] in out_pairs],\
                    [os.path.splitext(x)[0] for x in out_other1],\
                    [os.path.splitext(y)[0] for y in out_other2]

    def pair_r(path1,path2,exts1,exts2, with_root=False,with_ext=True):
        if(exts1=='' or exts2==''):
            print('\"ext1\" and \"ext2\" in funsion \"scanner.pair()\" must be set.')
            raise ValueError()
            return [],[],[]
        file1 = scanner.file_r(path1,exts1,False) 
        file2 = scanner.file_r(path2,exts2,False) 
        front1 = {os.path.splitext(f1)[0]:f1 for f1 in file1}
        front2 = {os.path.splitext(f2)[0]:f2 for f2 in file2}
        out_pairs = [[front1[k],front2[k]] for k in set(front1.keys()) & set(front2.keys())]
        out_other1 = [front1[k] for k in set(front1.keys()) - set(front2.keys())]
        out_other2 = [front2[k] for k in set(front2.keys()) - set(front1.keys())]

        if(with_root and with_ext):
            return [[os.path.join(path1,x),os.path.join(path2,y)] for [x,y] in out_pairs],\
                    [os.path.join(path1,x) for x in out_other1],\
                    [os.path.join(path2,y) for y in out_other2]
        elif(with_ext):
            return out_pairs,out_other1,out_other2
        elif(with_root):
            return [[os.path.join(path1,os.path.splitext(x)[0]),os.path.join(path2,os.path.splitext(y)[0])] \
                        for [x,y] in out_pairs],\
                    [os.path.join(path1,os.path.splitext(x)[0]) for x in out_other1],\
                    [os.path.join(path2,os.path.splitext(y)[0]) for y in out_other2]
        else:
            return [[os.path.splitext(x)[0],path2,os.path.splitext(y)[0]] \
                        for [x,y] in out_pairs],\
                    [os.path.splitext(x)[0] for x in out_other1],\
                    [os.path.splitext(y)[0] for y in out_other2]

    def text(path,sep=None):
        with open(path,'r') as fd:
            ls = fd.readlines()
        if(sep is None):
            sp_line = [s.split() for s in ls]
        else:
            sp_line = [s.split(sep) for s in ls]
        return [l for l in sp_line if len(l)!=0]
     
    #TO del
    def list(path,sep=''):
        if(os.path.isfile(path) is not True):
            print('Can not read list from file: ',path)
            return []
        list_from_file = []
        if(sep==''):
            with open(path) as fd:
                
                one_line = fd.readline()
                while(one_line):
                    one_line_list = one_line.split()
                    if(len(one_line_list)==0):
                        pass
                    else:
                        list_from_file.append(one_line_list)
                    one_line = fd.readline()
            return list_from_file
        else:
            with open(path) as fd:
                one_line = fd.readline()
                while(one_line):
                    one_line_list = one_line.split(sep)
                    if(len(one_line_list)==0):
                        pass
                    else:
                        list_from_file.append(one_line_list)
                    one_line = fd.readline()
            return list_from_file


#class VideoCaptureThreadReLoad(cv2.VideoCapture):
##class VideoCaptureThreadReLoad:


#    def __init__(self,cv_videocapture_param):
#        print(self)
#        cv2.VideoCapture.__init__(self)
#        print(self)
#        print('')
#        self.__init_param()
#        self.open(cv_videocapture_param)
        
#        return

#    def __init_param(self):
#        #self.__cvCap = cv2.VideoCapture()
#        #self.__isOpened = False

#        self.__cvMat = [[],[],[]]
#        self.__cvCap_read_index = 2
#        self.__user_read_index = 0
#        self.__ret = False

#        self.__thread_brk = False
#        self.__p_thread = None

#        self.__is_discarding_frame = True
        
#    def open(self, cv_videocapture_param):
#        print(self)      
#        self.release()
#        if(type(cv_videocapture_param)==str):
#            if(cv_videocapture_param.isnumeric()):
#                cv2.VideoCapture.open(self,int(cv_videocapture_param))
#                #self.__isOpened = self.__cvCap.open(int(cv_videocapture_param))
#            else:
#                cv2.VideoCapture.open(self,cv_videocapture_param)
#                #self.__isOpened = self.__cvCap.open(cv_videocapture_param)
#        else:
#            cv2.VideoCapture.open(self,cv_videocapture_param)
#            #self.__isOpened = self.__cvCap.open(cv_videocapture_param)

#        self.__ret,self.__cvMat[0] = cv2.VideoCapture.read(self)
#        #self.__ret,self.__cvMat[0] = self.__cvCap.read()
#        self.__cvMat[1] = self.__cvMat[0].copy()
#        self.__cvMat[2] = self.__cvMat[0].copy()
#        if(cv2.VideoCapture.isOpened(self)):
#            self.__thread_start()
#        #if( self.__isOpened):
#        #    self.__thread_start()
#        return cv2.VideoCapture.isOpened(self)

#    def __thread_start(self):
#        self.__p_thread = threading.Thread(target=VideoCaptureThread.get_img_in_thread,args = (self,),daemon=True)
#        self.__p_thread.start()
#        print('have start thread')
#        return

#    def get_img_in_thread(self):
#        while(self.__thread_brk is not True and self.__isOpened):
#            #print('getting')
#            self.__ret,self.__cvMat[self.__cvCap_read_index] = cv2.VideoCapture.read(self)
#            #self.__ret,self.__cvMat[self.__cvCap_read_index] = self.__cvCap.read()
#            if(self.__ret):
#                #print('ret true')
#                next_cvCap_read_index = self.__cvCap_read_index + 1
#                if(next_cvCap_read_index == 3):
#                    next_cvCap_read_index = 0
#                if(next_cvCap_read_index != self.__user_read_index):
#                    self.__cvCap_read_index = next_cvCap_read_index
#                else:
#                    self.__is_discarding_frame = True
#            else:
#                #print('ret false end')
#                return
#        #print('thread end')
            
#    def read(self):
#        #if(self.__isOpened is not True):
#        if(cv2.VideoCapture.isOpened(self)):
#            print('Video open fail')
#            return False,None
#        else:
#            ret = self.__ret
#            image = None
#            if(ret):
#                image = self.__cvMat[self.__user_read_index].copy()
#                next_user_read_index = self.__user_read_index + 1
#                if(next_user_read_index == 3):
#                    next_user_read_index = 0
#                if(next_user_read_index!= self.__cvCap_read_index):
#                    self.__user_read_index = next_user_read_index
#                else:
#                    self.__is_discarding_frame = False
#            return ret,image

#    def __thread_release(self):
#        self.__thread_brk = True
#        if(self.__p_thread is not None):
#            self.__p_thread.join()
#        return
        
#    def release(self):
#        self.__thread_release()
#        self.__init_param()
#        #if(self.__cvCap.isOpened()):
#            #self.__cvCap.release()
#        print(self)
#        print(cv2.VideoCapture.isOpened(self))
#        if(cv2.VideoCapture.isOpened(self)):
#            cv2.VideoCapture.release(self)

#        return 

#    #def isOpened(self):
#    #    return self.__isOpened

#    #def is_discarding_frame(self):
#    #    return self.__is_discarding_frame

#    #def __del__(self):
#    #    self.release()
#    #    return
    
#    #def get(self,propId):
#    #    return self.__cvCap.get(propId)

#class VideoCaptureThread():

#    def __init__(self,cv_videocapture_param):
#        self.__init_param()
#        self.open(cv_videocapture_param)
#        return

#    def __init_param(self):
#        self.__cvCap = cv2.VideoCapture()
#        self.__isOpened = False

#        self.__cvMat = [[],[],[]]
#        self.__cvCap_read_index = 2
#        self.__user_read_index = 0
#        self.__ret = False

#        self.__thread_brk = False
#        self.__p_thread = None

#        self.__is_discarding_frame = True
        
#    def open(self, cv_videocapture_param):      
#        self.release()
#        if(type(cv_videocapture_param)==str):
#            if(cv_videocapture_param.isnumeric()):
#                self.__isOpened = self.__cvCap.open(int(cv_videocapture_param))
#            else:
#                self.__isOpened = self.__cvCap.open(cv_videocapture_param)
#        else:
#            self.__isOpened = self.__cvCap.open(cv_videocapture_param)

#        self.__ret,self.__cvMat[0] = self.__cvCap.read()
#        self.__cvMat[1] = self.__cvMat[0].copy()
#        self.__cvMat[2] = self.__cvMat[0].copy()
#        if(self.__isOpened):
#            self.__thread_start()
#        return self.__isOpened

#    def __thread_start(self):
#        self.__p_thread = threading.Thread(target=VideoCaptureThread.get_img_in_thread,args = (self,),daemon=True)
#        self.__p_thread.start()
#        print('have start thread')
#        return

#    def get_img_in_thread(self):
#        while(self.__thread_brk is not True and self.__isOpened):
#            #print('getting')
#            self.__ret,self.__cvMat[self.__cvCap_read_index] = self.__cvCap.read()
#            if(self.__ret):
#                #print('ret true')
#                next_cvCap_read_index = self.__cvCap_read_index + 1
#                if(next_cvCap_read_index == 3):
#                    next_cvCap_read_index = 0
#                if(next_cvCap_read_index != self.__user_read_index):
#                    self.__cvCap_read_index = next_cvCap_read_index
#                else:
#                    self.__is_discarding_frame = True
#            else:
#                #print('ret false end')
#                return
#        #print('thread end')
            
#    def read(self):
#        if(self.__isOpened is not True):
#            print('Video open fail')
#            return False,None
#        else:
#            ret = self.__ret
#            image = None
#            if(ret):
#                image = self.__cvMat[self.__user_read_index].copy()
#                next_user_read_index = self.__user_read_index + 1
#                if(next_user_read_index == 3):
#                    next_user_read_index = 0
#                if(next_user_read_index!= self.__cvCap_read_index):
#                    self.__user_read_index = next_user_read_index
#                else:
#                    self.__is_discarding_frame = False
#            return ret,image

#    def __thread_release(self):
#        self.__thread_brk = True
#        if(self.__p_thread is not None):
#            self.__p_thread.join()
#        return
        
#    def release(self):
#        self.__thread_release()
#        self.__init_param()
#        if(self.__cvCap.isOpened()):
#            self.__cvCap.release()
#        return 

#    def isOpened(self):
#        return self.__isOpened

#    def is_discarding_frame(self):
#        return self.__is_discarding_frame

#    def __del__(self):
#        self.release()
#        return
    
#    def get(self,propId):
#        return self.__cvCap.get(propId)
      
  
class FPS:
    def __init__(self, num=30):
        self.__num=num
        self.__time_point = [time.time()]*num
        self.__index = 0
        self.__next_index = 1
        self.__count = 0
        self.__s_time = 0.000001
        return

    def click(self):
        if(self.__count!=self.__num):
            if(self.__index!=0):
                self.__time_point[self.__index] = time.time()
                self.__s_time = max((self.__time_point[self.__index] - self.__time_point[0]),0.000001)
            else:
                time0  = time.time()
                self.__s_time = max((time0 - self.__time_point[0]),0.000001)
                self.__time_point[self.__index] = time0
            self.__count += 1
        else:
            self.__time_point[self.__index] = time.time()
            self.__s_time = max((self.__time_point[self.__index] - self.__time_point[self.__next_index]),0.000001)
        self.__index = self.__next_index
        self.__next_index += 1
        if(self.__next_index==self.__num):
            self.__next_index = 0
        return

    #without click
    def get_only(self):
        return self.__count/self.__s_time

    #with click
    def get(self):
        self.click()
        return self.get_only()


#_colorwheel_RY = 15
#_colorwheel_YG = 6
#_colorwheel_GC = 4
#_colorwheel_CB = 11
#_colorwheel_BM = 13
#_colorwheel_MR = 6
#_colorwheel_list = [[0,255*i/_colorwheel_RY,255] for i in range(_colorwheel_RY)]
#_colorwheel_list = _colorwheel_list + [[0,255,255-i/_colorwheel_YG] for i in range(_colorwheel_YG)]
#_colorwheel_list = _colorwheel_list + [[255*i/_colorwheel_GC,255,0] for i in range(_colorwheel_GC)]
#_colorwheel_list = _colorwheel_list + [[255,255-i/_colorwheel_CB,0] for i in range(_colorwheel_CB)]
#_colorwheel_list = _colorwheel_list + [[255,0,255*i/_colorwheel_BM] for i in range(_colorwheel_BM)]
#_colorwheel_list = _colorwheel_list + [[255-255*i/_colorwheel_MR,0,255] for i in range(_colorwheel_MR)]
#_colorwheel = np.array(_colorwheel_list,np.float32)
#_ncols = _colorwheel_RY + _colorwheel_YG + _colorwheel_GC \
#    +_colorwheel_CB + _colorwheel_BM + _colorwheel_MR 

#_colorRing = cv2.imread(os.path.join(os.path.dirname(__file__), 'colorRing.png'))

#class calcOpticalFlow:
#    def flowToColor(flow,maxmotion=0):

#        #print(np.isnan(flow)[np.isnan(flow)==True])
#        unknown_place = np.abs(flow)>1e9
#        OK_place = np.bitwise_not(unknown_place)
#        flow_unknown_0 = flow.copy()
#        flow_unknown_0[unknown_place] = 0.0
#        fx = flow_unknown_0[:,:,0]
#        fy = flow_unknown_0[:,:,1]

#        if(maxmotion==0):
#            #print(type(fx))
#            #sq_value = 
#            #sq_value[sq_value<0.001] = 0.001
#            #print(np.max(sq_value))
#            rad = np.sqrt(fx*fx+fy*fy)
#            maxrad = rad.max()
#            rad = rad/maxrad
#        else:
#            maxrad = maxmotion
#            rad = np.sqrt(fx*fx+fy*fy)/maxrad

#        if(maxrad==0):#all are 0
#            maxrad = 1.0
#            rad = np.zeros([fx.shape[0],fx.shape[1]])

#        fx_norm = fx/maxrad
#        fy_norm = fy/maxrad
        
#        global _colorwheel
#        global _ncols

#        a = np.arctan2(-fy_norm,-fx_norm)/np.pi
#        fk = (a+1)/2*(_ncols-1)
#        k0 = fk.astype(np.int)
#        k1 = np.mod(k0+1,_colorwheel.shape[0])


#        f = fk - k0      
  
#        #print(k0.shape)
#        #print(k1.shape)
#        #print(f.shape)
        

#        col0 = _colorwheel[k0]/255.0
#        col1 = _colorwheel[k1]/255.0

#        col0 = col0.reshape([col0.shape[0],col0.shape[1],3])
#        col1 = col1.reshape([col1.shape[0],col1.shape[1],3])

#        #print(col0.shape)
#        #print(col1.shape)


#        f = f.reshape([f.shape[0],f.shape[1],1])


#        col = (1-f) * col0 + f * col1
#        rad_where_big_1 = rad>1
#        rad_where_not_big_1 = np.bitwise_not(rad_where_big_1)

#        for i in range(3):
#            col[:,:,i][rad_where_big_1] *=0.75
#            col[:,:,i][rad_where_not_big_1] = 1-rad[rad_where_not_big_1]*(1-col[:,:,i][rad_where_not_big_1])
#            #col[rad_where_not_big_1][0] = 1-rad[rad_where_not_big_1]*(1-col[rad_where_not_big_1][0])
#            #col[rad_where_not_big_1][1] = 1-rad[rad_where_not_big_1]*(1-col[rad_where_not_big_1][1])
#            #col[rad_where_not_big_1][2] = 1-rad[rad_where_not_big_1]*(1-col[rad_where_not_big_1][2])

#        output_color = (255.0*col).astype(np.uint8).reshape(flow.shape[0],flow.shape[1],3)
#        return output_color

#    def drawFlow(img,flow,stride = 10):
#        for y in range(0,img.shape[0],stride):
#            for x in range(0,img.shape[1],stride):
#                cv2.line(img,(x,y),(x+int(flow[y][x][0]),y+int(flow[y][x][1])),(0,255,0),1)
#        return

#    def colorRing():
#        global _colorRing
#        return _colorRing

#overload "read_thread_fun" "write_thread_fun", and then "__init__" class with these two function
class _IOThread:
    def __init__(self, read_param, use_write, cache,read_thread_target,write_thread_targe):
        self._read_param = read_param
        self._use_write = use_write
        self._cache = cache
        self._read_thread_target = read_thread_target
        self._write_thread_target = write_thread_targe
        self._reinit(self._read_thread_target, self._write_thread_target)

    def _reinit(self, read_target, write_target):
        self._p_fps_read = FPS()
        self._p_fps_write = FPS()

        self._i_read_param = 0
        self._len_read_param = len(self._read_param)

        self._q_read = queue.Queue(self._cache)
        self._q_write = queue.Queue(self._cache)
        self._read_thread_brk = False
        self._write_thread_brk = False

        self._p_fps_read.click()
        self._p_fps_write.click()

        if(self._len_read_param!=0):
            self._p_read_thread = threading.Thread(target=read_target,args = (self,),daemon=True)
            self._p_read_thread.start()
        else:
            self._p_read_thread = None
        if(self._use_write):
            self._p_write_thread = threading.Thread(target=write_target,args = (self,),daemon=True)
            self._p_write_thread.start()
        else:
            self._p_write_thread = None

        self._err = []

    def reset(self, read_param, use_write, cache):
        self.waitEnd()
        self._read_param = read_param
        self._use_write = use_write
        self._cache = cache
        self._reinit(self._read_thread_target,self._write_thread_target)

    def read_thread_fun(self):
        raise NotImplementedError
        for one_read_param in self._read_param:
            if(self._read_thread_brk):
                break
            if(self._q_read.full()):
                #sleeping will make it faster
                time.sleep(self._cache/self._p_fps_read.get_only()/2)
                pass
            #TODO
            #self._q_read.put(vocXmlRead(self._read_param[self._i_read_param]))
            self._i_read_param += 1
        return 
            
    def write_thread_fun(self):
        raise NotImplementedError
        while(self._write_thread_brk is not True or self._q_write.empty() is not True):
            if(self._q_write.empty()):
                #sleeping will make it faster
                time.sleep(self._cache/self._p_fps_write.get_only()/2)
                continue
            one_write_param = self._q_write.get()
            #TODO
            #vocXmlWrite(one_write_param[0],one_write_param[1])
        return  
 
    def read(self):
        if(self._p_read_thread is None):
            raise IOError
        if(self._i_read_param==self._len_read_param and self._q_read.empty()):
            print('warning: [*IO]: all are read.')
            return None
        out = self._q_read.get()
        self._p_fps_read.click()
        return out

    def write(self, *args):
        if(self._p_write_thread is None):
            raise IOError
        self._q_write.put(args)
        self._p_fps_write.click()
        return 

    def waitEnd(self):
        if(self._p_read_thread is not None):
            print('read thread waiting')
            self._read_thread_brk = True
            self._p_read_thread.join()
            print('read thread end')
        
        if(self._p_write_thread is not None):
            print('write thread waiting')
            self._write_thread_brk = True
            self._p_write_thread.join()
            print('write thread end')

        #while(self._q_write.empty()is not True):
        #    self._write_thread_brk = True
        #    self._p_write_thread.join()
        return 

    def error(self):
        out = self._err
        self._err = []
        return out
 
def makedirs(path):
    if(os.path.isdir(path) is not True):
        os.makedirs(path)
    return

class productionLineWorker:
    def __init__(self, func, maxsize=0, flow_type = 'FIFO', refresh_HZ = 1000):
        if(hasattr(func, '__call__')==False):
            raise TypeError
        self.func = func
        self.maxsize = maxsize
        self.flow_type = flow_type # or 'LIFO'
        self.refresh_HZ = refresh_HZ

class productionLine:
    def __init__(self, workers, maxsize = 0, flow_type='FIFO'):
        # param
        self._output_maxsize = maxsize
        self._flow_type = flow_type

        # workers
        self._workers = workers
        self._num_workers = len(self._workers)

        # data queue        
        _q_list = []
        for i,x in enumerate(self._workers):
            # x = productionLineWorker()
            if(x.flow_type=='FIFO'):
                _q_list.append(queue.Queue(x.maxsize))
            elif(x.flow_type=='LIFO'):
                _q_list.append(queue.LifoQueue(x.maxsize))
            else:
                raise TypeError('workers[{0}].flow_type=={1}'.format(i,x.flow_type))
        if(flow_type=='FIFO'):
            _q_list.append(queue.Queue(maxsize))
        elif(x.flow_type=='LIFO'):
            _q_list.append(queue.LifoQueue(maxsize))
        else:
            raise TypeError('productionLine.flow_type=={0}'.format(flow_type))
        self._q_list = _q_list

        # flags
        self._have_joined = False
        self._put_mutex = threading.Lock()
        self._to_brk_thread = [False]*self._num_workers
        self._brk_with_join = [True]*self._num_workers

        # lock
        _thread_mutex = []
        for i in range(self._num_workers):
            _thread_mutex.append(threading.Lock())
        self._thread_mutex = _thread_mutex

        # thread
        p_thread = []
        for i in range(self._num_workers):
            p_thread.append(threading.Thread(target=productionLine.thread_function,args=(self,i),daemon=True))
        for p in p_thread:
            p.start()
        self._p_thread = p_thread

    def thread_function(self,worker_index):
        lock = self._thread_mutex[worker_index]
        que = self._q_list[worker_index]
        worker = self._workers[worker_index]
        next_que = self._q_list[worker_index+1]
        while(self._to_brk_thread[worker_index]==False):
            if(que.empty()):
                time.sleep(1.0/worker.refresh_HZ)
            else:
                lock.acquire()
                data = que.get()
                output = worker.func(*data)
                next_que.put(output)
                lock.release()
        if(self._brk_with_join[worker_index]):
            while(que.empty()==False):
                data = que.get()
                output = worker.func(*data)
                next_que.put(output)
        return 
        
    def put(self,*data):
        self._put_mutex.acquire()
        if(self._have_joined):
            raise RuntimeError('productionLine has been joined.')
        elif(len(self._q_list)==0):
            raise RuntimeError('no worker in productionLine.')
        else:
            self._q_list[0].put(data)
        self._put_mutex.release()
        return 

    def release(self):
        self._put_mutex.acquire()
        self._have_joined = True
        for i in range(self._num_workers):
            self._brk_with_join[i] = False
            self._to_brk_thread[i] = True
            if(self._q_list[i+1].empty()==False):
                self._q_list[i+1].get()
            self._p_thread[i].join()
        self._put_mutex.release()
        return 

    def join(self):
        self._put_mutex.acquire()
        self._have_joined = True
        for i in range(self._num_workers):
            self._brk_with_join[i] = True
            self._to_brk_thread[i] = True
            self._p_thread[i].join()
        self._put_mutex.release()
        return 

    def waitFinish(self):
        self._put_mutex.acquire()
        for i in range(len(self._q_list)-1):
            worker = self._workers[i]
            que = self._q_list[i]
            lock = self._thread_mutex[i]
            while(1):
                lock.acquire()
                if(que.empty()):
                    lock.release()
                    break
                lock.release()
                time.sleep(1.0 / worker.refresh_HZ)
        self._put_mutex.release()
        return

    def get(self):
        if(self.empty()):
            time.sleep(1.0 / self._workers[self._num_workers].refresh_HZ)
            return False,None
        else:
            return True,self._q_list[self._num_workers].get()

    def empty(self):
        return self._q_list[self._num_workers].empty()

    def full(self):
        return self._q_list[self._num_workers].full()







#class imgIO(_IOThread):
#    def __init__(self, read_param=None,use_write = True, cache=10):
#        if(read_param is None):
#            _IOThread.__init__(self, [], use_write, cache, imgIO.read_thread_fun, imgIO.write_thread_fun)
#        else:
#            _IOThread.__init__(self, read_param, use_write, cache, imgIO.read_thread_fun, imgIO.write_thread_fun)
#        #self._read_param = read_param
#        #self._use_write = use_write
#        #self._cache = cache
#        #self._read_thread_target = imgIO.read_thread_fun
#        #self._write_thread_target = imgIO.write_thread_fun
#        #self._reinit(self._read_thread_target, self._write_thread_target)

#    def read_thread_fun(self):
#        for one_read_param in self._read_param:
#            if(self._read_thread_brk):
#                break
#            if(self._q_read.full()):
#                time.sleep(self._cache/self._p_fps_read.get_only()/2)
#                pass
#            if(type(one_read_param)==str):
#                one_img = cv2.imread(one_read_param)
#                #print('1----'+ str(one_read_param))
#                #print(one_img.dtype)
#                #print(one_img.shape)
#            elif(len(one_read_param)==2):
#                one_img = cv2.imread(one_read_param[0],one_read_param[1])
#                #print('2----'+str(one_read_param))
#                #print(one_img.dtype)
#                #print(one_img.shape)
#            else:
#                one_img = cv2.imread(one_read_param[0])
#                #print('3----'+str(one_read_param))
#                #print(one_img.dtype)
#                #print(one_img.shape)
#            if(one_img is None):
#                print('Can not read image with param: '+str(one_read_param))
#                raise IOError
#            self._q_read.put(one_img)
#            self._i_read_param += 1
#        return 
            
#    def write_thread_fun(self):
#        while(self._write_thread_brk is not True or self._q_write.empty() is not True):
#            if(self._q_write.empty()):
#                time.sleep(self._cache/self._p_fps_write.get_only()/2)
#                continue
#            one_write_param = self._q_write.get()
#            if(len(one_write_param)==3):
#                ret = cv2.imwrite(one_write_param[0],one_write_param[1],one_write_param[2])
#            else:
#                ret = cv2.imwrite(one_write_param[0],one_write_param[1])
#            if(ret==False):
#                err_str = 'image write fail: '+one_write_param[0]
#                self._err.append(err_str)
#                raise IOError
#        return 




##################################################################
#test module
##################################################################

def __test_FPS():
    import time
    p_fps = FPS()
    key=-1
    for x in range(100):
        time.sleep(1/30)
        print('fps = {:.6f}'.format(p_fps.get()))
    return    

def __test_scanner():
    path_to_scanner = r'G:\obj_mask_10'
    scanner_file = scanner.file(path_to_scanner)
    print('scanner_file = ')
    print(scanner_file)
    print()
    scanner_file_txt = scanner.file(path_to_scanner,'.txt')
    print('scanner_file(\'.txt\') = ')
    print(scanner_file_txt)
    print()
    scanner_file_r_img_full = scanner.file_r(path_to_scanner,'.jpg.png.jpeg')
    print('scanner_file_r_img_full[0:5] = ')
    print(scanner_file_r_img_full[0:5])
    print()
    scanner_folder = scanner.folder(path_to_scanner)
    print('scanner_folder = ')
    print(scanner_folder)
    print()
    scanner_list = scanner.text(os.path.join(path_to_scanner,'label.txt'))
    print('scanner_list[0:5] = ')
    print(scanner_list[0:5])
    print()
 
#def __test_calcOpticalFlow():
#    cap = cv2.VideoCapture(r'G:\hand\246.mp4')
#    imRsr = imResizer((720,480),RESIZE_TYPE.ROUNDUP,cv2.INTER_LINEAR)
#    fps = FPS()
#    key=0
#    ret = False
#    begin_num = 900
#    while((ret is not True) or (begin_num>0)):
#        begin_num -=1
#        ret ,preimg = cap.read()
#        prevgray = cv2.cvtColor(preimg,cv2.COLOR_BGR2GRAY)
#    while(key!=27):
#        ret ,img = cap.read()
#        if(ret):
#            #cal flow
#            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#            flow = cv2.calcOpticalFlowFarneback(prevgray, gray,None, 0.5, 3, 15, 3, 5, 1.2, 0);
#            #draw
#            drew_img = img.copy()
#            calcOpticalFlow.drawFlow(drew_img,flow,20)
#            img2show = imRsr.imResize(drew_img)
#            cv2.putText(img2show,str(img.shape[1])+' x '+str(img.shape[0]),(360,30),cv2.FONT_HERSHEY_COMPLEX,0.75,(0,255,0))
#            cv2.putText(img2show,str(round(fps.get(),3)),(525,30),cv2.FONT_HERSHEY_COMPLEX,0.75,(0,0,255))       
#            cv2.imshow('show',img2show)
#            #make flow color
#            color_flow = calcOpticalFlow.flowToColor(flow)
#            color2show = imRsr.imResize(color_flow)
#            cv2.imshow('color_flow',color2show)
#            cv2.imshow('color_ring',calcOpticalFlow.colorRing())

#        else:
#            break
#        key= cv2.waitKey(1)
#        prevgray = gray

def __test_productionLine():

    def readimg(img_path,i):
        time.sleep(2/1000)
        return np.zeros([720,1280,3],np.uint8),i
    
    def dealimg(img,i):
        cv2.line(img,(i,0),(i,10000),(0,0,255),3)
        time.sleep(3/1000)
        return img,i

    def saveimg(img,saved_index):
        time.sleep(1/1000)
        return saved_index

    begin = time.time()
    for i in range(720):
        img,_i = readimg('',i)
        img,_ii = dealimg(img,_i)
        saved_index = saveimg(img,_ii)
        #print(i,saved_index)
        if(i%72==0):
            print(saved_index)
    end = time.time()
    print('serial:    all result = {0}    total = {1:.4f}s    avg_ms = {2:.4f}'.format(720,end-begin,(end-begin)/720))

    print('')

    begin = time.time()

    workers = [productionLineWorker(readimg,30),\
               productionLineWorker(dealimg,30),\
               productionLineWorker(saveimg,30)]
    p_line = productionLine(workers)
    saved_indices = []    
    for i in range(720):
        p_line.put('',i)
        ret,got = p_line.get()
        if(ret):
            saved_indices.append(got)
    p_line.waitFinish()
    ret,got = p_line.get()
    while(ret):
        saved_indices.append(got)
        ret,got = p_line.get()
    print('len = '+str(len(saved_indices)))        

    for i,index in enumerate(saved_indices):
        if(i%72==0):
            print(index)
    end = time.time()
    print('multithreading:    all result = {0}    total = {1:.4f}s    avg_ms = {2:.4f}'.format(len(saved_indices),end-begin,(end-begin)/len(saved_indices)))
    
    return

if __name__=='__main__':

    save_folder_name = 'pyBoost_test_output'
    #__test_FPS()
    #__test_scanner()
    __test_productionLine()
    #__test_VideoCaptureThread()
    #__test_calcOpticalFlow()
    #__test_vocXMLIO(save_folder_name)

