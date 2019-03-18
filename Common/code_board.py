
import cv2
import numpy as np
import threading
import time
import queue

class VideoCaptureThreadReLoad(cv2.VideoCapture):
    def __init__(self,cv_videocapture_param, reopen_seconds=1):
        cv2.VideoCapture.__init__(self)

        self._cv_videocapture_param = cv_videocapture_param
        self._reopen_seconds=1

        self._init_param()
        self.open(cv_videocapture_param)

    def _init_param(self):
        self._qcvCapRead = queue.Queue(30)
        self._lastCapRead = None
       
        self._brk_read_thread = False
        self._p_read_thread = None

        self._linked = False

        self._brk_reopen_thread = False
        self._p_reopen_thread = None

        self._capSetParam = {}

    def set(self, propId, value):
        self._capSetParam[propId] = value
        flag = cv2.VideoCapture.set(self, propId, value)
        return flag

    def open(self,cv_videocapture_param):
        self._cv_videocapture_param = cv_videocapture_param
        cv2.VideoCapture.open(self,cv_videocapture_param)
        cvReadResult = cv2.VideoCapture.read(self)
        self._lastCapRead = cvReadResult
        self._qcvCapRead.put(cvReadResult)
        if(cv2.VideoCapture.isOpened(self)):
            self._read_thread_start()
            self._linked = True
        self._reopen_thread_start()
        return cv2.VideoCapture.isOpened(self)

    def read_in_thread(self):
        while(self._brk_read_thread is not True):
            if(self._linked):
                cvReadResult = cv2.VideoCapture.read(self)
                if(cvReadResult[0]):
                    self._lastCapRead = cvReadResult
                    if(self._qcvCapRead.full() is not True):
                        self._qcvCapRead.put(cvReadResult)
                else:
                    self._linked = False
                    cv2.VideoCapture.release(self)
        return

    def _read_thread_start(self):
        self._p_read_thread = threading.Thread(\
            target=VideoCaptureThreadReLoad.read_in_thread,\
            args = (self,),\
            daemon=True)
        self._p_read_thread.start()
        print('read from VideoCapture with thread.')
        return

    def _read_thread_release(self):
        self._brk_read_thread = True
        if(self._p_read_thread is not None):
            self._p_read_thread.join()
            self._p_read_thread = None
        return

    def reopen_in_thread(self):
        while(self._brk_reopen_thread is not True):
            if(self._linked is not True):
                cv2.VideoCapture.open(self,self._cv_videocapture_param)
                if(cv2.VideoCapture.isOpened(self)):
                    for k in self._capSetParam:
                        cv2.VideoCapture.set(self,k,self._capSetParam[k])
                self._linked = cv2.VideoCapture.isOpened(self)
                time.sleep(1)
        return

    def _reopen_thread_start(self):
        self._p_reopen_thread = threading.Thread(\
            target=VideoCaptureThreadReLoad.reopen_in_thread,\
            args = (self,),\
            daemon=True)
        self._p_reopen_thread.start()
        return

    def _reopen_thread_release(self):
        self._brk_reopen_thread = True
        if(self._p_reopen_thread is not None):
            self._p_reopen_thread.join()
            self._p_reopen_thread = None
        return

    def read(self):
        if(self._qcvCapRead.qsize()==0):
            ret = self._lastCapRead[0]
            image =  self._lastCapRead[1].copy()
        else:
            ret, image = self._qcvCapRead.get()
        return ret,image

    def release(self):
        self._read_thread_release()
        self._reopen_thread_release()
        self._init_param()
        cv2.VideoCapture.release(self)
        return


class VideoCaptureThread222:
    def __init__(self, _cvCap):
        self.cvCap = _cvCap
        # self.cvCap = cv2.VideoCapture()

        self._cvReadRetAndMat = [(False,None),(False,None),(False,None)]
        self._cvCap_read_index = 2
        self._user_read_index = 0
        self._cvReadRetAndMat[0] = self.cvCap.read()
        self._cvReadRetAndMat[1] = self.cvCap.read()
        self._cvReadRetAndMat[2] = self.cvCap.read()

        self._brk_read_thread = False
        self._p_read_thread = threading.Thread(\
            target=VideoCaptureThread222.read_thread_fun,args = (self,),daemon=True)
        self._p_read_thread.start()

    def read_thread_fun(self):
        while(self._brk_read_thread is not True):
            this_cvReadRetAndMat = self.cvCap.read()
            if(this_cvReadRetAndMat[0]):
                self._cvReadRetAndMat[self._cvCap_read_index] = this_cvReadRetAndMat
                next_cvCap_read_index = self._cvCap_read_index + 1
                if(next_cvCap_read_index == 3):
                    next_cvCap_read_index = 0
                if(next_cvCap_read_index != self._user_read_index):
                    self._cvCap_read_index = next_cvCap_read_index
            else:
                self._cvReadRetAndMat = [(False, None),(False, None),(False, None)]
                self.cvCap.release()
                break
            time.sleep(0.0001)
        self._cvReadRetAndMat = [(False, None),(False, None),(False, None)]
        self.cvCap.release()
        # print('have release camera')
        return
            
    def read(self):
        ret,image = self._cvReadRetAndMat[self._user_read_index]
        next_user_read_index = self._user_read_index + 1
        if(next_user_read_index == 3):
            next_user_read_index = 0
        if(next_user_read_index!= self._cvCap_read_index):
            self._user_read_index = next_user_read_index
        return ret,image

    def release(self):
        self._brk_read_thread = True
        if(self._p_read_thread is not None):
            self._p_read_thread.join()
            self._p_read_thread = None
        return 

class VideoCaptureTreadRelink:
    def __init__(self,_open_param, _video_param_dict = None, _default_img = None,_heartbeat_seconds = 1):
        self._open_param = _open_param
        self._video_param_dict = _video_param_dict
        self._default_img = _default_img
        self._heartbeat_seconds = _heartbeat_seconds
        self._default_frame = (False,_default_img)   
 
        self.cvCap = cv2.VideoCapture(self._open_param)
        if(self.cvCap.isOpened() and self._video_param_dict is not None):
            for k in self._video_param_dict:
                self.cvCap.set(k,self._video_param_dict[k])
        self._p_VideoCaptureThread222 = VideoCaptureThread222(self.cvCap)
        self._linked = self.cvCap.isOpened()

        self._brk_reopen_thread = False
        self._p_reopen_thread = threading.Thread(\
            target=VideoCaptureTreadRelink.reopen_thread_fun, args = (self,), daemon=True)
        self._p_reopen_thread.start()

    def reopen_thread_fun(self):
        while(self._brk_reopen_thread is not True):
            if(self._linked is not True):
                self._p_VideoCaptureThread222.release()
                self.cvCap = cv2.VideoCapture(self._open_param)
                if(self.cvCap.isOpened() and self._video_param_dict is not None):
                    for k in self._video_param_dict:
                        self.cvCap.set(k,self._video_param_dict[k])
                self._p_VideoCaptureThread222 = VideoCaptureThread222(self.cvCap)
                self._linked = self.cvCap.isOpened()
            time.sleep(self._heartbeat_seconds)
        return

     
    def read(self):
        if(self._linked):
            ret,img = self._p_VideoCaptureThread222.read()
        else:
            ret,img = self._default_frame
        if(ret is not True):
            self._p_VideoCaptureThread222.release()
            self._linked = False
            ret,img = self._default_frame
        return ret,img            

    def release(self):
        self._p_VideoCaptureThread222.release()
        self._brk_reopen_thread = True
        if(self._p_reopen_thread is not None):
            self._p_reopen_thread.join()
        self._linked = False
        return
        
    def isLinked(self):
        return self._linked


class VideoCaptureThreadReLoad(cv2.VideoCapture):
    def __init__(self,cv_videocapture_param, reopen_seconds=1):
        cv2.VideoCapture.__init__(self)

        self._cv_videocapture_param = cv_videocapture_param
        self._reopen_seconds=1

        self._init_param()
        self.open(cv_videocapture_param)

    def _init_param(self):
        self._qcvCapRead = queue.Queue(30)
        self._lastCapRead = None
       
        self._brk_read_thread = False
        self._p_read_thread = None

        self._linked = False

        self._brk_reopen_thread = False
        self._p_reopen_thread = None

        self._capSetParam = {}

    def set(self, propId, value):
        self._capSetParam[propId] = value
        flag = cv2.VideoCapture.set(self, propId, value)
        return flag

    def open(self,cv_videocapture_param):
        self._cv_videocapture_param = cv_videocapture_param
        cv2.VideoCapture.open(self,cv_videocapture_param)
        cvReadResult = cv2.VideoCapture.read(self)
        self._lastCapRead = cvReadResult
        self._qcvCapRead.put(cvReadResult)
        if(cv2.VideoCapture.isOpened(self)):
            self._read_thread_start()
            self._linked = True
        self._reopen_thread_start()
        return cv2.VideoCapture.isOpened(self)

    def read_in_thread(self):
        while(self._brk_read_thread is not True):
            if(self._linked):
                cvReadResult = cv2.VideoCapture.read(self)
                if(cvReadResult[0]):
                    self._lastCapRead = cvReadResult
                    if(self._qcvCapRead.full() is not True):
                        self._qcvCapRead.put(cvReadResult)
                else:
                    self._linked = False
                    cv2.VideoCapture.release(self)
        return

    def _read_thread_start(self):
        self._p_read_thread = threading.Thread(\
            target=VideoCaptureThreadReLoad.read_in_thread,\
            args = (self,),\
            daemon=True)
        self._p_read_thread.start()
        # print('read from VideoCapture with thread.')
        return

    def _read_thread_release(self):
        self._brk_read_thread = True
        if(self._p_read_thread is not None):
            self._p_read_thread.join()
            self._p_read_thread = None
        return

    def reopen_in_thread(self):
        while(self._brk_reopen_thread is not True):
            if(self._linked is not True):
                cv2.VideoCapture.open(self,self._cv_videocapture_param)
                if(cv2.VideoCapture.isOpened(self)):
                    for k in self._capSetParam:
                        cv2.VideoCapture.set(self,k,self._capSetParam[k])
                self._linked = cv2.VideoCapture.isOpened(self)
                time.sleep(1)
        return

    def _reopen_thread_start(self):
        self._p_reopen_thread = threading.Thread(\
            target=VideoCaptureThreadReLoad.reopen_in_thread,\
            args = (self,),\
            daemon=True)
        self._p_reopen_thread.start()
        return

    def _reopen_thread_release(self):
        self._brk_reopen_thread = True
        if(self._p_reopen_thread is not None):
            self._p_reopen_thread.join()
            self._p_reopen_thread = None
        return

    def read(self):
        if(self._qcvCapRead.qsize()==0):
            ret = self._lastCapRead[0]
            image =  self._lastCapRead[1].copy()
        else:
            ret, image = self._qcvCapRead.get()
        return ret,image

    def release(self):
        self._read_thread_release()
        self._reopen_thread_release()
        self._init_param()
        cv2.VideoCapture.release(self)
        return


if __name__== '__main__':
    # set_dict = {cv2.CAP_PROP_FRAME_WIDTH:1980,cv2.CAP_PROP_FRAME_HEIGHT:1980}

    #cvCap = cv2.VideoCapture(0)
    #for k in set_dict:
    #    cvCap.set(k,set_dict[k])

    #cap = VideoCaptureThread222(cvCap)
    # cap = VideoCaptureTreadRelink(0,set_dict,np.zeros((1280,720,3),np.uint8))

    cap = cv2.VideoCapture(0)
    print(cap.isOpened())   
    key=0
    ttt = time.time()
    time.sleep (0.3)
    iii = 0
    while(key!=27):
        ret ,img = cap.read()
        if(iii%30==0):
            temp_ttt = time.time()
            print(str(30/(temp_ttt-ttt))+'    '+str(ret))
            ttt = temp_ttt
        #if(ret):
        cv2.imshow('img',img)
        key= cv2.waitKey(30)
        iii += 1
    cap.release()

    




    print('finish')   

  

    # print(conf.get('123','1'))





