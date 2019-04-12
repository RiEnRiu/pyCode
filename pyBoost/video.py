#-*-coding:utf-8-*-
import cv2
import numpy as np
import threading
import time
import os

class _VideoCaptureBase():
    def __init__(self, filename=None):
        if(filename is None):
            self._cvCap = cv2.VideoCapture()
        else:
            self._cvCap = cv2.VideoCapture(filename)
    
    def get(self,propId):
        return self._cvCap.get(propId)

    def grab(self):
        return self._cvCap.grab()

    def isOpened(self):
        return self._cvCap.isOpened()

    def open(self,filename):
        return self._cvCap.open(filename)

    def read(self,image=None):
        return self._cvCap.read(image)

    def release(self):
        return self._cvCap.release()

    def retrieve(self, image=None, flag=None):
        return self._cvCap.retrieve(image,flag)

    def set(self, propId, value):
        return self._cvCap.set(propId,value)

class VideoCaptureThread(_VideoCaptureBase):
    def __init__(self, filename=None, delay=0, refresh_HZ=10000):
        _VideoCaptureBase.__init__(self,filename)
        self._refresh_HZ = refresh_HZ
        self._delay = delay
        self._sleep_seconds = 0 if self._delay == 0 else self._delay / 1000.0

        self._cvReadRetAndMat = [(False, None), (False, None), (False, None)]
        self._cvCap_read_index = 2
        self._user_read_index = 0

        if self.isOpened():
            self._brk_read_thread = False
            self._p_read_thread = threading.Thread(\
                target=VideoCaptureThread.read_thread_fun, args=(self,), daemon=True)
            self._p_read_thread.start()
    
    def open(self,filename):
        self.release()
        _VideoCaptureBase.open(self,filename)
        if self.isOpened():
            self._brk_read_thread = False
            self._p_read_thread = threading.Thread(\
                target=VideoCaptureThread.read_thread_fun, args=(self,), daemon=True)
            self._p_read_thread.start()
            return True
        else:
            return False

    def read_thread_fun(self):
        while self._brk_read_thread == False:
            this_cvReadRetAndMat = _VideoCaptureBase.read(self)
            self._cvReadRetAndMat[self._cvCap_read_index] = this_cvReadRetAndMat
            next_cvCap_read_index = self._cvCap_read_index + 1
            if next_cvCap_read_index == 3:
                next_cvCap_read_index = 0
            if next_cvCap_read_index != self._user_read_index:
                self._cvCap_read_index = next_cvCap_read_index
            time.sleep(1.0 / self._refresh_HZ)
        self._cvReadRetAndMat[0] = (False, None)
        self._cvReadRetAndMat[1] = (False, None)
        self._cvReadRetAndMat[2] = (False, None)
        return

    def read(self):
        ret, image = self._cvReadRetAndMat[self._user_read_index]
        next_user_read_index = self._user_read_index + 1
        if next_user_read_index == 3:
            next_user_read_index = 0
        if next_user_read_index != self._cvCap_read_index:
            self._user_read_index = next_user_read_index
        if self._sleep_seconds != 0:
            time.sleep(self._sleep_seconds)
        return ret, image

    def release(self):
        self._brk_read_thread = True
        if self._p_read_thread is not None:
            self._p_read_thread.join()
            self._p_read_thread = None
        _VideoCaptureBase.release(self)
        self._cvReadRetAndMat[0] = (False, None)
        self._cvReadRetAndMat[1] = (False, None)
        self._cvReadRetAndMat[2] = (False, None)
        self._cvCap_read_index,self._user_read_index = 2,0
        return

class VideoCaptureRelink(_VideoCaptureBase):
    def __init__(self, filename=None, delay=0):
        _VideoCaptureBase.__init__(self, filename)
        self._filename = filename
        self._delay = delay
        self._sleep_seconds = 0 if self._delay==0 else self._delay/1000.0
        self._set_dict = {}
        self._is_linked = self.isOpened()
        self._p_relink_thread = None
    
    def relink_thread_fun(self):
        _VideoCaptureBase.release(self)
        if _VideoCaptureBase.open(self, self._filename):
            for k in self._set_dict.keys():
                _VideoCaptureBase.set(self, k, self._set_dict[k])
            self._is_linked = True
        return

    def open(self, filename):
        _VideoCaptureBase.release(self)
        self._is_linked = _VideoCaptureBase.open(self, filename)
        self._set_dict = {}
        return self._is_linked

    def read(self):
        if self._is_linked == False:
            ret,img = False,None
        else:
            ret, img = _VideoCaptureBase.read(self)
        if ret == False:
            if self._p_relink_thread is None or self._p_relink_thread.is_alive() == False:
                self._is_linked = False
                self._p_relink_thread = threading.Thread(\
                    target=VideoCaptureRelink.relink_thread_fun,\
                    args=(self,),
                    daemon = True)
                self._p_relink_thread.start()
        if self._sleep_seconds!=0:
            time.sleep(self._sleep_seconds)
        return ret, img

    def release(self):
        if self._p_relink_thread is not None:
            self._p_relink_thread.join()
            self._p_relink_thread = None
        _VideoCaptureBase.release(self)
        return

    def set(self,propId,value):
        self._set_dict[propId] = value
        return _VideoCaptureBase.set(self,propId,value)


class VideoCaptureThreadRelink(VideoCaptureRelink):
    def __init__(self, filename=None, delay=0, refresh_HZ=10000):
        VideoCaptureRelink.__init__(self,filename)
        self._refresh_HZ = refresh_HZ
        self._delay = delay
        self._sleep_seconds = 0 if self._delay == 0 else self._delay / 1000.0

        self._cvReadRetAndMat = [(False, None), (False, None), (False, None)]
        self._cvCap_read_index = 2
        self._user_read_index = 0

        if self.isOpened():
            self._brk_read_thread = False
            self._p_read_thread = threading.Thread(\
                target=VideoCaptureThread.read_thread_fun, args=(self,), daemon=True)
            self._p_read_thread.start()
    
    def open(self,filename):
        self.release()
        VideoCaptureRelink.open(self,filename)
        if self.isOpened():
            self._brk_read_thread = False
            self._p_read_thread = threading.Thread(\
                target=VideoCaptureThread.read_thread_fun, args=(self,), daemon=True)
            self._p_read_thread.start()
            return True
        else:
            return False

    def read_thread_fun(self):
        while self._brk_read_thread == False:
            this_cvReadRetAndMat = VideoCaptureRelink.read(self)
            self._cvReadRetAndMat[self._cvCap_read_index] = this_cvReadRetAndMat
            next_cvCap_read_index = self._cvCap_read_index + 1
            if next_cvCap_read_index == 3:
                next_cvCap_read_index = 0
            if next_cvCap_read_index != self._user_read_index:
                self._cvCap_read_index = next_cvCap_read_index
            time.sleep(1.0 / self._refresh_HZ)
        self._cvReadRetAndMat[0] = (False, None)
        self._cvReadRetAndMat[1] = (False, None)
        self._cvReadRetAndMat[2] = (False, None)
        return

    def read(self):
        ret, image = self._cvReadRetAndMat[self._user_read_index]
        next_user_read_index = self._user_read_index + 1
        if next_user_read_index == 3:
            next_user_read_index = 0
        if next_user_read_index != self._cvCap_read_index:
            self._user_read_index = next_user_read_index
        if self._sleep_seconds != 0:
            time.sleep(self._sleep_seconds)
        return ret, image

    def release(self):
        self._brk_read_thread = True
        if self._p_read_thread is not None:
            self._p_read_thread.join()
            self._p_read_thread = None
        VideoCaptureRelink.release(self)
        self._cvReadRetAndMat[0] = (False, None)
        self._cvReadRetAndMat[1] = (False, None)
        self._cvReadRetAndMat[2] = (False, None)
        self._cvCap_read_index,self._user_read_index = 2,0
        return


_colorwheel_RY = 15
_colorwheel_YG = 6
_colorwheel_GC = 4
_colorwheel_CB = 11
_colorwheel_BM = 13
_colorwheel_MR = 6
_colorwheel_list = [[0,255*i/_colorwheel_RY,255] for i in range(_colorwheel_RY)]
_colorwheel_list = _colorwheel_list + [[0,255,255-i/_colorwheel_YG] for i in range(_colorwheel_YG)]
_colorwheel_list = _colorwheel_list + [[255*i/_colorwheel_GC,255,0] for i in range(_colorwheel_GC)]
_colorwheel_list = _colorwheel_list + [[255,255-i/_colorwheel_CB,0] for i in range(_colorwheel_CB)]
_colorwheel_list = _colorwheel_list + [[255,0,255*i/_colorwheel_BM] for i in range(_colorwheel_BM)]
_colorwheel_list = _colorwheel_list + [[255-255*i/_colorwheel_MR,0,255] for i in range(_colorwheel_MR)]
_colorwheel = np.array(_colorwheel_list,np.float32)
_ncols = _colorwheel_RY + _colorwheel_YG + _colorwheel_GC \
    +_colorwheel_CB + _colorwheel_BM + _colorwheel_MR 

_colorRing = cv2.imread(os.path.join(os.path.dirname(__file__), 'colorRing.png'))

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


def opticalFlowToColorMap(flow, maxmotion=0):

    #print(np.isnan(flow)[np.isnan(flow)==True])
    unknown_place = np.abs(flow)>1e9
    OK_place = np.bitwise_not(unknown_place)
    flow_unknown_0 = flow.copy()
    flow_unknown_0[unknown_place] = 0.0
    fx = flow_unknown_0[:,:,0]
    fy = flow_unknown_0[:,:,1]

    if(maxmotion==0):
        #print(type(fx))
        #sq_value = 
        #sq_value[sq_value<0.001] = 0.001
        #print(np.max(sq_value))
        rad = np.sqrt(fx*fx+fy*fy)
        maxrad = rad.max()
        rad = rad/maxrad
    else:
        maxrad = maxmotion
        rad = np.sqrt(fx*fx+fy*fy)/maxrad

    if(maxrad==0):#all are 0
        maxrad = 1.0
        rad = np.zeros([fx.shape[0],fx.shape[1]])

    fx_norm = fx/maxrad
    fy_norm = fy/maxrad
        
    global _colorwheel
    global _ncols

    a = np.arctan2(-fy_norm,-fx_norm)/np.pi
    fk = (a+1)/2*(_ncols-1)
    k0 = fk.astype(np.int)
    k1 = np.mod(k0+1,_colorwheel.shape[0])


    f = fk - k0      
  
    #print(k0.shape)
    #print(k1.shape)
    #print(f.shape)
        

    col0 = _colorwheel[k0]/255.0
    col1 = _colorwheel[k1]/255.0

    col0 = col0.reshape([col0.shape[0],col0.shape[1],3])
    col1 = col1.reshape([col1.shape[0],col1.shape[1],3])

    #print(col0.shape)
    #print(col1.shape)


    f = f.reshape([f.shape[0],f.shape[1],1])


    col = (1-f) * col0 + f * col1
    rad_where_big_1 = rad>1
    rad_where_not_big_1 = np.bitwise_not(rad_where_big_1)

    for i in range(3):
        col[:,:,i][rad_where_big_1] *=0.75
        col[:,:,i][rad_where_not_big_1] = 1-rad[rad_where_not_big_1]*(1-col[:,:,i][rad_where_not_big_1])
        #col[rad_where_not_big_1][0] = 1-rad[rad_where_not_big_1]*(1-col[rad_where_not_big_1][0])
        #col[rad_where_not_big_1][1] = 1-rad[rad_where_not_big_1]*(1-col[rad_where_not_big_1][1])
        #col[rad_where_not_big_1][2] = 1-rad[rad_where_not_big_1]*(1-col[rad_where_not_big_1][2])

    output_color = (255.0*col).astype(np.uint8).reshape(flow.shape[0],flow.shape[1],3)
    return output_color

def drawOpticalFlow(img,flow,stride = 10):
    for y in range(0,img.shape[0],stride):
        for x in range(0,img.shape[1],stride):
            cv2.line(img,(x,y),(x+int(flow[y][x][0]),y+int(flow[y][x][1])),(0,255,0),1)
    return

def colorRing():
    global _colorRing
    return _colorRing









if __name__=='__main__':

    ##################################################################
    #test module
    ##################################################################
    import sys
    sys.path.append('../')
    import pyBoost as pb
    def test_VideoCaptureThread():
        cap = pb.video.VideoCaptureThread(r'G:\hand\246.mp4')   
        key=0
        fps = pb.FPS()
        while key!=27 :
            print(fps.get())
            ret ,img = cap.read()
            if(ret):
                cv2.imshow('img',img)
            key= cv2.waitKey(1)
        cap.release()
        print('finish')

    def test_VideoCaptureRelink():
        cap = pb.video.VideoCaptureRelink(r'G:\hand\246.mp4')   
        key=0
        fps = pb.FPS()
        while key!=27 :
            print(fps.get())
            ret ,img = cap.read()
            if(ret):
                cv2.imshow('img',img)
            key= cv2.waitKey(1)
        cap.release()
        print('finish')

    def test_VideoCaptureThreadRelink():
        cap = pb.video.VideoCaptureThreadRelink(r'G:\hand\246.mp4')   
        key=0
        fps = pb.FPS()
        while(key!=27):
            print(fps.get())
            ret ,img = cap.read()
            if(ret):
                cv2.imshow('img',img)
            key= cv2.waitKey(1)
        cap.release()
        print('finish')

    def test_opticalFlow():
        cap = cv2.VideoCapture(r'G:\hand\246.mp4')
        fps = pb.FPS()
        key=0
        ret = False
        begin_num = 900
        while((ret == False) or (begin_num>0)):
            begin_num -=1
            ret ,preimg = cap.read()
            prevgray = cv2.cvtColor(preimg,cv2.COLOR_BGR2GRAY)
        while(key!=27):
            ret ,img = cap.read()
            if(ret):
                #cal flow
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prevgray, gray,None, 0.5, 3, 15, 3, 5, 1.2, 0);
                #draw
                drew_img = img.copy()
                pb.video.drawOpticalFlow(drew_img,flow,20)
                img2show = cv2.resize(drew_img,(720,480))
                cv2.putText(img2show,str(img.shape[1])+' x '+str(img.shape[0]),(360,30),cv2.FONT_HERSHEY_COMPLEX,0.75,(0,255,0))
                cv2.putText(img2show,str(round(fps.get(),3)),(525,30),cv2.FONT_HERSHEY_COMPLEX,0.75,(0,0,255))       
                cv2.imshow('show',img2show)
                #make flow color
                color_flow = pb.video.opticalFlowToColorMap(flow)
                color2show = cv2.resize(color_flow,(720,480))
                cv2.imshow('color_flow',color2show)
                cv2.imshow('color_ring',pb.video.colorRing())

            else:
                break
            key= cv2.waitKey(1)
            prevgray = gray

    #####################################################################
    save_folder_name = 'pyBoost_test_output'
    #test_VideoCaptureThread()
    test_VideoCaptureRelink()
    #test_VideoCaptureThreadRelink()
    #test_opticalFlow()