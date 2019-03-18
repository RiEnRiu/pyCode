
import cv2
import numpy as np
import threading
import time
import os


class VideoCaptureThread:
    def __init__(self, _cvCap, _delay):
        self.cvCap = _cvCap
        # self.cvCap = cv2.VideoCapture()

        self._cvReadRetAndMat = [(False, None), (False, None), (False, None)]
        self._cvCap_read_index = 2
        self._user_read_index = 0
        self._cvReadRetAndMat[0] = self.cvCap.read()
        self._cvReadRetAndMat[1] = self.cvCap.read()
        self._cvReadRetAndMat[2] = self.cvCap.read()

        self._brk_read_thread = False
        self._p_read_thread = threading.Thread( \
            target=VideoCaptureThread.read_thread_fun, args=(self,), daemon=True)
        self._p_read_thread.start()

    def read_thread_fun(self):
        while (self._brk_read_thread is not True):
            this_cvReadRetAndMat = self.cvCap.read()
            if (this_cvReadRetAndMat[0]):
                self._cvReadRetAndMat[self._cvCap_read_index] = this_cvReadRetAndMat
                next_cvCap_read_index = self._cvCap_read_index + 1
                if (next_cvCap_read_index == 3):
                    next_cvCap_read_index = 0
                if (next_cvCap_read_index != self._user_read_index):
                    self._cvCap_read_index = next_cvCap_read_index
            else:
                self._cvReadRetAndMat = [(False, None), (False, None), (False, None)]
                self.cvCap.release()
                break
            time.sleep(0.0001)
        self._cvReadRetAndMat = [(False, None), (False, None), (False, None)]
        self.cvCap.release()
        # print('have release camera')
        return

    def read(self):
        ret, image = self._cvReadRetAndMat[self._user_read_index]
        next_user_read_index = self._user_read_index + 1
        if (next_user_read_index == 3):
            next_user_read_index = 0
        if (next_user_read_index != self._cvCap_read_index):
            self._user_read_index = next_user_read_index
        return ret, image

    def release(self):
        self._brk_read_thread = True
        if (self._p_read_thread is not None):
            self._p_read_thread.join()
            self._p_read_thread = None
        return

    def get(self):
        return self.cvCap.get()


class VideoCaptureThreadRelink:
    def __init__(self, _open_param, _video_param_dict=None, _default_img=None, _heartbeat_seconds=1):
        self._open_param = _open_param
        self._video_param_dict = _video_param_dict
        self._default_img = _default_img
        self._heartbeat_seconds = _heartbeat_seconds
        self._default_frame = (False, _default_img)

        self.cvCap = cv2.VideoCapture(self._open_param)
        if (self.cvCap.isOpened() and self._video_param_dict is not None):
            for k in self._video_param_dict:
                self.cvCap.set(k, self._video_param_dict[k])
        self._p_VideoCaptureThread = VideoCaptureThread(self.cvCap)
        self._linked = self.cvCap.isOpened()

        self._brk_reopen_thread = False
        self._p_reopen_thread = threading.Thread(\
            target=VideoCaptureThreadRelink.reopen_thread_fun, args=(self,), daemon=True)
        self._p_reopen_thread.start()

    def reopen_thread_fun(self):
        while (self._brk_reopen_thread is not True):
            if (self._linked is not True):
                self._p_VideoCaptureThread.release()
                try:
                    self.cvCap = cv2.VideoCapture(self._open_param)
                except Exception as e:
                    self.cvCap = cv2.VideoCapture()
                if (self.cvCap.isOpened() and self._video_param_dict is not None):
                    for k in self._video_param_dict:
                        self.cvCap.set(k, self._video_param_dict[k])
                self._p_VideoCaptureThread = VideoCaptureThread(self.cvCap)
                self._linked = self.cvCap.isOpened()
            time.sleep(self._heartbeat_seconds)
        return

    def read(self):
        if (self._linked):
            ret, img = self._p_VideoCaptureThread.read()
            if (ret is not True):
                self._p_VideoCaptureThread.release()
                self._linked = False
                ret, img = self._default_frame
        else:
            ret, img = self._default_frame
        return ret, img

    def release(self):
        self._p_VideoCaptureThread.release()
        self._brk_reopen_thread = True
        if (self._p_reopen_thread is not None):
            self._p_reopen_thread.join()
            self._p_reopen_thread = None
        self._linked = False
        return

    def set(self,propId,value):
        self._video_param_dict[propId] = value
        return self.cvCap.set(propId,value)

    def get(self):
        return self.cvCap.get()

    def isOpened(self):
        return self.cvCap.isOpened()


class VideoCaptureRelink:
    def __init__(self, _open_param, _video_param_dict=None, _heartbeat_seconds=1):
        self._open_param = _open_param
        self._video_param_dict = _video_param_dict
        self._heartbeat_seconds = _heartbeat_seconds

        self._default_frame = (False, None)

        self.cvCap = cv2.VideoCapture(self._open_param)
        if (self.cvCap.isOpened() and self._video_param_dict is not None):
            for k in self._video_param_dict:
                self.cvCap.set(k, self._video_param_dict[k])
        self._linked = self.cvCap.isOpened()

        self._brk_reopen_thread = False
        self._p_reopen_thread = threading.Thread(\
            target=VideoCaptureRelink.reopen_thread_fun, args=(self,), daemon=True)
        self._p_reopen_thread.start()

    def reopen_thread_fun(self):
        while (self._brk_reopen_thread is not True):
            if (self._linked is not True):
                self.cvCap.release()
                try:
                    self.cvCap = cv2.VideoCapture(self._open_param)
                except Exception as e:
                    self.cvCap = cv2.VideoCapture()
                if (self.cvCap.isOpened() and self._video_param_dict is not None):
                    for k in self._video_param_dict:
                        self.cvCap.set(k, self._video_param_dict[k])
                self._linked = self.cvCap.isOpened()
            time.sleep(self._heartbeat_seconds)
        return

    def read(self):
        if (self._linked):
            ret, img = self.cvCap.read()
            if (ret is not True):
                self.cvCap.release()
                self._linked = False
                ret, img = self._default_frame
        else:
            ret, img = self._default_frame
        return ret, img

    def set(self,propId,value):
        self._video_param_dict[propId] = value
        return self.cvCap.set(propId,value)

    def release(self):
        self._brk_reopen_thread = True
        if (self._p_reopen_thread is not None):
            self._p_reopen_thread.join()
        self.cvCap.release()
        self._linked = False
        return

    def get(self):
        return self.cvCap.get()

    def isOpened(self):
        return self.cvCap.isOpened()


class imShowerThread:
    __img_dict = dict()
    __destroy_set = set()
    __brk_show_thread = False
    __p_show_thread = None

    def show_thread_fun():
        while(imShowerThread.__brk_show_thread==False):
            # print(imShowerThread.__destroy_set,set(imShowerThread.__img_dict.keys()))
            # destroy
            num_win_destroy = 0
            win_destroy_keys = list(imShowerThread.__destroy_set)
            for win in win_destroy_keys:
                if(imShowerThread.__img_dict.get(win) is not None):
                    imShowerThread.__img_dict.pop(win)
                try:
                    cv2.destroyWindow(win)
                    num_win_destroy += 1
                except Exception as e:
                    # print(e)
                    imShowerThread.__destroy_set.remove(win)

            # show
            num_win_showing = 0
            win_show_keys = list(imShowerThread.__img_dict.keys())
            for win in win_show_keys:
                img = imShowerThread.__img_dict[win]
                try:
                    cv2.imshow(win,img)
                    num_win_showing += 1
                except Exception as e:
                    print(e)
                    imShowerThread.__destroy_set.add(win)
                    continue

            if(num_win_showing==0 and num_win_destroy==0):
                time.sleep(0.01)
            else:
                cv2.waitKey(10)
        return

    def show(winname,mat):
        if(imShowerThread.__p_show_thread is None):
            imShowerThread.__p_show_thread = threading.Thread(target=imShowerThread.show_thread_fun,daemon=True)
            imShowerThread.__p_show_thread.start()
        this_img_to_show = np.zeros(mat.shape,mat.dtype)
        this_img_to_show = this_img_to_show + mat
        imShowerThread.__img_dict[winname] = this_img_to_show
        # print(imShowerThread.__destroy_set,set(imShowerThread.__img_dict.keys()))

        return

    def destroyWindow(winname):
        imShowerThread.__destroy_set.add(winname)
        return

    def destroyAllWindows():
        imShowerThread.__destroy_set.update(imShowerThread.__img_dict.keys())
        return



#Bug!
#class VideoCaptureThreadReLoad(cv2.VideoCapture):
#    def __init__(self,cv_videocapture_param, reopen_seconds=1):
#        cv2.VideoCapture.__init__(self)

#        self._cv_videocapture_param = cv_videocapture_param
#        self._reopen_seconds=1

#        self._init_param()
#        self.open(cv_videocapture_param)

#    def _init_param(self):
#        self._qcvCapRead = queue.Queue(30)
#        self._lastCapRead = None
       
#        self._brk_read_thread = False
#        self._p_read_thread = None

#        self._linked = False

#        self._brk_reopen_thread = False
#        self._p_reopen_thread = None

#        self._capSetParam = {}

#    def set(self, propId, value):
#        self._capSetParam[propId] = value
#        flag = cv2.VideoCapture.set(self, propId, value)
#        return flag

#    def open(self,cv_videocapture_param):
#        self._cv_videocapture_param = cv_videocapture_param
#        cv2.VideoCapture.open(self,cv_videocapture_param)
#        cvReadResult = cv2.VideoCapture.read(self)
#        self._lastCapRead = cvReadResult
#        self._qcvCapRead.put(cvReadResult)
#        if(cv2.VideoCapture.isOpened(self)):
#            self._read_thread_start()
#            self._linked = True
#        self._reopen_thread_start()
#        return cv2.VideoCapture.isOpened(self)

#    def read_in_thread(self):
#        while(self._brk_read_thread is not True):
#            if(self._linked):
#                cvReadResult = cv2.VideoCapture.read(self)
#                if(cvReadResult[0]):
#                    self._lastCapRead = cvReadResult
#                    if(self._qcvCapRead.full() is not True):
#                        self._qcvCapRead.put(cvReadResult)
#                else:
#                    self._linked = False
#                    cv2.VideoCapture.release(self)
#        return

#    def _read_thread_start(self):
#        self._p_read_thread = threading.Thread(\
#            target=VideoCaptureThreadReLoad.read_in_thread,\
#            args = (self,),\
#            daemon=True)
#        self._p_read_thread.start()
#        print('read from VideoCapture with thread.')
#        return

#    def _read_thread_release(self):
#        self._brk_read_thread = True
#        if(self._p_read_thread is not None):
#            self._p_read_thread.join()
#            self._p_read_thread = None
#        return

#    def reopen_in_thread(self):
#        while(self._brk_reopen_thread is not True):
#            if(self._linked is not True):
#                cv2.VideoCapture.open(self,self._cv_videocapture_param)
#                if(cv2.VideoCapture.isOpened(self)):
#                    for k in self._capSetParam:
#                        cv2.VideoCapture.set(self,k,self._capSetParam[k])
#                self._linked = cv2.VideoCapture.isOpened(self)
#                time.sleep(1)
#        return

#    def _reopen_thread_start(self):
#        self._p_reopen_thread = threading.Thread(\
#            target=VideoCaptureThreadReLoad.reopen_in_thread,\
#            args = (self,),\
#            daemon=True)
#        self._p_reopen_thread.start()
#        return

#    def _reopen_thread_release(self):
#        self._brk_reopen_thread = True
#        if(self._p_reopen_thread is not None):
#            self._p_reopen_thread.join()
#            self._p_reopen_thread = None
#        return

#    def read(self):
#        if(self._qcvCapRead.qsize()==0):
#            ret = self._lastCapRead[0]
#            image =  self._lastCapRead[1].copy()
#        else:
#            ret, image = self._qcvCapRead.get()
#        return ret,image

#    def release(self):
#        self._read_thread_release()
#        self._reopen_thread_release()
#        self._init_param()
#        cv2.VideoCapture.release(self)
#        return

#to del
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
#        self.__p_thread = Thread(target=VideoCaptureThread.get_img_in_thread,args = (self,),daemon=True)
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

class calcOpticalFlow:
    def flowToColor(flow,maxmotion=0):

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

    def drawFlow(img,flow,stride = 10):
        for y in range(0,img.shape[0],stride):
            for x in range(0,img.shape[1],stride):
                cv2.line(img,(x,y),(x+int(flow[y][x][0]),y+int(flow[y][x][1])),(0,255,0),1)
        return

    def colorRing():
        global _colorRing
        return _colorRing



##################################################################
#test module
##################################################################
def __test_VideoCaptureThread():
    from pyBoost import FPS
    cap = VideoCaptureThread(cv2.VideoCapture(r'G:\hand\246.mp4'))   
    key=0
    fps = FPS()
    while(key!=27):
        print(fps.get())
        ret ,img = cap.read()
        if(ret):
            cv2.imshow('img',img)
        key= cv2.waitKey(1)
    cap.release()
    print('finish')

def __test_VideoCaptureThreadRelink():
    from pyBoost import FPS
    cap = VideoCaptureThreadRelink(r'G:\hand\246.mp4')   
    key=0
    fps = FPS()
    while(key!=27):
        print(fps.get())
        ret ,img = cap.read()
        if(ret):
            cv2.imshow('img',img)
        key= cv2.waitKey(1)
    cap.release()
    print('finish')

def __test_calcOpticalFlow():
    from pyBoost import FPS
    cap = cv2.VideoCapture(r'G:\hand\246.mp4')
    imRsr = imResizer((720,480),RESIZE_TYPE.ROUNDUP,cv2.INTER_LINEAR)
    fps = FPS()
    key=0
    ret = False
    begin_num = 900
    while((ret is not True) or (begin_num>0)):
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
            calcOpticalFlow.drawFlow(drew_img,flow,20)
            img2show = imRsr.imResize(drew_img)
            cv2.putText(img2show,str(img.shape[1])+' x '+str(img.shape[0]),(360,30),cv2.FONT_HERSHEY_COMPLEX,0.75,(0,255,0))
            cv2.putText(img2show,str(round(fps.get(),3)),(525,30),cv2.FONT_HERSHEY_COMPLEX,0.75,(0,0,255))       
            cv2.imshow('show',img2show)
            #make flow color
            color_flow = calcOpticalFlow.flowToColor(flow)
            color2show = imRsr.imResize(color_flow)
            cv2.imshow('color_flow',color2show)
            cv2.imshow('color_ring',calcOpticalFlow.colorRing())

        else:
            break
        key= cv2.waitKey(1)
        prevgray = gray


if __name__=='__main__':
    save_folder_name = 'videoBoost_test_output'
    #__test_VideoCaptureThread()
    __test_VideoCaptureThreadRelink()
    #__test_calcOpticalFlow()


