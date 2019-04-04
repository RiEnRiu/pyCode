import sys
sys.path.append('../Common')
from pyBoost import FPS
from imBoost import imResizer,RESIZE_TYPE

import os
import cv2
import argparse
from datetime import datetime

import json


from time import time


#class AA:
#    A()

#class A:
#    def __init__(self):
#        print(self)
#        #self.p()
#        return


#    def p(self):
        
#        print('this is A')
#        return

#class B(A):

#    def __init__(self):
#        print(self)
#        A.__init__(self)
#        print(self)
#        print('')
#        self.p()
#        return


#    def p(self):
        
#        print('this is B')
#        print('and then ')
#        print(self)
#        A.p(self)
    

if __name__=='__main__':

    #bb = B()
    ##bb.p()
    #input()

    #key=0
    ##while(key!=27):
    #begin = time()
    #key = waitKey(1000)
    #print(time()-begin)
    #print(key)
    #print('ready to exit')
    #sys.exit()
    
    #ccc = cv2.VideoCapture(1)
    #ccc.release()
    #input()

    parser = argparse.ArgumentParser()
    parser.add_argument('--cam', type = str, required=True, help = 'Where is the stream data such as\nrtsp://[usr]:[pw]@[ip]/h264/ch1/main/av_stream')
    parser.add_argument('--thread',type = int, default=1, help = 'read image by thread, \"1\"')

    parser.add_argument('--wh', type = str, default = '[720,480]',help = 'Roundly resize to show, \"[720,480]\".')
    parser.add_argument('--show', type = str, default = '[1,1,1]', help = 'format = [date,orgsize,fps], \"[1,1,1]\"')#[B,G,R]
    parser.add_argument('--flow', type = str, default = '[0,0,0]',help = 'format = [flow,color,ring], \"[0,0,0]\"')

    args = parser.parse_args()

    if(args.thread == 1):
        cap = VideoCaptureThreadReLoad(args.cam)
    elif(args.cam.isnumeric()):
        cap = cv2.VideoCapture(int(args.cam))
        print(cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G')))
        # print(cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('Y','U','Y','2')))
        print(cap.set(cv2.CAP_PROP_FRAME_WIDTH,640))
        print(cap.set(cv2.CAP_PROP_FRAME_HEIGHT,640))

        print(cap.set(cv2.CAP_PROP_EXPOSURE,-6))
        # print(cap.set(cv2.CAP_PROP_BRIGHTNESS,128))
        #cap.set(cv2.CAP_PROP_FOCUS,0)

        #cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,False)
        #cap.set(cv2.CAP_PROP_AUTOFOCUS,False)

        #cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
        #cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,-1)
        #cap.set(cv2.CAP_PROP_FPS,30)
    else:
        cap = cv2.VideoCapture(args.cam)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
        #cap.set(cv2.CAP_PROP_FPS,30)

    print('fps = {}'.format(cap.get(cv2.CAP_PROP_FPS)))
    print('foc = {}'.format(cap.get(cv2.CAP_PROP_FOCUS)))
    print('bri = {}'.format(cap.get(cv2.CAP_PROP_BRIGHTNESS)))
    print('con = {}'.format(cap.get(cv2.CAP_PROP_CONTRAST)))
    print('sat = {}'.format(cap.get(cv2.CAP_PROP_SATURATION)))
    print('gai = {}'.format(cap.get(cv2.CAP_PROP_GAIN)))
    print('exp = {}'.format(cap.get(cv2.CAP_PROP_EXPOSURE)))
    print('auto Foc = {}'.format(cap.get(cv2.CAP_PROP_AUTOFOCUS)))
    print('auto Exp = {}'.format(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)))
    print('mode = {}'.format(cap.get(cv2.CAP_PROP_MODE)))

    if(cap.isOpened()is not True):
        sys.exit('Can not open: '+args.cam)

    is_show = json.loads(args.show)
    is_flow = json.loads(args.flow)

    date = datetime.now()
    imRsr = imResizer(json.loads(args.wh),RESIZE_TYPE.ROUNDUP,cv2.INTER_LINEAR)
    fps = FPS()

    t_path,full_name = os.path.split(args.cam)
    front,ext = os.path.splitext(full_name)

    #pregray
    if(is_flow[0]!=0 or is_flow[1]!=0):
        ret ,img = cap.read()
        while(ret is not True):
            ret ,img = cap.read()
        img2show = imRsr.imResize(img)
        pregray = cv2.cvtColor(img2show,cv2.COLOR_BGR2GRAY)

    key=0
    while(key!=27):
        ret ,img = cap.read()
        if(ret is not True):
            break
        img2show = imRsr.imResize(img)
        #flow
        if(is_flow[0]!=0 or is_flow[1]!=0):
            gray = cv2.cvtColor(img2show,cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(pregray, gray,None, 0.5, 3, 15, 3, 5, 1.2, 0)
            if(is_flow[0]!=0):
                img2show_flow = img2show.copy()
                calcOpticalFlow.drawFlow(img2show_flow,flow,20)
                cv2.imshow('flow',img2show_flow)
            if(is_flow[1]!=0):
                color_flow = calcOpticalFlow.flowToColor(flow)
                cv2.imshow('flow_color',color_flow)
            pregray = gray
        if(is_flow[2]!=0):
            cv2.imshow('color_ring',calcOpticalFlow.colorRing())
        #info
        if(is_show[0] == 1):
            cv2.putText(img2show,str(datetime.now())[:-3],(10,30),cv2.FONT_HERSHEY_COMPLEX,0.75,(255,0,0))
        if(is_show[1] == 1):
            cv2.putText(img2show,str(img.shape[1])+' x '+str(img.shape[0]),(360,30),cv2.FONT_HERSHEY_COMPLEX,0.75,(0,255,0))
        if(is_show[2] == 1):
            cv2.putText(img2show,str(round(fps.get(),3)),(525,30),cv2.FONT_HERSHEY_COMPLEX,0.75,(0,0,255))            
        cv2.imshow(front,img2show)
        key= cv2.waitKey(1)
        #print(key)

    #print('main end')
    cap.release()
    
    #del cap


#{
#    "data":"run_no_sense"
#}
#{
#    "data":"pay_no_sense"
#}


#{
#    "status": 100000,
#    "msg": "success",
#    "data": [
#        "wwsp-wwxxs-dz-yw-60g,1.000,671,270,1247,581",
#        "bl-blht-dz-ya-6.7g,0.999,510,515,1115,1146",
#        "glg-glgblzbg-hz-mcxcw-45g,0.999,1052,573,1487,1105",
#        "glg-glgblzbg-hz-mcxcw-45g,0.969,102,53,147,105"
#    ]
#}

#{
#    "status": 100001,
#    "msg": "Bad Request"
#}

#{
#    "status": 100002,
#    "msg": "No goods"
#}






    