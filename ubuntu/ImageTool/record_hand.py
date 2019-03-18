import sys
import os
import cv2
import argparse
import datetime
import json

#sys.path.append('../Common')
#from pyBoost import *

if __name__=='__main__':
    #cap
    cap0 = cv2.VideoCapture(0)
    print(cap0.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G')))
    print(cap0.set(cv2.CAP_PROP_FRAME_WIDTH,1280))
    print(cap0.set(cv2.CAP_PROP_FRAME_HEIGHT,720))
    print(cap0.set(cv2.CAP_PROP_FPS,30))

    #cap0.set(cv2.CAP_PROP_BRIGHTNESS,256)
    #cap0.set(cv2.CAP_PROP_SATURATION,)
    #cap0.set(cv2.CAP_PROP_AUTO_EXPOSURE,-7)
    #cap0.set(cv2.CAP_PROP_EXPOSURE,-8)    
    
    #print(cap0.set(cv2.CAP_PROP_AUTO_EXPOSURE,False))
    #print(cap0.set(cv2.CAP_PROP_AUTOFOCUS,False))
    #print(cap0.set(cv2.CAP_PROP_FOCUS,0))
    
    print(cap0.get(cv2.CAP_PROP_FOURCC))
    print(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(cap0.get(cv2.CAP_PROP_FPS))
    
    #print(cap0.get(cv2.CAP_PROP_AUTO_EXPOSURE))
    #print(cap0.get(cv2.CAP_PROP_AUTOFOCUS))
    #print(cap0.get(cv2.CAP_PROP_FOURCC))
    #print(cap0.get(cv2.CAP_PROP_AUTOFOCUS))
    print(cap0.get(cv2.CAP_PROP_AUTO_EXPOSURE))
    print(cap0.get(cv2.CAP_PROP_EXPOSURE))
    print(cap0.get(cv2.CAP_PROP_EXPOSUREPROGRAM))
    #print(cap0.get(cv2.CAP_PROP_AUTOFOCUS))


    tt = datetime.datetime.now()
    t_str = str(tt)[0:10]+'_'+str(tt.hour)+'-'+str(tt.minute)+'.avi'
    file0 = './v0_'+t_str+'.avi'
    print(file0)
    videoFourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    videoFPS = 28
    videowh = (int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #capW0 = cv2.VideoWriter(file0,videoFourcc,videoFPS,videowh)

    #cap1 = cv2.VideoCapture(1)
    #cap1.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
    #cap1.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    #cap1.set(cv2.CAP_PROP_BRIGHTNESS,256)
    #cap1.set(cv2.CAP_PROP_EXPOSURE,-7)
    #file1 = './v1_'+t_str+'.avi'
    #capW1 = cv2.VideoWriter(file1,videoFourcc,videoFPS,videowh)

    #fps = FPS()

    while(1):
        ret0 ,img0 = cap0.read()
        if(ret0 is not True):
            break
        #capW0.write(img0)
        cv2.imshow('v0',img0)

        #print(cap0.get(cv2.CAP_PROP_FOCUS))

        #ret1 ,img1 = cap1.read()
        #if(ret1 is not True):
        #    break
        #capW1.write(img1)
        #cv2.imshow('v1',img1)


        key = cv2.waitKey(1)
        if(key==27):
            break

    cap0.release()
    capW0.release()
    #cap1.release()
    #capW1.release()
        


    