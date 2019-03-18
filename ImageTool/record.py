import sys
sys.path.append('../Common')
from pyBoost import *
import os
import cv2
import argparse
from datetime import datetime

import json

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam', type = str, required=True, help = 'Where is the stream data such as\nrtsp://[usr]:[pw]@[ip]/h264/ch1/main/av_stream')
    parser.add_argument('--thread',type = int, default=1, help = 'read image by thread, \"1\"')

    parser.add_argument('--save',type = str, help = 'Where to save the video,\"./RECORD_OUTPUT/record_[datetime].mp4\"')
    parser.add_argument('--fps',type = int, default=25, help = 'FPS of saved video, \"25\"')
    parser.add_argument('--fourcc',type = str,default='MJPG', choices = ['MJPG','MP42','DIV3','DIVX','U263','I263','FLV1','-1'], \
                        help = '\"MJPG\" \"MP42\" \"DIV3\" \"DIVX\" \"U263\" \"I263\" \"FLV1\" "-1"')   
    parser.add_argument('--rtype', type = int, default=1, help = '[0 = stretch.] [1 = round up.] [2 = round up and crop] [3 = round down] [4 = round down and fill black] [5 = round down and fill self], \"2\"')
    parser.add_argument('--wh', type = str,help = '[width,height] such as \'[720,480]\' to save video')
    parser.add_argument('--inter', default = 3,  type = int, help = '[0 = INTER_NEAREST] [1 = INTER_LINEAR] [2 = INTER_CUBIC] [3 = INTER_AREA] [4 = INTER_LANCZOS4], \"3\"')
   

    args = parser.parse_args()

    #cap
    if(args.thread == 1):
        cap = VideoCaptureThread(args.cam)
    elif(args.cam.isnumeric()):
        cap = cv2.VideoCapture(int(args.cam))
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1280)
        #cap.set(cv2.CAP_PROP_EXPOSURE,-7)
        #cap.set(cv2.CAP_PROP_BRIGHTNESS,512)
                

    else:
        cap = cv2.VideoCapture(args.cam)

    if(cap.isOpened()is not True):
        sys.exit('Can not open: '+args.cam)

    #save
    if(args.save is not None):
        videoSave = args.save
    else:
        datetimeStr = str(datetime.now())[:-7]
        dateStr,timeStr= datetimeStr.split()
        hStr,mStr,sStr = timeStr.split(':')
        videoSave = './RECORD_OUTPUT/record_'+dateStr+'_'+hStr+'-'+mStr+'-'+sStr+'.avi'

    if(os.path.isdir(os.path.split(videoSave)[0]) is not True):
        os.makedirs(os.path.split(videoSave)[0])

    #fps
    videoFPS = args.fps
    #if(videoFPS>cap.get(cv2.CAP_PROP_FPS)):
    #    videoFPS = cap.get(cv2.CAP_PROP_FPS)

    #print(cap.get(cv2.CAP_PROP_FPS))
    #input()

    #fourcc
    #print(args.fourcc)
    #print(len(args.fourcc))
    #input()
    if(len(args.fourcc)==4):
        videoFourcc = cv2.VideoWriter_fourcc(args.fourcc[0],args.fourcc[1],args.fourcc[2],args.fourcc[3])
    else:
        videoFourcc = -1

    #wh
    if(args.wh is not None):
        videowh = tuple(json.loads(args.wh))
        rszr = imResizer(videowh,args.rtype,args.inter)
    else:
        videowh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        rszr = None

    capWriter = cv2.VideoWriter(videoSave,videoFourcc,videoFPS,videowh)
    key=0
    while(key!=27):
        ret ,img = cap.read()
        if(ret is not True):
            break
        if(rszr is not None):
            videoFrame = rszr.imResize(img)
        else:
            videoFrame = img
        capWriter.write(videoFrame)
        cv2.imshow('video',videoFrame)
        key=cv2.waitKey(1)
        #key = waitKey(int(1000/videoFPS))
    cap.release()
    capWriter.release()
    

    