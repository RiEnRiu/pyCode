import sys
sys.path.append('../Common')
from pyBoost import *
import os
import cv2
import argparse
import json
import multiprocessing

def decodeOneVideo(param):
    full_path = param[0]
    full_name = param[1]
    args = param[2]
    one_video = os.path.join(full_path,full_name)
    cap = cv2.VideoCapture(one_video)
    if(cap.isOpened() is not True):
        print(one_video+'  -->  open fail.')
        return 
    front_name,ext = os.path.splitext(full_name)
    save_path = os.path.join(args.save,front_name)
    if(os.path.isdir(save_path) is not True):
        os.makedirs(save_path)
    if(args.saveFlow is not None):
        save_path_flow = os.path.join(args.saveFlow,front_name)
        if(os.path.isdir(save_path_flow) is not True):
            os.makedirs(save_path_flow)

    index = args.first
    ret = False    

    if(args.wh is not None):     
        imresizer = imResizer(tuple(json.loads(args.wh)), args.rtype, args.inter)
        #frame 0
        while(ret is not True):
            ret,img = cap.read()
        img2save = imresizer.imResize(img)
        cv2.imwrite(os.path.join(save_path,front_name+'_frame_'+str(index)+args.format),img2save)
        #flow
        if(args.saveFlow is not True):
            pregray = cv2.cvtColor(img2save,cv2.COLOR_BGR2GRAY)
        index +=1
        #start
        while(1):
            ret,img = cap.read()
            if(ret is not True):
                break
            img2save = imresizer.imResize(img)
            cv2.imwrite(os.path.join(save_path,front_name+'_frame_'+str(index)+args.format),img2save)
            #flow
            if(args.saveFlow is not True):
                gray = cv2.cvtColor(img2save,cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(pregray, gray,None, 0.5, 3, 15, 3, 5, 1.2, 0)
                cv2.imwrite(os.path.join(save_path_flow,front_name+'_flow_'+str(index)+args.format),calcOpticalFlow.flowToColor(flow))
                pregray = gray
            index +=1
    else:
        #frame 0
        while(ret is not True):
            ret,img = cap.read()
        cv2.imwrite(os.path.join(save_path,front_name+'_frame_'+str(index)+args.format),img)
        #flow
        if(args.saveFlow is not None):
            pregray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        index +=1
        #start
        while(1):
            ret,img = cap.read()
            if(ret is not True):
                break
            cv2.imwrite(os.path.join(save_path,front_name+'_frame_'+str(index)+args.format),img)
            #flow
            if(args.saveFlow is not None):
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(pregray, gray,None, 0.5, 3, 15, 3, 5, 1.2, 0)
                cv2.imwrite(os.path.join(save_path_flow,front_name+'_flow_'+str(index)+args.format),calcOpticalFlow.flowToColor(flow))
                pregray = gray
            index +=1
    print(one_video+'  -->  save '+str(index-args.first)+' ('+str(args.first)+'-'+str(index-1)+')'+' frames.')
    return 


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type = str,required=True, help = '[video file] or [path].')
    parser.add_argument('--save', type = str,required=True, help = 'Where to save.')
    parser.add_argument('--saveFlow',type = str, help = 'Where to save color images of video optical flow.')
    parser.add_argument('--format', default = '.jpg', choices = ['.jpg', '.png'], type = str, help = '\".jpg\"')
    parser.add_argument('--rtype', type = int, help = '[0 = stretch.] [1 = round up.] [2 = round up and crop] [3 = round down] [4 = round down and fill black] [5 = round down and fill self], "1"')
    parser.add_argument('--wh', type = str, help = '[width,height] such as \'[720,480]\' for resized images')
    #parser.add_argument('--width', type = int, help = 'Width for saved images')
    #parser.add_argument('--height', type = int, help = 'Height for saved images')
    parser.add_argument('--inter', default = 3, type = int, help = '[0 = INTER_NEAREST] [1 = INTER_LINEAR] [2 = INTER_CUBIC] [3 = INTER_AREA] [4 = INTER_LANCZOS4], \"3\"')
    parser.add_argument('--first', default=10001, type=int, help='the fisrt index, \"10001\"')
    parser.add_argument('--process',default=int(multiprocessing.cpu_count())-1,type=int,help='use how many process, \"$core$-1\"')
    args = parser.parse_args()

    #make save path
    if(os.path.isdir(args.save) is not True):
        os.makedirs(args.save)
    if(args.saveFlow is not None):
        if(os.path.isdir(args.saveFlow) is not True):
            os.makedirs(args.saveFlow)

    #get list
    full_paths = []
    full_names = []
    if(os.path.isfile(args.video)):
        (filepath,tempfilename) = os.path.split(args.video)
        full_names.append(tempfilename)
        full_paths.append(filepath)
    else:
        tempfilenames = os.listdir(args.video)
        for tempfilename in tempfilenames:
            full_path_name = os.path.join(args.video,tempfilename)
            if(os.path.isfile(full_path_name)):
                cap = cv2.VideoCapture(full_path_name)
                if(cap.isOpened()):
                    full_paths.append(args.video)
                    full_names.append(tempfilename)

    #adjust process
    if(args.process<=1 or int(multiprocessing.cpu_count())-1<=1):
        num_process =1
    elif(len(full_names)<min(int(multiprocessing.cpu_count())-1,args.process)):
        num_process = len(full_names)
    else:
        num_process = min(int(multiprocessing.cpu_count())-1,args.process)

    #do it 
    if (num_process == 1):
        for i in range(len(full_names)):
            decodeOneVideo((full_paths[i],full_names[i],args))
    else:
        param = list(zip(full_paths,full_names,[args]*len(full_names)))
        pool = multiprocessing.Pool(num_process) 
        for x in pool.imap_unordered(decodeOneVideo,param):
            pass
        pool.close()
        pool.join()
        