import sys
sys.path.append('../Common')
from pyBoost import *
import os
import cv2
import argparse
from datetime import datetime



def get_fit_size(fit_str):
    index_begin1 = 0
    index_end1 = 0
    index_begin2 = 0
    index_end2 = 0
    index_for = 0
    for i in range(index_for,len(fit_str)):
        if(fit_str[i].isnumeric()):
            index_begin1 = i
            index_for = i+1
            break
    for i in range(index_for,len(fit_str)):
        if(fit_str[i].isnumeric() is not True):
            index_end1 = i
            index_for = i
            break
    for i in range(index_for,len(fit_str)):
        if(fit_str[i].isnumeric()):
            index_begin2 = i
            index_for = i+1
            break
    for i in range(index_for,len(fit_str)):
        if(fit_str[i].isnumeric() is not True):
            index_end2 = i
            index_for = i+1
            break
    return (int(fit_str[index_begin1:index_end1]),int(fit_str[index_begin2:index_end2]))






if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs', type = str,required=True, help = '[camera] or [video] or [images list] or [images path](not recommend)')
    parser.add_argument('--save',type = str, default = '', help = 'where to save to color flow')
    parser.add_argument('--NotDisplay',type = int, default = 0, help = '1 = not Display, 0 = display')
    parser.add_argument('--size',type = str,default = '[720,480]',help = 'roundly resize to \'--size=[w,h]\' to display.')
    parser.add_argument('--thread',type = int, default = 0, help = 'use thread to get images')

    parser.add_argument('--NPdate', help = 'Do not print date')#B
    parser.add_argument('--NPsize',help = 'Do not print source size.')#G
    parser.add_argument('--NPfps', help = 'Do not print fps.')#R

    args = parser.parse_args()

    
    if(args.thread==1):
        cap = VideoCaptureThread(args.imgs)
    elif(args.imgs.isnumeric()):
        cap = cv2.VideoCapture(int(args.imgs))
    else:
        cap = cv2.VideoCapture(args.imgs)

    if(cap.isOpened()is not True):
        is_video = False
        if(os.path.isfile(args.imgs)):
            imgs_list = scanner.list(args.imgs)
        elif(os.path.isdir(args.imgs)):
            imgs_list = scanner.file(args.imgs,'.jpg.png.jpeg',True)
    else:
        is_video = True

    date = datetime.now()

    fps = FPS()

    imRsr = Resizer(get_fit_size(args.size),RESIZE_TYPE.ROUNDUP,cv2.INTER_LINEAR)

    t_path,full_name = os.path.split(args.imgs)
    front,ext = os.path.splitext(full_name)

    fgbg = cv2.createBackgroundSubtractorMOG2()

    #boxed = [0]*20

    if(is_video):
        key=0
        ret = False
        begin_num = 225
        while((ret is not True) or (begin_num>0)):
            begin_num -=1
            ret ,preimg = cap.read()
            prevgray = cv2.cvtColor(preimg,cv2.COLOR_BGR2GRAY)
        while(key!=27):
            ret ,img = cap.read()
            if(ret):
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prevgray, gray,None, 0.5, 3, 15, 3, 5, 1.2, 0);

                ycrcb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                #print((ycrcb_image).dtype)
                #cv2.imshow('ycrcb_image',ycrcb_image[:,:,1])
                temp_v ,fuse = cv2.threshold(ycrcb_image[:,:,1],0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                fgmask = fgmask = fgbg.apply(img)
                
                fx = flow[:,:,0][np.bitwise_and(fuse!=0,fgmask!=0)]

                #fy = flow[:,:,0][np.bitwise_and(fusefuse!=0,fgmaskfuse!=0)]
                print(str(fx.min())+'    '+str(fx.max()))
                
                #if(fx.sum()>500000):
                #    print('shen')
                #elif(fx.sum()<-500000):
                #    print('suo')
                #else:
                #    print('')

                #print(img.shape)
                #print(flow.shape)

                drew_img = calcOpticalFlow.drawFlow(img,flow,20)

                img2show = imRsr.imResize(drew_img)
                cv2.putText(img2show,str(datetime.now())[:-3],(10,30),cv2.FONT_HERSHEY_COMPLEX,0.75,(255,0,0))
                cv2.putText(img2show,str(img.shape[1])+' x '+str(img.shape[0]),(360,30),cv2.FONT_HERSHEY_COMPLEX,0.75,(0,255,0))
                cv2.putText(img2show,str(round(fps.get(),3)),(525,30),cv2.FONT_HERSHEY_COMPLEX,0.75,(0,0,255))       
                #print(fps.get())     
                cv2.imshow(front,img2show)

                #color_flow = calcOpticalFlow.MotionToColor(flow)
                ##print(color_flow.shape)
                #color2show = imRsr.imResize(color_flow)
                ##print(color_flow.shape)

                #resized_img = imRsr.imResize(img)

                ##cv2.imshow('output_mask',output_mask)

                ##bgSubtractor.apply
                ##bgSubtractor(color2show)


                #color2show[:,:,0][output_mask==0]=0
                #color2show[:,:,1][output_mask==0]=0
                #color2show[:,:,2][output_mask==0]=0

                #color2show[:,:,0][fgmask==0]=0
                #color2show[:,:,1][fgmask==0]=0
                #color2show[:,:,2][fgmask==0]=0

                #cv2.imshow('color_flow',color2show)



                #print(calcOpticalFlow.colorRing())
                cv2.imshow('color_ring',calcOpticalFlow.colorRing())

            else:
                break
            key= cv2.waitKey(1)
            prevgray = gray
        cap.release()
    else:
        pass

















