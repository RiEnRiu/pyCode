#-*-coding:utf-8-*-
import sys
import os
__pyBoost_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__pyBoost_root_path)
import pyBoost as pb

import cv2
import numpy as np
import datetime
import argparse

RECORD_CONFIG = \
{
    # open param
    'OPEN':(r'D:\dataset\PRINT-ATTACK Database\org\real_access.avi'),\
    'PUBLIC_CAPTURE_SET':{'cv2.CAP_PROP_FRAME_WIDTH':'1280',\
                        'cv2.CAP_PROP_FRAME_HEIGHT':'720',\
                        'cv2.CAP_PROP_FPS':'25',\
                        'cv2.CAP_PROP_FOURCC':'cv2.VideoWriter_fourcc(*\'MJPG\')'},\
    'PRIVATE_CAPTURE_SET':(),\

    # save image param
    'SAVE_FILE_NAME':'v{0}_{1}_{2}.jpg',\

    # save video param
    'PUBLIC_WRITER_SET':{'filename':'v{0}_{1}.avi',\
                        'fourcc':'cv2.VideoWriter_fourcc(*\'MJPG\')',\
                        'fps':None,\
                        'frameSize':None},\
    'PRIVATE_WRITER_SET':()
}

def decode_config(config=RECORD_CONFIG):
    result = config.copy()
    if type(result['OPEN'])==str:
        result['OPEN'] = (result['OPEN'],)
    result['PUBLIC_CAPTURE_SET'] = {eval(k):eval(config['PUBLIC_CAPTURE_SET'][k]) for k in config['PUBLIC_CAPTURE_SET'].keys()}

    result['PRIVATE_CAPTURE_SET']=[{} for x in result['OPEN']]
    if config.get('PRIVATE_CAPTURE_SET') is not None:
        for i, one_open_set in enumerate(config['PRIVATE_CAPTURE_SET']):
            if i>= len(sult['PRIVATE_CAPTURE_SET']):
                break
            result['PRIVATE_CAPTURE_SET'][i]={eval(k):eval(one_open_set[k]) for k in one_open_set.keys()}

    result['PUBLIC_WRITER_SET'] = config['PUBLIC_WRITER_SET'].copy()
    if result['PUBLIC_WRITER_SET'].get('fourcc') is not None:
        result['PUBLIC_WRITER_SET']['fourcc'] = eval(result['PUBLIC_WRITER_SET']['fourcc'])

    result['PRIVATE_WRITER_SET'] = [result['PUBLIC_WRITER_SET'].copy() for x in result['OPEN']]
    if config.get('PRIVATE_WRITER_SET') is not None:
        for i, one_private_writer_set in enumerate(config['PRIVATE_WRITER_SET']):
            if i>= len(sult['PRIVATE_WRITER_SET']):
                break
            if one_private_writer_set.get('fourcc') is not None:
                one_private_writer_set['fourcc'] = eval(one_private_writer_set['fourcc'])
            result['PRIVATE_WRITER_SET'][i].update(one_private_writer_set)
    return result



def get_date_time_string():
    date_time_str = str(datetime.datetime.now())[:-7]
    date_str,time_str= date_time_str.split()
    h,m,s = time_str.split(':')
    return date_str,h+'-'+m+'-'+s # such as 2019-11-11, 11-11-11
    
    
def record_images(cap_list,save_dir,config):
    print('************** Control ***************')
    print('q    quit')
    print('s    save images')
    print('**************************************')
    ret,key = True,0
    save_path = os.path.join(save_dir, config['SAVE_FILE_NAME'])
    count = 10000
    date_time = os.path.basename(save_dir)
    while key!=ord('q') and key!=ord('Q'):
        imgs = []
        ret = False
        for i, cap in enumerate(cap_list):
            _ret,_img = cap.read()
            ret = ret or _ret
            if _ret:
                imgs.append((i,_ret, _img))
                cv2.imshow('view{0}'.format(i),_img)
            else:
                cv2.destroyWindow('view{0}'.format(i))
        key = cv2.waitKey(1)
        if key==ord('s') or key==ord('S'):
            for i,_ret,_img in imgs:
                if _ret:
                    s_path = save_path.format(i,date_time,count)
                    cv2.imwrite(s_path,_img)
                    cv2.imshow('view{0}'.format(i), np.ones(_img.shape,dtype=np.float32))
                    print('Saved: {0}'.format(s_path))
            cv2.waitKey(5)
        count += 1
    return

# TODO
def record_videos(cap_list,save_dir,config):
    print('************** Control ***************')
    print('q    quit')
    print('r    run video recording')
    print('s    stop video recording')
    print('**************************************')
    return
    ret,key = True,0
    save_path = os.path.join(save_dir, config['SAVE_FILE_NAME'])
    count = 10000
    while ret and key!=ord('q') and key!=ord('Q'):
        imgs = []
        ret = False
        for i, cap in cap_list:
            _ret,_img = cap.read()
            ret = ret or _ret
            if _ret:
                imgs.append((i,_ret, _img))
                cv2.imshow('view{0}'.format(i),_img)
            else:
                cv2.destroyWindow('view{0}'.format(i))
        key = cv2.waitKey(1)
        if key==ord('s') or key==ord('S'):
            for i,_ret,_img in imgs:
                if _ret:
                    s_path = save_path.format(i,count)
                    cv2.imwrite(s_path,_img)
                    print('Saved: {0}'.format(s_path))
        count += 1
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type = str,  help = 'Where to save. \"$pwd$\"')
    parser.add_argument('-i','--image',nargs='?',default='NOT_MENTIONED',help='To save images? \"OFF\"')
    parser.add_argument('-v','--video',nargs='?',default='NOT_MENTIONED',help='To save videos? \"OFF\"')
    parser.add_argument('-t','--thread',nargs='?',default='NOT_MENTIONED',help='open camera in thread? \"OFF\"')
    parser.add_argument('-q','--quiet',nargs='?',default='NOT_MENTIONED',help='Accept all the configuration? \"OFF\"')
    args = parser.parse_args()

    # mode
    if args.image == 'NOT_MENTIONED' and args.video == 'NOT_MENTIONED':
        parser.print_help()
        print('')
        sys.exit('Choose one mode to record: --image or --video')
    elif args.video != 'NOT_MENTIONED':
        sys.exit('--video mode can not be use now')


    # save dir
    date_str,time_str = get_date_time_string()
    if args.image != 'NOT_MENTIONED':
        date_time = date_str+'_'+time_str
        save_dir = date_time if args.save is None else os.path.join(args.save, date_time)
    else:
        save_dir = date_str if args.save is None else os.path.join(args.save, date_str)
    pb.makedirs(save_dir)
    if os.path.isdir(save_dir)==False:
        sys.exit('Invalid save dir: {0}'.format(save_dir))


    # config
    for k in RECORD_CONFIG.keys():
        print('{0}: {1}'.format(k,RECORD_CONFIG[k]))
    if args.quiet =='NOT_MENTIONED':
        print('')
        print('Accept all the configurations above? [Y/N]?')
        str_in = input()
        if(str_in !='Y' and str_in !='y'):
            sys.exit('user quit')
    config = decode_config(RECORD_CONFIG)
    #for k in config.keys():
    #    print('{0}: {1}'.format(k,config[k]))

    # cap
    if args.thread == 'NOT_MENTIONED':
        cap_list = [pb.video.VideoCaptureRelink(x) for x in config['OPEN']]
    else:
        cap_list = [pb.video.VideoCaptureThreadRelink(x) for x in config['OPEN']]
    
    if args.image != 'NOT_MENTIONED':
        record_images(cap_list,save_dir,config)
    else:
        pass
        #TODO
        #record_videos(cap_list,save_dir,config)




'''
cv2.CAP_PROP_POS_MSEC	            Current position of the video file in milliseconds.
cv2.CAP_PROP_POS_FRAMES	            0-based index of the frame to be decoded/captured next.
cv2.CAP_PROP_POS_AVI_RATIO	        Relative position of the video file: 0=start of the film, 1=end of the film.
cv2.CAP_PROP_FRAME_WIDTH	        Width of the frames in the video stream.
cv2.CAP_PROP_FRAME_HEIGHT	        Height of the frames in the video stream.
cv2.CAP_PROP_FPS	                Frame rate.
cv2.CAP_PROP_FOURCC	                4-character code of codec. see?VideoWriter::fourcc?.
cv2.CAP_PROP_FRAME_COUNT	        Number of frames in the video file.
cv2.CAP_PROP_FORMAT	                Format of the Mat objects returned by?VideoCapture::retrieve().
cv2.CAP_PROP_MODE	                Backend-specific value indicating the current capture mode.
cv2.CAP_PROP_BRIGHTNESS	            Brightness of the image (only for those cameras that support).
cv2.CAP_PROP_CONTRAST	            Contrast of the image (only for cameras).
cv2.CAP_PROP_SATURATION	            Saturation of the image (only for cameras).
cv2.CAP_PROP_HUE	                Hue of the image (only for cameras).
cv2.CAP_PROP_GAIN	                Gain of the image (only for those cameras that support).
cv2.CAP_PROP_EXPOSURE	            Exposure (only for those cameras that support).
cv2.CAP_PROP_CONVERT_RGB	        Boolean flags indicating whether images should be converted to RGB.
cv2.CAP_PROP_WHITE_BALANCE_BLUE_U	Currently unsupported.
cv2.CAP_PROP_RECTIFICATION	        Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently).
cv2.CAP_PROP_MONOCHROME	
cv2.CAP_PROP_SHARPNESS	
cv2.CAP_PROP_AUTO_EXPOSURE	        DC1394: exposure control done by camera, user can adjust reference level using this feature.
cv2.CAP_PROP_GAMMA	
cv2.CAP_PROP_TEMPERATURE	
cv2.CAP_PROP_TRIGGER	
cv2.CAP_PROP_TRIGGER_DELAY	
cv2.CAP_PROP_WHITE_BALANCE_RED_V	
cv2.CAP_PROP_ZOOM	
cv2.CAP_PROP_FOCUS	
cv2.CAP_PROP_GUID	
cv2.CAP_PROP_ISO_SPEED	
cv2.CAP_PROP_BACKLIGHT	
cv2.CAP_PROP_PAN	
cv2.CAP_PROP_TILT	
cv2.CAP_PROP_ROLL	
cv2.CAP_PROP_IRIS	
cv2.CAP_PROP_SETTINGS	            Pop up video/camera filter dialog (note: only supported by DSHOW backend currently. The property value is ignored)
cv2.CAP_PROP_BUFFERSIZE	            
cv2.CAP_PROP_AUTOFOCUS	            
cv2.CAP_PROP_SAR_NUM	            Sample aspect ratio: num/den (num)
cv2.CAP_PROP_SAR_DEN	            Sample aspect ratio: num/den (den)
cv2.CAP_PROP_BACKEND	            Current backend (enum VideoCaptureAPIs). Read-only property.
cv2.CAP_PROP_CHANNEL	            Video input or Channel Number (only for those cameras that support)
cv2.CAP_PROP_AUTO_WB	            enable/ disable auto white-balance
cv2.CAP_PROP_WB_TEMPERATURE	        white-balance color temperature
'''