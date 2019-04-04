import cv2
import numpy
import time
import os
import datetime

RECORD_CONFIG = \
{
    'OPEN':(0,1),\
    'PUBLIC_CAPTURE_SET':{cv2.CAP_PROP_FRAME_WIDTH:1280,\
                        cv2.CAP_PROP_FRAME_HEIGHT:720,\
                        cv2.CAP_PROP_FPS:25,\
                        cv2.CAP_PROP_FOURCC:cv2.VideoWriter_fourcc(*'MJPG')},\
    'PUBLIC_WRITER_SET':{'filename':'v{0}_{1}_{2}.avi',\
                        'fourcc':cv2.VideoWriter_fourcc(*'MJPG'),\
                        'fps':None,\
                        'frameSize':None},\
    'PRIVATE_CAPTURE_SET':({},{}),\
    'PRIVATE_WRITER_SET':({},{}),\
    'SAVE_FILE_EXT':'.jpg'

    
}



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    #TODO
    parser.add_argument('--dir', type = str,required=True, help = 'Where is the images?')
    parser.add_argument('--wh', type = str,required=True, help = '[width,height] such as \'[720,480]\' for resized images')
    parser.add_argument('--rtype', type = int, default = 1, help = '[0 = stretch.] [1 = round up.] [2 = round up and crop] [3 = round down] [4 = round down and fill black] [5 = round down and fill self], \"1\"')
    parser.add_argument('--inter', default = 1,  type = int, help = '[0 = INTER_NEAREST] [1 = INTER_LINEAR] [2 = INTER_CUBIC] [3 = INTER_AREA] [4 = INTER_LANCZOS4], \"1\"')
    parser.add_argument('--save', type = str,  help = 'Where to save. \"$dir$\"')
    parser.add_argument('--wait', type = int, default=1, help = 'Where to save. \"$dir$\"')
    parser.add_argument('-q','--quiet',nargs='?',default='NOT_MENTIONED',help='Sure to cover the source file? \"OFF\"')
    parser.add_argument('-r','--recursion',nargs='?',default='NOT_MENTIONED',help='Whether scan file with recursion? \"OFF\"')

   

    args = parser.parse_args()



to_open_list = [1,2]

to_open_list = ['/home/lee/PycharmProject/noSensePay_bg/log/2019-03-14_15-57-2/rendered_2019-03-14_15-57-2_v0.avi',\
'/home/lee/PycharmProject/noSensePay_bg/log/2019-03-14_15-57-2/rendered_2019-03-14_15-57-2_v1.avi']

cap_list = []
for x in to_open_list:
    cap = cv2.VideoCapture(x)
    if cap.isOpened():
        #TODO: set
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
        cap_list.append(cap)


tt = datetime.datetime.now()
datetime = str(tt)[0:10] + '_' + str(tt.hour) + '-' + str(tt.minute)+ '-' + str(tt.second)
save_path = './'+datetime
print(save_path)
if os.path.isdir(save_path)==False:
    os.makedirs(save_path)

ret = True
key = 0
img_list = [0]*len(cap_list)
index = 10000
while key!=ord('q') and key!=ord('Q'):
    #read
    img_list = [cap.read() for cap in cap_list]
    #show
    for i,x in enumerate(img_list):
        if x[0]==True:
            cv2.imshow('v'+str(i),x[1])
    #save
    key = cv2.waitKey(44)
    if key==ord('s') or key==ord('S'):
        for i,x in enumerate(img_list):
            if x[0]==True:
                cv2.imwrite('{0}/{1}_v{2}_{3}.jpg'.format(save_path,datetime,i,index),x[1]) 
        index += 1





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