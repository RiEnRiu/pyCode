import numpy as np
import Common.pyBoost as pb
import cv2

cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','U','Y','2'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
cap.set(cv2.CAP_PROP_FPS,30)
ret = True

import time


# fps = pb.FPS()

index = 0
key = 0
while(ret==True and key!=27):
    begin = time.time()
    # ret = cap.grab()
    ret,img = cap.read()
    cv2.imshow('img',img)
    key = cv2.waitKey(1)
    if(index%10==0):
        print(1/(time.time()-begin))
    index+=1

