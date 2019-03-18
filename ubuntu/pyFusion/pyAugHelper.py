from cvAugmentor import *
import sys
sys.path.append('../Common')
from pyBoost import *

import cv2
import numpy as np 
import argparse




global img
global opt_bar
global opt_max_number
global param_bar
global param
global aug_mat

user_img_view_size = (750,450)

img = np.zeros([450,750,3],np.uint8)
opt_bar = 0
opt_max_number = 21
param_bar = 10

param = [10]
aug_mat =[img.copy()]




def makeDefaultImg():
    global img
    global opt_bar
    global opt_max_number
    global param_bar
    global param
    global aug_mat

    img[:,:,:]=255
    j=0
    while(j<450/50+1):
        cv2.line(img,(0,int(j*50)),(750,int(j*50)),(0,0,0),3)
        j+=1
    j=0
    while(j<750/50+1):
        cv2.line(img,(int(j*50),0),(int(j*50),450),(0,0,0),3)
        j+=1

    _box_size = 50-5
    img[(3*50+3):((3*50+3)+_box_size),(6*50+3):((6*50+3)+_box_size),0] = 255 
    img[(3*50+3):((3*50+3)+_box_size),(6*50+3):((6*50+3)+_box_size),1] = 0
    img[(3*50+3):((3*50+3)+_box_size),(6*50+3):((6*50+3)+_box_size),2] = 255

    img[(3*50+3):((3*50+3)+_box_size),(7*50+3):((7*50+3)+_box_size),0] = 0
    img[(3*50+3):((3*50+3)+_box_size),(7*50+3):((7*50+3)+_box_size),1] = 0
    img[(3*50+3):((3*50+3)+_box_size),(7*50+3):((7*50+3)+_box_size),2] = 255

    img[(3*50+3):((3*50+3)+_box_size),(8*50+3):((8*50+3)+_box_size),0] = 0 
    img[(3*50+3):((3*50+3)+_box_size),(8*50+3):((8*50+3)+_box_size),1] = 255
    img[(3*50+3):((3*50+3)+_box_size),(8*50+3):((8*50+3)+_box_size),2] = 255

    img[(4*50+3):((4*50+3)+_box_size),(6*50+3):((6*50+3)+_box_size),0] = 0 
    img[(4*50+3):((4*50+3)+_box_size),(6*50+3):((6*50+3)+_box_size),1] = 0
    img[(4*50+3):((4*50+3)+_box_size),(6*50+3):((6*50+3)+_box_size),2] = 0

    img[(4*50+3):((4*50+3)+_box_size),(7*50+3):((7*50+3)+_box_size),0] = 255 
    img[(4*50+3):((4*50+3)+_box_size),(7*50+3):((7*50+3)+_box_size),1] = 255
    img[(4*50+3):((4*50+3)+_box_size),(7*50+3):((7*50+3)+_box_size),2] = 255

    img[(4*50+3):((4*50+3)+_box_size),(8*50+3):((8*50+3)+_box_size),0] = 255/2 
    img[(4*50+3):((4*50+3)+_box_size),(8*50+3):((8*50+3)+_box_size),1] = 255/2
    img[(4*50+3):((4*50+3)+_box_size),(8*50+3):((8*50+3)+_box_size),2] = 255/2

    img[(5*50+3):((5*50+3)+_box_size),(6*50+3):((6*50+3)+_box_size),0] = 255 
    img[(5*50+3):((5*50+3)+_box_size),(6*50+3):((6*50+3)+_box_size),1] = 0
    img[(5*50+3):((5*50+3)+_box_size),(6*50+3):((6*50+3)+_box_size),2] = 0

    img[(5*50+3):((5*50+3)+_box_size),(7*50+3):((7*50+3)+_box_size),0] = 255 
    img[(5*50+3):((5*50+3)+_box_size),(7*50+3):((7*50+3)+_box_size),1] = 255
    img[(5*50+3):((5*50+3)+_box_size),(7*50+3):((7*50+3)+_box_size),2] = 0

    img[(5*50+3):((5*50+3)+_box_size),(8*50+3):((8*50+3)+_box_size),0] = 0 
    img[(5*50+3):((5*50+3)+_box_size),(8*50+3):((8*50+3)+_box_size),1] = 255
    img[(5*50+3):((5*50+3)+_box_size),(8*50+3):((8*50+3)+_box_size),2] = 0


def on_trackbar_share_opt_bar(value):
    global img
    global opt_bar
    global opt_max_number
    global param_bar
    global param
    global aug_mat

    opt_bar = value

    aug_text = 'AUG_CONFIG = '
    fanwei = 'Range = '
    shuoming = 'Explanation: '
    cv2.imshow('original', img)
    if(opt_bar == 0):
        #Rotate
        param[0] = 1.0 / 100.0*360.0 * param_bar
        aug_mat = cvAugmentor.imRotate(img, param)
        cv2.putText(aug_mat[0], aug_text+'Rotate('+str(param[0])+')', (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "(---,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming+"Rotate by one angle (degree)", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 1):
        #FlipUD
        aug_mat = cvAugmentor.imFlipUD(img)
        cv2.putText(aug_mat[0], aug_text + "FlipUD", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "(NULL)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "Flip up down", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 2):
        #FlipLR
        aug_mat = cvAugmentor.imFlipLR(img)
        cv2.putText(aug_mat[0], aug_text + "FlipLR", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "(NULL)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "Flip left right", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 3):
        #Flip
        aug_mat = cvAugmentor.imFlip(img)
        cv2.putText(aug_mat[0], aug_text + "Flip", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "(NULL)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "Random flip", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 4):
        #Crop
        param[0] = 1.0 / 100.0* param_bar
        if(param[0]<0.01):
             param[0] = 0.01
        aug_mat = cvAugmentor.imCrop(img, param)
        cv2.putText(aug_mat[0], aug_text + "Crop(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "(0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "Random crop image by area percentage", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 5):
        #AffineX
        param[0] = 1.0 / 50.0* (param_bar - 50)
        aug_mat = cvAugmentor.imAffineX(img, param)
        cv2.putText(aug_mat[0], aug_text + "AffineX(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[-1,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "X axis affine", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 6):
        #AffineY
        param[0] = 1.0 / 50.0* (param_bar - 50)
        aug_mat = cvAugmentor.imAffineY(img, param)
        cv2.putText(aug_mat[0], aug_text + "AffineY(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[-1,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "Y axis affine", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 7):
        #AffineY
        param[0] = 1.0 / 50.0* (param_bar - 50)
        aug_mat = cvAugmentor.imAffine(img, param)
        cv2.putText(aug_mat[0], aug_text + "Affine(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[-1,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "X or Y axis affine", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 8):
        #Noise
        param[0] = 1.0 / 100.0* (param_bar)
        aug_mat = cvAugmentor.imNoise(img, param)
        cv2.putText(aug_mat[0], aug_text + "Noise(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "Add gauss noise", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 9):
        #Hue
        param[0] = 1.0 / 100.0*360.0 * param_bar
        aug_mat = cvAugmentor.imHue(img, param)
        cv2.putText(aug_mat[0], aug_text + "Hue(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "(---,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "Hue angle (degree)", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 10):
        #Saturation
        param[0] = 1.0 / 50.0* (param_bar - 50)
        aug_mat = cvAugmentor.imSaturation(img, param)
        cv2.putText(aug_mat[0], aug_text + "Saturation(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "Saturation", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 11):
        #Lightness
        param[0] = 1.0 / 50.0* (param_bar - 50)
        aug_mat = cvAugmentor.imLightness(img, param)
        cv2.putText(aug_mat[0], aug_text + "Lightness(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "Lightness", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 12):
        #PerspectiveUL
        param[0] = 1.0 / 100.0* param_bar
        aug_mat = cvAugmentor.imPerspectiveUL(img, param)
        cv2.putText(aug_mat[0], aug_text + "PerspectiveUL(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "UL Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 13):
        #PerspectiveU
        param[0] = 1.0 / 100.0* param_bar
        aug_mat = cvAugmentor.imPerspectiveU(img, param)
        cv2.putText(aug_mat[0], aug_text + "PerspectiveU(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "U  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 14):
        #PerspectiveUR
        param[0] = 1.0 / 100.0* param_bar
        aug_mat = cvAugmentor.imPerspectiveUR(img, param)
        cv2.putText(aug_mat[0], aug_text + "PerspectiveUR(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "UR Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 15):
        #PerspectiveL
        param[0] = 1.0 / 100.0* param_bar
        aug_mat = cvAugmentor.imPerspectiveL(img, param)
        cv2.putText(aug_mat[0], aug_text + "PerspectiveL(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "L  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 16):
        #Perspective
        param[0] = 1.0 / 100.0* param_bar
        aug_mat = cvAugmentor.imPerspective(img, param)
        cv2.putText(aug_mat[0], aug_text + "Perspective(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "Random perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 17):
        #PerspectiveR
        param[0] = 1.0 / 100.0* param_bar
        aug_mat = cvAugmentor.imPerspectiveR(img, param)
        cv2.putText(aug_mat[0], aug_text + "PerspectiveR(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "R  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 18):
        #PerspectiveDL
        param[0] = 1.0 / 100.0* param_bar
        aug_mat = cvAugmentor.imPerspectiveDL(img, param)
        cv2.putText(aug_mat[0], aug_text + "PerspectiveDL(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "DL Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 19):
        #PerspectiveD
        param[0] = 1.0 / 100.0* param_bar
        aug_mat = cvAugmentor.imPerspectiveD(img, param)
        cv2.putText(aug_mat[0], aug_text + "PerspectiveD(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "D  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 20):
        #PerspectiveDR
        param[0] = 1.0 / 100.0* param_bar
        aug_mat = cvAugmentor.imPerspectiveDR(img, param)
        cv2.putText(aug_mat[0], aug_text + "PerspectiveDR(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "DR Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    #elif(opt_bar == 21):
    #    #Distort
    #    param[0] = 1.0 / 20.0* param_bar
    #    cvAugmentor.imDistort(img, aug_mat, param)
    #    cv2.putText(aug_mat[0], aug_text + "Distort(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.putText(aug_mat[0], fanwei + "[0,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.putText(aug_mat[0], shuoming + "Add distort", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.imshow("augmented", aug_mat[0])
    #elif(opt_bar == 21):
    #    #Pyramid
    #    aug_mat[0].fill(255 / 2)
    #    cv2.putText(aug_mat[0], aug_text + "Pyramid(donw,up)", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.putText(aug_mat[0], fanwei + "(0,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.putText(aug_mat[0], shuoming + "Build image pyramid", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.imshow("augmented", aug_mat[0])
    #elif(opt_bar == 23):#NotBackup
    #    aug_mat[0].setTo(Scalar(255 / 2, 255 / 2, 255 / 2))
    #    cv2.putText(aug_mat[0], aug_text + "NotBackup", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.putText(aug_mat[0], fanwei + "(NULL)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.putText(aug_mat[0], shuoming + "A flag means do not keep prestep images", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.imshow("augmented", aug_mat[0])
    return


def on_trackbar_share_param_bar(value):
    global img
    global opt_bar
    global opt_max_number
    global param_bar
    global param
    global aug_mat

    param_bar = value

    aug_text = 'AUG_CONFIG = '
    fanwei = 'Range = '
    shuoming = 'Explanation: '
    cv2.imshow('original', img)
    if(opt_bar == 0):
        #Rotate
        param[0] = 1.0 / 100.0*360.0 * param_bar
        aug_mat = cvAugmentor.imRotate(img, param)
        cv2.putText(aug_mat[0], aug_text+'Rotate('+str(param[0])+')', (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "(---,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming+"Rotate by one angle (degree)", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 1):
        #FlipUD
        aug_mat = cvAugmentor.imFlipUD(img)
        cv2.putText(aug_mat[0], aug_text + "FlipUD", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "(NULL)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "Flip up down", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 2):
        #FlipLR
        aug_mat = cvAugmentor.imFlipLR(img)
        cv2.putText(aug_mat[0], aug_text + "FlipLR", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "(NULL)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "Flip left right", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 3):
        #Flip
        aug_mat = cvAugmentor.imFlip(img)
        cv2.putText(aug_mat[0], aug_text + "Flip", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "(NULL)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "Random flip", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 4):
        #Crop
        param[0] = 1.0 / 100.0* param_bar
        if(param[0]<0.01):
             param[0] = 0.01
        aug_mat = cvAugmentor.imCrop(img, param)
        cv2.putText(aug_mat[0], aug_text + "Crop(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "(0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "Random crop image by area percentage", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 5):
        #AffineX
        param[0] = 1.0 / 50.0* (param_bar - 50)
        aug_mat = cvAugmentor.imAffineX(img, param)
        cv2.putText(aug_mat[0], aug_text + "AffineX(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[-1,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "X axis affine", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 6):
        #AffineY
        param[0] = 1.0 / 50.0* (param_bar - 50)
        aug_mat = cvAugmentor.imAffineY(img, param)
        cv2.putText(aug_mat[0], aug_text + "AffineY(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[-1,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "Y axis affine", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 7):
        #AffineY
        param[0] = 1.0 / 50.0* (param_bar - 50)
        aug_mat = cvAugmentor.imAffine(img, param)
        cv2.putText(aug_mat[0], aug_text + "Affine(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[-1,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "X or Y axis affine", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 8):
        #Noise
        param[0] = 1.0 / 100.0* (param_bar)
        aug_mat = cvAugmentor.imNoise(img, param)
        cv2.putText(aug_mat[0], aug_text + "Noise(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "Add gauss noise", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 9):
        #Hue
        param[0] = 1.0 / 100.0*360.0 * param_bar
        aug_mat = cvAugmentor.imHue(img, param)
        cv2.putText(aug_mat[0], aug_text + "Hue(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "(---,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "Hue angle (degree)", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 10):
        #Saturation
        param[0] = 1.0 / 50.0* (param_bar - 50)
        aug_mat = cvAugmentor.imSaturation(img, param)
        cv2.putText(aug_mat[0], aug_text + "Saturation(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "Saturation", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 11):
        #Lightness
        param[0] = 1.0 / 50.0* (param_bar - 50)
        aug_mat = cvAugmentor.imLightness(img, param)
        cv2.putText(aug_mat[0], aug_text + "Lightness(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "Lightness", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 12):
        #PerspectiveUL
        param[0] = 1.0 / 100.0* param_bar
        aug_mat = cvAugmentor.imPerspectiveUL(img, param)
        cv2.putText(aug_mat[0], aug_text + "PerspectiveUL(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "UL Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 13):
        #PerspectiveU
        param[0] = 1.0 / 100.0* param_bar
        aug_mat = cvAugmentor.imPerspectiveU(img, param)
        cv2.putText(aug_mat[0], aug_text + "PerspectiveU(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "U  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 14):
        #PerspectiveUR
        param[0] = 1.0 / 100.0* param_bar
        aug_mat = cvAugmentor.imPerspectiveUR(img, param)
        cv2.putText(aug_mat[0], aug_text + "PerspectiveUR(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "UR Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 15):
        #PerspectiveL
        param[0] = 1.0 / 100.0* param_bar
        aug_mat = cvAugmentor.imPerspectiveL(img, param)
        cv2.putText(aug_mat[0], aug_text + "PerspectiveL(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "L  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 16):
        #Perspective
        param[0] = 1.0 / 100.0* param_bar
        aug_mat = cvAugmentor.imPerspective(img, param)
        cv2.putText(aug_mat[0], aug_text + "Perspective(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "Random perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 17):
        #PerspectiveR
        param[0] = 1.0 / 100.0* param_bar
        aug_mat = cvAugmentor.imPerspectiveR(img, param)
        cv2.putText(aug_mat[0], aug_text + "PerspectiveR(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "R  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 18):
        #PerspectiveDL
        param[0] = 1.0 / 100.0* param_bar
        aug_mat = cvAugmentor.imPerspectiveDL(img, param)
        cv2.putText(aug_mat[0], aug_text + "PerspectiveDL(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "DL Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 19):
        #PerspectiveD
        param[0] = 1.0 / 100.0* param_bar
        aug_mat = cvAugmentor.imPerspectiveD(img, param)
        cv2.putText(aug_mat[0], aug_text + "PerspectiveD(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "D  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    elif(opt_bar == 20):
        #PerspectiveDR
        param[0] = 1.0 / 100.0* param_bar
        aug_mat = cvAugmentor.imPerspectiveDR(img, param)
        cv2.putText(aug_mat[0], aug_text + "PerspectiveDR(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat[0], shuoming + "DR Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat[0])
    #elif(opt_bar == 21):
    #    #Distort
    #    param[0] = 1.0 / 20.0* param_bar
    #    cvAugmentor.imDistort(img, aug_mat, param)
    #    cv2.putText(aug_mat[0], aug_text + "Distort(" + str(param[0]) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.putText(aug_mat[0], fanwei + "[0,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.putText(aug_mat[0], shuoming + "Add distort", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.imshow("augmented", aug_mat[0])
    #elif(opt_bar == 21):
    #    #Pyramid
    #    aug_mat[0].fill(255 / 2)
    #    cv2.putText(aug_mat[0], aug_text + "Pyramid(donw,up)", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.putText(aug_mat[0], fanwei + "(0,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.putText(aug_mat[0], shuoming + "Build image pyramid", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.imshow("augmented", aug_mat[0])
    #elif(opt_bar == 23):#NotBackup
    #    aug_mat[0].setTo(Scalar(255 / 2, 255 / 2, 255 / 2))
    #    cv2.putText(aug_mat[0], aug_text + "NotBackup", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.putText(aug_mat[0], fanwei + "(NULL)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.putText(aug_mat[0], shuoming + "A flag means do not keep prestep images", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.imshow("augmented", aug_mat[0])
    return


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    #'common cfg'
    parser.add_argument('--img',type=str, default='', help='Image you want to view')
    opt = parser.parse_args()
    if(opt.img == ''):
        makeDefaultImg()
        #bl = np.zeros((img.shape[0],img.shape[1],4),np.uint8)
        #bl[:,:,0:3] = img
        #img = bl
    else:
        img = Resizer(user_img_view_size,\
        RESIZE_TYPE.ROUNDUP,cv2.INTER_LINEAR).imResize(cv2.imread(opt.img))

    key_value = 0
    while(1):
        cv2.namedWindow('original',cv2.WINDOW_AUTOSIZE|cv2.WINDOW_KEEPRATIO|cv2.WINDOW_GUI_EXPANDED)
        cv2.createTrackbar('option','original',opt_bar,opt_max_number-1,on_trackbar_share_opt_bar)
        cv2.createTrackbar('param','original',param_bar,100,on_trackbar_share_param_bar)
        on_trackbar_share_opt_bar(opt_bar)
        on_trackbar_share_param_bar(param_bar)
        key_value = cv2.waitKey(0)
        
        if (key_value == ord('d') or key_value == ord('D')):
            opt_bar +=1
            if (opt_bar >= opt_max_number):
                opt_bar = 0
        elif (key_value == ord('a') or key_value == ord('A')):
            opt_bar -=1
            if (opt_bar < 0):
                opt_bar = opt_max_number - 1
        elif(key_value == ord('s') or key_value == ord('S')):
            param_bar +=10
            if(param_bar>100):
                param_bar-=100
        elif(key_value == ord('W') or key_value == ord('w')):
            param_bar -=10
            if(param_bar<0):
                param_bar+=100

        if (key_value == 27 or key_value == ord('\r')):
            break
    cv2.destroyAllWindows()

        






