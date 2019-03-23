#-*-coding:utf-8-*-
import sys
import os
__pyBoost_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__pyBoost_root_path)
import pyBoost as pb

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

param = 10
aug_mat = img.copy()




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
        param = 1.0 / 100.0*360.0 * param_bar
        aug_mat = pb.img.rotate(img, param)
        cv2.putText(aug_mat, aug_text+'Rotate('+str(param)+')', (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "(---,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming+"Rotate by one angle (degree)", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 1):
        #FlipUD
        aug_mat = pb.img.flipUD(img)
        cv2.putText(aug_mat, aug_text + "FlipUD", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "(NULL)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "Flip up down", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 2):
        #FlipLR
        aug_mat = pb.img.flipLR(img)
        cv2.putText(aug_mat, aug_text + "FlipLR", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "(NULL)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "Flip left right", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 3):
        #Flip
        aug_mat = pb.img.flip(img)
        cv2.putText(aug_mat, aug_text + "Flip", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "(NULL)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "Random flip", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 4):
        #Crop
        param = 1.0 / 100.0* param_bar
        if(param<0.01):
             param = 0.01
        aug_mat = pb.img.crop(img, param)
        cv2.putText(aug_mat, aug_text + "Crop(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "(0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "Random crop image by area percentage", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 5):
        #AffineX
        param = 1.0 / 50.0* (param_bar - 50)
        aug_mat = pb.img.affineX(img, param)
        cv2.putText(aug_mat, aug_text + "AffineX(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[-1,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "X axis affine", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 6):
        #AffineY
        param = 1.0 / 50.0* (param_bar - 50)
        aug_mat = pb.img.affineY(img, param)
        cv2.putText(aug_mat, aug_text + "AffineY(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[-1,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "Y axis affine", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 7):
        #AffineY
        param = 1.0 / 50.0* (param_bar - 50)
        aug_mat = pb.img.affine(img, param)
        cv2.putText(aug_mat, aug_text + "Affine(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[-1,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "X or Y axis affine", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 8):
        #Noise
        param = 1.0 / 100.0* (param_bar)
        aug_mat = pb.img.add_noise(img, param)
        cv2.putText(aug_mat, aug_text + "Noise(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "Add gauss noise", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 9):
        #Hue
        param = 1.0 / 100.0*360.0 * param_bar
        aug_mat = pb.img.adjust_hue(img, param)
        cv2.putText(aug_mat, aug_text + "Hue(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "(---,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "Hue angle (degree)", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 10):
        #Saturation
        param = 1.0 / 50.0* (param_bar - 50)
        aug_mat = pb.img.adjust_saturation(img, param)
        cv2.putText(aug_mat, aug_text + "Saturation(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "Saturation", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 11):
        #Lightness
        param = 1.0 / 50.0* (param_bar - 50)
        aug_mat = pb.img.adjust_lightness(img, param)
        cv2.putText(aug_mat, aug_text + "Lightness(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "Lightness", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 12):
        #PerspectiveUL
        param = 1.0 / 100.0* param_bar
        aug_mat = pb.img.perspectiveUL(img, param)
        cv2.putText(aug_mat, aug_text + "PerspectiveUL(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "UL Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 13):
        #PerspectiveU
        param = 1.0 / 100.0* param_bar
        aug_mat = pb.img.perspectiveU(img, param)
        cv2.putText(aug_mat, aug_text + "PerspectiveU(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "U  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 14):
        #PerspectiveUR
        param = 1.0 / 100.0* param_bar
        aug_mat = pb.img.perspectiveUR(img, param)
        cv2.putText(aug_mat, aug_text + "PerspectiveUR(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "UR Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 15):
        #PerspectiveL
        param = 1.0 / 100.0* param_bar
        aug_mat = pb.img.perspectiveUL(img, param)
        cv2.putText(aug_mat, aug_text + "PerspectiveL(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "L  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 16):
        #Perspective
        param = 1.0 / 100.0* param_bar
        aug_mat = pb.img.perspective(img, param)
        cv2.putText(aug_mat, aug_text + "Perspective(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "Random perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 17):
        #PerspectiveR
        param = 1.0 / 100.0* param_bar
        aug_mat = pb.img.perspectiveR(img, param)
        cv2.putText(aug_mat, aug_text + "PerspectiveR(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "R  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 18):
        #PerspectiveDL
        param = 1.0 / 100.0* param_bar
        aug_mat = pb.img.perspectiveDL(img, param)
        cv2.putText(aug_mat, aug_text + "PerspectiveDL(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "DL Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 19):
        #PerspectiveD
        param = 1.0 / 100.0* param_bar
        aug_mat = pb.img.perspectiveD(img, param)
        cv2.putText(aug_mat, aug_text + "PerspectiveD(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "D  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 20):
        #PerspectiveDR
        param = 1.0 / 100.0* param_bar
        aug_mat = pb.img.perspectiveDR(img, param)
        cv2.putText(aug_mat, aug_text + "PerspectiveDR(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "DR Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    #elif(opt_bar == 21):
    #    #Distort
    #    param = 1.0 / 20.0* param_bar
    #    pb.img.imDistort(img, aug_mat, param)
    #    cv2.putText(aug_mat, aug_text + "Distort(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.putText(aug_mat, fanwei + "[0,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.putText(aug_mat, shuoming + "Add distort", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.imshow("augmented", aug_mat)
    #elif(opt_bar == 21):
    #    #Pyramid
    #    aug_mat.fill(255 / 2)
    #    cv2.putText(aug_mat, aug_text + "Pyramid(donw,up)", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.putText(aug_mat, fanwei + "(0,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.putText(aug_mat, shuoming + "Build image pyramid", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.imshow("augmented", aug_mat)
    #elif(opt_bar == 23):#NotBackup
    #    aug_mat.setTo(Scalar(255 / 2, 255 / 2, 255 / 2))
    #    cv2.putText(aug_mat, aug_text + "NotBackup", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.putText(aug_mat, fanwei + "(NULL)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.putText(aug_mat, shuoming + "A flag means do not keep prestep images", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.imshow("augmented", aug_mat)
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
        param = 1.0 / 100.0*360.0 * param_bar
        aug_mat = pb.img.rotate(img, param)
        cv2.putText(aug_mat, aug_text+'Rotate('+str(param)+')', (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "(---,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming+"Rotate by one angle (degree)", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 1):
        #FlipUD
        aug_mat = pb.img.flipUD(img)
        cv2.putText(aug_mat, aug_text + "FlipUD", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "(NULL)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "Flip up down", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 2):
        #FlipLR
        aug_mat = pb.img.flipLR(img)
        cv2.putText(aug_mat, aug_text + "FlipLR", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "(NULL)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "Flip left right", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 3):
        #Flip
        aug_mat = pb.img.flip(img)
        cv2.putText(aug_mat, aug_text + "Flip", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "(NULL)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "Random flip", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 4):
        #Crop
        param = 1.0 / 100.0* param_bar
        if(param<0.01):
             param = 0.01
        aug_mat = pb.img.crop(img, param)
        cv2.putText(aug_mat, aug_text + "Crop(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "(0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "Random crop image by area percentage", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 5):
        #AffineX
        param = 1.0 / 50.0* (param_bar - 50)
        aug_mat = pb.img.affineX(img, param)
        cv2.putText(aug_mat, aug_text + "AffineX(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[-1,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "X axis affine", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 6):
        #AffineY
        param = 1.0 / 50.0* (param_bar - 50)
        aug_mat = pb.img.affineY(img, param)
        cv2.putText(aug_mat, aug_text + "AffineY(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[-1,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "Y axis affine", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 7):
        #AffineY
        param = 1.0 / 50.0* (param_bar - 50)
        aug_mat = pb.img.affine(img, param)
        cv2.putText(aug_mat, aug_text + "Affine(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[-1,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "X or Y axis affine", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 8):
        #Noise
        param = 1.0 / 100.0* (param_bar)
        aug_mat = pb.img.add_noise(img, param)
        cv2.putText(aug_mat, aug_text + "Noise(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "Add gauss noise", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 9):
        #Hue
        param = 1.0 / 100.0*360.0 * param_bar
        aug_mat = pb.img.adjust_hue(img, param)
        cv2.putText(aug_mat, aug_text + "Hue(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "(---,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "Hue angle (degree)", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 10):
        #Saturation
        param = 1.0 / 50.0* (param_bar - 50)
        aug_mat = pb.img.adjust_saturation(img, param)
        cv2.putText(aug_mat, aug_text + "Saturation(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "Saturation", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 11):
        #Lightness
        param = 1.0 / 50.0* (param_bar - 50)
        aug_mat = pb.img.adjust_lightness(img, param)
        cv2.putText(aug_mat, aug_text + "Lightness(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "Lightness", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 12):
        #PerspectiveUL
        param = 1.0 / 100.0* param_bar
        aug_mat = pb.img.perspectiveUL(img, param)
        cv2.putText(aug_mat, aug_text + "PerspectiveUL(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "UL Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 13):
        #PerspectiveU
        param = 1.0 / 100.0* param_bar
        aug_mat = pb.img.perspectiveU(img, param)
        cv2.putText(aug_mat, aug_text + "PerspectiveU(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "U  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 14):
        #PerspectiveUR
        param = 1.0 / 100.0* param_bar
        aug_mat = pb.img.perspectiveUR(img, param)
        cv2.putText(aug_mat, aug_text + "PerspectiveUR(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "UR Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 15):
        #PerspectiveL
        param = 1.0 / 100.0* param_bar
        aug_mat = pb.img.perspectiveL(img, param)
        cv2.putText(aug_mat, aug_text + "PerspectiveL(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "L  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 16):
        #Perspective
        param = 1.0 / 100.0* param_bar
        aug_mat = pb.img.perspective(img, param)
        cv2.putText(aug_mat, aug_text + "Perspective(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "Random perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 17):
        #PerspectiveR
        param = 1.0 / 100.0* param_bar
        aug_mat = pb.img.perspectiveR(img, param)
        cv2.putText(aug_mat, aug_text + "PerspectiveR(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "R  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 18):
        #PerspectiveDL
        param = 1.0 / 100.0* param_bar
        aug_mat = pb.img.perspectiveDL(img, param)
        cv2.putText(aug_mat, aug_text + "PerspectiveDL(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "DL Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 19):
        #PerspectiveD
        param = 1.0 / 100.0* param_bar
        aug_mat = pb.img.perspectiveD(img, param)
        cv2.putText(aug_mat, aug_text + "PerspectiveD(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "D  Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    elif(opt_bar == 20):
        #PerspectiveDR
        param = 1.0 / 100.0* param_bar
        aug_mat = pb.img.perspectiveDR(img, param)
        cv2.putText(aug_mat, aug_text + "PerspectiveDR(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, fanwei + "[0,1]", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.putText(aug_mat, shuoming + "DR Perspective", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
        cv2.imshow("augmented", aug_mat)
    #elif(opt_bar == 21):
    #    #Distort
    #    param = 1.0 / 20.0* param_bar
    #    pb.img.imDistort(img, aug_mat, param)
    #    cv2.putText(aug_mat, aug_text + "Distort(" + str(param) + ")", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.putText(aug_mat, fanwei + "[0,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.putText(aug_mat, shuoming + "Add distort", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.imshow("augmented", aug_mat)
    #elif(opt_bar == 21):
    #    #Pyramid
    #    aug_mat.fill(255 / 2)
    #    cv2.putText(aug_mat, aug_text + "Pyramid(donw,up)", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.putText(aug_mat, fanwei + "(0,+++)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.putText(aug_mat, shuoming + "Build image pyramid", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.imshow("augmented", aug_mat)
    #elif(opt_bar == 23):#NotBackup
    #    aug_mat.setTo(Scalar(255 / 2, 255 / 2, 255 / 2))
    #    cv2.putText(aug_mat, aug_text + "NotBackup", (30, 30), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.putText(aug_mat, fanwei + "(NULL)", (30, 60), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.putText(aug_mat, shuoming + "A flag means do not keep prestep images", (30, 90), 1, 1.5, (255/2, 0 , 255/2), 2)
    #    cv2.imshow("augmented", aug_mat)
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
        img = pb.img.imResizer(user_img_view_size, pb.img.IMRESIZE_ROUNDUP,cv2.INTER_LINEAR).\
                                imResize(cv2.imread(opt.img))

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

        






