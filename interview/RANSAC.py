
import numpy as np
import random
import math

import cv2

def lineRANSAC(points,iter,threshDist, inlierRatio):
    #Total number of points
    number = points.shape[0]
    #Best fitting line with largest number of inliers
    bestInNum = 0
    #parameters for best fitting line
    bestParameter1=0
    bestParameter2=0
    all_points = [p for p in points]
    for i in range(iter):
        #Randomly select 2 points
        random.shuffle(all_points)
        sample = all_points[0:2]
        #Compute the distances between all points with the fitting line 
        kLine = sample[1]-sample[0]
        #kLine =[sample[0][0]-sample[1][0],sample[0][1]-sample[1][1]]# two points relative distance
        lenKLine = math.sqrt(kLine[0]*kLine[0]+kLine[1]*kLine[1])   
        kLineNorm = kLine/lenKLine
        #Ax+By+C=0 A=-kLineNorm(2),B=kLineNorm(1)
        normVector = np.array([-kLineNorm[1],kLineNorm[0]])

        distance = np.array(np.matmul(normVector,(points - np.array([sample[0].tolist()]*number)).transpose()))
        #Compute the inliers with distances smaller than the threshold
        inlierIdx = (np.abs(distance)<=threshDist)
        inlierNum = inlierIdx.sum()
        #Update the number of inliers and fitting model if better model is found     
        if inlierNum>=inlierRatio*number and inlierNum>bestInNum:
            bestInNum = inlierNum
            parameter1 = (sample[1][1]-sample[0][1])/(sample[1][0]-sample[0][0])
            parameter2 = sample[0][1]-parameter1*sample[0][0]
            bestParameter1=parameter1
            bestParameter2=parameter2

    return [bestParameter1,bestParameter2]

if __name__=='__main__':
    
    points = [\
              [ 240,  33],\
              [ 163,  44],\
              [  97,  93],\
              [ 219,  95],\
              [ 188, 140],\
              [ 299, 122],\
              [ 183, 191],\
              [ 119, 238],\
              [  41, 300],\
              [ 262, 180],\
              [ 323, 175],\
              [ 241, 241],\
              [ 212, 285],\
              [ 158, 330],\
              [  75, 377],\
              [  39, 554],\
              [  74, 529],\
              [  91, 502],\
              [ 129, 491],\
              [ 149, 450],\
              [ 170, 420],\
              [ 194, 414],\
              [ 270, 385],\
              [ 210, 387],\
              [ 236, 367],\
              [ 256, 351],\
              [ 282, 318],\
              [ 294, 303],\
              [ 320, 300],\
              [ 330, 269],\
              [ 360, 257],\
              [ 419, 272],\
              [ 381, 228],\
              [ 408, 197],\
              [ 416, 170],\
              [ 370, 526],\
              [ 415, 535],\
              [ 445, 488],\
              [ 346, 432],\
              [ 412, 383],\
              [ 466, 342],\
              [ 510, 320],\
              [ 539, 274],\
              [ 481, 221],\
              [ 442, 162],\
              [ 455, 148],\
              [ 490, 130],\
              [ 497, 119],\
              [ 512,  98],\
              [ 552, 66]\
             ]
    points = np.array(points)


    #y = ax+b
    [a,b] = lineRANSAC(points,100,10,0.1)
    print([a,b])

    xMax = points[:,0].max()
    yMax = points[:,1].max()
    img = np.zeros((yMax+10,xMax+10,3),np.uint8)
    for p in points:
        cv2.circle(img,(p[0],p[1]),1,(0,0,255),1)
    p4 = [(0,int(b)),(int(-b/a),0),(int(xMax),int(a*xMax+b)),(int((yMax-b)/a),int(yMax))]
    p2 = []
    for i in range(4):
        if p4[i][0]>=0 and p4[i][1]>=0:
            p2.append(p4[i])
    cv2.line(img,p2[0],p2[1],(255,0,0),3)
    cv2.imshow('img',img)
    cv2.waitKey(0)
