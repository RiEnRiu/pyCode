import cv2
import numpy as np


if __name__=='__main__':
    img = cv2.imread('./ans_card.jpg')
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img_gray = cv2.GaussianBlur(img_gray,(3, 3), 2, 2)
    #img_can = cv2.Canny(img_gray,200,250)
    #img_can_dilate = cv2.dilate(img_can,cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    img_gray[300:315,20:260] = img_gray[285:300,20:260]
    img_gray[315:330,20:260] = img_gray[285:300,20:260]

    circles_param = cv2.HoughCircles(img_gray,cv2.HOUGH_GRADIENT,1,30,50,250,15,8,20)[0]

    y_sorted_ind = np.argsort(circles_param[:,1])
    circles_param = circles_param[y_sorted_ind, :]
    for ii in range(5):
        part = circles_param[ii*5:ii*5+5,:]
        x_sorted_ind = np.argsort(part[:,0])
        circles_param[ii*5:ii*5+5,:] = part[x_sorted_ind, :]

    ans_abcde = np.array([['A','B','C','D','E']]*5).reshape(25)

    if circles_param is not None:
        for i,cir in enumerate(circles_param):
            bb = [int(cir[0]-cir[2]),int(cir[1]-cir[2]),int(cir[0]+cir[2]),int(cir[1]+cir[2])]
            cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),(255,0,0))
            cv2.circle(img,(int(cir[0]),int(cir[1])),int(cir[2]),(0,0,255))
            cv2.putText(img,str(i),(int(cir[0]),int(cir[1])),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255))  
            bb_roi = img_gray[bb[1]:bb[3],bb[0]:bb[2]]
            score = bb_roi.astype(np.float).sum()/bb_roi.size
            if score<=160:
                print(ans_abcde[i]) 

    cv2.imshow('img',img)
    cv2.imshow('gray',img_gray)
    cv2.waitKey(0)

    

