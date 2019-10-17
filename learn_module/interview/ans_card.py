import cv2
import numpy as np

def softmax(scores):
    row_max = scores.max(axis = 1).reshape(-1,1)
    scores = scores - row_max
    scores_exp = np.exp(scores)
    scores_exp_row_sum = scores_exp.sum(axis=1).reshape(-1,1)
    softmaxs = scores_exp / scores_exp_row_sum
    return softmaxs

def read_ans_card():




if __name__=='__main__':
    img = cv2.imread('./ans_card.jpg')

    # to gray image
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # mask invalid area
    img_gray[300:315,20:260] = img_gray[285:300,20:260]
    img_gray[315:330,20:260] = img_gray[285:300,20:260]

    # find circles
    circles_param = cv2.HoughCircles(img_gray,cv2.HOUGH_GRADIENT,1,30,50,250,15,8,20)[0]

    # sort circle
    y_sorted_ind = np.argsort(circles_param[:,1])
    circles_param = circles_param[y_sorted_ind, :]
    for ii in range(5):
        part = circles_param[ii*5:ii*5+5,:]
        x_sorted_ind = np.argsort(part[:,0])
        circles_param[ii*5:ii*5+5,:] = part[x_sorted_ind, :]

    ans_abcde = ['A','B','C','D','E']
    if(circles_param is not None):
        # find ans
        bbs = [[c[0]-c[2],c[1]-c[2],c[0]+c[2],c[1]+c[2]] for c in circles_param]
        bbs = np.array(bbs,np.int32)
        bb_rois = [img_gray[bb[1]:bb[3],bb[0]:bb[2]] for bb in bbs]
        scores = [-r.astype(np.float).sum()/r.size for r in bb_rois]
        scores = np.array(scores, np.float32).reshape(-1,5)
        props = softmax(scores)
        for i, prop in enumerate(props):
            ans_i = prop.argmax()
            ans = ans_abcde[ans_i]
            print('answer {0}: {1}'.format(i+1, ans))
        
        # print circle box
        for i,cir,bb,p in zip(range(len(circles_param)),\
                              circles_param,\
                              bbs,\
                              props.reshape(25)):
            cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),(255,0,0))
            cv2.circle(img,(int(cir[0]),int(cir[1])),int(cir[2]),(0,0,255))
            cv2.putText(img,str(i),(int(bb[0]),int(bb[1])),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255))  
            cv2.putText(img,str(int(p*100)),(int(cir[0]),int(cir[1])),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255))  


    cv2.imshow('img',img)
    cv2.imshow('gray',img_gray)
    cv2.waitKey(0)

    

