import cv2

import sys
sys.path.append('../Common')
from pyBoost import *

from tqdm import tqdm

images_path = scanner.file('./test_hand','.jpg.png.jpeg',True)



protoFile = './model/pose_deploy.prototxt'
weightsFile = './model/pose_iter_102000.caffemodel'
nPoints = 22
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]


net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

#cap = VideoCaptureThread(0)


#while(1):
for one_image_path in tqdm(images_path,ncols=50):

    frame = cv2.imread(one_image_path)
    #ret,frame = cap.read()
    


    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    aspect_ratio = frameWidth/frameHeight

    threshold = 0.05
    #threshold = -0.1

    # input image dimensions for the network
    inHeight = 368
    inWidth = int(((aspect_ratio*inHeight)*8)//8)

    #print(((frame/255.0*2.0)-1.0).shape)
    #input()
    #inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    inpBlob = cv2.dnn.blobFromImage((frame*1.0/256.0).astype(np.float32), 1.0, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)


    net.setInput(inpBlob)

    output = net.forward()

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        print(minVal)

        if prob > threshold :
            cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(point[0]), int(point[1])))
        else :
            points.append(None)

        
        #rows = probMap.shap[0]
        #cols = probMap.shap[1]
        #probMap_nebor = [probMap.copy(),probMap.copy(),probMap.copy(),probMap.copy(),probMap.copy(),probMap.copy(),probMap.copy(),probMap.copy()]
        #probMap_nebor[0][1:rows,1:cols] = probMap[0:rows-1,0:cols-1]
        #probMap_nebor[1][1:rows,:] = probMap[0:rows-1,:]
        #probMap_nebor[2][1:rows,0:cols-1] = probMap[0:rows-1,1:cols]
        #probMap_nebor[3][:,1:cols] = probMap[:,0:cols-1]
        #probMap_nebor[4][:,0:cols-1] = probMap[:,1:cols]
        #probMap_nebor[5][0:rows-1,1:cols] = probMap[1:rows,0:cols-1]
        #probMap_nebor[6][0:rows-1,:] = probMap[1:rows,:]
        #probMap_nebor[7][0:rows-1,0:cols-1] = probMap[1:rows,1:cols]
        
        #for i in range(8):
        #    probMap_flag = 

        

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)


    cv2.imshow('Output-Keypoints', frameCopy)
    cv2.imshow('Output-Skeleton', frame)

    key = cv2.waitKey(1000)
    if(key==27):
        break

cap.release()










