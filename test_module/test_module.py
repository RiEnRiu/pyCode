import numpy as np
import cv2
import glob

import sys
#sys.path.append('D:\project\pyCode\Common')
#import pyBoost as pb
#import vocBoost as vb


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#images = glob.glob('*.jpg')
#fname = r'D:\project\pyCode\tf_module\v0_2019-02-21_17-58-9\v0_2019-02-21_17-58-9_frame_10375.jpg'
fname = r'D:\project\pyCode\tf_module\v0_2019-02-21_17-58-9\v0_2019-02-21_17-58-9_frame_10285.jpg'


#for fname in images:
img1 = cv2.imread(fname)

#_,img1 = cv2.VideoCapture(r'.\video\v1_2019-02-19_15-46-11.avi').read()
gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (6,4),None)


# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)

    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    imgpoints.append(corners2)

    pts1 = np.array([x[0] for x in corners2])

    # Draw and display the corners
    img1 = cv2.drawChessboardCorners(img1, (6,4), corners2,ret)
    cv2.imshow('img1',img1)
    cv2.waitKey(0)


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#images = glob.glob('*.jpg')
#fname = r'D:\project\pyCode\tf_module\v1_2019-02-21_17-58-9\v1_2019-02-21_17-58-9_frame_10375.jpg'
fname = r'D:\project\pyCode\tf_module\v1_2019-02-21_17-58-9\v1_2019-02-21_17-58-9_frame_10285.jpg'


#for fname in images:
img2 = cv2.imread(fname)
#_,img2 = cv2.VideoCapture(r'.\video\v0_2019-02-19_15-46-11.avi').read()
gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (6,4),None)


# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)

    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    imgpoints.append(corners2)


    pts2 = np.array([x[0] for x in corners2])

    # Draw and display the corners
    img2 = cv2.drawChessboardCorners(img2, (6,4), corners2,ret)
    cv2.imshow('img2',img2)
    cv2.waitKey(0)

#pts1 = np.int32(pts1)

#cv2.putText(img1,'0',(int(pts1[0][0]),int(pts1[0][1])),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),3)


#1.0.2.6.0 ����
#pts1_list = [None]*24
#for i in range(0,6):
#    pts1_list[i] = pts1[i+18]
#for i in range(6,12):
#    pts1_list[i] = pts1[i+6]
#for i in range(12,18):
#    pts1_list[i] = pts1[i-6]
#for i in range(18,24):
#    pts1_list[i] = pts1[i-18]
#pts1 = np.array(pts1_list,np.int32)

#1.0.2.4.3 ����
#pts1_list = [None]*24
#for i in range(0,6):
#    pts1_list[i] = pts1[5-i]
#for i in range(6,12):
#    pts1_list[i] = pts1[18-i]
#for i in range(12,18):
#    pts1_list[i] = pts1[29-i]
#for i in range(18,24):
#    pts1_list[i] = pts1[41-i]
#pts1 = np.array(pts1_list,np.int32)


#1.0.2.3.3 ����
#pts1_list = [None]*24
#for i in range(0,6):
#    pts1_list[i] = pts1[5-i]
#for i in range(6,12):
#    pts1_list[i] = pts1[18-i]
#for i in range(12,18):
#    pts1_list[i] = pts1[29-i]
#for i in range(18,24):
#    pts1_list[i] = pts1[41-i]
#pts1 = np.array(pts1_list,np.int32)


#1.0.0.2.0 ����
#pts1_list = [None]*24
#for i in range(0,6):
#    pts1_list[i] = pts1[5-i]
#for i in range(6,12):
#    pts1_list[i] = pts1[18-i]
#for i in range(12,18):
#    pts1_list[i] = pts1[29-i]
#for i in range(18,24):
#    pts1_list[i] = pts1[41-i]
#pts1 = np.array(pts1_list,np.int32)

#1.0.0.5.0 ����
#pts1_list = [None]*24
#for i in range(0,6):
#    pts1_list[i] = pts1[5-i]
#for i in range(6,12):
#    pts1_list[i] = pts1[18-i]
#for i in range(12,18):
#    pts1_list[i] = pts1[29-i]
#for i in range(18,24):
#    pts1_list[i] = pts1[41-i]
#pts1 = np.array(pts1_list,np.int32)


#1.0.3.7.5 ����
#pts1_list = [None]*24
#for i in range(0,6):
#    pts1_list[i] = pts1[5-i]
#for i in range(6,12):
#    pts1_list[i] = pts1[18-i]
#for i in range(12,18):
#    pts1_list[i] = pts1[29-i]
#for i in range(18,24):
#    pts1_list[i] = pts1[41-i]
#pts1 = np.array(pts1_list,np.int32)


#1.0.3.5.4 ����
#pts1_list = [None]*24
#for i in range(0,6):
#    pts1_list[i] = pts1[5-i]
#for i in range(6,12):
#    pts1_list[i] = pts1[18-i]
#for i in range(12,18):
#    pts1_list[i] = pts1[29-i]
#for i in range(18,24):
#    pts1_list[i] = pts1[41-i]
#pts1 = np.array(pts1_list,np.int32)


#print([((int(x[0][0]),int(x[0][1])),(int(x[1][0]),int(x[1][1]))) for x in zip(pts1,pts2)])

#for i in range(24):
#    pts1_list[i] = pts1[23-i]
#pts1 = np.array(pts1_list,np.int32)



pairs = \
[((1312, 735), (1056, 716)), ((1298, 714), (1075, 675)), ((1284, 692), (1094, 636)), ((1270, 671), (1113, 598)), ((1256, 651), (1132, 561)), ((1243, 630), (1151, 525)), ((1206, 652), (1094, 738)), ((1293, 747), (1113, 697)), ((1279, 725), (1132, 658)), ((1266, 704), (1150, 620)), ((1252, 682), (1168, 583)), ((1238, 661), (1187, 547)), ((1274, 759), (1133, 760)), ((1260, 737), (1151, 719)), ((1246, 715), (1170, 680)), ((1233, 694), (1187, 642)), ((1219, 673), (1205, 604)), ((1206, 652), (1223, 568)), ((1255, 772), (1172, 782)), ((1241, 749), (1190, 741)), ((1227, 727), (1207, 702)), ((1214, 705), (1225, 663)), ((1200, 684), (1241, 626)), ((1187, 663), (1259, 589))] + \
[((1192, 687), (1204, 366)), ((1193, 706), (1202, 437)), ((1194, 726), (1199, 512)), ((1195, 746), (1195, 590)), ((1196, 766), (1190, 674)), ((1198, 786), (1184, 764)), ((1211, 687), (1127, 366)), ((1212, 707), (1124, 437)), ((1213, 726), (1118, 511)), ((1214, 746), (1112, 589)), ((1216, 766), (1105, 672)), ((1217, 786), (1096, 760)), ((1230, 687), (1052, 366)), ((1232, 707), (1046, 437)), ((1233, 726), (1039, 510)), ((1234, 746), (1030, 587)), ((1236, 766), (1020, 670)), ((1238, 786), (1010, 757)), ((1249, 687), (976, 367)), ((1251, 707), (969, 436)), ((1252, 726), (959, 509)), ((1254, 746), (950, 587)), ((1256, 766), (936, 667)),((1257, 786), (925, 754))] + \
[((1639, 677), (663, 660)), ((1618, 655), (687, 632)), ((1598, 632), (711, 604)), ((1578, 610), (734, 575)), ((1558, 588), (756, 548)), ((1539, 567), (778, 521)), ((1498, 595), (690, 681)), ((1619, 692), (714, 652)), ((1598, 669), (737, 623)), ((1578, 647), (760, 595)), ((1558, 624), (782, 567)), ((1539, 602), (804, 540)), ((1599, 708), (717, 702)), ((1578, 685), (740, 672)), ((1558, 662), (763, 644)), ((1538, 639), (787, 615)), ((1518, 616), (809, 587)), ((1498, 595), (831,559)), ((1578, 724), (743, 723)), ((1557, 700), (767, 693)), ((1537, 677), (790, 664)), ((1517, 654), (813, 635)), ((1498, 631), (836, 607)), ((1478, 609), (857, 579))] + \
[((1470, 660), (692, 550)), ((1467, 637), (702, 509)), ((1463, 614), (713, 470)), ((1459, 591), (722, 432)), ((1455, 569), (731, 395)), ((1452, 547), (740, 359)), ((1409, 550), (737, 556)), ((1449, 662), (746, 515)), ((1445, 639), (756, 475)), ((1442, 616), (765, 438)), ((1438, 593), (774, 401)), ((1434, 570), (782, 365)), ((1427, 664), (782, 561)), ((1423, 640), (791, 520)), ((1420, 618), (799, 481)), ((1416, 595), (807, 443)), ((1412, 572), (815, 407)), ((1409, 550), (823,371)), ((1405, 666), (827, 567)), ((1401, 643), (835, 526)), ((1398, 619), (842, 486)), ((1394, 597), (850, 449)), ((1391, 574), (857, 412)), ((1387, 552), (863, 376))] + \
[((1221, 453), (1202, 567)), ((1218, 482), (1207, 593)), ((1214, 512), (1213, 618)), ((1211, 543), (1218, 644)), ((1207, 576), (1224, 670)), ((1203, 609), (1229, 695)), ((1283, 618), (1177, 572)), ((1257, 456), (1182, 597)), ((1254, 485), (1187, 623)), ((1252, 515), (1193, 648)), ((1249, 547), (1198, 674)), ((1246, 579), (1204, 700)), ((1294, 459), (1152, 577)), ((1292, 488), (1157, 602)), ((1290, 518), (1162, 627)), ((1287, 551), (1167, 653)), ((1285, 583), (1173, 679)), ((1283, 618), (1178, 705)), ((1331, 462), (1127, 581)), ((1329, 491), (1132, 607)), ((1328, 523), (1137, 632)), ((1326, 554), (1142, 657)), ((1325, 587), (1148, 683)), ((1323, 623), (1153, 709))] + \
[((1842, 356), (811, 564)), ((1812, 325), (829, 547)), ((1781, 296), (846, 532)), ((1749, 266), (863, 516)), ((1718, 237), (880, 500)), ((1688, 208), (896, 485)), ((1642, 247), (827, 579)), ((1822, 378), (845, 563)), ((1791, 347), (862, 547)), ((1759, 316), (879, 531)), ((1728, 286), (896, 516)), ((1697, 256), (912, 500)), ((1802, 400), (843, 595)), ((1769, 369), (861, 578)), ((1737, 338), (878, 562)), ((1706, 307), (895, 546)), ((1674, 277), (911, 531)), ((1642, 247), (928,515)), ((1780, 423), (860, 611)), ((1747, 392), (877, 594)), ((1715, 360), (894, 578)), ((1682, 330), (911, 561)), ((1650, 298), (927, 546)), ((1618, 268), (944, 530))] + \
[((1181, 328), (1214, 611)), ((1156, 357), (1226, 629)), ((1130, 387), (1239, 647)), ((1104, 418), (1251, 665)), ((1075, 451), (1264, 683)), ((1046, 485), (1277, 702)), ((1131, 546), (1197, 624)), ((1220, 354), (1209, 642)), ((1196, 383), (1222, 660)), ((1171, 414), (1235, 678)), ((1144, 446), (1248, 697)), ((1117, 480), (1260, 715)), ((1259, 380), (1180, 636)), ((1236, 410), (1192, 654)), ((1211, 442), (1205, 673)), ((1186, 475), (1217, 691)), ((1159, 509), (1230, 710)), ((1131, 546), (1243, 728)), ((1299, 407), (1162, 649)), ((1276, 438), (1175, 667)), ((1253, 470), (1188, 686)), ((1228, 504), (1200, 704)), ((1202, 540), (1213, 723)), ((1175, 577), (1226, 742))]



#xml_root = r'D:\project\pyCode\tf_module\2019-02-21_17-58-9'
#xmls = pb.scanner.file(xml_root,'.xml',False,True)
#pairs = []
#for xml in xmls:
#    if(xml[0:2]=='v1'):
#        continue
#    v0 = vb.vocXmlRead(xml_root+'/'+xml)
#    v1 = vb.vocXmlRead(xml_root+'/v1'+xml[2:])
#    b0 = v0.objs[0]
#    b1 = v1.objs[0]
#    pairs.append(((b0.xmin,b0.ymin),(b1.xmax,b1.ymin)))
#    pairs.append(((b0.xmax,b0.ymax),(b1.xmin,b1.ymax)))


pts1 = np.array([p[0] for p in pairs],np.int32)
pts2 = np.array([p[1] for p in pairs],np.int32)


#pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]



               
#img1,img2 = img2,img1
#points_all = [((802, 607), (1903, 473)), ((802, 776), (1903, 727)), ((960, 776), (1654, 727)), ((960, 607), (1654, 473)), ((854, 618), (1822, 523)), ((854, 784), (1822, 765)), ((1007, 784), (1581, 765)), ((1007, 618), (1581, 523)), ((921, 504), (1613, 478)), ((921, 696), (1613, 670)), ((1094, 696), (1413, 670)), ((1094, 504), (1413, 478)), ((879, 483), (1653, 444)), ((879, 661), (1653, 633)), ((1048, 661), (1443, 633)), ((1048, 483), (1443, 444)), ((827, 442), (1712, 392)), ((827, 629), (1712, 563)), ((1022, 629), (1498, 563)), ((1022, 442), (1498, 392)), ((762, 381), (1732, 368)), ((762, 564), (1732, 534)), ((968, 564), (1513, 534)), ((968, 381), (1513, 368)), ((742, 356), (1745, 351)), ((742, 535), (1745, 505)), ((940, 535), (1530, 505)), ((940, 356), (1530, 351)), ((704, 314), (1753, 313)), ((704, 478), (1753, 463)), ((918, 478), (1533, 463)), ((918, 314), (1533, 313)), ((695, 296), (1748, 308)), ((695, 454), (1748, 450)), ((905, 454), (1535, 450)), ((905, 296), (1535, 308)), ((687, 273), (1739, 305)), ((687, 423), (1739, 439)), ((889, 423), (1530, 439)), ((889, 273), (1530, 305)), ((682, 262), (1729, 307)), ((682, 414), (1729, 446)), ((899, 414), (1520, 446)), ((899, 262), (1520, 307)), ((691, 269), (1711, 324)), ((691, 426), (1711, 459)), ((913, 426), (1515, 459)), ((913, 269), (1515, 324)), ((703, 276), (1682, 360)), ((703, 441), (1682, 504)), ((922, 441), (1479, 504)), ((922, 276), (1479, 360)), ((712, 287), (1656, 385)), ((712, 459), (1656, 530)), ((938, 459), (1462, 530)), ((938, 287), (1462, 385)), ((755, 309), (1594, 432)), ((755, 504), (1594, 588)), ((999, 504), (1413, 588)), ((999, 309), (1413, 432)), ((799, 320), (1564, 457)), ((799, 534), (1564, 616)), ((1038, 534), (1381, 616)), ((1038, 320), (1381, 457)), ((835, 340), (1523, 480)), ((835, 555), (1523, 645)), ((1061, 555), (1346, 645)), ((1061, 340), (1346, 480)), ((937, 76), (1412, 518)), ((937, 403), (1412, 654)), ((1245, 403), (1277, 654)), ((1245, 76), (1277, 518)), ((832, 46), (1437, 500)), ((832, 344), (1437, 631)), ((1151, 344), (1301, 631)), ((1151, 46), (1301, 500))]
#pts1 = np.array([x[0] for x in points_all],np.int32)
#pts2 = np.array([x[1] for x in points_all],np.int32)

#F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
## We select only inlier points
#pts1 = pts1[mask.ravel()==1]
#pts2 = pts2[mask.ravel()==1]

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)

    index = 0
    col = [(255,0,0),(0,255,0),(0,0,255),(255,255,255)]

    for r,pt1,pt2 in zip(lines,pts1,pts2):
        if(index<4):
            color = col[index]
        else:
            color = tuple(np.random.randint(0,255,3).tolist())
        index+=1
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)


    #for p in pts1:
    #    x0,y0 = map(int, [0, -r[2]/r[1] ])
    #    x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        
    #    cv2.circle(img1,((x0+x1)//2,(y0+y1)//2),3,col[index],3)
    #    index += 1
    #    if(index>=4):
    #        break

    return img1,img2


img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image

#pts1 = np.array([pts1[0],pts1[3],pts1[18],pts1[21]])

#rect = [int(min(pts1[0][0],pts1[1][0],pts1[2][0],pts1[3][0])),\
#        int(min(pts1[0][1],pts1[1][1],pts1[2][1],pts1[3][1])),\
#        int(max(pts1[0][0],pts1[1][0],pts1[2][0],pts1[3][0])),\
#        int(max(pts1[0][1],pts1[1][1],pts1[2][1],pts1[3][1]))]




rect = [1037,453,1037+248,453+343]

cv2.rectangle(img5,(rect[0],rect[1]),(rect[2],rect[3]),(0,0,255),1,1)

pts1 = np.array([(rect[0],rect[1]),(rect[0],rect[3]),(rect[2],rect[3]),(rect[2],rect[1])])

cv2.circle(img5,(int(pts1[0][0]),int(pts1[0][1])),3,(255,0,0),3)
cv2.circle(img5,(int(pts1[1][0]),int(pts1[1][1])),3,(0,255,0),3)
cv2.circle(img5,(int(pts1[2][0]),int(pts1[2][1])),3,(0,0,255),3)
cv2.circle(img5,(int(pts1[3][0]),int(pts1[3][1])),3,(255,255,255),3)

for p in pts1:
    cv2.circle(img5,(int(p[0]),int(p[1])),10,(0,0,0),10)




lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
print(pts1.reshape(-1,1,2))
lines2 = lines2.reshape(-1,3)

img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)


cv2.imshow('img5',img5)
cv2.imshow('img3',img3)
cv2.waitKey(0)

