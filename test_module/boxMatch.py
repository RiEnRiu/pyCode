import cv2
import numpy as np

from sklearn.utils.linear_assignment_ import linear_assignment

class boxMatch:
    def __init__(self,mapping=(1,0,3,2),pts_pairs=None,F=None):

        # 4 points clockwise order. first view is [0,1,2,3], and another is list(mapping)
        self._mapping = np.array(mapping,np.uint8)
        self._inv_mapping = 3-self._mapping

        if pts_pairs is None and F is None:
            raise ValueError("pts_pairs or F must be set")
        if F is not None:
            self._F = np.array(F,dtype=np.float32)
        else:
            self._pts_pairs = pts_pairs
            pts1_list,pts2_list = [],[]
            for pair in self._pts_pairs:
                pts1_list.append(pair[0])
                pts2_list.append(pair[1])
            pts1 = np.int32(pts1_list)
            pts2 = np.int32(pts2_list)
            self._F,mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

        # calculate camera point
        v2_cam_pt = np.matmul(\
                        np.mat([[self._F[0,0],self._F[1,0]],[self._F[0,1],self._F[1,1]]]).I,\
                        [[-self._F[2,0]],[-self._F[2,1]]])
        self._v2_cam_pt = (v2_cam_pt[0,0],v2_cam_pt[1,0])
        self._FT = np.mat(self._F).T
        v1_cam_pt = np.matmul(\
                        np.mat([[self._FT[0,0],self._FT[1,0]],[self._FT[0,1],self._FT[1,1]]]).I,\
                        [[-self._FT[2,0]],[-self._FT[2,1]]])
        self._v1_cam_pt = (v1_cam_pt[0,0],v1_cam_pt[1,0])

        self._v1_cam_pt_int = (int(self._v1_cam_pt[0]),int(self._v1_cam_pt[1]))
        self._v2_cam_pt_int = (int(self._v2_cam_pt[0]),int(self._v2_cam_pt[1]))

    def boxMatchDegree(self,box1,box2,whichImage=None):
        #                  tl                tr                br                bl
        box1_pts = [[box1[0],box1[1]],[box1[2],box1[1]],[box1[2],box1[3]],[box1[0],box1[3]]]
        box2_pts = [[box2[0],box2[1]],[box2[2],box2[1]],[box2[2],box2[3]],[box2[0],box2[3]]]

        # mapping
        pts1 = np.array(box1_pts)
        # pts2 = np.array(box2_pts)[self._mapping]
        pts2 = np.array(box2_pts)

        d = 0
        nd = 0
        if whichImage is None or whichImage==1:
            d_max = -10
            d_min = 10
            lines_from_v1_to_v2 = cv2.computeCorrespondEpilines(pts1, 1, self._F).reshape((4,3))
            lines_v1_to_v2_order = lines_from_v1_to_v2[self._mapping]
            a0,b0,c0,a1,b1,c1,a2,b2,c2,a3,b3,c3 = lines_v1_to_v2_order.reshape(-1)
            # a0 b0  0  0     xmin     -c0
            #  0 b1 a1  0  *  ymin  =  -c1
            #  0  0 a2 b2     xmax     -c2
            #  1  0 -1  0     ymax     -100
            mat_a = np.array([[a0,b0,0,0],\
                             [0,b1,a1,0],\
                             [0,0,a2,b2],\
                             [1,0,-1,0]])
            mat_b = np.array([[-c0],[-c1],[-c2],[-100]])
            xmin,ymin,xmax,ymax = np.linalg.solve(mat_a, mat_b)
            pts1_to_pts2 = np.array([[xmin,ymin],[xmax,ymin],\
                                     [xmax,ymax],[xmin,ymax]])

            for p0,p1 in zip(pts2,pts1_to_pts2):
                x = [p0[0]-self._v2_cam_pt[0],p0[1]-self._v2_cam_pt[1]]
                y = [p1[0]-self._v2_cam_pt[0],p1[1]-self._v2_cam_pt[1]]
                dist = (np.dot(x,y))/np.linalg.norm(y)/np.linalg.norm(x)
                if(dist>d_max):
                    d_max = dist
                if(dist<d_min):
                    d_min = dist
                d += dist
            d = d - d_max -d_min
            nd += 2
            # d = d - d_min
            # nd += 3

        if whichImage is None or whichImage==2:
            d_max = -10
            d_min = 10
            lines_from_v2_to_v1 = cv2.computeCorrespondEpilines(pts2, 2, self._F).reshape((4,3))
            lines_v2_to_v1_order = lines_from_v2_to_v1[self._mapping]
            a0,b0,c0,a1,b1,c1,a2,b2,c2,a3,b3,c3 = lines_v2_to_v1_order.reshape(-1)
            # a0 b0  0  0     xmin     -c0
            #  0 b1 a1  0  *  ymin  =  -c1
            #  0  0 a2 b2     xmax     -c2
            #  1  0 -1  0     ymax     -100
            mat_a = np.array([[a0,b0,0,0],\
                             [0,b1,a1,0],\
                             [0,0,a2,b2],\
                             [1,0,-1,0]])
            mat_b = np.array([[-c0],[-c1],[-c2],[-100]])
            xmin,ymin,xmax,ymax = np.linalg.solve(mat_a, mat_b)
            pts2_to_pts1 = np.array([[xmin,ymin],[xmax,ymin],\
                                     [xmax,ymax],[xmin,ymax]])

            for p0,p1 in zip(pts1,pts2_to_pts1):
                x = [p0[0]-self._v1_cam_pt[0],p0[1]-self._v1_cam_pt[1]]
                y = [p1[0]-self._v1_cam_pt[0],p1[1]-self._v1_cam_pt[1]]
                dist = (np.dot(x,y))/np.linalg.norm(y)/np.linalg.norm(x)
                if(dist>d_max):
                    d_max = dist
                if(dist<d_min):
                    d_min = dist
                d += dist
            d = d - d_max -d_min
            nd += 2

        return d/nd

    def getF(self):
        return self._F

    def computeCorrespondEpilines(self,points,whichImages):
        return cv2.computeCorrespondEpilines(np.array(points),whichImages,self._F)

    def camPoint(self, whichImage):
        if(whichImage==1):
            return self._v1_cam_pt_int
        else:
            return self._v2_cam_pt_int

    #TODO: speed up
    def boxesMatchMatrix(self,boxes1,boxes2,whichImage=None):
        r = np.zeros([len(boxes1),len(boxes2)])
        for i,bb1 in enumerate(boxes1):
            for j,bb2 in enumerate(boxes2):
                r[i,j] = self.boxMatchDegree(bb1,bb2,whichImage)
        return r

    def findMatchPairs(self,boxes1,boxes2,thresh=0.9): #cos(25)~=0.94
        matched_mat = self.boxesMatchMatrix(boxes1,boxes2)
        matched_pairs = linear_assignment(-matched_mat)
        #del low thresh
        matches = []
        for m in matched_pairs:
            if(matched_mat[m[0],m[1]] < thresh):
                matches.append(False)
            else:
                matches.append(True)
        return matched_pairs[matches]

if __name__=='__main__':

    # param
    F = np.array([[-2.46197462e-08,1.17730694e-06,-6.10152251e-04],\
                  [ 1.14557456e-06,-1.30211655e-07, -8.50494174e-04],\
                  [-6.34708223e-04, -9.16125703e-04,  1.00000000e+00]])
    mapping = (1,0,3,2)
    mapping_list = list(mapping)

    # boxes
    box1 = [816,268,985,418]
    # box2 = [697,345,797,472]
    box2 = [697+300,345+300,797+300,472+300]
    box1 = [int(x) for x in box1]
    box2 = [int(x) for x in box2]

    # imgs
    img1 = cv2.imread('/home/lee/PycharmProject/noSensePay_two_cam/video_2019-02-26_18-9-53_v0_frame_10247.jpg')
    img2 = cv2.imread('/home/lee/PycharmProject/noSensePay_two_cam/video_2019-02-26_18-9-53_v1_frame_10247.jpg')
    img1 = cv2.resize(img1,(1280,720))
    img2 = cv2.resize(img2,(1280,720))

    # draw boxes
    cv2.rectangle(img1, (box1[0],box1[1]),(box1[2],box1[3]),(0,0,255),3)
    cv2.rectangle(img2, (box2[0],box2[1]),(box2[2],box2[3]),(0,0,255),3)
    pts1 = np.array([[box1[0], box1[1]], [box1[2], box1[1]], [box1[2], box1[3]], [box1[0], box1[3]]])
    pts2 = np.array([[box2[0], box2[1]], [box2[2], box2[1]], [box2[2], box2[3]], [box2[0], box2[3]]])
    color = [(255,0,0),(0,255,0),(0,0,255),(255,255,255)]
    for i,p in enumerate(pts2):
        cv2.circle(img2, (p[0],p[1]),3,color[i],3)

    # draw line from v2 in v1
    lines_from_v2_to_v1 = cv2.computeCorrespondEpilines(pts2, 2, F).reshape((4, 3))
    color = [(255,0,0),(0,255,0),(0,0,255),(255,255,255)]
    for i,a_b_c in enumerate(lines_from_v2_to_v1):
        a,b,c = a_b_c
        p0 = (0,int(-c/b))
        p1 = (10000,int((-c-a*10000)/b))
        print(p0,p1)
        cv2.line(img1, p0,p1,color[i],3,3)

    # 100 weight box from v2 in v1
    lines_v2_to_v1_order = lines_from_v2_to_v1[mapping_list]
    a0, b0, c0, a1, b1, c1, a2, b2, c2, a3, b3, c3 = lines_v2_to_v1_order.reshape(-1)
    # a0 b0  0  0     xmin     -c0
    #  0 b1 a1  0  *  ymin  =  -c1
    #  0  0 a2 b2     xmax     -c2
    #  1  0 -1  0     ymax     -100
    mat_a = np.array([[a0, b0, 0, 0], \
                      [0, b1, a1, 0], \
                      [0, 0, a2, b2], \
                      [1, 0, -1, 0]])
    mat_b = np.array([[-c0], [-c1], [-c2], [-100]])
    xmin, ymin, xmax, ymax = np.linalg.solve(mat_a, mat_b)

    cv2.rectangle(img1, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)


    # box match
    bm = boxMatch(mapping,F=F)
    cv2.circle(img1, bm.camPoint(1),3,(255,255,255),3)
    cv2.circle(img2, bm.camPoint(2),3,(255,255,255),3)
    print(bm.boxMatchDegree(box1,box2,1))
    print(bm.boxMatchDegree(box1,box2,2))
    print(bm.boxMatchDegree(box1,box2))
    print(bm.findMatchPairs([box1],[box2],0.98))

    cv2.imshow('img1',cv2.resize(img1,(0,0),fx=1,fy=1))
    cv2.imshow('img2',cv2.resize(img2,(0,0),fx=1,fy=1))
    cv2.waitKey(0)





