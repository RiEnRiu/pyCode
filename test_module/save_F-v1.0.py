import argparse
import cv2
import numpy as np
import os
import json
import sys


def transf(x):
    tt = np.zeros((24, 2), np.float32)
    for i in range(len(x)):
        if (i + 1) % 6 == 0 and i > 0:
            tt[i - 5:i + 1] = x[i - 5:i + 1][::-1]
    return tt


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    index = 0
    col = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        if (index < 4):
            color = col[index]
        else:
            color = tuple(np.random.randint(0, 255, 3).tolist())
        index += 1
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


parser = argparse.ArgumentParser(description='test F')
parser.add_argument('--cap1', default=0,
                    type=int, action='store', help='number means which cap will be opened')
parser.add_argument('--cap2', default=1,
                    type=int, action='store', help='number means which cap will be opened')
parser.add_argument('--save', default='cap_F/',
                    help='Directory for saving parameter')
parser.add_argument('--w', default=1280,
                    type=int, help='widht of cap')
parser.add_argument('--h', default=720,
                    type=int, help='height of cap')
args = parser.parse_args()

if not os.path.exists(args.save):
    os.mkdir(args.save)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
ptse1 = []  # 2d points in image plane.
ptse2 = []

cap0 = cv2.VideoCapture(args.cap1)
cap1 = cv2.VideoCapture(args.cap2)

w = args.w
h = args.h
videoFourcc = cv2.VideoWriter_fourcc(*'MJPG')
cap0.set(cv2.CAP_PROP_FOURCC, videoFourcc)
cap0.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
cap1.set(cv2.CAP_PROP_FOURCC, videoFourcc)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

count = 1
F = 1
while True:
    ret1, img1 = cap0.read()
    ret2, img2 = cap1.read()
    print('origin:', ret1, ret2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret11, corners11 = cv2.findChessboardCorners(gray1, (6, 4), None)
    ret22, corners22 = cv2.findChessboardCorners(gray2, (6, 4), None)
    print('detec corner:', ret11, ret22)
    # cv2.imshow('img1', img1)
    # cv2.imshow('img2', img2)
    # If found, add object points, image points (after refining them)
    if ret11 and ret22:
        objpoints.append(objp)
        corners1 = cv2.cornerSubPix(gray1, corners11, (11, 11), (-1, -1), criteria)
        corners2 = cv2.cornerSubPix(gray2, corners22, (11, 11), (-1, -1), criteria)

        corners1 = corners1.reshape(24, 2)
        corners2 = corners2.reshape(24, 2)
        if ((corners1[0][0] - corners1[18][0]) * (corners2[0][0] - corners2[18][0])) <= 0:
            corners2 = transf(corners2)
        else:
            corners2 = corners2[::-1]
        # corners2 = transf(corners2)
        try:
            img1 = cv2.drawChessboardCorners(img1, (6, 4), corners1, ret11)
            img2 = cv2.drawChessboardCorners(img2, (6, 4), corners2, ret22)
        finally:
            print('done')
        cv2.imshow('image1', img1)
        cv2.imshow('image2', img2)
        # cv2.waitKey(0)
        if cv2.waitKey(10) & 0xFF == ord('s'):
            ptse1.append(corners1)
            ptse2.append(corners2)
            pts1 = np.int32(ptse1)
            pts2 = np.int32(ptse2)
            pts1 = pts1.reshape(-1, 2)
            pts2 = pts2.reshape(-1, 2)
            # get F
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
            print('yes')
            # only inlier points
            # pts1 = pts1[mask.ravel() == 1]
            # pts2 = pts2[mask.ravel() == 1]

            # Find epilines corresponding to points in left image (first image) and
            # drawing its lines on right image
            lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
            lines1 = lines1.reshape(-1, 3)
            # img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
            img5, img6 = drawlines(gray1, gray2, lines1, pts1, pts2)
            lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
            lines2 = lines2.reshape(-1, 3)
            # img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
            img3, img4 = drawlines(gray2, gray1, lines2, pts2, pts1)
            cv2.imshow('1', img5)
            cv2.imshow('2', img3)
            # cv2.waitKey(0)
        if cv2.waitKey(10) & 0xFF == ord('w'):
            cv2.imwrite('v0_{}.jpg'.format(count), img5)
            cv2.imwrite('v1_{}.jpg'.format(count), img3)
    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

v1_cam_pt = np.matmul(np.mat([[F[0, 0], F[1, 0]], [F[0, 1], F[1, 1]]]).I, [[-F[2, 0]], [-F[2, 1]]])
v1_cam_pt = (v1_cam_pt[0, 0], v1_cam_pt[1, 0])
FT = np.mat(F).T
v0_cam_pt = np.matmul(np.mat([[FT[0, 0], FT[1, 0]], [FT[0, 1], FT[1, 1]]]).I, [[-FT[2, 0]], [-FT[2, 1]]])
v0_cam_pt = (v0_cam_pt[0, 0], v0_cam_pt[1, 0])
a = dict(capid=[args.cap1, args.cap2],
         F=F.tolist(),
         pts1=pts1.tolist(),
         pts2=pts2.tolist(),
         size=(w, h),
         expoint=(v0_cam_pt, v1_cam_pt)
         )
json_str = json.dumps(a, indent=4)
with open('{}/data.json'.format(args.save), 'w') as json_file:
    json_file.write(json_str)
np.save('F.npy', F)
cv2.destroyAllWindows()
