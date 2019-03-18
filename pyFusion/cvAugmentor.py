import cv2
import numpy as np
import random
import math

#out a list of augmented images, even if it output only one images
#input image of CV_8UC1,CV_8UC3,CV_8UC4
class cvAugmentor:
    #Rotate
    def __imRotate_one(img,angle):
        is_1c = len(img.shape) == 2
        diagonal = int(math.sqrt(img.shape[0]*img.shape[0]+img.shape[1]*img.shape[1])) + 1
        bg_shape = list(img.shape)
        bg_shape[0] = diagonal
        bg_shape[1] = diagonal
        bg = np.zeros(bg_shape,img.dtype)
        bg_roi_x1 = int((diagonal-img.shape[1])/2)
        bg_roi_y1 = int((diagonal-img.shape[0])/2)
        bg_roi_x2 = bg_roi_x1+img.shape[1]
        bg_roi_y2 = bg_roi_y1+img.shape[0]
        if(is_1c):
            bg[bg_roi_y1:bg_roi_y2,bg_roi_x1:bg_roi_x2] = img
        else:
            bg[bg_roi_y1:bg_roi_y2,bg_roi_x1:bg_roi_x2,:] = img
        r = cv2.getRotationMatrix2D((diagonal/2,diagonal/2),angle,1.0)
        rot_output = cv2.warpAffine(bg,r,(bg.shape[1],bg.shape[0]))
       
        c = math.fabs(math.cos(angle/180*math.pi))
        s = math.fabs(math.sin(angle/180*math.pi))
        true_w = int(img.shape[1] * c + img.shape[0]*s)
        true_h = int(img.shape[0] * c + img.shape[1]*s)
        rot_output_roi_x1 = int((diagonal-true_w)/2)
        rot_output_roi_y1 = int((diagonal-true_h)/2)
        rot_output_roi_x2 = rot_output_roi_x1+true_w
        rot_output_roi_y2 = rot_output_roi_y1+true_h

        if(is_1c):
            return rot_output[rot_output_roi_y1:rot_output_roi_y2,rot_output_roi_x1:rot_output_roi_x2]
        else:
            return rot_output[rot_output_roi_y1:rot_output_roi_y2,rot_output_roi_x1:rot_output_roi_x2,:]

    def imRotate(img,angles):
        output = []
        for angle in angles:
            output.append(cvAugmentor.__imRotate_one(img,angle))
        return output
           
    #Flip
    def imFlipUD(img):
        return [cv2.flip(img,0)]
    def imFlipLR(img):
        return [cv2.flip(img,1)]
    def imFlip(img):
        return [cv2.flip(img,random.randint(0,1))]
    
    #Crop
    def __imCrop_one(img,rate):
        imgshape = img.shape
        is_1c = len(img.shape) == 2
        if(rate>1):
            rate = 1.0
        if(rate<0):
            rate = 0.0
        
        maxShape = [imgshape[0] * (1.0-rate)-1.0,imgshape[1] * (1.0-rate)-1.0]
        if(maxShape[0]<=0 or maxShape[1]<=0):
            return img.copy()
        x1 = random.randint(0,int(maxShape[1]))
        y1 = random.randint(0,int(maxShape[0]))
        x2 = x1+ int(imgshape[1]*rate)
        y2 = y1+ int(imgshape[0]*rate)
        if(is_1c):
            return img[y1:y2,x1:x2].copy()
        else:
            return img[y1:y2,x1:x2,:].copy()

    def imCrop(img,rates):
        output = []
        for rate in rates:
            output.append(cvAugmentor.__imCrop_one(img,rate))
        return output

    #Affine
    def __imAffine_one(img,is_X,rate):
        img_cols = img.shape[1]
        img_rows = img.shape[0]
        src = np.array([[0.0,0.0],[img_cols,0.0],[0.0,img_rows]],np.float32)
        if(is_X):
            if(rate<0):
                offset_x = int(-rate * img_cols)
                dist = np.array([[src[0][0]+offset_x,src[0][1]],\
                [src[1][0]+offset_x,src[1][1]],[src[2][0],src[2][1]]],\
                np.float32)
                r = cv2.getAffineTransform(src,dist)
                output = cv2.warpAffine(img,r,(img_cols+offset_x,img_rows))
            else:
                offset_x = int(rate * img_cols)
                dist = np.array([[src[0][0],src[0][1]],\
                [src[1][0],src[1][1]],[src[2][0]+offset_x,src[2][1]]],\
                np.float32)
                r = cv2.getAffineTransform(src,dist)
                output = cv2.warpAffine(img,r,(img_cols+offset_x,img_rows))
        else:
            if(rate<0):
                offset_y = int(-rate * img_rows)
                dist = np.array([[src[0][0],src[0][1]+offset_y],\
                [src[1][0],src[1][1]],[src[2][0],src[2][1]+offset_y]],\
                np.float32)
                r = cv2.getAffineTransform(src,dist)
                output = cv2.warpAffine(img,r,(img_cols,img_rows+offset_y))
            else:
                offset_y = int(rate * img_rows)
                dist = np.array([[src[0][0],src[0][1]],\
                [src[1][0],src[1][1]+offset_y],[src[2][0],src[2][1]]],\
                np.float32)
                r = cv2.getAffineTransform(src,dist)
                output = cv2.warpAffine(img,r,(img_cols,img_rows+offset_y))
        return output

    def imAffine(img,rates):
        output = []
        for rate in rates:
            output.append(cvAugmentor.__imAffine_one(img,random.randint(0,1),rate))
        return output
      
    def imAffineX(img,rates):
        output = []
        for rate in rates:
            output.append(cvAugmentor.__imAffine_one(img,True,rate))
        return output
        
    def imAffineY(img,rates):
        output = []
        for rate in rates:
            output.append(cvAugmentor.__imAffine_one(img,False,rate))
        return output

    #Noise
    def imNoise(img,rates):
        output = []
        is_4c = len(img.shape) == 3
        if(is_4c):
            is_4c = img.shape[2] == 4
        if(is_4c is not True):
            for rate in rates:
                noise = np.random.randint(-255,255,img.size)
                f_noise = noise.reshape(img.shape).astype(np.float32)
                added_noise = (f_noise*rate+img)
                added_noise[added_noise>255]=255
                added_noise[added_noise<0]=0
                output.append(added_noise.astype(img.dtype))
        else:
            for rate in rates:
                noise = np.random.randint(-255,255,(img.shape[0],img.shape[1],3))
                f_noise = noise.reshape((img.shape[0],img.shape[1],3)).astype(np.float32)
                added_noise = img.copy()
                added_noise[:,:,0:3] = (f_noise*rate+img[:,:,0:3]).astype(np.uint8)
                added_noise[added_noise>255]=255
                added_noise[added_noise<0]=0
                output.append(added_noise)
        return output

    #Hue
    def imHue(img,anlges):
        is_4c = img.shape[2] == 4
        if(is_4c):
            bgr = img[:,:,0:3]
            one_output = np.zeros(img.shape,img.dtype)
            one_output[:,:,3] = img[:,:,3]
        else:
            bgr = img
            one_output = np.zeros(img.shape,img.dtype)
        
        hls = cv2.cvtColor(bgr,cv2.COLOR_BGR2HLS)
        
        output = []
        for anlge in anlges:
            temp_hls = hls.copy()
            f_temp_h = temp_hls[:,:,0]*2.0+anlge
            f_temp_h[f_temp_h>360] = f_temp_h[f_temp_h>360]-360
            f_temp_h[f_temp_h<0] = f_temp_h[f_temp_h<0]+360
            f_temp_h /=2.0
            temp_hls[:,:,0] = f_temp_h
            one_output[:,:,0:3] = cv2.cvtColor(temp_hls,cv2.COLOR_HLS2BGR)
            output.append(one_output)
        return output

    #Lightness
    def imLightness(img,rates):
        is_4c = img.shape[2] == 4
        if(is_4c):
            bgr = img[:,:,0:3]
            one_output = np.zeros(img.shape,img.dtype)
            one_output[:,:,3] = img[:,:,3]
        else:
            bgr = img
            one_output = np.zeros(img.shape,img.dtype)
        
        hls = cv2.cvtColor(bgr,cv2.COLOR_BGR2HLS)
        
        output = []
        for rate in rates:
            temp_hls = hls.copy()
            f_temp_l = temp_hls[:,:,1]/255.0+rate
            f_temp_l[f_temp_l>1] = 1.0
            f_temp_l[f_temp_l<0] = 0.0
            temp_hls[:,:,1] = f_temp_l*255.0
            one_output[:,:,0:3] = cv2.cvtColor(temp_hls,cv2.COLOR_HLS2BGR)
            output.append(one_output)
        return output

    #Saturation
    def imSaturation(img,rates):
        is_4c = img.shape[2] == 4
        if(is_4c):
            bgr = img[:,:,0:3]
            one_output = np.zeros(img.shape,img.dtype)
            one_output[:,:,3] = img[:,:,3]
        else:
            bgr = img
            one_output = np.zeros(img.shape,img.dtype)
        
        hls = cv2.cvtColor(bgr,cv2.COLOR_BGR2HLS)
        
        output = []
        for rate in rates:
            temp_hls = hls.copy()
            f_temp_s = temp_hls[:,:,2]/255.0+rate
            f_temp_s[f_temp_s>1] = 1.0
            f_temp_s[f_temp_s<0] = 0.0
            temp_hls[:,:,2] = f_temp_s*255.0
            one_output[:,:,0:3] = cv2.cvtColor(temp_hls,cv2.COLOR_HLS2BGR)
            output.append(one_output.copy())
        return output

    #Perspective
    def __imPerspective_one(img,type,rate):
        img_cols = img.shape[1]
        img_rows = img.shape[0]
        src = np.array([[0.0,0.0],[img_cols,0.0],[img_cols,img_rows],[0.0,img_rows]],np.float32)
        dist = src.copy()
        
        offset_x = rate*img_cols
        offset_y = rate*img_rows

        if(type == 'U'):
            dist[0,0] += offset_x
            dist[1,0] -= offset_x
        elif(type == 'UR'):
            dist[1,0] -= offset_x
            dist[1,1] += offset_y
        elif(type == 'R'):
            dist[1,1] += offset_y
            dist[2,1] -= offset_y
        elif(type == 'DR'):
            dist[2,0] -= offset_x
            dist[2,1] -= offset_y
        elif(type == 'D'):
            dist[2,0] -= offset_x
            dist[3,0] += offset_x
        elif(type == 'DL'):
            dist[3,0] += offset_x 
            dist[3,1] -= offset_y
        elif(type == 'L'):
            dist[0,1] += offset_y
            dist[3,1] -= offset_y
        elif(type == 'UL'):
            dist[0,0] += offset_x 
            dist[0,1] += offset_y
        r = cv2.getPerspectiveTransform(src,dist)
        return cv2.warpPerspective(img,r,(img_cols, img_rows))

    def imPerspective(img,rates):
        type_erum = ('U','UR','R','DR','D','DL','L','UL')
        output = []
        type_np = np.random.randint(0,7,len(rates),np.int)
        #print(type_erum[type_np[0]])
        for i in range(len(rates)):
            output.append(cvAugmentor.__imPerspective_one(img,type_erum[type_np[i]],rates[i]))
        return output

    def imPerspectiveU(img,rates):
        output = []
        for rate in rates:
            output.append(cvAugmentor.__imPerspective_one(img,'U',rate))
        return output

    def imPerspectiveUR(img,rates):
        output = []
        for rate in rates:
            output.append(cvAugmentor.__imPerspective_one(img,'UR',rate))
        return output

    def imPerspectiveR(img,rates):
        output = []
        for rate in rates:
            output.append(cvAugmentor.__imPerspective_one(img,'R',rate))
        return output

    def imPerspectiveDR(img,rates):
        output = []
        for rate in rates:
            output.append(cvAugmentor.__imPerspective_one(img,'DR',rate))
        return output

    def imPerspectiveD(img,rates):
        output = []
        for rate in rates:
            output.append(cvAugmentor.__imPerspective_one(img,'D',rate))
        return output

    def imPerspectiveDL(img,rates):
        output = []
        for rate in rates:
            output.append(cvAugmentor.__imPerspective_one(img,'DL',rate))
        return output

    def imPerspectiveL(img,rates):
        output = []
        for rate in rates:
            output.append(cvAugmentor.__imPerspective_one(img,'L',rate))
        return output

    def imPerspectiveUL(img,rates):
        output = []
        for rate in rates:
            output.append(cvAugmentor.__imPerspective_one(img,'UL',rate))
        return output

    def imPerstep(img):
        return [img.copy()]

    #Distort
    #def __imDistort_one(img,rate):
    #    pass     

    #def imDistort(img,rates):
    #    pass    

    #def imPyramid(img,down_number,up_number):
    #    if(down_number<0):
    #        down_number = 0
    #    if(up_number<0):
    #        up_number = 0
    #    n = down_number + up_number + 1
    #    output = [[]]*n
    #    output[down_number] = img.copy()
    #    for i in range(down_number):
    #        true_i = down_number-1-i
    #        output[true_i] = cv2.pyrDown(output[true_i+1])
    #    for i in range(down_number+1,n):
    #        output[i] = cv2.pyrUp(output[i-1])
    #    return output