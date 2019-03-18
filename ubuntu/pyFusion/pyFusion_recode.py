import cv2
import numpy as np
import argparse
import os
import datetime
import random
import math
from tqdm import tqdm

import datetime

import sys
sys.path.append('..\Common')
from pyBoost import *
from vocBoost import *



class FUSION_TYPE:
    PASTE = 0
    CV2_NORMAL_CLONE = 1
    CV2_MIXED_CLONE = 2
    SMOOTH_BOUNDARY = 3
#    JACOBI_POISSON = 4


#def _solvePoissonJacobi(self, mix_image, X, Y, mask, _itr, _th):
 
#    lap_x = cv2.filter2D(X, cv2.CV_32F, np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]], dtype=np.float32))
#    lap_y = cv2.filter2D(Y, cv2.CV_32F, np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32))
 
#    lap = lap_x + lap_y
 
#    diff0 = 10000000.0
#    max_diff = 10000000.0
 
#    mix_image_32FC3 = mix_image.astype(np.float32)
 
#    clone_mix_image_32FC3 = [mix_image_32FC3.copy(), mix_image_32FC3.copy()]
#    i_clone = 0
#    i_pre_clone = 1
 
#    mask01_c1 = (mask / 255.0).astype(np.float32)
#    mask01 = cv2.cvtColor(mask01_c1, cv2.COLOR_GRAY2BGR)
#    invmask01 = (1.0 - mask01).astype(np.float32)
 
#    while (_itr > 0):
#        _itr -= 1
 
#        clone_mix_image_32FC3[i_clone] = cv2.filter2D(clone_mix_image_32FC3[i_pre_clone], cv2.CV_32F, self.__K)
 
#        clone_mix_image_32FC3[i_clone] = (((clone_mix_image_32FC3[i_clone] + lap) * 0.25) * mask01\
#        + clone_mix_image_32FC3[i_pre_clone] * invmask01).astype(np.float32)

#        out_255 = clone_mix_image_32FC3[i_clone]>255.0
#        out_0 = clone_mix_image_32FC3[i_clone]<0
#        clone_mix_image_32FC3[i_clone][out_255] = 255.0
#        clone_mix_image_32FC3[i_clone][out_0] = 0.0
        
#        max_diff = (np.abs(clone_mix_image_32FC3[i_clone] - clone_mix_image_32FC3[i_pre_clone])).max()
 
#        if (math.fabs(diff0 - max_diff) / diff0 < _th):
#            break
#        diff0 = max_diff
 
#        if (i_clone == 0):
#            i_clone = 1
#            i_pre_clone = 0
#        else:
#            i_clone = 0
#            i_pre_clone = 1
 
 
#    result = clone_mix_image_32FC3[i_clone]
 
 
#    return result

def imFusion(src,dst,mask,p,flag,param=None):
    cv_xmin = max(int(p[0]-src.shape[1]/2),1)
    cv_xmax = int(cv_xmin + src.shape[1])
    cv_ymin = max(int(p[1]-src.shape[0]/2),1)
    cv_ymax = int(cv_ymin + src.shape[0])

    #print(src.shape,mask.shape,(cv_ymax-cv_ymin,cv_xmax-cv_xmin),dst.shape)
    #print(cv_xmin,cv_ymin,cv_xmax,cv_ymax)

    #print(p)
    #print(src.shape,dst.shape,mask.shape,(cv_ymin,cv_ymax,cv_xmin,cv_xmax))

    #if(cv_xmin>=dst.shape[1] or cv_xmax>=dst.shape[1]\
    #    or cv_ymin>=dst.shape[0] or cv_ymax>=dst.shape[0]):
    #    print('fuion ,OUT!!!')
    #    input()

    #if(cv_xmin>=dst.shape[1] or cv_xmax>=dst.shape[1]\
    #    or cv_ymin>=dst.shape[0] or cv_ymax>=dst.shape[0]):
    #    print('fuion, OUT!!!')
    #    input()


    #if have alpha channel
    if(src.shape[2] == 4):
        obj = src[:,:,0:3]
        alpha = src[:,:,3]/255.0
        bg = dst.copy()
        bg_roi = bg[cv_ymin:cv_ymax,cv_xmin:cv_xmax,:]
        #print((bg_roi.shape,src.shape,mask.shape))

        alpha_c1 = alpha.reshape([alpha.shape[0],alpha.shape[1],1])
        alpha_c3 = np.concatenate((alpha_c1,)*3,2)

        mask_c1 = mask.reshape([mask.shape[0],mask.shape[1],1])
        mask_c3 = np.concatenate((mask_c1,)*3,2)

        #print(np.concatenate(np.concatenate(alpha_c1,alpha_c1,2),alpha_c1,2).shape)
        #input()
        if(flag==FUSION_TYPE.PASTE):
            mask_not_zero = mask_c3!=0
            bg_roi[mask_not_zero] = (bg_roi*(1-alpha_c3) + alpha_c3 * obj).astype(np.uint8)[mask_not_zero]
            return bg
        elif(flag==FUSION_TYPE.CV2_NORMAL_CLONE):
            mask_not_zero = mask_c3!=0
            bg_roi[mask_not_zero] = (bg_roi*(1-alpha_c3) + alpha_c3 * obj).astype(np.uint8)[mask_not_zero]
            return cv2.seamlessClone(obj,bg,mask.copy(),p,cv2.NORMAL_CLONE)
        elif(flag==FUSION_TYPE.CV2_MIXED_CLONE):
            mask_not_zero = mask_c3!=0
            bg_roi[mask_not_zero] = (bg_roi*(1-alpha_c3) + alpha_c3 * obj).astype(np.uint8)[mask_not_zero]
            return cv2.seamlessClone(obj,bg,mask.copy(),p,cv2.MIXED_CLONE)
        elif(flag==FUSION_TYPE.SMOOTH_BOUNDARY):

            #cv2.imshow('bg',bg)
            #cv2.imshow('bg_roi',bg_roi)
            #cv2.imshow('obj',obj)
            #cv2.imshow('mask',mask)
            #cv2.waitKey(0)
            
            mask_smooth = cv2.blur(mask_c3,(param['ksize'],param['ksize']))
            alpha_c3 = alpha_c3 * mask_smooth / 255.0
            mask_smooth_not_zero = mask_smooth!=0
            bg_roi[mask_smooth_not_zero] = (bg_roi*(1-alpha_c3) + alpha_c3 * obj).astype(np.uint8)[mask_smooth_not_zero]
            return bg
        elif(flag==FUSION_TYPE.JACOBI_POISSON):
            pass
    else:#have no alpha channel
        obj = src
        bg = dst.copy()
        bg_roi = bg[cv_ymin:cv_ymax,cv_xmin:cv_xmax,:]
        if(flag==FUSION_TYPE.PASTE):
            mask_not_zero = mask!=0
            bg_roi[mask_not_zero] = obj[mask_not_zero]
            return bg
        elif(flag==FUSION_TYPE.CV2_NORMAL_CLONE):
            return cv2.seamlessClone(obj,bg,mask.copy(),p,cv2.NORMAL_CLONE)
        elif(flag==FUSION_TYPE.CV2_MIXED_CLONE):
            return cv2.seamlessClone(obj,bg,mask.copy(),p,cv2.MIXED_CLONE)
        elif(flag==FUSION_TYPE.SMOOTH_BOUNDARY):
            mask_smooth = cv2.blur(mask,(param['ksize'],param['ksize']))
            alpha = mask_smooth / 255.0
            mask_smooth_not_zero = mask_smooth!=0
            bg_roi[mask_smooth_not_zero] = (bg_roi*(1-alpha) + alpha * obj).astype(np.uint8)[mask_smooth_not_zero]
            return bg
        elif(flag==FUSION_TYPE.JACOBI_POISSON):
            pass
    return 

def scanImage(bgpath,objpath,maskpath=''):
    #bg
    if(os.path.isfile(bgpath)):
        bg_list = scanner.list(bgpath)
    else:
        bg_list = scanner.file(bgpath,'.jpg.png.jpeg',True)

    #obj mask label
    if(os.path.isfile(objpath) and maskpath==''):
        list_from_text = scanner.list(objpath)
        png_list = [x for x in list_from_text if x[0][-4:]=='.png']
        obj_list = [x[0] for x in png_list]
        if(len(png_list[0])==2):
            label_list = [x[1] for x in png_list]
        else:
            label_list = [os.path.split(os.path.split(x)[0])[1] for x in obj_list]
        mask_list = ['']*len(obj_list)
    elif(os.path.isdir(objpath) and maskpath==''):
        obj_list = scanner.file_r(objpath,'.png',True)
        label_list = [os.path.split(os.path.split(x)[0])[1] for x in obj_list]
        mask_list = ['']*len(obj_list)
    elif(os.path.isfile(objpath) and maskpath!=''):
        list_from_text = scanner.list(objpath)
        obj_list = [x[0] for x in list_from_text]
        if(len(list_from_text[0])==2):
            label_list = [x[1] for x in list_from_text]
        else:
            label_list = [os.path.split(os.path.split(x)[0])[1] for x in obj_list]
        mask_list = ['']*len(obj_list)
        for i in range(len(obj_list)-1,-1,-1):
            name = os.path.splitext(os.path.split(obj_list[i])[1])[0]
            label = label_list[i]
            mask = os.path.join(maskpath,label,name+'.png')
            if(os.path.isfile(mask)):
                mask_list[i] = mask
            else:
                obj_list.pop(i)
                label_list.pop(i)
                mask_list.pop(i)
    elif(os.path.isdir(objpath) and maskpath!=''):
        obj_list = scanner.file_r(objpath,'.jpg.png.jpeg',True)
        label_list = [os.path.split(os.path.split(x)[0])[1] for x in obj_list]
        mask_list = ['']*len(obj_list)
        for i in range(len(obj_list)-1,-1,-1):
            name = os.path.splitext(os.path.split(obj_list[i])[1])[0]
            label = label_list[i]
            mask = os.path.join(maskpath,label,name+'.png')
            #print(mask)
            #input()
            if(os.path.isfile(mask)):
                mask_list[i] = mask
            else:
                obj_list.pop(i)
                label_list.pop(i)
                mask_list.pop(i)


    print('scan backgroud = {}, object = {}, mask = {}'.format(len(bg_list),len(obj_list),len(mask_list)))
    return bg_list,list(zip(obj_list,mask_list,label_list))


#systematic sampling
def randSolusion(bg_list,obj_mask_label,num,min,max):
    sample_dict = {}
    for i,x in enumerate(obj_mask_label):
        p_get = sample_dict.get(x[2])
        if(p_get is None):
            sample_dict[x[2]] = [i]
        else:
            p_get.append(i)
    
    label_list = list(sample_dict.keys())

    obj_index = [[]]*num
    for i in range(num):
        len_obj = np.random.randint(min,max+1)
        this_fusion_labels = np.random.choice(label_list,len_obj)
        for label in this_fusion_labels:
            obj_index[i] = obj_index[i]+ list(np.random.choice(sample_dict[label],1))

    obj_rand_list = [[obj_mask_label[x] for x in index] for index in obj_index]    
    bg_rand_list = [str(x) for x in (np.random.choice(bg_list,num))]

    return [[x,y] for x,y in zip(bg_rand_list,obj_rand_list)]
    
def setIO(solution):
    img_list = []
    for x in solution:
        img_list.append(x[0])#x[0] == bg
        for y in x[1]:#y == [obj, mask, label]
            if(y[1]!=''):
                img_list.append(y[0])#y[0] == obj 
                img_list.append((y[1],cv2.IMREAD_GRAYSCALE))#y[1] == mask
            else:
                img_list.append((y[0],cv2.IMREAD_UNCHANGED))
    p_imgIO = imgIO(img_list,True,20)
    p_vocXmlIO = vocXmlIO([],True,20)
    return p_imgIO, p_vocXmlIO

def readImg(one_solu,p_imgIO):
    bg = p_imgIO.read()
    objs = []
    masks = []
    labels = []
    for obj_mask_label in one_solu[1]:
        obj = p_imgIO.read()
        if(obj_mask_label[1]==''):
            mask = obj[:,:,3]
        else:
            mask = p_imgIO.read()
        objs.append(obj)
        masks.append(mask)
        labels.append(obj_mask_label[2])
    return bg,objs,masks,labels

def getObjSize(obj_size_path):
    if(os.path.isfile(obj_size_path)):
        return {x[0]:float(x[1])*float([2]) for x in scanner.list(obj_size_path)}
    else:
        return {}

def resizeObjs(bg,objs,masks,labels,sizeDict,rate = [0.30,0.35]):
    bgSize = bg.shape[0] * bg.shape[1]
    objsImgSize = [x.size - x[x<=5].size/2.0 for x in masks]
    allObjsImgSize = sum(objsImgSize)
    if(len(sizeDict)!=0):
        objsRealSize = [sizeDict[x]*1000 for x in labels]
    else:
        objsRealSize = None

    allRate = np.random.randint(int(rate[0]*100),int(rate[1]*100+1))/100.0
    objsRate = []
    if(objsRealSize is not None):
        t1_rate = [x/y for x,y in zip(objsRealSize,objsImgSize)]
        #print(t1_rate)
        #print(objsImgSize)
        allObjsImgSize = sum(objsImgSize)
        t_rate = allRate /(allObjsImgSize/bgSize)
        #print(t_rate)
        objsRate = [t_rate * x for x in t1_rate]
        #print(objsRate)
    else:
        objsRate = [allRate /(allObjsImgSize/bgSize)]*len(objs)

    objsRate = [math.sqrt(x) for x in objsRate]

    objs = [cv2.resize(obj,(int(obj.shape[1]*drate),int(obj.shape[0]*drate))) \
            for obj,drate in zip(objs,objsRate)]

    masks = [cv2.resize(mask,(obj.shape[1],obj.shape[0])) for mask,obj in zip(masks,objs)]
    masks = [cv2.threshold(mask,127,255,cv2.CV_8UC1)[1] for mask in masks]

    return objs,masks

def isAllowBox(box,exitBndboxes,thresh=[0,0.2]):
    for ebox in exitBndboxes:
        opix = overlapPixes(ebox,box)
        coverRate = max(opix/boxPixes(ebox),opix/boxPixes(box))
        if(coverRate>thresh[1] or coverRate < thresh[0]):
            return False
    return True

def findPlace(bg,objs,masks,labels,gradSize = 15):
    grads = [(i,j) for i in range(gradSize) for j in range(gradSize)]
    bndboxes = []
    for i in range(len(objs)):
        isfind = False
        while(isfind is not True):
            obj = objs[i]
            mask = masks[i]
            label = labels[i]
            allow_tl = (bg.shape[1] - obj.shape[1],bg.shape[0] - obj.shape[0])
            #print(allow_tl)
            random.shuffle(grads)
            isfind = False
            for grad in grads:
                tl_x = min(int(allow_tl[0]/gradSize) * grad[0] + random.random()*gradSize, allow_tl[0]-1)
                tl_y = min(int(allow_tl[1]/gradSize) * grad[1] + random.random()*gradSize, allow_tl[1]-1)
                tl_x = max(tl_x,1)
                tl_y = max(tl_y,1)
                this_box = (int(tl_x),int(tl_y),int(tl_x+obj.shape[1]),int(tl_y+obj.shape[0]))
                if(isAllowBox(this_box,bndboxes,[0,0.2])):
                    isfind = True
                    bndboxes.append(this_box)
                    #print((this_box[3]-this_box[1],this_box[2]-this_box[0]),obj.shape,mask.shape,bg.shape,this_box)
                    #input()
                    break
            if(isfind is not True):
                print('Narrow object to find a suitable place')
                objs[i] = cv2.resize(objs[i],(int(objs[i].shape[1]*0.9),int(objs[i].shape[0]*0.9)))
                #print(masks[i])
                masks[i] = cv2.resize(masks[i],(objs[i].shape[1],objs[i].shape[0]))
    masks = [cv2.threshold(x,127,255,cv2.CV_8UC1)[1] for x in masks]
    bndboxes = [vocXmlobj(xmin=x[0],ymin=x[1],xmax=x[2]-1,ymax=x[3]-1,name=y) for x,y in zip(bndboxes,labels)]
    return objs,masks,bndboxes

def cutBlackEdig(objs,masks):
    for i in range(len(masks)):
        mask = masks[i].astype(np.int)
        mask_col_sum = mask.sum(0)
        mask_row_sum = mask.sum(1)

        xmin = 0
        ymin = 0
        xmax = mask_col_sum.shape[0]
        ymax = mask_row_sum.shape[0]

        for index in range(mask_col_sum.shape[0]):
            if(mask_col_sum[index]!=0):
                xmin = index
                break
        for index in range(mask_col_sum.shape[0]-1,-1,-1):
            if(mask_col_sum[index]!=0):
                xmax = index
                break
        for index in range(mask_row_sum.shape[0]):
            if(mask_row_sum[index]!=0):
                ymin = index
                break
        for index in range(mask_row_sum.shape[0]-1,-1,-1):
            if(mask_row_sum[index]!=0):
                ymax = index
                break

        if(xmin==0 and ymin == 0 and xmax == mask_col_sum.shape[0] \
            and ymax == mask_row_sum.shape[0]):
            pass
        else:
            objs[i] = objs[i][ymin:ymax,xmin:xmax,:]
            masks[i] = masks[i][ymin:ymax,xmin:xmax]
    return objs,masks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pyFusion')
    ## 'common cfg'
    parser.add_argument('--bg', type=str, required=True, help='[text list] or [path]')
    parser.add_argument('--obj', type=str, required=True, help='[text list] or [path]')
    parser.add_argument('--mask', default='', type=str, help='[path], default = \'\'')
    parser.add_argument('--obj_size', default='obj_size.txt', type=str, help='the obj real size')
    parser.add_argument('--save', type=str, required=True, help='path to save')
    # 'fusion parameter'
    parser.add_argument('--num', type=int, required=True, help='numer of out images.')
    parser.add_argument('--min', default=5, type=int, help='min numer of objs in each image')
    parser.add_argument('--max', default=6, type=int, help='mix numer of objs in each image')
    parser.add_argument('--date', default='TODAY', type=str, help='it can be set.For example: \'2018-08-13\'')
    parser.add_argument('--first', default=10001, type=int, help='the fisrt index. For example: \'10001\'')

    # 'resize parameter'
    parser.add_argument('--resize', type = int, default = None, help = '[0 = stretch.] [1 = round up.] [2 = round up and crop] [3 = round down] [4 = round down and fill black] [5 = round down and fill self]')
    parser.add_argument('--width', type = int,  default = None, help = 'Width for saved images')
    parser.add_argument('--height', type = int,  default = None, help = 'Height for saved images')

    ## progrom parser
    #parser.add_argument('--process', default=multiprocessing.cpu_count() - 1, type=int, help='use how many process.')
    args = parser.parse_args()

    #date
    if(args.date=='TODAY'):
        date = str(datetime.datetime.now())
        date = date[0:date.find(' ')]
    else:
        date = args.date

    #first index
    save_index = args.first

    #read real size
    if(os.path.isfile(args.obj_size)):
        objSize = {x[0]:float(x[1])*float(x[2]) for x in scanner.list(args.obj_size)}
    else:
        objSize =  {}

    #read images path
    bg_list,obj_mask_label = scanImage(args.bg,args.obj,args.mask)

    #make solution
    solution = randSolusion(bg_list,obj_mask_label,args.num,args.min,args.max)

    #IO
    pImgIO,pVocXmlIO = setIO(solution)
     
    #resizer    
    if(args.resize is not None and args.width is not None and args.height is not None):
        p_rszr = vocResizer((args.width,args.height),args.resize,cv2.INTER_LINEAR)
        print('Resizer is ready.')
    else:
        p_rszr=None

    #make save path
    if(os.path.isdir(os.path.join(args.save,'JPEGImages')) is not True):
        os.makedirs(os.path.join(args.save,'JPEGImages'))
    if(os.path.isdir(os.path.join(args.save,'Annotations')) is not True):
        os.makedirs(os.path.join(args.save,'Annotations'))
    
    for i in tqdm(range(args.num),ncols=50):
        #read images
        bg,objs,masks,labels = readImg(solution[i],pImgIO)

        #cut the black edge
        objs,masks = cutBlackEdig(objs,masks)

        #resize
        objs, masks = resizeObjs(bg,objs,masks,labels,objSize,[0.4,0.5])

        #print(bg.size,objs[0].size)

        #findPlace (may resize again)
        objs,masks,p_vocXmlobj_list = findPlace(bg,objs,masks,labels,15)
        
        #do fusion
        fusion = bg
        for obj, mask ,xmlobj in zip(objs,masks,p_vocXmlobj_list):
            #print(obj.shape,mask.shape,(xmlobj.ymax+1-xmlobj.ymin,xmlobj.xmax+1-xmlobj.xmin),bg.shape)
            #print(xmlobj.xmin,xmlobj.ymin,xmlobj.xmax+1,xmlobj.ymax+1)

            #if(xmlobj.xmin>=bg.shape[1] or xmlobj.xmax>=bg.shape[1]\
            #    or xmlobj.ymin>=bg.shape[0] or xmlobj.ymax>=bg.shape[0]):
            #    print('xmlobj, OUT!!!')
            #    input()

            cen_point = (int((xmlobj.xmin + xmlobj.xmax + 1)/2),int((xmlobj.ymin + xmlobj.ymax + 1)/2))
            #print(cen_point)
            #cen_point = (int((xmlobj.xmin+xmlobj.xmax+1)/2),int((xmlobj.ymin+xmlobj.ymax+1)/2))
            fusion = imFusion(obj,fusion,mask,cen_point,FUSION_TYPE.SMOOTH_BOUNDARY,{'ksize':16})
        #make vocXml
        p_vocXml = vocXml(objs=p_vocXmlobj_list)
        p_vocXml.width = fusion.shape[1]
        p_vocXml.height = fusion.shape[0]
        p_vocXml.folder='JPEGImages'
        p_vocXml.filename = 'fusion_{}_{}.jpg'.format(date,save_index)
        p_vocXml.path = os.path.join(args.save,'JPEGImages',p_vocXml.filename)
        #resize output
        if(p_rszr is not None):
            #print('resized')
            fusion = p_rszr.imResize(fusion)
            ret,p_vocXml = p_rszr.vocXmlResize(p_vocXml)
        #output
        pImgIO.write(p_vocXml.path,fusion)
        pVocXmlIO.write(os.path.join(args.save,'Annotations',\
                        'fusion_{}_{}.xml'.format(date,save_index)),p_vocXml)
        save_index += 1
