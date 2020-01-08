#-*-coding:utf-8-*-

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
# class "vocEvaluator" is depend on the license above


import xml.etree.ElementTree as ET
import os
import random
import numpy as np
import pickle
import cv2

from scipy.optimize import linear_sum_assignment

import pyBoost as pb

class bndbox():
    def __init__(self, _xmin, _ymin, _xmax, _ymax):
        if _xmax + 1 <_xmin or _ymax + 1< _ymin:
            print('init bndbox Error : ')
            raise ValueError('xmax < xmin or ymax < ymin')
        self.xmin = _xmin
        self.ymin = _ymin
        self.xmax = _xmax
        self.ymax = _ymax

    def area(self):
        return (self.xmax - self.xmin + 1) * (self.ymax - self.ymin + 1)

    def contains(self,pt):
        return self.xmin <= pt[0] and\
                pt[0] < self.xmax + 1 and\
                self.ymin <= pt[1] and\
                pt[1] < self.ymax + 1

    def empty(self):
        return self.ymax-self.ymin<=-1 or self.height<=-1

    def roi(self,img):
        return img[self.ymin:self.ymax+1,self.xmin:self.xmax+1]

    def tolist(self):
        return [self.xmin,self.ymin,self.xmax,self.ymax]

    def __getitem__(self,i):
        if i==0 or i==-4:
            return self.xmin
        elif i==1 or i==-3:
            return self.ymin
        elif i==2 or i==-2:
            return self.xmax
        elif i==3 or i==-1:
            return self.ymax
        else:
            raise IndexError('bndbox out of range')

    def tocvRect(self):
        return pb.img.cvRect(self.xmin,self.ymin,self.xmax-self.xmin+1,self.ymax-self.ymin+1)

    def copy(self):
        return bndbox(_xmin=self.xmin,_ymin=self.ymin,\
                      _xmax=self.xmax,_ymax=self.ymax)

class vocXmlobj(bndbox):
    def __init__(self,name,xmin,ymin,xmax,ymax,\
                pose = 'Unspecified',truncated = 0,difficult = 0):
        bndbox.__init__(self,xmin,ymin,xmax,ymax)
        self.name = name
        self.pose = pose
        self.truncated = truncated
        self.difficult = difficult

    def copy(self):
        return vocXmlobj(name=self.name,xmin=self.xmin,ymin=self.ymin,\
                         xmax=self.xmax,ymax=self.ymax,pose=self.pose,\
                         truncated=self.truncated,difficult=self.difficult)

class vocXml:
    def __init__(self,width,height,depth,objs = None,\
                segmented=0,folder='JPEGImages',filename='Unknown',\
                path = './JEPGImages',database = 'Unknown',verified=None):
        self.folder = folder
        self.filename = filename
        self.path = path  
        self.database = database
        self.width = width
        self.height = height
        self.depth = depth
        self.segmented = segmented
        if objs is None:
            # self.objs = 0*[vocXmlobj()]
            self.objs = []
        else:
            self.objs = objs
        self.verified=verified

    def copy(self):
        return vocXml(width=self.width,height=self.height,depth=self.depth,\
                      objs=[x.copy() for x in self.objs],\
                      segmented=self.segmented,folder=self.folder,\
                      filename=self.filename,path=self.path,\
                      database=self.databas,verified=self.verified)

def bndboxMatch(objs1,objs2, giou_thresh):
    giou_matrix = gIouMatrix(objs1,objs2)
    result = linear_sum_assignment(-giou_matrix)
    out = []
    for m0,m1 in zip(result[0],result[1]):
        if giou_matrix[m0,m1]>=giou_thresh:
            out.append([m0,m1])
    return out


def xml_read(path):
    #Unimportant infomations will be loaded with try, set default if load fail 
    xml_tree = ET.parse(path) 
    xml_root = xml_tree.getroot()

    w = int(xml_root.find('size').find('width').text)
    h = int(xml_root.find('size').find('height').text)
    d = int(xml_root.find('size').find('depth').text)

    out = vocXml(width=w,height=h,depth=d)

    p_xml_verified = xml_root.attrib.get('verified')
    if p_xml_verified is not None:
        out.verified = p_xml_verified

    p_xml_segmented = xml_root.find('segmented')
    if p_xml_segmented is not None:
        out.segmented = int(p_xml_segmented.text)

    p_xml_element = xml_root.find('folder')
    if p_xml_element is not None and p_xml_element.text is not None:
        out.folder = p_xml_element.text

    p_xml_filename = xml_root.find('filename')
    if p_xml_filename is not None:
        out.filename = p_xml_filename.text

    p_xml_element = xml_root.find('path')
    if p_xml_element is not None and p_xml_element.text is not None:
        out.path = p_xml_element.text

    p_xml_element = xml_root.find('source')
    if p_xml_element is not None:
        p_xml_element_1 = p_xml_element.find('database')
        if p_xml_element_1 is not None and p_xml_element_1.text is not None:
            out.database = p_xml_element_1.text

    #objects
    xml_obj_all = xml_root.findall('object')
    for xml_obj_one in xml_obj_all:
        _name = xml_obj_one.find('name').text
        _xmin = int(xml_obj_one.find('bndbox').find('xmin').text)
        _ymin = int(xml_obj_one.find('bndbox').find('ymin').text)
        _xmax = int(xml_obj_one.find('bndbox').find('xmax').text)
        _ymax = int(xml_obj_one.find('bndbox').find('ymax').text)
        xml_obj_one_dict = vocXmlobj(name=_name,xmin=_xmin,ymin=_ymin,xmax=_xmax,ymax=_ymax)        
        xml_obj_one_dict.pose = xml_obj_one.find('pose').text
        xml_obj_one_dict.truncated = int(xml_obj_one.find('truncated').text)
        xml_obj_one_dict.difficult = int(xml_obj_one.find('difficult').text)
        out.objs.append(xml_obj_one_dict)
    return out

def xml_write(filename,vocxml_info):
    flag = False
    with open(filename,'w') as xml_file:
        flag = True
        try:
            if vocxml_info.verified is None:
                xml_file.write('<annotation>\n')
            else:
                xml_file.write('<annotation verified=\"{0}\">\n'.format(vocxml_info.verified))
            xml_file.write('    <folder>{0}</folder>\n'.format(vocxml_info.folder))
            xml_file.write('    <filename>{0}</filename>\n'.format(vocxml_info.filename))
            xml_file.write('    <path>{0}</path>\n'.format(vocxml_info.path))
            xml_file.write('    <source>\n')
            xml_file.write('        <database>{0}</database>\n'.format(vocxml_info.database))
            xml_file.write('    </source>\n')
            xml_file.write('    <size>\n')
            xml_file.write('        <width>{0}</width>\n'.format(int(vocxml_info.width)))
            xml_file.write('        <height>{0}</height>\n'.format(int(vocxml_info.height)))
            xml_file.write('        <depth>{0}</depth>\n'.format(int(vocxml_info.depth)))
            xml_file.write('    </size>\n')
            xml_file.write('    <segmented>{0}</segmented>\n'.format(vocxml_info.segmented))
            for  one_obj in vocxml_info.objs:
                xml_file.write('    <object>\n')
                xml_file.write('        <name>{0}</name>\n'.format(one_obj.name))
                xml_file.write('        <pose>{0}</pose>\n'.format(one_obj.pose))
                xml_file.write('        <truncated>{0}</truncated>\n'.format(one_obj.truncated))
                xml_file.write('        <difficult>{0}</difficult>\n'.format(one_obj.difficult))
                xml_file.write('        <bndbox>\n')
                xml_file.write('            <xmin>{0}</xmin>\n'.format(int(one_obj.xmin)))
                xml_file.write('            <ymin>{0}</ymin>\n'.format(int(one_obj.ymin)))
                xml_file.write('            <xmax>{0}</xmax>\n'.format(int(one_obj.xmax)))
                xml_file.write('            <ymax>{0}</ymax>\n'.format(int(one_obj.ymax)))
                xml_file.write('        </bndbox>\n')
                xml_file.write('    </object>\n')
            xml_file.write('</annotation>\n')
        except Exception as e:
            print('Error while saving \"{0}\"'.format(filename))
            flag = False
    return flag

#inplace
#return True if nothing is changed
def adjust_bndbox(xml_info):
    no_change = True
    for i in range(len(xml_info.objs)-1,-1,-1):
        one_bndbox = xml_info.objs[i]
        if one_bndbox.xmin<1:
            one_bndbox.xmin = 1
            no_change = False
        if one_bndbox.ymin<1:
            one_bndbox.ymin = 1
            no_change = False
        if one_bndbox.xmax>=xml_info.width:
            one_bndbox.xmax = xml_info.width - 1
            no_change = False
        if one_bndbox.ymax>xml_info.height:
            one_bndbox.ymax = xml_info.height - 1
            no_change = False
        if one_bndbox.xmin>=one_bndbox.xmax or one_bndbox.ymin>=one_bndbox.ymax:
            xml_info.objs.pop(i)
    return no_change

VOCRESIZE_STRETCH = pb.img.IMRESIZE_STRETCH
VOCRESIZE_ROUNDUP = pb.img.IMRESIZE_ROUNDUP
VOCRESIZE_ROUNDUP_CROP = pb.img.IMRESIZE_ROUNDUP_CROP
VOCRESIZE_ROUNDDOWN = pb.img.IMRESIZE_ROUNDDOWN
VOCRESIZE_ROUNDDOWN_FILL_BLACK = pb.img.IMRESIZE_ROUNDDOWN_FILL_BLACK
VOCRESIZE_ROUNDDOWN_FILL_SELF = pb.img.IMRESIZE_ROUNDDOWN_FILL_SELF


imResize = pb.img.imResize
pntResize = pb.img.pntResize
pntRecover = pb.img.pntRecover

def xmlResize(src,img_src_shape,dsize,fx=None,fy=None,rtype=None):
    dst = src.copy()
    for obj in dst.objs:
        obj.xmin,obj.ymin = pntResize((obj.xmin,obj.ymin),
                            img_src_shape,dsize,fx=fx,fy=fy,rtype=rtype)
        obj.xmax,obj.ymax = pntResize((obj.xmax,obj.ymax),
                            img_src_shape,dsize,fx=fx,fy=fy,rtype=rtype)
    adjust_bndbox(dst)
    return dst
    
def xmlRecover(xml_dst,img_src_shape,dsize,fx=None,fy=None,rtype=None):
    src = xml_dst.copy()
    for obj in dst.objs:
        obj.xmin,obj.ymin = pntRecover((obj.xmin,obj.ymin),
                            img_src_shape,dsize,fx=fx,fy=fy,rtype=rtype)
        obj.xmax,obj.ymax = pntRecover((obj.xmax,obj.ymax),
                            img_src_shape,dsize,fx=fx,fy=fy,rtype=rtype)
    adjust_bndbox(src)
    return src

def vocResize(img_src,xml_src,dsize,fx=None,fy=None,interpolation=None,rtype=None):
    img_dst = imResize(img_src,dsize,fx=fx,fy=fy,interpolation=interpolation,rtype=rtype)
    xml_dst = xmlResize(xml_src,img_src.shape,dsize,fx=fx,fy=fy,rtype=rtype)
    return img_dst,xml_dst

#"dets_file" is pickle file with each dump [["xmlName" "label" "confidence" "x_min" "y_min" "x_max" "y_max"],...]
class vocEvaluator:
    def __init__(self, dets_file, annotations_path, IOU_thresh = 0.5):
        self.__iouThresh = IOU_thresh
        self.__detsFile = dets_file
        self.__annoPath = annotations_path

        #annotions
        self.__annoClsName, self.__annoRecs = self.__load_annotations(self.__annoPath)

        #detections
        self.__detsClsName, self.__detsImgName, self.__dets_list = self.__load_dets(self.__detsFile)

        #ground true about detections
        self.__clsName, self.__recs = self.__load_dets_recs(self.__annoRecs, self.__detsImgName)

        #calculation
        self.__ap_conf_rec_prec = self.__calAPConfRecPrec(self.__clsName,self.__recs, self.__dets_list, self.__iouThresh)

    def __load_dets(self,dets_file):
        print('Load detections...')
        if dets_file[-4:]=='.txt':
            out_dets_list = [[x[0],x[1],float(x[2]),int(x[3]),int(x[4]),int(x[5]),int(x[6])] for x in pb.scan_text(dets_file)]
            out_dets_cls_name = list(set([x[1] for x in out_dets_list]))
            out_dets_img_name = list(set([x[0] for x in out_dets_list]))
        else:
            with open(dets_file,'rb') as f_pkl:
                out_dets_list = []
                while(1):
                    try:
                        one_list = pickle.load(f_pkl)
                        out_dets_list = out_dets_list + one_list
                    except Exception as e:
                        out_dets_cls_name = list(set([x[1] for x in out_dets_list]))
                        out_dets_img_name = list(set([x[0] for x in out_dets_list]))
                        break
        return out_dets_cls_name, out_dets_img_name, out_dets_list

    def __load_dets_recs(self,recs,imagenames):
        out_recs = {}
        for imagename in imagenames:
            out_recs[imagename] = recs[imagename]

        dets_cls_set = set()
        for key in list(out_recs.keys()):
            for obj in out_recs[key].objs:
                dets_cls_set.add(obj.name)

        return list(dets_cls_set),out_recs

    def __get_cls_recs(self,recs, cls):
        class_recs = {}
        npos = 0
        for imagename in list(recs.keys()):
            R = [obj for obj in recs[imagename].objs if obj.name == cls]
            bbox = np.array([[x.xmin,x.ymin,x.xmax,x.ymax] for x in R])
            difficult = np.array([x.difficult for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox, 'difficult': difficult, 'det': det}
        return npos,class_recs

    def __load_annotations(self,annotations_path):
        cachefile = './EVALCACHE/annots.pkl'
        if os.path.isfile(cachefile):
            print('Find a cache file: '+cachefile)
            print('Load it ? [Y/N]')
            str_in = input()
            if str_in=='Y'or str_in =='y':
                annotations_path = cachefile
        if os.path.isfile(annotations_path):
            print('Load annotations from: '+annotations_path)
            # load
            with open(annotations_path, 'rb') as f:
                recs = pickle.load(f)
        else:
            # load annots
            anno_name_list = pb.scan_file(annotations_path,'.xml',False,True)
            recs = {}
            print('Reading annotations... ')
            p_vocXmlIO = vocXmlIO([os.path.join(annotations_path,x) for x in anno_name_list],False,3000)
            for anno_name in tqdm(anno_name_list):
                #recs[anno_name[:-4]] = xml_read(os.path.join(annotations_path,anno_name))
                recs[anno_name[:-4]] = p_vocXmlIO.read()
            p_vocXmlIO.waitEnd()
                #if i % 100 == 0:
                    #logger.info('Reading annotation for {:d}/{:d}'.format(i + 1, len(anno_name_list)))
            # save
            if os.path.isdir('./EVALCACHE') is not True:
                os.makedirs('./EVALCACHE')

            #logger.info('Saving cached annotations to {:s}'.format(cachefile))
            with open(cachefile, 'wb') as f:
                pickle.dump(recs, f)
        cls_set = set()
        for key in list(recs.keys()):
            for obj in recs[key].objs:
                cls_set.add(obj.name)
        return list(cls_set),recs
        
    def __get_sort_cls_dets(self,dets_list,cls):
        #image_ids = [x[0] for x in dets_list]
        #confidence = np.array([x[2] for x in dets_list],np.float32)
        #BB = np.array([[float(z) for z in x[3:7]] for x in dets_list])
        #for i,det in enumerate(dets_list):
        #    if det[1] != cls):#label is not this class 
        #        confidence[i]==0
        image_ids = [x[0] for x in dets_list if x[1]==cls]
        confidence = np.array([x[2] for x in dets_list if x[1]==cls],np.float32)
        BB = np.array([[float(z) for z in x[3:7]] for x in dets_list if x[1]==cls])
        if image_ids==[]:
            return [],np.array([]),np.array([])

        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
        confidence = np.array([confidence[x] for x in sorted_ind],np.float32)
        return image_ids,confidence,BB

    def __calAPConfRecPrec(self, cls_list,recs,dets_list,ovthresh):
        out_dict = {}
        print('Calculating for each class...')
        for cls in tqdm(cls_list):
            npos, class_recs = self.__get_cls_recs(recs, cls)
            image_ids,confidence,BB = self.__get_sort_cls_dets(dets_list,cls)

            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            if nd==0:
                out_dict[cls] ={'ap':0,'conf':confidence,'rec':confidence,'prec':confidence}
                continue
            for d in range(nd):
                R = class_recs[image_ids[d]]
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)

                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                           (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                           (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
        
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)
        
                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            #print((tp,npos))
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self.__calClsAP(rec, prec)
            out_dict[cls] = {'ap':ap,'conf':confidence,'rec':rec,'prec':prec}
        return out_dict

    def __calClsAP(self, rec, prec):
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
        
    def report_total(self,save_path):
        if os.path.isdir(save_path) is not True:
            os.makedirs(save_path)
        with open(os.path.join(save_path,'ALLCLS.csv'),'w') as fd:
            print('Begin to save all classes report...')
            #body data
            cls_list = list(self.__ap_conf_rec_prec.keys())
            n_cls = len(cls_list)
            out_mat = np.zeros((n_cls+1,23),np.float32)
            for i,cls in tqdm(enumerate(cls_list)):
                data_dict = self.__ap_conf_rec_prec[cls]
                ap = data_dict['ap']
                conf = data_dict['conf']
                rec = data_dict['rec']
                prec = data_dict['prec']
                out_mat[i][0] = ap
                len_conf = conf.shape[0]
                i_begin = 0
                if len_conf==0:
                    continue
                for i_rate in range(11):
                    rate = (10-i_rate)/10
                    for i_conf in range(i_begin,len_conf):
                        if i_conf==len_conf-1 or conf[i_conf]<rate:
                            i_begin = i_conf
                            break
                    if i_begin==0:
                        out_mat[i][i_rate*2+1] = 0
                        out_mat[i][i_rate*2+2] = 1
                    else:
                        out_mat[i][i_rate*2+1] = rec[max(i_begin-1,0)]
                        out_mat[i][i_rate*2+2] = prec[max(i_begin-1,0)]
            #mean
            out_mat[n_cls,:] = out_mat[0:n_cls,:].sum(0).astype(np.float32)/n_cls

            #write head
            fd.write('\"\" AP recall precision recall precision recall precision'+\
            ' recall precision recall precision recall precision recall precision'+\
            ' recall precision recall precision recall precision recall precision\n')
            fd.write('confidence \"\" 1 1 0.9 0.9 0.8 0.8 0.7 0.7 0.6 0.6 0.5 0.5 0.4 0.4 0.3 0.3 0.2 0.2 0.1 0.1 0 0\n')
            #write body
            for i,cls in enumerate(cls_list+['mean']):
                fd.write(cls)
                for j in range(23):
                    fd.write(' {:.4f}'.format(out_mat[i][j]))
                fd.write('\n')
        return

    def report_detail(self,save_path):
        if os.path.isdir(save_path) is not True:
            os.makedirs(save_path)
        cls_list = list(self.__ap_conf_rec_prec.keys())
        print('Begin to save each class report...')
        for i, cls in tqdm(enumerate(cls_list)):
            ap_conf_rec_prec = self.__ap_conf_rec_prec[cls]
            with open(os.path.join(save_path,cls+'.csv'),'w') as fd:
                fd.write('ap = {:.6f}\n'.format(ap_conf_rec_prec['ap']))
                conf = ap_conf_rec_prec['conf']
                rec = ap_conf_rec_prec['rec']
                prec = ap_conf_rec_prec['prec']
                for i in range(len(conf)):
                    fd.write('confidence = {:.6f}, recall = {:.6f}, precision = {:.6f}\n'.format(\
                    conf[i],rec[i],prec[i]))
        return

    def release(self):
        del self.__dets_list
        self.__dets_list = []
        del self.__annoRecs
        self.__annoRecs = []
        return

#cv2.seamlessClone(src,dst,mask,p,flags,blend)
#TODO:blend
BLEND_SPARSE = 0
BLEND_DENSE = 1
BLEND_ABREAST = 2
BLEND_OPT_ROTATE = 3
BLEND_OPT_FLIP = 4
BLEND_OPT_FLIPUD = 5
BLEND_OPT_FLIPLR = 6
BLEND_OPT_AFFINE = 7
BLEND_OPT_PERSPECTIVE = 8
BLEND_OPT_HUE = 9
BLEND_OPT_LIGHTNESS = 10
BLEND_OPT_SATURATION = 11

class blender():
    def __init__(self):
        self.type = BLEND_SPARSE
        self.size = 512
        self.obj_size = [32,64]
        self.opts = \
           {BLEND_OPT_ROTATE:0.1,\
            BLEND_OPT_FLIP:0.1,\
            BLEND_OPT_AFFINE:0.1,\
            BLEND_OPT_PERSPECTIVE:0.1,\
            BLEND_OPT_HUE:0.05,\
            BLEND_OPT_LIGHTNESS:0.05,\
            BLEND_OPT_SATURATION:0.05}
        self.opt_range = \
           {BLEND_OPT_ROTATE:(0,360),\
            BLEND_OPT_AFFINE:(-0.05,0.05),\
            BLEND_OPT_PERSPECTIVE:(0,0.05),\
            BLEND_OPT_HUE:(-10,10),\
            BLEND_OPT_LIGHTNESS:(-0.1,0.1),\
            BLEND_OPT_SATURATION:(-0.1,0.1)}
        self.cover = 0.1
        self.harmonic = 0.5
        pass

    

    def blend(bg,objs):
        # random opt and change objs

        # get obj size and find place

        # do blend
        image, xml = None, None
        return image, xml



def maskToBndbox(mask):
    if len(mask.shape)<2:
        return None
    elif len(mask.shape)==2:
        mask_c1 = mask
    else:
        mask_c1 = mask[:,:,-1]
    col_add = mask_c1.sum(0)
    raw_add = mask_c1.sum(1)
    i_col = np.where(col_add!=0)[0]
    i_raw = np.where(raw_add!=0)[0]
    if i_col.shape[0]==0 or i_raw.shape[0]==0:
        return None
    else:
        return [i_col[0],i_raw[0],i_col[-1],i_raw[-1]]
  
def maskROI(img,mask):
    bb = maskToBndbox(mask)
    if bb is None:
        return None
    else:
        return img[bb[1]:bb[3]+1,bb[0]:bb[2]+1]


def bndboxArea(bb):
    return (bb[2]-bb[0]+1)*(bb[3]-bb[1]+1)

def bndboxIntersect(bb1, bb2):
    #inters
    ixmin = max(bb1[0], bb2[0])
    iymin = max(bb1[1], bb2[1])
    ixmax = min(bb1[2], bb2[2])
    iymax = min(bb1[3], bb2[3])
    iw = max(ixmax - ixmin + 1., 0.)
    ih = max(iymax - iymin + 1., 0.)
    return int(iw * ih)

def bndboxIou(bb1,bb2):
    #ixmin = max(bb1[0], bb2[0])
    #iymin = max(bb1[1], bb2[1])
    #ixmax = min(bb1[2], bb2[2])
    #iymax = min(bb1[3], bb2[3])
    #iw = ixmax - ixmin + 1
    #ih = iymax - iymin + 1
    #inters = iw * ih
    #uni = (bb2[2] - bb2[0] + 1.) * (bb2[3] - bb2[1] + 1.) + (bb1[2] - bb1[0] + 1.) * (bb1[3] - bb1[1] + 1.) - inters
    #return inters/uni
    inters = bndboxIntersect(bb1,bb2)
    uni = bndboxArea(bb1) + bndboxArea(bb2) - inters
    return inters/uni



def iouMatrix(bbs1,bbs2):
    bbs1_area = [bndboxArea(x) for x in bbs1]
    bbs2_area = [bndboxArea(x) for x in bbs2]
    
    iou_matrix = np.zeros((len(bbs1),len(bbs2)),dtype=np.float64)
    
    for i1,a1 in enumerate(bbs1_area):
        for i2,a2 in enumerate(bbs2_area):
            inters = bndboxIntersect(det,trk)
            uni = a1 + a2 - inters
            iou_matrix[i1,i2] = inters/uni
    return iou_matrix

def bndboxGIou(bb1,bb2):
    inters = bndboxIntersect(bb1,bb2)
    areaA = bndboxArea(bb1)
    areaB = bndboxArea(bb2)
    uni = areaA + areaB - inters
    areaC = bndboxArea([min(bb1[0],bb2[0]),\
                        min(bb1[1],bb2[1]),\
                        max(bb1[2],bb2[2]),\
                        max(bb1[3],bb2[3])])

    # if inters>0:
    #     areaC = areaC-(areaC-uni)/2
    # else:
    #     ixmin = max(bb1[0], bb2[0])
    #     iymin = max(bb1[1], bb2[1])
    #     ixmax = min(bb1[2], bb2[2])
    #     iymax = min(bb1[3], bb2[3])
    #     iw = ixmax - ixmin + 1.
    #     ih = iymax - iymin + 1.
    #     areaC = areaC - (areaC-abs(iw*ih))/2


    return inters/uni - (areaC-uni)/areaC

def bndboxPR(bb1,bb2):
    xc1,yc1 = (bb1[0]+bb1[2])/2,(bb1[1]+bb1[3])/2
    xc2,yc2 = (bb2[0]+bb2[2])/2,(bb2[1]+bb2[3])/2
    w1,h1 = bb1[2]-bb1[0],bb1[3]-bb1[1]
    w2,h2 = bb2[2]-bb2[0],bb2[3]-bb2[1]
    mw,mh = (w1+w2)/2,(h1+h2)/2
    dx,dy = xc2-xc1,yc2-yc1
    kw = w2/w1
    kh = h2/h1
    r = ((dx/mw)**2+(dy/mh)**2+(kw-kh)**2)**0.5
    return r
    


    ixmin = max(bb1[0], bb2[0])
    iymin = max(bb1[1], bb2[1])
    ixmax = min(bb1[2], bb2[2])
    iymax = min(bb1[3], bb2[3])
    iw = ixmax - ixmin + 1
    ih = iymax - iymin + 1
    s1 = bndboxArea(bb1)
    s2 = bndboxArea(bb2)
    all = bndboxArea([min(bb1[0],bb2[0]),\
                    min(bb1[1],bb2[1]),\
                    max(bb1[2],bb2[2]),\
                    max(bb1[3],bb2[3])])
    if iw<0 or ih<0:
        inter =  int(-abs(iw*ih))
    else:
        inter = iw*ih
    c = (all - (s1 + s2 - inter)) / 2
    return (c-inter)/(all-c)
        

    


def gIouMatrix(bbs1,bbs2):
    bbs1_area = [bndboxArea(x) for x in bbs1]
    bbs2_area = [bndboxArea(x) for x in bbs2]
    giou_matrix = np.zeros((len(bbs1),len(bbs2)),dtype=np.float64)
    for i1,areaA in enumerate(bbs1_area):
        for i2,areaB in enumerate(bbs2_area):
            bb1,bb2 = bbs1[i1],bbs2[i2]
            inters = bndboxIntersect(bb1,bb2)
            uni = areaA + areaB - inters
            areaC = bndboxArea([min(bb1[0],bb2[0]),\
                        min(bb1[1],bb2[1]),\
                        max(bb1[2],bb2[2]),\
                        max(bb1[3],bb2[3])])
            giou_matrix[i1,i2] = inters/uni - (areaC-uni)/areaC
    return giou_matrix


def bndboxExIntersect(bb1,bb2):
    ixmin = max(bb1[0], bb2[0])
    iymin = max(bb1[1], bb2[1])
    ixmax = min(bb1[2], bb2[2])
    iymax = min(bb1[3], bb2[3])
    iw = ixmax - ixmin + 1
    ih = iymax - iymin + 1
    if iw<0 or ih<0:
        return int(-abs(iw*ih))
    else:
        return iw*ih

def bndboxExIou(bb1,bb2):
    ixmin = max(bb1[0], bb2[0])
    iymin = max(bb1[1], bb2[1])
    ixmax = min(bb1[2], bb2[2])
    iymax = min(bb1[3], bb2[3])
    iw = ixmax - ixmin + 1
    ih = iymax - iymin + 1
    isOver = 1
    if iw<0:
        iw = -iw
        isOver = -1
    if ih<0:
        ih = -ih
        isOver = -1
    inters = isOver * iw * ih
    uni = (bb2[2] - bb2[0] + 1.) * (bb2[3] - bb2[1] + 1.) +\
            (bb1[2] - bb1[0] + 1.) * (bb1[3] - bb1[1] + 1.) - inters
    return inters/uni

def exIouMatrix(bbs1,bbs2):
    bbs1_area = [bndboxArea(x) for x in bbs1]
    bbs2_area = [bndboxArea(x) for x in bbs2]
    iou_matrix = np.zeros((len(bbs1),len(bbs2)),dtype=np.float64)
    for i1,a1 in enumerate(bbs1_area):
        for i2,a2 in enumerate(bbs2_area):
            inters = bndboxExIntersect(bbs1[i1],bbs2[i2])
            uni = a1 + a2 - inters
            iou_matrix[i1,i2] = inters/uni
    return iou_matrix
    
    
#format='name.jpg' 'class' 'xmin' 'ymin' 'xmax' 'ymax'
#HAVEEXT: whether have file extension name
#ONECLASS: if 'detFile' have no 'class', you can set 'class' by this parameter
def detToXml(detFile,imgPath,savePath,HAVEEXT=True,ONECLASS=''):
    imgFolder = os.path.split(imgPath)[1]
    detFileInfo = pb.scan_text(detFile)

    boxes_dict = {}
    if ONECLASS=='':
        for oneLine in detFileInfo:
            p_boxes_dict = boxes_dict.get(oneLine[0])
            if p_boxes_dict is None:
                boxes_dict[oneLine[0]] = [oneLine[1:]]
            else:
                p_boxes_dict.append(oneLine[1:])
    else:
        for oneLine in detFileInfo:
            p_boxes_dict = boxes_dict.get(oneLine[0])
            if p_boxes_dict is None:
                boxes_dict[oneLine[0]] = [[ONECLASS]+oneLine[1:]]
            else:
                p_boxes_dict.append([ONECLASS]+oneLine[1:])

    if HAVEEXT:
        for key in boxes_dict:
            imgFullPath = os.path.join(imgPath,key)
            imgShape = cv2.imread(os.path.join(imgPath,key)).shape
            p_vocXml = vocXml(folder=imgFolder,filename=key,path=imgFullPath,\
                                width=imgShape[1],height=imgShape[0],\
                                objs = [vocXmlobj(name=x[0],xmin=int(x[1]),ymin=int(x[2]),xmax=int(x[3]),ymax=int(x[4]))\
                                        for x in boxes_dict[key]])
            frontName = os.path.splitext(key)[0]
            xmlSavePath = os.path.join(savePath,frontName+'.xml')
            adjust_bndbox(p_vocXml)
            xml_write(xmlSavePath,p_vocXml)
    else:
        for key in boxes_dict:
            imgFullName = key+'.jpg'
            imgFullPath = os.path.join(imgPath,imgFullName)
            imgShape = cv2.imread(os.path.join(imgPath,imgFullName)).shape
            p_vocXml = vocXml(folder=imgFolder,filename=imgFullName,path=imgFullPath,\
                                width=imgShape[1],height=imgShape[0],\
                                objs = [vocXmlobj(name=x[0],xmin=int(x[1]),ymin=int(x[2]),xmax=int(x[3]),ymax=int(x[4]))\
                                        for x in boxes_dict[key]])
            xmlSavePath = os.path.join(savePath,key+'.xml')
            adjust_bndbox(p_vocXml)
            xml_write(xmlSavePath,p_vocXml)



if __name__=='__main__':
    ##################################################################
    #test module
    ##################################################################

    import sys
    sys.path.append('../')
    import pyBoost as pb

    def test_adjust_bndbox():
        read_path = r'E:\fusion\frcnn_hand'
        xml_file_name = pb.deep_scan_file(read_path,'.xml',False,True)
        for one in xml_file_name:
            xml_info = pb.voc.xml_read(os.path.join(read_path,one))
            flag = pb.voc.adjust_bndbox(xml_info)
            if flag!=0:
                print('['+str(flag)+'], path = '+ one)
        return

    def test_vocEvaluator(save_folder_name):
        p_vocEvaluator = pb.voc.vocEvaluator('./test_07_19_dets.pkl','./Annotations',0.5)
        p_vocEvaluator.report_total(os.path.join(save_folder_name,'vocEvaluator'))
        p_vocEvaluator.report_detail(os.path.join(save_folder_name,'vocEvaluator'))

    def test_detToXml(save_folder_name):
        detFile = r'D:\dataset\hand_dataset\detection.txt'
        imgPath = r'D:\dataset\hand_dataset\voc\JPEGImages'
        save_path = os.path.join(save_folder_name,'detToXml','Annotations')
        pb.makedirs(save_path)
        pb.voc.detToXml(detFile,imgPath,save_path)
        return 

    def test_boxiou():
        import cv2
        import numpy as np 
        import argparse
        import os
        parser = argparse.ArgumentParser()
        parser.add_argument('--img',type=str, default='', help='Image you want to view')
        args = parser.parse_args()

        # load img
        user_img_view_size = (1280,720) # hwc
        if args.img == '':
            img = np.zeros((user_img_view_size[1],user_img_view_size[0],3),np.uint8)
        else:
            img = cv2.imread(args.img)
            if img is None:
                img = np.zeros((user_img_view_size[1],user_img_view_size[0],3),np.uint8)
            elif img.shape!=user_img_view_size:
                img = pb.img.imResize(img,user_img_view_size,rtype=pb.img.IMRESIZE_ROUNDDOWN)
                #img = pb.img.imResizer(pb.img.IMRESIZE_ROUNDDOWN,\
                #                        user_img_view_size).\
                #                        imResize(img)

        # global data
        global_data = {}
        global_data['background'] = img
        global_data['target_box'] = None
        global_data['predict_boxes'] = []
        global_data['bg_with_box'] = img.copy()
        global_data['img_to_show'] = img.copy()

        global_data['show_name'] = 'bndbox iou viewer'
        global_data['is_l_down'] = False
        global_data['is_m_down'] = False
        global_data['is_r_down'] = False
        global_data['l_point'] = (0,0)
        global_data['m_point'] = (0,0)
        global_data['r_point'] = (0,0)

        def draw_iou_info(img, target_box, predict_boxes):
            for predict_box in predict_boxes:
                draw_one_predict_box(img, target_box, predict_box)
            if target_box is not None:
                cv2.rectangle(img,(target_box[0],target_box[1]),\
                              (target_box[2]+1,target_box[3]+1),(255, 255, 255))
            return
            
        def draw_one_predict_box(img, target_box, predict_box):
            cv2.rectangle(img,(predict_box[0],predict_box[1]),\
                              (predict_box[2]+1,predict_box[3]+1),(0, 255, 0))
            if target_box is not None:
                iou_value = float(pb.voc.bndboxIou(target_box,predict_box))
                exiou_value = float(pb.voc.bndboxExIou(target_box,predict_box))
                giou_value = float(pb.voc.bndboxGIou(target_box,predict_box))
                PR_value = float(pb.voc.bndboxPR(target_box,predict_box))
                iou_str = 'iou = {0:5.4}'.format(iou_value)
                exiou_str = 'exiou = {0:5.4}'.format(exiou_value)
                giou_str = 'giou = {0:5.4}'.format(giou_value)
                PR_str = 'PR = {0:5.4}'.format(PR_value)
                y_bias = 30
                iou_p = (int(predict_box[0]*0.9+predict_box[2]*0.1),int(predict_box[1]*0.8+predict_box[3]*0.2))
                exiou_p = (iou_p[0],iou_p[1]+y_bias)
                giou_p = (exiou_p[0],exiou_p[1]+y_bias)
                PR_p = (giou_p[0],giou_p[1]+y_bias)
                cv2.putText(img, iou_str, iou_p, \
                            cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
                cv2.putText(img, exiou_str, exiou_p, \
                            cv2.FONT_HERSHEY_COMPLEX,0.5, (0,255,0),1)
                cv2.putText(img, giou_str, giou_p, \
                            cv2.FONT_HERSHEY_COMPLEX,0.5, (0,255,0),1)
                cv2.putText(img, PR_str, PR_p, \
                            cv2.FONT_HERSHEY_COMPLEX,0.5, (0,255,0),1)
            return

        def to_box_and_points(p1,p2):
            xmin = min(p1[0],p2[0])
            ymin = min(p1[1],p2[1])
            xmax = max(p1[0],p2[0])
            ymax = max(p1[1],p2[1])
            if xmin==xmax or ymin==ymax:
                bb = None
                return None, None, None
            else:
                return [xmin,ymin,xmax,ymax], (xmin,ymin+1), (xmax,ymax+1)

        def mouse_call_back(event, x, y, flags, global_data):
            #global_data = param['global_data']
            if event == cv2.EVENT_MOUSEMOVE:
                img_to_show = global_data['bg_with_box'].copy()
                img_to_show = cv2.line(img_to_show,(x,0),(x,img_to_show.shape[0]),(0,255,0))
                img_to_show = cv2.line(img_to_show,(0,y),(img_to_show.shape[1],y),(0,255,0))
                if global_data['is_l_down']:
                    _, p1,p2 = to_box_and_points(global_data['l_point'],(x,y))
                    img_to_show = cv2.rectangle(img_to_show,p1,p2,(0, 255, 0))
                if global_data['is_m_down']:
                    _, p1,p2 = to_box_and_points(global_data['m_point'],(x,y))
                    img_to_show = cv2.rectangle(img_to_show,p1,p2,(0, 0, 255))
                if global_data['is_r_down']:
                    _, p1,p2 = to_box_and_points(global_data['r_point'],(x,y))
                    img_to_show = cv2.rectangle(img_to_show,p1,p2,(255, 255, 255))
                global_data['img_to_show'] = img_to_show
            elif event == cv2.EVENT_LBUTTONDOWN:
                global_data['is_l_down'] = True
                global_data['l_point'] = (x,y)
            elif event == cv2.EVENT_MBUTTONDOWN:
                global_data['is_m_down'] = True
                global_data['m_point'] = (x,y)
            elif event == cv2.EVENT_RBUTTONDOWN:
                global_data['is_r_down'] = True
                global_data['r_point'] = (x,y)
            elif event == cv2.EVENT_LBUTTONUP:
                global_data['is_l_down'] = False
                xmin = min(global_data['l_point'][0],x)
                ymin = min(global_data['l_point'][1],y)
                xmax = max(global_data['l_point'][0],x)
                ymax = max(global_data['l_point'][1],y)
                if xmin==xmax or ymin==ymax:
                    bb = None
                    return
                else:
                    bb = [xmin,ymin,xmax,ymax]                           
                global_data['predict_boxes'].append(bb)
                global_data['img_to_show'] = global_data['bg_with_box']
                draw_one_predict_box(global_data['img_to_show'], global_data['target_box'],bb)
            elif event == cv2.EVENT_MBUTTONUP:
                global_data['is_m_down'] = False
                xmin = min(global_data['m_point'][0],x)
                ymin = min(global_data['m_point'][1],y)
                xmax = max(global_data['m_point'][0],x)
                ymax = max(global_data['m_point'][1],y)
                if xmin==xmax or ymin==ymax:
                    bb = None
                    return
                else:
                    bb = [xmin,ymin,xmax,ymax]
                target_box_changed = False
                predict_box_changed = False
                if global_data['target_box'] is not None:
                    if pb.voc.bndboxIntersect(global_data['target_box'], bb) >0.1:
                        global_data['target_box'] = None
                        target_box_changed = True
                predict_boxes = global_data['predict_boxes']
                for i in range(len(predict_boxes)-1,-1,-1):
                    predict_box = predict_boxes[i]
                    if pb.voc.bndboxIntersect(predict_box, bb) >0.1:
                        predict_boxes.pop(i)
                        predict_box_changed = True
                if target_box_changed or predict_box_changed:
                    global_data['bg_with_box'] = global_data['background'].copy()
                    global_data['img_to_show'] = global_data['bg_with_box']
                    draw_iou_info(global_data['img_to_show'], global_data['target_box'],global_data['predict_boxes'])
            elif event == cv2.EVENT_RBUTTONUP:
                global_data['is_r_down'] = False
                xmin = min(global_data['r_point'][0],x)
                ymin = min(global_data['r_point'][1],y)
                xmax = max(global_data['r_point'][0],x)
                ymax = max(global_data['r_point'][1],y)
                if xmin==xmax or ymin==ymax:
                    bb = None
                    return
                else:
                    bb = [xmin,ymin,xmax,ymax]
                global_data['target_box'] = bb
                global_data['bg_with_box'] = global_data['background'].copy()
                global_data['img_to_show'] = global_data['bg_with_box']
                draw_iou_info(global_data['img_to_show'], global_data['target_box'],global_data['predict_boxes'])
            return

        cv2.namedWindow(global_data['show_name'])
        param = {}
        param['global_data'] = global_data
        cv2.setMouseCallback(global_data['show_name'],mouse_call_back, global_data)
        while 1:
            cv2.imshow(global_data['show_name'], global_data['img_to_show'])
            key = cv2.waitKey(10)
            if key==ord('q') or key==ord('Q') or key==27:
                break
        return

    #####################################################################
    save_folder_name = 'pyBoost_test_output'
    #test_adjust_bndbox()
    #test_vocEvaluator(save_folder_name)
    #test_detToXml(save_folder_name)
    test_boxiou()

