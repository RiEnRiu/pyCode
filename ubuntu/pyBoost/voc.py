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
import pyBoostBase as pbBase
import img as pbimg
import os
import random
import numpy as np
import pickle
import cv2

class vocXmlobj:
    def __init__(self,name = 'Unknow',pose = 'Unspecified',truncated = 0,\
                difficult = 0,xmin = 0,ymin = 0, xmax = 0,ymax = 0):
        self.name = name
        self.pose = pose
        self.truncated = truncated
        self.difficult = difficult
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def copyTo(self,inputOutput):
        inputOutput.name = self.name
        inputOutput.pose = self.pose 
        inputOutput.truncated = self.truncated 
        inputOutput.difficult = self.difficult 
        inputOutput.xmin = self.xmin 
        inputOutput.ymin = self.ymin 
        inputOutput.xmax = self.xmax 
        inputOutput.ymax = self.ymax

    def copy(self):
        out = vocXmlobj(self.name,self.pose,self.truncated,self.difficult,\
                        self.xmin,self.ymin,self.xmax,self.ymax)
        return out


class vocXml:
    def __init__(self,folder='Unknown',filename='Unknown',path = 'Unknown',database = 'Unknown',\
                width = 0,height = 0,depth = 3,segmented = 0,objs = None):
        self.folder = folder
        self.filename = filename
        self.path = path  
        self.database = database
        self.width = width
        self.height = height
        self.depth = depth
        self.segmented = segmented
        if objs is None:
            self.objs = 0*[vocXmlobj()]
        else:
            self.objs = objs

    def copyTo(self,inputOutput):
        inputOutput.folder = self.folder 
        inputOutput.filename = self.filename 
        inputOutput.path = self.path 
        inputOutput.database = self.database
        inputOutput.width = self.width 
        inputOutput.height = self.height 
        inputOutput.depth = self.depth
        inputOutput.segmented = self.segmented 
        inputOutput.objs = [x.copy() for x in self.objs]

    def copy(self):
        out = vocXml(self.folder,self.filename,self.path,self.database,\
                    self.width,self.height,self.depth,self.segmented)
        out.objs = [x.copy() for x in self.objs]
        return out

def splitVocDataset(path, train,val,test):
    pairs,o1,o2 = pbBase.scan_pair(os.path.join(path,'JPEGImages'),os.path.join(path,'Annotations'),'.jpg.jpeg.png','.xml',False, True)
    names = [x[1][:-4] for x in pairs]
    random.shuffle(names)
    len_names = len(names)
    len_train = int(len_names*train)
    len_val = int(len_names*val)
    len_test = len_names - len_train - len_val
    len_trainval = len_train+len_val
    save_path = os.path.join(path,'ImageSets','Main')
    if os.path.isdir(save_path) is not True:
        os.makedirs(save_path)
    with open(os.path.join(save_path,'trainval.txt'),'w') as f_trainval:
        for i in range(0,len_trainval-1):
            f_trainval.write(names[i]+'\n')
        if len_trainval!=0:
            f_trainval.write(names[len_trainval-1])
    with open(os.path.join(save_path,'train.txt'),'w') as f_train:
        for i in range(0,len_train-1):
            f_train.write(names[i]+'\n')
        if len_train!=0:
            f_train.write(names[len_train-1])
    with open(os.path.join(save_path,'val.txt'),'w') as f_val:
        for i in range(len_train,len_trainval-1):
            f_val.write(names[i]+'\n')
        if len_val!=0:
            f_val.write(names[len_trainval-1])
    with open(os.path.join(save_path,'test.txt'),'w') as f_test:
        for i in range(len_trainval,len_names-1):
            f_test.write(names[i]+'\n')
        if len_test!=0:
            f_test.write(names[len_names-1])
    return len_train,len_val,len_test

def vocXmlRead(path):
    #Unimportant infomations will be loaded with try, set default if load fail 
    out = vocXml()
    xml_tree = ET.parse(path) 
    xml_root = xml_tree.getroot()

    p_xml_element = xml_root.find('folder')
    if p_xml_element is not None and p_xml_element.text is not None:
        out.folder = p_xml_element.text

    out.filename = xml_root.find('filename').text

    p_xml_element = xml_root.find('path')
    if p_xml_element is not None and p_xml_element.text is not None:
        out.path = p_xml_element.text

    p_xml_element = xml_root.find('source')
    if p_xml_element is not None:
        p_xml_element_1 = p_xml_element.find('database')
        if p_xml_element_1 is not None and p_xml_element_1.text is not None:
            out.database = p_xml_element_1.text

    out.width = int(xml_root.find('size').find('width').text)
    out.height = int(xml_root.find('size').find('height').text)
    out.depth = int(xml_root.find('size').find('depth').text)
    out.segmented = int(xml_root.find('segmented').text)
    #objects
    xml_obj_all = xml_root.findall('object')
    for xml_obj_one in xml_obj_all:
        xml_obj_one_dict = vocXmlobj()
        xml_obj_one_dict.name = xml_obj_one.find('name').text
        xml_obj_one_dict.pose = xml_obj_one.find('pose').text
        xml_obj_one_dict.truncated = int(xml_obj_one.find('truncated').text)
        xml_obj_one_dict.difficult = int(xml_obj_one.find('difficult').text)
        xml_obj_one_dict.xmin = int(xml_obj_one.find('bndbox').find('xmin').text)
        xml_obj_one_dict.ymin = int(xml_obj_one.find('bndbox').find('ymin').text)
        xml_obj_one_dict.xmax = int(xml_obj_one.find('bndbox').find('xmax').text)
        xml_obj_one_dict.ymax = int(xml_obj_one.find('bndbox').find('ymax').text)
        out.objs.append(xml_obj_one_dict)
    
    return out

def vocXmlWrite(filename,vocxml_info):
    xml_file = open(filename,'w')
    xml_file.write('<annotation>'+'\n')
    xml_file.write('    <folder>'+vocxml_info.folder+'</folder>'+'\n')
    xml_file.write('    <filename>'+vocxml_info.filename+'</filename>'+'\n')
    xml_file.write('    <path>' +vocxml_info.path+ '</path>' + '\n')
    xml_file.write('    <source>' + '\n')
    xml_file.write('        <database>'+vocxml_info.database+'</database>'  + '\n')
    xml_file.write('    </source>' + '\n')
    xml_file.write('    <size>' + '\n')
    xml_file.write('        <width>' + str(vocxml_info.width)+ '</width>' + '\n')
    xml_file.write('        <height>'  +str(vocxml_info.height)+ '</height>' + '\n')
    xml_file.write('        <depth>' +str(vocxml_info.depth)+ '</depth>' + '\n')
    xml_file.write('    </size>' + '\n')
    xml_file.write('    <segmented>'+str(vocxml_info.segmented)+'</segmented>' + '\n')
    for  one_obj in vocxml_info.objs:
        xml_file.write('    <object>' + '\n')
        xml_file.write('        <name>'+one_obj.name+ '</name>'  + '\n')
        xml_file.write('        <pose>'+one_obj.pose+'</pose>' + '\n')
        xml_file.write('        <truncated>'+str(one_obj.truncated)+'</truncated>' + '\n')
        xml_file.write('        <difficult>'+str(one_obj.difficult)+'</difficult>'  + '\n')
        xml_file.write('        <bndbox>' + '\n')
        xml_file.write('            <xmin>'+str(one_obj.xmin)+'</xmin>'  + '\n')
        xml_file.write('            <ymin>'+str(one_obj.ymin)+ '</ymin>' + '\n')
        xml_file.write('            <xmax>'+str(one_obj.xmax)+ '</xmax>' + '\n')
        xml_file.write('            <ymax>'+str(one_obj.ymax)+ '</ymax>' + '\n')
        xml_file.write('        </bndbox>'  + '\n')
        xml_file.write('    </object>' + '\n')
    xml_file.write('</annotation>' + '\n')
    xml_file.close()
    return

#inplace
#return = 0, there is no changing; 
#return = 1, there is something changing;
#return = -1, there is no objects after adjusting
def adjustBndbox(xml_info):
    out_flag = 0
    list_to_del = []
    for i,one_bndbox in enumerate(xml_info.objs):
        if one_bndbox.xmin<1:
            one_bndbox.xmin = 1
            out_flag = 1
        if one_bndbox.ymin<1:
            one_bndbox.ymin = 1
            out_flag = 1
        if one_bndbox.xmax>=xml_info.width:
            one_bndbox.xmax = xml_info.width - 1
            out_flag = 1
        if one_bndbox.ymax>xml_info.height:
            one_bndbox.ymax = xml_info.height - 1
            out_flag = 1
        if one_bndbox.xmin>=one_bndbox.xmax or one_bndbox.ymin>=one_bndbox.ymax:
            list_to_del.append(i)
    if list_to_del!=[]:
        for i in reversed(list_to_del):
            temp = xml_info.objs.pop(i)
        return -1
    else:
        return out_flag

class vocResizer(pbimg.imResizer):
    def __init__(self, resize_type, dsize, interpolation):
        pbimg.imResizer.__init__(self, resize_type, dsize, interpolation)

    def xmlResizeInDisk(self, xml_src, xml_dst, path_replace = None):
        xml_src_info = vocXmlRead(xml_src)
        ret,xml_dist_info = self.xmlResize(xml_src_info)
        if ret is not True:
            print('There is no object after resized: '+xml_src)
        vocXmlWrite(xml_dst,xml_dist_info)
        return

    def xmlResize(self,xml_src_info,path_replace = None):
        xml_dist_info = xml_src_info.copy()
        if path_replace is not None:
            xml_dist_info.path = path_replace
            xml_dist_info.folder = os.path.split(os.path.split(path_replace)[0])[1]#[part_of_path]/[folder]/[image]
        # ((cv_w, cv_h), (save_w,save_h), (fx, bx), (fy, by))
        param = self._transParam(xml_src_info.height, xml_src_info.width)
        xml_dist_info.width, xml_dist_info.height = param[1]
        for i,one_bndbox in enumerate(xml_dist_info.objs):
            one_bndbox.xmin = int(one_bndbox.xmin*param[2][0]+param[2][1])
            one_bndbox.ymin = int(one_bndbox.ymin*param[3][0]+param[3][1])
            one_bndbox.xmax = int(one_bndbox.xmax*param[2][0]+param[2][1])
            one_bndbox.ymax = int(one_bndbox.ymax*param[3][0]+param[3][1])
        #must check bndboxes range
        #must be in [1,w-1] and [1,h-1]
        flag = adjustBndbox(xml_dist_info)
        return flag!=-1,xml_dist_info

    def xmlRecover(self,xml_dist_info, im_src_shape, path_replace = None):
        vocxml_info = xml_dist_info.copy()
        if path_replace is not None:
            vocxml_info.path = path_replace
            vocxml_info.folder = os.path.split(os.path.split(path_replace)[0])[1]#[part_of_path]/[folder]/[image]
        vocxml_info.width, vocxml_info.height = self.im_src_shape[1], self.im_src_shape[0]
        for i,one_bndbox in enumerate(vocxml_info.objs):
            one_bndbox.xmin = int((one_bndbox.xmin-param[2][1])/param[2][0])
            one_bndbox.ymin = int((one_bndbox.ymin-param[3][1])/param[3][0])
            one_bndbox.xmax = int((one_bndbox.xmax-param[2][1])/param[2][0])
            one_bndbox.ymax = int((one_bndbox.ymax-param[3][1])/param[3][0])
        #must check bndboxes range
        #must be in [1,w-1] and [1,h-1]
        flag = adjustBndbox(vocxml_info)
        return flag!=-1,vocxml_info

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
            out_dets_list = [[x[0],x[1],float(x[2]),int(x[3]),int(x[4]),int(x[5]),int(x[6])] for x in pbBase.scan_text(dets_file)]
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
            anno_name_list = pbBase.scan_file(annotations_path,'.xml',False,True)
            recs = {}
            print('Reading annotations... ')
            p_vocXmlIO = vocXmlIO([os.path.join(annotations_path,x) for x in anno_name_list],False,3000)
            for anno_name in tqdm(anno_name_list):
                #recs[anno_name[:-4]] = vocXmlRead(os.path.join(annotations_path,anno_name))
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



class bndbox():
    def __init__(self, _xmin, _ymin, _xmax, _ymax):
        if _xmax + 1 <_xmin or _ymax + 1< _ymin:
            print('init bndbox Error : ')
            raise ValueError('xmax < xmin or ymax < ymin')
        self.xmin = int(_xmin)
        self.ymin = int(_ymin)
        self.xmax = int(_xmax)
        self.ymax = int(_ymax)

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
            raise IndexError('cvRect bndbox out of range')

    def tocvRect(self):
        return pbimg.cvRect(self.xmin,self.ymin,self.xmax-self.xmin+1,self.ymax-self.ymin+1)

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
            inters = bndboxExIntersect(det,trk)
            uni = a1 + a2 - inters
            iou_matrix[i1,i2] = inters/uni
    return iou_matrix
    
    
#format='name.jpg' 'class' 'xmin' 'ymin' 'xmax' 'ymax'
#HAVEEXT: whether have file extension name
#ONECLASS: if 'detFile' have no 'class', you can set 'class' by this parameter
def detToXml(detFile,imgPath,savePath,HAVEEXT=True,ONECLASS=''):
    imgFolder = os.path.split(imgPath)[1]
    detFileInfo = pbBase.scan_text(detFile)

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
            adjustBndbox(p_vocXml)
            vocXmlWrite(xmlSavePath,p_vocXml)
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
            adjustBndbox(p_vocXml)
            vocXmlWrite(xmlSavePath,p_vocXml)



if __name__=='__main__':
    ##################################################################
    #test module
    ##################################################################

    import sys
    sys.path.append('../')
    import pyBoost as pb

    def test_adjustBndbox():
        read_path = r'E:\fusion\frcnn_hand'
        xml_file_name = pb.scan_file_r(read_path,'.xml',False,True)
        for one in xml_file_name:
            xml_info = pb.voc.vocXmlRead(os.path.join(read_path,one))
            flag = pb.voc.adjustBndbox(xml_info)
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


    #####################################################################
    save_folder_name = 'pyBoost_test_output'
    test_adjustBndbox()
    #test_vocEvaluator(save_folder_name)
    #test_detToXml(save_folder_name)

