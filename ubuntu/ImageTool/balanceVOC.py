
import sys
sys.path.append('../Common')
from pyBoost import *
import numpy as np
import xml.etree.ElementTree as ET

import math
import os

import argparse
import shutil

import datetime


def check_xml(xmlPath,labelCount):

    et = ET.parse(xmlPath)
    element = et.getroot()
    element_objs = element.findall('object')

    for element_obj in element_objs:
        node = element_obj.find('name')
        label=node.text
        got = labelCount.get(label)
        if(got is None):
            labelCount[label] = 1
        else:
            labelCount[label] = got + 1

    return labelCount


def countLabel(path,label=None):

    pairs,other1,other2 = scanner.pair(os.path.join(path,'JPEGImages'),os.path.join(path,'Annotations'),'.jpg.jpeg','.xml',True)

    label_dict = {}
    sample_size = len(pairs)
    label_size = 0
    A = [[]]*sample_size
    others_dict = {}
    sample_name = []
    #no appointing label
    if(label is None):
        for i, pair in enumerate(pairs):
            element_objs = ET.parse(pair[1]).getroot().findall('object')
            this_img_objs_count = [0]*label_size
            for element_obj in element_objs:
                label_name=element_obj.find('name').text
                got = label_dict.get(label_name)
                if(got is None):
                    this_img_objs_count += [1]
                    label_dict[label_name] = label_size
                    label_size +=1
                else:
                    this_img_objs_count[got]+=1
            A[i] = this_img_objs_count
            _t_path,_full_name = os.path.split(pair[1])
            _t_sample_name,_ext = os.path.splitext(_full_name)
            sample_name.append([_t_sample_name])
    else:
        #havs appointing labels
        label_size = len(label)
        for i,one_label in enumerate(label):
            label_dict[one_label] = i
        for i, pair in enumerate(pairs):
            element_objs = ET.parse(pair[1]).getroot().findall('object')
            this_img_objs_count = [0]*label_size
            for element_obj in element_objs:
                label_name=element_obj.find('name').text
                got = label_dict.get(label_name)
                if(got is None):
                    this_img_objs_count += [1]
                    others_dict[label_name] = label_size
                    label_size +=1
                else:
                    this_img_objs_count[got]+=1
            A[i] = this_img_objs_count
            _t_path,_full_name = os.path.split(pair[1])
            _t_sample_name,_ext = os.path.splitext(_full_name)
            sample_name.append([_t_sample_name])
    #make list as a matrix
    for i,this_img_objs_count in  enumerate(A):
        if(len(this_img_objs_count)!=label_size):
            A[i] = [0]*label_size
            A[i][0:len(this_img_objs_count)] = this_img_objs_count
    return A,sample_name,label_dict,others_dict
          

def __score_label(np_count):
    count_expand = np.array([np_count,list(range(len(np_count)))])
    sorted_count_expand = count_expand[:,count_expand[0].argsort()]
    
def balanceVOC_greedy(A,std_var_thresh):
    np_A_t = np.array(A).transpose()
    np_solution = np.array([[1]]*len(A))
    np_obj_count = np.matmul(np_A_t,np_solution)
    while(math.sqrt(np_obj_count.var())>std_var_thresh):
        label_score = (np_obj_count - np_obj_count.sum()/np_obj_count.size)/np_obj_count.var()
        sample_score = np.matmul(label_score.transpose(),np_A_t).transpose()
        sample_score[np_solution==0] = 0
        max_id = sample_score.argmax()
        np_solution[max_id] = 0
        np_obj_count = np.matmul(np_A_t,np_solution)
    return np_solution.tolist()
 

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str,required=True, help = 'Where is the VOC data.')
    parser.add_argument('--save',type = str,required=True, help = 'Where to save the solution.')
    parser.add_argument('--thresh', type = int, default=10, help = 'Threshold to stop balance, \"10\"')
    parser.add_argument('--move',type = int ,default = 0,help = 'Whether move files, \"0\"')
    args = parser.parse_args()

    begin = datetime.datetime.now()
    A,sample_name,label_dict,others_dict = countLabel(args.dir)
    #A,sample_name,label_dict,others_dict = countLabel(r'G:\fusion_2018-08-20')
    end = datetime.datetime.now()
    print('time for statistics = '+str((end - begin).total_seconds()))
    #input()

    begin = datetime.datetime.now()
    solution = balanceVOC_greedy(A,args.thresh)
    end = datetime.datetime.now()
    print('time for balance = '+str((end - begin).total_seconds()))

    label_name_list = list(label_dict.keys())

    #create dirs
    if(os.path.isdir(args.save) is not True):
        os.makedirs(args.save)

    #save solution
    line0 = [['names']+label_name_list+['solution']]
    linex = [x+y+z for x,y,z in zip(np.array(sample_name,dtype=np.str).tolist(),np.array(A,dtype=np.str).tolist(),np.array(solution,dtype=np.str).tolist())]    
    mat2save = line0+linex
    np.savetxt(os.path.join(args.save,'solu.txt'),mat2save,'%s')
   
    #save report
    with open(os.path.join(args.save,'report.txt'),'w') as fd:
        npA = np.array(A)
        count = npA.sum(0)
        fd.write('----Before Balance----\n')
        fd.write('min: '+str(count.min())+'\n')
        fd.write('max: '+str(count.max())+'\n')
        fd.write('nonZeroClass: '+str(np.count_nonzero(count))+'\n')
        fd.write('allClass: '+str(len(count))+'\n')
        fd.write('number: '+str(npA.shape[0])+'\n')
        for i in range(len(count)):
            fd.write(label_name_list[i]+': '+str(count[i])+'\n')
        
        fd.write('\n')
        
        new_count = np.matmul(npA.transpose(),np.array(solution))
        fd.write('----After Balance----\n')
        fd.write('min: '+str(new_count.min())+'\n')
        fd.write('max: '+str(new_count.max())+'\n')
        fd.write('nonZeroClass: '+str(np.count_nonzero(new_count))+'\n')
        fd.write('allClass: '+str(len(new_count))+'\n')      
        fd.write('number: '+str(solution.count([1]))+'\n')
        for i in range(len(new_count)):
            fd.write(label_name_list[i]+': '+str(new_count[i][0])+'\n')

    #move files
    if(args.move!=0):
        src_jpg_path =  os.path.join(args.dir,'JPEGImages')
        src_xml_path =  os.path.join(args.dir,'Annotations')
        jpg_path  = os.path.join(args.save,'JPEGImages')
        xml_path = os.path.join(args.save,'Annotations')
        if(os.path.isdir(jpg_path)is not True):
            os.makedirs(jpg_path)
        if(os.path.isdir(jpg_path)is not True):
            sys.exit('Can not create path: '+jpg_path)
        if(os.path.isdir(xml_path)is not True):
            os.makedirs(xml_path)
        if(os.path.isdir(xml_path)is not True):
            sys.exit('Can not create path: '+xml_path)
        for i in range(len(solution)):
            if(solution[i][0] is 0):
                shutil.move(os.path.join(src_jpg_path,sample_name[i][0]+'.jpg'),jpg_path)
                shutil.move(os.path.join(src_xml_path,sample_name[i][0]+'.xml'),xml_path)
