import sys
sys.path.append('../Common')
from pyBoost import *
from vocBoost import *
import os
import argparse

from tqdm import tqdm

import multiprocessing

import xml.etree.ElementTree as ET

def rename_xml(xml_src,filename):
    #filename
    xml_front_path,xml_old_name  = os.path.split(xml_src)
    xml_dst = os.path.join(xml_front_path,filename)

    #path
    img_path = xml_root.find('path').text
    img_front_path,img_name  = os.path.split(img_path)
    img_path = os.path.join(img_front_path,filename)

    read_xml = vocXmlRead(xml_src)
    read_xml.filename = filename
    read_xml.path = img_path
    vocXmlWrite(xml_dst,read_xml)

    #read_xml = vocXmlIO(xml_src)
    #read_xml.filename = filename
    #read_xml.path = img_path
    #read_xml.write(xml_dst)
    return True


    #xml_tree = ET.parse(xml_src) 
    #xml_root = xml_tree.getroot()

    ##filename
    #xml_front_path,xml_old_name  = os.path.split(xml_src)
    #xml_dst = os.path.join(xml_front_path,filename)

    ##path
    #img_path = xml_root.find('path').text
    #img_front_path,img_name  = os.path.split(img_path)
    #img_path = os.path.join(img_front_path,filename)

    ##width height
    #width =  xml_root.find('size').find('width').text
    #height = xml_root.find('size').find('height').text
    #depth = xml_root.find('size').find('depth').text

    ##obj
    #label=[]
    #tlbr = []
    #xml_obj_all = xml_root.findall('object')
    #for xml_obj_one in xml_obj_all:
    #    #label name
    #    label.append(xml_obj_one.find('name').text)
    #    tlbr.append((xml_obj_one.find('bndbox').find('xmin').text,\
    #        xml_obj_one.find('bndbox').find('ymin').text,\
    #        xml_obj_one.find('bndbox').find('xmax').text,\
    #        xml_obj_one.find('bndbox').find('ymax').text))

    #if(os.path.isfile(xml_src)):
    #    os.remove(xml_src)
    #if(os.path.isfile(xml_dst)):
    #    os.remove(xml_dst)

    #xml_file = open(xml_dst,'w')
    #xml_file.write('<?xml version=\"1.0\" encoding=\"utf-8\"?>'+'\n')
    #xml_file.write('<annotation>'+'\n')
    #xml_file.write('    <folder>JPEGImages</folder>'+'\n')
    #xml_file.write('    <filename>'+filename+'</filename>'+'\n')
    #xml_file.write('    <path>' + img_path + '</path>' + '\n')
    #xml_file.write('    <source>' + '\n')
    #xml_file.write('        <database>Unknown</database>'  + '\n')
    #xml_file.write('    </source>' + '\n')
    #xml_file.write('    <size>' + '\n')
    #xml_file.write('        <width>' +width+ '</width>' + '\n')
    #xml_file.write('        <height>'  +height+ '</height>' + '\n')
    #xml_file.write('        <depth>' +depth+  '</depth>' + '\n')
    #xml_file.write('    </size>' + '\n')
    #xml_file.write('    <segmented>0</segmented>' + '\n')
    #for  i in range(len(label)):
    #    xml_file.write('    <object>' + '\n')
    #    xml_file.write('        <name>'+label[i]+ '</name>'  + '\n')
    #    xml_file.write('        <pose>Unspecified</pose>' + '\n')
    #    xml_file.write('        <truncated>0</truncated>' + '\n')
    #    xml_file.write('        <difficult>0</difficult>'  + '\n')
    #    xml_file.write('        <bndbox>' + '\n')
    #    xml_file.write('            <xmin>'+tlbr[i][0]+'</xmin>'  + '\n')
    #    xml_file.write('            <ymin>'+tlbr[i][1]+ '</ymin>' + '\n')
    #    xml_file.write('            <xmax>'+tlbr[i][2]+ '</xmax>' + '\n')
    #    xml_file.write('            <ymax>'+tlbr[i][3]+ '</ymax>' + '\n')
    #    xml_file.write('        </bndbox>'  + '\n')
    #    xml_file.write('    </object>' + '\n')
    #xml_file.write('</annotation>' + '\n')
    #xml_file.close()
        
    #return True

   
def getRenameList(pairs,prefix,first,suffix):
    src_name_lower=['']*len(pairs)
    src_name_lower_set = []
    src_name_lower_dist = {}
    for i,one_pair in enumerate(pairs):
        front_path,file_full_name = os.path.split(one_pair[0])
        file_front_name,file_ext = os.path.splitext(file_full_name)
        src_name_lower[i] = file_front_name.lower()
        src_name_lower_dist[src_name_lower[i]] = i
    src_name_lower_set = set(src_name_lower)

    #print('make dist pairs')
    dist_pairs = [[]]*len(pairs)
    dist_name = ['']*len(pairs)
    dist_name_lower = ['']*len(pairs)
    dist_name_lower_set = []
    dist_name_lower_dist = {}
    for i,name_numer in enumerate(range(first,first+len(pairs))):
        file_front_name = prefix+str(name_numer)+suffix
        dist_name[i] = file_front_name
        dist_pairs[i] = (os.path.join(img_front_path,file_front_name+'.jpg'),os.path.join(xml_front_path,file_front_name+'.xml'))
        dist_name_lower[i] = file_front_name.lower()
        dist_name_lower_dist[dist_name_lower[i]] = i
    dist_name_lower_set = set(dist_name_lower)

    src_and_dist =  dist_name_lower_set & src_name_lower_set
    src_rest = src_name_lower_set - src_and_dist
    disr_rest = dist_name_lower_set - src_and_dist

    name_pairs = list(zip(src_rest,disr_rest))
    rename_param = []
    for one_name_pairs in name_pairs:
        index1 = src_name_lower_dist[one_name_pairs[0]]
        index2 = dist_name_lower_dist[one_name_pairs[1]]
        rename_param.append((pairs[index1],\
        dist_pairs[index2],dist_name[index2]))
    return rename_param


def renamefun(param):
    src_pairs = param[0]
    dist_pairs=param[1]
    file_front_name = param[2]

    os.rename(src_pairs[0],dist_pairs[0])
    rename_xml(src_pairs[1],file_front_name+'.xml')

    return

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str,required=True, help = 'Where is the VOC data.')
    parser.add_argument('--prefix',type = str,required=True, help = 'name = prefix + number + suffix + .ext')
    parser.add_argument('--first',default=10001,type=int,help='the fisrt index, \"10001\"')
    parser.add_argument('--suffix',type = str,required=True, help = 'name = prefix + number + suffix + .ext')
    parser.add_argument('--process',default=multiprocessing.cpu_count()-1,type=int,help='use how many process, \"core-1\"')
    args = parser.parse_args()

    print('You are going to rename the files, such as to \''+args.prefix+str(args.first)+args.suffix+'.jpg\'.\nContinue? [Y/N]?')
    str_in = input()

    if(str_in != 'Y'and str_in != 'y'):
        sys.exit('user quit')

    img_front_path = os.path.join(args.dir,'JPEGImages')
    xml_front_path = os.path.join(args.dir,'Annotations')

    pairs,t_others1,t_others2 = scanner.pair(img_front_path,xml_front_path,'.jpg.png.jpeg','.xml',True)

    rename_param = getRenameList(pairs,args.prefix,str(args.first),args.suffix)

    if(args.process<=1 or int(multiprocessing.cpu_count())-1<=1):
        num_process =1
    elif(len(rename_param)<min(int(multiprocessing.cpu_count())-1,args.process)):
        num_process = max(len(rename_param),1)
    else:
        num_process = min(int(multiprocessing.cpu_count())-1,args.process)

    if(num_process == 1):
        for one_rename_param in tqdm(rename_param):
        #for one_pair in pairs:
            renamefun(one_rename_param)
    else:
        pool = multiprocessing.Pool(num_process) 
        for x in tqdm(pool.imap_unordered(renamefun,rename_param)):
        #for x in pool.imap_unordered(resizefun,param):
            pass
        pool.close()
        pool.join() 
    
