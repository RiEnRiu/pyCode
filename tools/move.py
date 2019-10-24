#-*-coding:utf-8-*-
import sys
import os
__pyBoost_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__pyBoost_root_path)
import pyBoost as pb

import random

import time
import cv2
import shutil
import argparse
import tqdm

   
def make_move_list(name_format,begin,num):
    name_list = [name_format.format(i) for i in range(begin,num+begin)]
    name_set = set()
    for x in name_list:
        if x in name_set:
            raise ValueError('Same file name: {0}'.format(os.path.basename(x)))
        name_set.add(x)
    return name_list
    

def move_files(source_dir,begin,args_format,args_quiet):
    save_dir, name_format = os.path.split(args.format)
    if save_dir=='' and args_quiet:
        print('You are going to cover the source files. Continue? [Y/N]?')
        str_in = input()
        if(str_in !='Y' and str_in !='y'):
            sys.exit('user quit')
    save_dir = source_dir if save_dir=='' else save_dir
    pb.makedirs(save_dir)
    # scan
    files = pb.scan_file(source_dir,with_root_dir=False,with_ext=True)
    name_list = make_move_list(name_format,begin,len(files))
    if len(name_list)==0:
        return
    # move to tmp file
    tmp_save_path_list = []
    file_ext_list = []
    for file_full_name in files:
        file_front_name, file_ext = pb.splitext(file_full_name)
        read_path = os.path.join(source_dir,file_full_name)
        while 1:
            save_path = os.path.join(save_dir,'{0}{1}'.format(str(random.random())[2:],file_ext))
            if not os.path.isfile(save_path):
                break
        tmp_save_path_list.append(save_path)
        file_ext_list.append(file_ext)
        shutil.move(read_path, save_path)   
    # move to dir
    for read_path,file_ext,name in zip(tmp_save_path_list,file_ext_list,name_list):
        save_path = os.path.join(save_dir,'{0}{1}'.format(name,file_ext))
        shutil.move(read_path,save_path)
    try:
        os.rmdir(source_dir)
    except Exception as e:
        pass  
    return 

        
def move_voc(source_dir,begin,args_format,args_quiet):
    save_dir, name_format = os.path.split(args.format)
    if save_dir=='' and args_quiet:
        print('You are going to cover the source voc. Continue? [Y/N]?')
        str_in = input()
        if(str_in !='Y' and str_in !='y'):
            sys.exit('user quit')
    save_dir = source_dir if save_dir=='' else save_dir
    save_jpg_dir = os.path.join(save_dir,'JPEGImages')
    save_xml_dir = os.path.join(save_dir,'Annotations')
    pb.makedirs(save_jpg_dir)
    pb.makedirs(save_xml_dir)
    jpg_dir = os.path.join(source_dir,'JPEGImages')
    xml_dir = os.path.join(source_dir,'Annotations')
    # scan
    relative_pairs, o1 ,o2 = pb.scan_pair(jpg_dir,xml_dir,'.jpg.jpeg.JPG.JPEG','.xml',False,True)
    if len(o1)!=0 or len(o2)!=0:
        print('There are other files not voc type...')
        for x in o1:
            print(x) 
        for x in o2:
            print(x)
    name_list = make_move_list(name_format,begin,len(relative_pairs))
    if len(name_list)==0:
        return
    # move to tmp file
    jpg_save_list = []
    xml_save_list = []
    jpg_ext_list = []
    xml_ext_list = []
    for jpg_full_name,xml_full_name in relative_pairs:
        jpg_ext = pb.splitext(jpg_full_name)[1]
        jpg_read_path = os.path.join(jpg_dir,jpg_full_name)
        while 1:
            jpg_save_path = os.path.join(save_jpg_dir,'{0}{1}'.format(str(random.random())[2:],jpg_ext))
            if not os.path.isfile(jpg_save_path):
                break
        jpg_save_list.append(jpg_save_path)
        jpg_ext_list.append(jpg_ext)
        shutil.move(jpg_read_path,jpg_save_path)

        xml_ext = pb.splitext(xml_full_name)[1]
        xml_read_path = os.path.join(xml_dir,xml_full_name)
        while 1:
            xml_save_path = os.path.join(save_xml_dir,'{0}{1}'.format(str(random.random())[2:],xml_ext))
            if not os.path.isfile(xml_save_path):
                break
        xml_save_list.append(xml_save_path)
        xml_ext_list.append(xml_ext)
        shutil.move(xml_read_path,xml_save_path)
        
    # move to dir
    jpg_save_list = []
    xml_save_list = []
    jpg_ext_list = []
    xml_ext_list = []
    for jpg_read_path,xml_read_path,jpg_ext,xml_ext,name in zip(jpg_save_list,\
                                                                xml_save_list,\
                                                                jpg_ext_list,\
                                                                xml_ext_list):
        jpg_save_path = os.path.join(save_jpg_dir,'{0}{1}'.format(name,jpg_ext))
        shutil.move(jpg_read_path,jpg_save_path)

        xml_save_path = os.path.join(save_xml_dir,'{0}{1}'.format(name,xml_ext))
        xml = pb.voc.xml_read(xml_read_path)
        xml.filename = name
        pb.voc.xml_write(xml_save_path,xml)
        os.remove(xml_read_path)
    # move to save_dir
    try:         
        os.rmdir(jpg_dir)
    except Exception as e:
        pass   
    try:         
        os.rmdir(xml_dir)
    except Exception as e:
        pass  
    try:         
        os.rmdir(source_dir)
    except Exception as e:
        pass  
    return 



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dir', type = str,required=True, help = 'Files or VOC directory.')
    parser.add_argument('-f','--format', type = str,required=True, help = 'name format, please use \"\", \"PATH/format\"(copy) or \"format\"(cover).')
    parser.add_argument('-b','--begin', type = int,required=True, default=10000, help='begin index. \"10000\"')
    parser.add_argument('-v','--voc',nargs='?',default='NOT_MENTIONED',help='Is it voc date set in $dir$? \"OFF\"')
    parser.add_argument('-q','--quiet',nargs='?',default='NOT_MENTIONED',help='Sure to cover the source file? \"OFF\"')
    args = parser.parse_args()
    
    try:
        args.format.format(1)
    except Exception as e:
        sys.exit('-f,--format, please use \"\"')
    if args.voc=='NOT_MENTIONED':
        print('Move files in: {0}'.format(args.dir))
        print('Save in: {0}'.format(os.path.join(args.dir,'{0}.*'.format(args.format))))
        move_files(args.dir,args.begin,args.format,args.quiet)
    else:
        print('Move an VOC data set in: {0}'.format(args.dir))
        print('Save in: {0}'.format(os.path.join(args.dir,'*','{0}.*'.format(args.format))))
        move_voc(args.dir,args.begin,args.format,args.quiet)



    
