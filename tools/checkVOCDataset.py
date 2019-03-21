import sys
import os
__pyBoost_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__pyBoost_root_path)
import pyBoost as pb


import argparse
import tqdm
import json
import shutil
import cv2

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str,required=True, help = 'Where is the VOC data.')
    parser.add_argument('--move',type=str,help = 'where the other files move to? \"$dir$/others"')
    parser.add_argument('-f','--fast',nargs='?',default='NOT_MENTIONED',help='skip checking image size to be faster. \"OFF\"')
    parser.add_argument('-q','--quiet',nargs='?',default='NOT_MENTIONED',help='sure to cover the source file. \"OFF\"')
    args = parser.parse_args()

    #quiet mode
    if args.quiet=='NOT_MENTIONED':
        print('You are going to cover the source files. Continue? [Y/N]?')
        str_in = input()
        if str_in !='Y' and str_in !='y':
            sys.exit('user quit')

    #read voc data list
    jpeg_path = os.path.join(args.dir,'JPEGImages')
    anno_path = os.path.join(args.dir,'Annotations')
    pairs, others_in_jpeg, others_in_anno = pb.scan_pair(jpeg_path,anno_path,'.jpg.jpeg','.xml',True,True)

    #make move path and move other files  
    if args.move is None:
        args.move = os.path.join(args.dir,'others')
    move_jpeg_dir = os.path.join(args.move,'JPEGImages')
    move_anno_dir = os.path.join(args.move,'Annotations')
    pb.makedirs(move_jpeg_dir)
    pb.makedirs(move_anno_dir)
    for x in others_in_jpeg:
        shutil.move(x,move_jpeg_dir)
    for x in others_in_anno:
        shutil.move(x,move_anno_dir)
    for f in pb.scan_folder(jpeg_path):
        shutil.move(os.path.join(jpeg_path,f),move_jpeg_dir)
    for f in pb.scan_folder(anno_path):
        shutil.move(os.path.join(anno_path,f),move_anno_dir)

    #TODO: to speed up 
    bad_img_size_list = []
    is_check_img_size = args.fast== 'NOT_MENTIONED'
    for img_path, xml_path in tqdm.tqdm(pairs,ncols=55):
        #read xml
        xml = pb.voc.vocXmlRead(xml_path)
        #check whether size matched
        if is_check_img_size:
            img_shape = cv2.imread(img_path).shape
            if img_shape[0]!=xml.height or img_shape[1]!=xml.width:
                bad_img_size_list.append((img_shape, xml, xml_path))
                continue
            elif len(img_shape)==3 and img_shape[2]!=xml.depth:
                bad_img_size_list.append((img_shape, xml, xml_path))
                continue

        #check voc
        no_change = pb.voc.adjustBndbox(xml)==0
        #move no obj or save changed
        if len(xml.objs)==0:#no obj
            shutil.move(img_path, move_jpeg_dir)
            shutil.move(xml_path, move_anno_dir)
        elif no_change==False:#is changed
            pb.voc.vocXmlWrite(xml_path,xml)            

    #print bad size images
    if len(bad_img_size_list)!=0:
        print('There are {0} images\' size not matched, such as:'.format(len(bad_img_size_list)))
        for img_shape, xml, xml_path in bad_img_size_list[:3]:
            print('{0}  img={1} vs xml=({2}, {3}, {4}).'.format(\
                    os.path.basename(xml_path),\
                    img_shape,\
                    xml.height, xml.width, xml.depth))

    #rmdir if is empty
    try:
        os.rmdir(move_jpeg_dir)
    except Exception as e:
        pass
    try:
        os.rmdir(move_anno_dir)
    except Exception as e:
        pass
    try:
        os.rmdir(args.move)
    except Exception as e:
        pass


          