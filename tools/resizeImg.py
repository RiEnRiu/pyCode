#-*-coding:utf-8-*-
import sys
import os
__pyBoost_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__pyBoost_root_path)
import pyBoost as pb

import cv2
import argparse
import json
import tqdm

def resize_img_only(args):
    print('Warning: Repeated encoding of images can lead to loss of information.')
    if args.save is None and args.quiet =='NOT_MENTIONED':
        print('You are going to cover the source files. Continue? [Y/N]?')
        str_in = input()
        if(str_in !='Y' and str_in !='y'):
            sys.exit('user quit')

    if args.recursion=='NOT_MENTIONED':
        file_relative = pb.scan_file(args.dir, '.jpg.png.jpeg', False,True)  
    else:
        file_relative = pb.deep_scan_file(args.dir, '.jpg.png.jpeg', False,True)

    if(args.save is None):
        read_img_list = [os.path.join(args.dir,x) for x in file_relative]
        save_img_list = read_img_list
    else:
        pb.makedirs(args.save)
        read_img_list = [os.path.join(args.dir,x) for x in file_relative]
        save_img_list = [os.path.join(args.save,x) for x in file_relative]
    
    wh = tuple(json.loads(args.wh))
    im_rszr = pb.img.imResizer(args.rtype,wh,args.inter)
    for read_path,save_path in tqdm.tqdm(zip(read_img_list,save_img_list)):
        img = cv2.imread(read_path,cv2.IMREAD_UNCHANGED)
        rimg = im_rszr.imResize(img)
        cv2.imwrite(save_path,rimg)
    return 

def resize_voc(args):
    print('Warning: Repeated encoding of images can lead to loss of information.')
    if args.save is None and args.quiet =='NOT_MENTIONED':
        print('You are going to cover the source files. Continue? [Y/N]?')
        str_in = input()
        if(str_in !='Y' and str_in !='y'):
            sys.exit('user quit')

    jpg_dir = os.path.join(args.dir,'JPEGImages')
    xml_dir = os.path.join(args.dir,'Annotations')

    relative_pairs, o1 ,o2 = pb.scan_pair(jpg_dir,xml_dir,'.jpg.jpeg','.xml',False,True)

    if(args.save is None):
        read_jpg_list = [os.path.join(jpg_dir,x[0]) for x in relative_pairs]
        read_xml_list = [os.path.join(xml_dir,x[1]) for x in relative_pairs]
        save_jpg_list = read_jpg_list
        save_xml_list = read_xml_list
    else:
        read_jpg_list = [os.path.join(jpg_dir,x[0]) for x in relative_pairs]
        read_xml_list = [os.path.join(xml_dir,x[1]) for x in relative_pairs]
        save_jpg_dir = os.path.join(args.save,'JPEGImages')
        save_xml_dir = os.path.join(args.save,'Annotations')
        pb.makedirs(save_jpg_dir)
        pb.makedirs(save_xml_dir)
        save_jpg_list = [os.path.join(save_jpg_dir,x[0]) for x in relative_pairs]
        save_xml_list = [os.path.join(save_xml_dir,x[1]) for x in relative_pairs]

    wh = tuple(json.loads(args.wh))
    voc_rszr = pb.voc.vocResizer(args.rtype,wh,args.inter)
    for imgrp,xmlrp,imgsp,xmlsp in \
            tqdm.tqdm(zip(read_jpg_list,read_xml_list,save_jpg_list,save_xml_list)):
        img = cv2.imread(imgrp,cv2.IMREAD_UNCHANGED)
        rimg = voc_rszr.imResize(img)
        cv2.imwrite(imgsp,rimg)
        voc_rszr.xmlResizeInDisk(xmlrp,xmlsp)
    return 


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str,required=True, help = 'Where is the images or voc data set?')
    parser.add_argument('--wh', type = str,required=True, help = '[width,height] such as \'[720,480]\' for resized images')
    parser.add_argument('--rtype', type = int, default = 1, help = '[0 = stretch.] [1 = round up.] [2 = round up and crop] [3 = round down] [4 = round down and fill black] [5 = round down and fill self], \"1\"')
    parser.add_argument('--inter', default = 1,  type = int, help = '[0 = INTER_NEAREST] [1 = INTER_LINEAR] [2 = INTER_CUBIC] [3 = INTER_AREA] [4 = INTER_LANCZOS4], \"1\"')
    parser.add_argument('--save', type = str,  help = 'Where to save. \"$dir$\"')
    parser.add_argument('-v','--voc',nargs='?',default='NOT_MENTIONED',help='Is it voc date set in $dir$? \"OFF\"')
    parser.add_argument('-q','--quiet',nargs='?',default='NOT_MENTIONED',help='Sure to cover the source file? \"OFF\"')
    parser.add_argument('-r','--recursion',nargs='?',default='NOT_MENTIONED',help='Scan file with recursion, invalid to voc data set. \"OFF\"')

    args = parser.parse_args()

    if args.voc=='NOT_MENTIONED':
        print('Resizing an images set in: {0}'.format(args.dir))
        print('Save in: {0}'.format(args.dir if args.save is None else args.save))
        resize_img_only(args)
    else:
        print('Resizing an VOC data set in: {0}'.format(args.dir))
        print('Save in: {0}'.format(args.dir if args.save is None else args.save))
        resize_voc(args)



    
