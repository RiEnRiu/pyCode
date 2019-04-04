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

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str,required=True, help = 'Where is the voc date set?')
    parser.add_argument('--wh', type = str,required=True, help = '[width,height] such as \'[720,480]\' for resized images')
    parser.add_argument('--rtype', type = int, default = 1, help = '[0 = stretch.] [1 = round up.] [2 = round up and crop] [3 = round down] [4 = round down and fill black] [5 = round down and fill self], \"1\"')
    parser.add_argument('--inter', default = 1,  type = int, help = '[0 = INTER_NEAREST] [1 = INTER_LINEAR] [2 = INTER_CUBIC] [3 = INTER_AREA] [4 = INTER_LANCZOS4], \"1\"')
    parser.add_argument('--save', type = str,  help = 'Where to save. \"$dir$\"')
    parser.add_argument('-q','--quiet',nargs='?',default='NOT_MENTIONED',help='Sure to cover the source file? \"OFF\"')
    args = parser.parse_args()

    print('Warning: Repeated encoding of images can lead to loss of information.')
    if args.save is None and args.quiet =='NOT_MENTIONED':
        print('You are going to cover the source files. Continue? [Y/N]?')
        str_in = input()
        if(str_in !='Y' and str_in !='y'):
            sys.exit('user quit')

    jpg_dir = os.path.join(args.dir,'JPGEImages')
    xml_dir = os.path.join(args.dir,'Annotations')

    jpg_relative, xml_relative, o1 ,o2 = pb.scan_pair(jpg_dir,xml_dir,'.jpg.jpeg','.xml',False,True)

    if(args.save is None):
        read_jpg_list = [os.path.join(jpg_dir,x) for x in jpg_relative]
        read_xml_list = [os.path.join(xml_dir,x) for x in xml_relative]
        save_jpg_list = read_jpg_list
        save_xml_list = read_xml_list
    else:
        read_jpg_list = [os.path.join(jpg_dir,x) for x in jpg_relative]
        read_xml_list = [os.path.join(xml_dir,x) for x in xml_relative]
        save_jpg_dir = os.path.join(jpg_dir,'JPGEImages')
        save_xml_dir = os.path.join(xml_dir,'Annotations')
        pb.makedirs(save_jpg_dir)
        pb.makedirs(save_xml_dir)
        save_jpg_list = [os.path.join(save_jpg_dir,x) for x in jpg_relative]
        save_xml_list = [os.path.join(save_xml_dir,x) for x in xml_relative]

    voc_rszr = pb.voc.vocResizer(args.rtype,json.loads(args.wh),args.inter)
    for imgrp,xmlrp,imgsp,xmlsp in \
            tqdm.tqdm(zip(read_jpg_list,read_xml_list,save_jpg_list,save_xml_list)):
        img = cv2.imread(imgrps,cv2.IMREAD_UNCHANGED)
        xml = pb.voc.xml_read(xmlrp)
        rimg = voc_rszr.imResize(img)
        rxml = voc_rszr.xmlResize(img)
        cv2.imwrite(imgsp,rimg)
        pb.voc.xml_write(xmlsp,rxml)

    
