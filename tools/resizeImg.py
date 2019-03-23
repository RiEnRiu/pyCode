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
    parser.add_argument('--dir', type = str,required=True, help = 'Where is the images?')
    parser.add_argument('--wh', type = str,required=True, help = '[width,height] such as \'[720,480]\' for resized images')
    parser.add_argument('--rtype', type = int, default = 1, help = '[0 = stretch.] [1 = round up.] [2 = round up and crop] [3 = round down] [4 = round down and fill black] [5 = round down and fill self], \"1\"')
    parser.add_argument('--inter', default = 1,  type = int, help = '[0 = INTER_NEAREST] [1 = INTER_LINEAR] [2 = INTER_CUBIC] [3 = INTER_AREA] [4 = INTER_LANCZOS4], \"1\"')
    parser.add_argument('--save', type = str,  help = 'Where to save. \"$dir$\"')
    parser.add_argument('-q','--quiet',nargs='?',default='NOT_MENTIONED',help='Sure to cover the source file? \"OFF\"')
    parser.add_argument('-r','--recursion',nargs='?',default='NOT_MENTIONED',help='Whether scan file with recursion? \"OFF\"')
    args = parser.parse_args()

    print('Warning: Repeated encoding of images can lead to loss of information.')
    if args.save is None and args.quiet =='NOT_MENTIONED':
        print('You are going to cover the source files. Continue? [Y/N]?')
        str_in = input()
        if(str_in !='Y' and str_in !='y'):
            sys.exit('user quit')

    if args.recursion=='NOT_MENTIONED':
        file_relative = pb.scan_file(args.dir, '.jpg.png.jpeg', False,True)  
    else:
        pb.scan_file_r(args.dir, '.jpg.png.jpeg', False,True)

    if(args.save is None):
        read_img_list = [os.path.join(args.dir,x) for x in file_relative]
        save_img_list = read_img_list
    else:
        read_img_list = [os.path.join(args.dir,x) for x in file_relative]
        save_img_list = [os.path.join(args.save,x) for x in file_relative]

    im_rszr = pb.img.imResizer(args.rtype,json.loads(args.wh),args.inter)
    for read_path,save_path in tqdm.tqdm(zip(read_img_list,save_img_list)):
        img = cv2.imread(read_path,cv2.IMREAD_UNCHANGED)
        rimg = im_rszr.imResize(img)
        cv2.imwrite(save_path,rimg)
    print('Have resized {0} images.'.format(len(read_img_list)))

    
