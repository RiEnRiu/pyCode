import sys
sys.path.append('../Common')
from pyBoost import *
from vocBoost import *
import os
import cv2
import argparse
import json
from tqdm import tqdm

import multiprocessing

def resizeCOVERfun(param):
    args = param[0]
    one_pair = param[1]
    wh = param[2]
    vocRsr = vocResizer(wh,args.rtype,args.inter)
    img = cv2.imread(one_pair[0],cv2.IMREAD_UNCHANGED)
    dst_img = vocRsr.imResize(img)
    cv2.imwrite(one_pair[0],dst_img)
    vocRsr.vocXmlResizeToDisk(one_pair[1],one_pair[1])
    return

def resizefun(param):
    args = param[0]
    one_pair=param[1]
    wh = param[2]
    vocRsr = vocResizer(wh,args.rtype,args.inter)
    _t_paht,img_full_name = os.path.split(one_pair[0])
    _t_paht,xml_full_name = os.path.split(one_pair[1])
    img = cv2.imread(one_pair[0], cv2.IMREAD_UNCHANGED)
    dst_img = vocRsr.imResize(img)
    img_new_path = os.path.join(args.save,'JPEGImages',img_full_name)
    cv2.imwrite(img_new_path,dst_img)
    vocRsr.vocXmlResizeToDisk(one_pair[1],os.path.join(args.save,'Annotations',xml_full_name),img_new_path)
    return



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str,required=True, help = 'Where is the VOC data.')
    parser.add_argument('--save', type = str,  help = 'Where to save. default is to cover the source files.')
    parser.add_argument('--rtype', type = int, default=1, help = '[0 = stretch.] [1 = round up.] [2 = round up and crop] [3 = round down] [4 = round down and fill black] [5 = round down and fill self], \"1\"')
    parser.add_argument('--wh', type = str, help = '[width,height] such as \'[720,480]\' for resized images')

    #parser.add_argument('--width', type = int, required=True, help = 'Width for saved images')
    #parser.add_argument('--height', type = int, required=True, help = 'Height for saved images')
    parser.add_argument('--inter', default = 3,  type = int, help = '[0 = INTER_NEAREST] [1 = INTER_LINEAR] [2 = INTER_CUBIC] [3 = INTER_AREA] [4 = INTER_LANCZOS4], \"3\"')
    parser.add_argument('--process',default=multiprocessing.cpu_count()-1,type=int,help='use how many process, \"core-1\"')
    args = parser.parse_args()

    if(args.save is None):
        print('You are going to cover the source files. Continue? [Y/N]?')
        str_in = input()
        if(str_in !='Y' and str_in !='y'):
            sys.exit('user quit')
    else:  
        if(os.path.isfile(args.save)):
            sys.exit('Can not create save path '+args.save)
        if(os.path.isdir(os.path.join(args.save,'JPEGImages')) is not True):
            os.makedirs(os.path.join(args.save,'JPEGImages'))
        if(os.path.isdir(os.path.join(args.save,'Annotations')) is not True):
            os.makedirs(os.path.join(args.save,'Annotations'))        

    pairs ,others1,others2 = scanner.pair(os.path.join(args.dir,'JPEGImages'),\
    os.path.join(args.dir,'Annotations'),'.jpg.jpeg','.xml',True)

    if(args.process<=1 or int(multiprocessing.cpu_count())-1<=1):
        num_process =1
    elif(len(pairs)<min(int(multiprocessing.cpu_count())-1,args.process)):
        num_process = len(pairs)
    else:
        num_process = min(int(multiprocessing.cpu_count())-1,args.process)

    if(args.save is None):
        if(num_process == 1):
            for one_pair in tqdm(pairs):
            #for one_pair in pairs:
                resizeCOVERfun((args,one_pair,json.loads(args.wh)))
        else:
            param = list(zip([args]*len(pairs), pairs, json.loads(args.wh)))
            pool = multiprocessing.Pool(num_process) 
            for x in tqdm( pool.imap_unordered(resizeCOVERfun,param)):
            #for x in pool.imap_unordered(resizeCOVERfun,param):
                pass
            pool.close()
            pool.join()            
    else:
        if(num_process == 1):
            for one_pair in tqdm(pairs):
            #for one_pair in pairs:
                resizefun((args,one_pair, json.loads(args.wh)))
        else:
            param = list(zip([args]*len(pairs), pairs, json.loads(args.wh)))
            pool = multiprocessing.Pool(num_process) 
            for x in tqdm( pool.imap_unordered(resizefun,param)):
            #for x in pool.imap_unordered(resizefun,param):
                pass
            pool.close()
            pool.join() 
    
