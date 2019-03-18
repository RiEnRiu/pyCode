import sys
sys.path.append('../Common')
from pyBoost import *
import os
import cv2
import argparse
import json
import multiprocessing
from tqdm import tqdm


def resizefun(param):
    args = param[0]
    read_path = param[1]
    save_path = param[2]
    wh = param[3]
    save_root, save_name = os.path.split(save_path)
    if(os.path.isdir(save_root) is not True):
        os.makedirs(save_root)
    vocRsr = imResizer(wh,args.rtype,args.inter)
    img = cv2.imread(read_path,cv2.IMREAD_UNCHANGED)
    dst_img = vocRsr.imResize(img)
    cv2.imwrite(save_path,dst_img)
    #print(one_pair[0]+'  -->  ['+str(dst_img.shape[1])+','+str(dst_img.shape[0])+']')
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str,required=True, help = 'Where is the img data.')
    parser.add_argument('--save', type = str,  help = 'Where to save. default is to cover the source files')
    parser.add_argument('--rtype', type = int, default = 1, help = '[0 = stretch.] [1 = round up.] [2 = round up and crop] [3 = round down] [4 = round down and fill black] [5 = round down and fill self], \"1\"')
    parser.add_argument('--wh', type = str,required=True, help = '[width,height] such as \'[720,480]\' for resized images')

    #parser.add_argument('--width', type = int, required=True, help = 'Width for saved images')
    #parser.add_argument('--height', type = int, required=True, help = 'Height for saved images')
    parser.add_argument('--inter', default = 3,  type = int, help = '[0 = INTER_NEAREST] [1 = INTER_LINEAR] [2 = INTER_CUBIC] [3 = INTER_AREA] [4 = INTER_LANCZOS4], \"3\"')
    parser.add_argument('--recursion',type = int,default = 1, help = 'whether scan file with recursion, \"1\"')
    parser.add_argument('--process',default=multiprocessing.cpu_count()-1,type=int,help='use how many process, \"core-1\"')
    args = parser.parse_args()

    if(args.save is None):
        print('You are going to cover the source files. Continue? [Y/N]?')
        str_in = input()
        if(str_in !='Y' and str_in !='y'):
            sys.exit('user quit')
    else:  
        if(os.path.isdir(args.save) is not True):
            os.makedirs(args.save)

    if(args.recursion!=0):
        file_relative = scanner.file_r(args.dir,'.jpg.png.jpeg',False)
    else:
        file_relative = scanner.file(args.dir,'.jpg.png.jpeg',False)

    if(args.save is None):
        read_img_list = [os.path.join(args.dir,x) for x in file_relative]
        save_img_list = read_img_list
    else:
        read_img_list = [os.path.join(args.dir,x) for x in file_relative]
        save_img_list = [os.path.join(args.save,x) for x in file_relative]

    param_list = list(zip([args]*len(read_img_list),\
                            read_img_list,\
                            save_img_list),\
                            [tuple(json.loads(args.wh))]*len(read_img_list))

    if(args.process<=1 or int(multiprocessing.cpu_count())-1<=1):
        num_process =1
    elif(len(param_list)<min(int(multiprocessing.cpu_count())-1,args.process)):
        num_process = len[param_list]
    else:
        num_process = min(int(multiprocessing.cpu_count())-1,args.process)

    if(num_process == 1):
        for one_param in tqdm(param_list):
        #for one_param in param_list:
            resizefun(one_param)
    else:
        pool = multiprocessing.Pool(num_process) 
        for x in tqdm(pool.imap_unordered(resizefun,param)):
        #for x in pool.imap_unordered(resizefun,param):
            pass
        pool.close()
        pool.join()  
    
