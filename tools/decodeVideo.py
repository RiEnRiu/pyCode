#-*-coding:utf-8-*-
import sys
import os
__pyBoost_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__pyBoost_root_path)
import pyBoost as pb

import cv2
import argparse
import json
import multiprocessing

def decode_one_video(param):
    video_path = param[0]
    save_dir = param[1]
    begin_index = param[2]
    ext = param[3]
    img_resizer_param = param[4]
    if img_resizer_param is None:
        img_resizer = None
    else:
        rtype = img_resizer_param['rtype']
        dsize = img_resizer_param['dsize']
        interpolation = img_resizer_param['interpolation']
        img_resizer = pb.img.imResizer(rtype,dsize,interpolation)

    video_full_name = os.path.basename(video_path)
    video_front_name,_ = pb.splitext(video_full_name)
    save_full_name = video_front_name+'_{0}'+ext
    save_full_dir = os.path.join(save_dir,video_front_name)
    pb.makedirs(save_full_dir)
    common_save_path = os.path.join(save_full_dir,save_full_name)
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        index = begin_index
        ret,img = cap.read()
        while ret:
            save_path = common_save_path.format(index)
            if img_resizer is not None:
                img = img_resizer.imResize(img)
            cv2.imwrite(save_path,img)
            ret,img = cap.read()
            index += 1
        print('{0}  -->  save {1} ({2}~{3}) frames.'.format(video_full_name,index-begin_index,begin_index,index-1))
    else:
        print(video_full_name + '  -->  open fail.')
    return 



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type = str,required=True, help = '[video file] or [video dir].')

    parser.add_argument('--save', type = str,default='', help = 'Where to save. \"$video$\"')
    parser.add_argument('--first', default=10001, type=int, help='the fisrt index, \"10001\"')
    parser.add_argument('--ext', default = '.jpg', choices = ['.jpg', '.png'], type = str, help = '\".jpg\"')

    parser.add_argument('--rtype', type = int, default=1, help = '[0 = stretch.] [1 = round up.] [2 = round up and crop] [3 = round down] [4 = round down and fill black] [5 = round down and fill self], "1"')
    parser.add_argument('--wh', type = str, help = '[width,height] such as \'[720,480]\' for resized images')
    parser.add_argument('--inter', default = 3, type = int, help = '[0 = INTER_NEAREST] [1 = INTER_LINEAR] [2 = INTER_CUBIC] [3 = INTER_AREA] [4 = INTER_LANCZOS4], \"3\"')

    parser.add_argument('--process',default=int(multiprocessing.cpu_count())-1,type=int,help='use how many process, \"$core$-1\"')
    args = parser.parse_args()

    multi_param = []

    #get video list
    if os.path.isfile(args.video) and pb.splitext(args.video)[1]!='':
        cap = cv2.VideoCapture(args.video)
        if cap.isOpened() and cap.read()[0]:
            cap.release()
            multi_param.append([args.video])
    else:
        all_files = pb.scan_file(args.video,None,True,True)
        for one_file in all_files:
            # if pb.splitext(one_file)[1]!='':
            ext = pb.splitext(one_file)[1]
            if ext=='.avi' or ext=='.mp4' or ext=='.AVI' or ext=='.MP4':
                cap = cv2.VideoCapture(one_file)
                if cap.isOpened() and cap.read()[0]:
                    cap.release()
                    multi_param.append([one_file])
   
    # save dir
    if args.save=='':
        for param in multi_param:
            param.append(os.path.dirname(param[0]))
    else:
        for param in multi_param:
            param.append(args.save)

    # begin index
    for param in multi_param:
        param.append(args.first)

    # ext
    for param in multi_param:
        param.append(args.ext)

    # img_resizer_params
    one_param = None if args.wh is None else\
                {'rtyep':args.rtype,\
                     'dsize':json.loads(args.wh),\
                     'interpolation':args.inter}
    for param in multi_param:
        param.append(one_param)

    # multiprocessing
    num_process = min(args.process, int(multiprocessing.cpu_count())-1)
    num_process = min(len(multi_param),num_process)
    if num_process<=1:
        decode_one_video(param_pack[0])
    else:
        pool = multiprocessing.Pool(num_process) 
        for x in pool.imap_unordered(decode_one_video, multi_param):
            pass
        pool.close()
        pool.join()
        