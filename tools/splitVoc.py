#-*-coding:utf-8-*-
import sys
import os
__pyBoost_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__pyBoost_root_path)
import pyBoost as pb
import json
import argparse
import random

def __write_split_file(save_path, name_list):
    with open(save_path, 'w') as fp:
        # can not make newline in VOC split file 
        for x in name_list[:-1]:
            fp.write(x+'\n')
        if len(name_list)!=0:
            fp.write(name_list[-1])
    return 

def split_voc_data_set(dir, train, val, test=None):
    # read list
    pairs,o1,o2 = pb.scan_pair(os.path.join(dir,'JPEGImages'),\
                                os.path.join(dir,'Annotations'),\
                                '.jpg.jpeg.png',\
                                '.xml',\
                                with_root_dir=False, \
                                with_ext=True)
    names = [x[1][:-4] for x in pairs]
    # random split
    random.shuffle(names)
    len_names = len(names)
    len_train = int(len_names*train)
    len_val = int(len_names*val)
    len_test = len_names - len_train - len_val
    train_list = names[:len_train]
    val_list = names[len_train:len_train+len_val]
    trainval_list = names[:len_train+len_val]
    test_list = names[len_train+len_val:]

    # write file
    save_dir = os.path.join(dir,'ImageSets','Main')
    pb.makedirs(save_dir)
    __write_split_file(os.path.join(save_dir,'trainval.txt'),trainval_list)
    __write_split_file(os.path.join(save_dir,'train.txt'),train_list)
    __write_split_file(os.path.join(save_dir,'val.txt'),val_list)
    __write_split_file(os.path.join(save_dir,'test.txt'),test_list)

    return len_train,len_val,len_test



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str,required=True, help = 'Where is the VOC data set?')
    parser.add_argument('--ratio', type = str,required=True, help = 'Set ratio as (train,val,test) such as (0.8,0.1,0.1)')
    args = parser.parse_args()

    ratio = json.loads(args.ratio)
    split_voc_data_set(args.dir, ratio[0],ratio[1], 1-ratio[0]-ratio[1])
          