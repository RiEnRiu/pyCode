#-*-coding:utf-8-*-
import sys
import os
__pyBoost_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__pyBoost_root_path)
import pyBoost as pb
import json
import argparse
import random
import shutil
import tqdm

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

def move_test_data_set(dir):
    # read test data list
    front_name_set = set(pb.scan_text(os.path.join(dir,'ImageSets','Main','test.txt')))
    if len(front_name_set)==0:
        return
    # make save path
    save_root_path = os.path.join(dir,'test_part')
    save_jpg_path = os.path.join(save_root_path,'JPEGImages')
    save_xml_path = os.path.join(save_root_path,'Annotations')
    pb.makedirs(save_jpg_path)
    pb.makedirs(save_xml_path)
    if os.path.isdir(save_jpg_path)==False or os.path.isdir(save_xml_path)==False:
        print('Move test data to {0} is not allowed'.format(save_root_path))

    # read all voc data list
    jpg_path = os.path.join(dir,'JPEGImages')
    xml_path = os.path.join(dir,'Annotations')
    img_exts = '.jpg.jpeg.png.JPG.JPEG.PNG'
    pairs,o1,o2 = pb.scan_pair(jpg_path, xml_path, img_exts, '.xml',\
                                with_root_dir=False, \
                                with_ext=True)
    pairs_front_name = {pb.splitext(p[0])[0]:p for p in pairs}
    # moving 
    print('Moving test data to {0} ......'.format(save_root_path))
    for front_name in tqdm.tqdm(front_name_set):
        p = pairs_front_name[front_name]
        jpg_full_path = os.path.join(jpg_path,p[0])
        xml_full_path = os.path.join(jpg_path,p[1])
        shutil.move(jpg_full_path,save_jpg_path)
        shutil.move(xml_full_path,save_xml_path)
    return 
    


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str,required=True, help = 'Where is the VOC data set?')
    parser.add_argument('--ratio', type = str,required=True, help = 'Set ratio as [train,val,test] such as [0.8,0.1,0.1]')
    parser.add_argument('-s','--split',nargs='?',default='NOT_MENTIONED',help='Move test data to $dir$\test ? \"OFF\"')
    args = parser.parse_args()

    ratio = json.loads(args.ratio)
    split_voc_data_set(args.dir, ratio[0],ratio[1], 1-ratio[0]-ratio[1])
    if args.split!='NOT_MENTIONED':
        move_test_data_set(args.dir)
            