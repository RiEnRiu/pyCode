import sys
sys.path.append('../Common')
from pyBoost import *
import os
import argparse

from tqdm import tqdm
   
#def renamefun(param):
#    os.rename(param[0],param[1])
#    return

def checkLabel(obj_path,mask_path,label_file_path):
    if(os.path.isfile(label_file_path) is not True):
        print('It is not an valid text file: '+label_file_path)
        return False
    label_from_file_list = list([x[0] for x in scanner.list(label_file_path)])
    label_from_file = set(label_from_file_list)
    if(len(label_from_file)!=len(label_from_file_list)):
        print('There is the same label in file: '+label_file_path)
        return False
    obj_other_file = scanner.file(obj_path)
    if(len(obj_other_file)!=0):
        print('There are redundant files : ')
        print(obj_other_file)
        return False
    
    obj_folder_name = set(scanner.folder(obj_path))
    if(mask_path!=''):
        mask_other_file = scanner.file(mask_path);
        if(len(mask_other_file)!=0):
            print('There are redundant files : ')
            print(mask_other_file)
            return False
        mask_folder_name = set(scanner.folder(mask_path))
        if(len(label_from_file-obj_folder_name)!=0):
            print('Missed label folder in : '+obj_path)
            print(label_from_file-obj_folder_name)
            return False
        if(len(label_from_file-mask_folder_name)!=0):
            print('Missed label folder in : '+mask_path)
            print(label_from_file-mask_folder_name)
            return False
        if(len(obj_folder_name-label_from_file)!=0):
            print('Redundant label folder in : '+obj_path)
            print(obj_folder_name-label_from_file)
            return False
        if(len(mask_folder_name-label_from_file)!=0):
            print('Redundant label folder in : '+mask_path)
            print(mask_folder_name-label_from_file)
            return False
    else:
        if(len(label_from_file-obj_folder_name)!=0):
            print('Missed label folder in : '+obj_path)
            print(label_from_file-obj_folder_name)
            return False
        if(len(obj_folder_name-label_from_file)!=0):
            print('Redundant label folder in : '+obj_path)
            print(obj_folder_name-label_from_file)
            return False
    print('Check label with NO ERROR.')
    return True

def getRenameList(obj_path,mask_path,first):
    out_put = []
    label = scanner.folder(obj_path)
    if(mask_path==''):
        for one_label in label:
            root_path = os.path.join(obj_path,one_label)
            obj_full_name_list = scanner.file(root_path,'.jpg.png.jpeg',False)
            dist_front_name = [one_label+'_object_'+str(first+i) for i in range(len(obj_full_name_list))]
            src_front_name = []
            src_ext = []
            src_dict = {}
            for i,one_obj_full_name in enumerate(obj_full_name_list):
                one_obj_front_name,one_obj_ext = os.path.splitext(one_obj_full_name)
                src_front_name.append(one_obj_front_name)
                src_ext.append(one_obj_ext)
                src_dict[one_obj_front_name] = i

            src_set = set(src_front_name)
            dist_set = set(dist_front_name)
            rest_src = src_set - dist_set
            rest_dist = dist_set - src_set 

            rename_pairs = list(zip(rest_src,rest_dist))

            for one_rename_pair in rename_pairs:
                one_index = src_dict[one_rename_pair[0]]
                one_ext = src_ext[one_index]
                out_pair = [os.path.join(root_path,obj_full_name_list[one_index]),os.path.join(root_path,one_rename_pair[1]+one_ext)]
                out_put.append(out_pair)

    else:
        for one_label in label:

            obj_root_path = os.path.join(obj_path,one_label)
            mask_root_path = os.path.join(mask_path,one_label)
            
            pairs_full_name_list,_others1,_others2 = scanner.pair(obj_root_path,mask_root_path,'.jpg.png.jpeg','.png',False)

            print(pairs_full_name_list)

            obj_full_name_list = [x for [x,y] in pairs_full_name_list]
            mask_full_name_list = [y for [x,y] in pairs_full_name_list]



            dist_front_name = [one_label+'_object_'+str(first+i) for i in range(len(pairs_full_name_list))]
            src_front_name = []
            src_ext = []
            src_dict = {}
            for i,one_obj_full_name in enumerate(obj_full_name_list):
                one_obj_front_name,one_obj_ext = os.path.splitext(one_obj_full_name)
                src_front_name.append(one_obj_front_name)
                one_mask_front_name,one_mask_ext = os.path.splitext(mask_full_name_list[i])
                src_ext.append([one_obj_ext,one_mask_ext])
                src_dict[one_obj_front_name] = i

            src_set = set(src_front_name)
            dist_set = set(dist_front_name)
            rest_src = src_set - dist_set
            rest_dist = dist_set - src_set 

            rename_pairs = list(zip(rest_src,rest_dist))

            for one_rename_pair in rename_pairs:
                one_index = src_dict[one_rename_pair[0]]
                one_obj_ext,one_mask_ext = src_ext[one_index]
                out_pair_obj = [os.path.join(obj_root_path,obj_full_name_list[one_index]),os.path.join(obj_root_path,one_rename_pair[1]+one_obj_ext)]
                out_pair_mask = [os.path.join(mask_root_path,mask_full_name_list[one_index]),os.path.join(mask_root_path,one_rename_pair[1]+one_mask_ext)]
                out_put.append(out_pair_obj)
                out_put.append(out_pair_mask)
    return out_put
        
        
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', type = str,required=True, help = 'Where is the obj data.')
    parser.add_argument('--mask',type = str,default='', help = 'Where is the mask data')
    parser.add_argument('--label',type = str,default='', help = 'Where is the label, set it to check the data')
    parser.add_argument('--first',default=10001,type=int,help='the fisrt index. default = 10001')
    args = parser.parse_args()

    print('You are going to rename the object files in: '+args.obj)
    if(args.mask!=""):
        print('You are going to rename the mask files in: '+args.mask+'\n')
    if(args.label!=""):
        print('Check the label set in file: '+args.label+'\n')
    print('Continue? [Y/N]?')
    str_in = input()       
    if(str_in != 'Y'and str_in != 'y'):
        sys.exit('user quit')

    if(args.label!=''):
        if(checkLabel(args.obj,args.mask,args.label)is not True):
            sys.exit()

    rename_param = getRenameList(args.obj,args.mask,args.first)
    
    for one_rename_param in tqdm(rename_param):
        os.rename(one_rename_param[0],one_rename_param[1])
