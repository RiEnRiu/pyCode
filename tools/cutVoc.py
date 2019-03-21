import sys
import os
__pyBoost_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__pyBoost_root_path)
import pyBoost as pb

import argparse
import tqdm
import cv2




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str,required=True, help = 'Where is the VOC data set?')
    parser.add_argument('--save',type=str,help = 'Where files will be saved? \"$dir$/cutVoc"')
    parser.add_argument('--broaden',type=float,help = 'How many magnification of length and width? \"[1]\"')
    args = parser.parse_args()

    # check args
    if args.broaden is not None and args.broaden<=0:
        raise ValueError('broaden <= 0')
    rate = None if args.broaden is None else (args.broaden-1.0)/2.0 

    #read voc data list
    jpeg_path = os.path.join(args.dir,'JPEGImages')
    anno_path = os.path.join(args.dir,'Annotations')
    pairs, others_in_jpeg, others_in_anno = pb.scan_pair(jpeg_path,anno_path,'.jpg.jpeg','.xml',True,True)

    #print others
    if len(others_in_jpeg)!=0:
        print('There are {0} unmatched files in \"$dir$/JPEGImages, such as:'.format(len(others_in_jpeg)))
        for x in others_in_jpeg[:3]:
            print(x)
    if len(others_in_anno)!=0:
        print('There are {0} unmatched files in \"$dir$/Annotations, such as:'.format(len(others_in_anno)))
        for x in others_in_anno[:3]:
            print(x)
    folders_in_jpeg = pb.scan_folder(jpeg_path)
    folders_in_anno = pb.scan_folder(anno_path)
    if len(folders_in_jpeg)!=0:
        print('There are {0} folders in \"$dir$/JPEGImages, such as:'.format(len(folders_in_jpeg)))
        for x in folders_in_jpeg[:3]:
            print(x)
    if len(folders_in_anno)!=0:
        print('There are {0} folders in \"$dir$/Annotations, such as:'.format(len(folders_in_anno)))
        for x in folders_in_anno[:3]:
            print(x)

    #make save dir
    save_root_path = os.path.join(args.dir,'cutVoc') if args.save is None else args.save
    
    #TODO: to speed up 
    save_path_dict = {}
    for img_path, xml_path in tqdm.tqdm(pairs,ncols=55):
        #read
        xml = pb.voc.vocXmlRead(xml_path)
        if len(xml.objs)==0:
            continue
        img = cv2.imread(img_path)
        #cut and save
        objs_index = 10000
        for obj in xml.objs:
            obj_save_path = save_path_dict.get(obj.name)
            if obj_save_path is None:
                obj_save_path = os.path.join(save_root_path,obj.name)
                pb.makedirs(obj_save_path)
                save_path_dict[obj.name] = obj_save_path
            #broaden bndbox
            if rate is not None:
                obj_w = obj.xmax+1-obj.xmin
                obj_h = obj.ymax+1-obj.ymin
                obj.xmin = max(0,int(obj.xmin-obj_w*rate))
                obj.ymin = max(0,int(obj.ymin-obj_h*rate))
                obj.xmax = min(img.shape[1]-1,int(obj.xmax+obj_w*rate))
                obj.ymax = min(img.shape[0]-1,int(obj.ymax+obj_h*rate))
            img_obj = img[obj.ymin:obj.ymax+1,obj.xmin:obj.xmax+1]
            save_full_name = '{0}_{1}.jpg'.format(os.path.basename(xml_path)[:-4], objs_index)
            obj_save_full_path = os.path.join(obj_save_path, save_full_name)
            if cv2.imwrite(obj_save_full_path, img_obj)==False:
                print('Fail to save {0} to {1}'.format(os.path.basename(img_path), obj_save_full_path))
            objs_index += 1