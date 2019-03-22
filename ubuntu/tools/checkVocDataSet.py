import sys
import os
__pyBoost_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__pyBoost_root_path)
import pyBoost as pb


import argparse
import tqdm
import shutil
import cv2

def check_voc(voc_root_path,to_check_size):
    #read voc data list
    jpeg_path = os.path.join(voc_root_path,'JPEGImages')
    anno_path = os.path.join(voc_root_path,'Annotations')
    pairs, others_in_jpeg, others_in_anno = pb.scan_pair(jpeg_path,anno_path,'.jpg.jpeg','.xml',True,True)

    #make move path and move other files  
    move_jpeg_dir = os.path.join(voc_root_path,'others','JPEGImages')
    move_anno_dir = os.path.join(voc_root_path,'others','Annotations')
    pb.makedirs(move_jpeg_dir)
    pb.makedirs(move_anno_dir)
    for x in others_in_jpeg:
        shutil.move(x,move_jpeg_dir)
    for x in others_in_anno:
        shutil.move(x,move_anno_dir)
    for f in pb.scan_folder(jpeg_path):
        shutil.move(os.path.join(jpeg_path,f),move_jpeg_dir)
    for f in pb.scan_folder(anno_path):
        shutil.move(os.path.join(anno_path,f),move_anno_dir)

    #adjust bndbox
    bad_img_size_list = []
    for img_path, xml_path in tqdm.tqdm(pairs,ncols=55):
        #read xml
        xml = pb.voc.vocXmlRead(xml_path)
        #check whether size matched
        if to_check_size:
            img_shape = cv2.imread(img_path).shape
            if img_shape[0]!=xml.height or img_shape[1]!=xml.width:
                bad_img_size_list.append((img_shape, xml, xml_path))
                continue
            elif len(img_shape)==3 and img_shape[2]!=xml.depth:
                bad_img_size_list.append((img_shape, xml, xml_path))
                continue
        #check voc
        no_change = pb.voc.adjustBndbox(xml)==0
        #move no obj or save changed
        if len(xml.objs)==0:#no obj
            shutil.move(img_path, move_jpeg_dir)
            shutil.move(xml_path, move_anno_dir)
        elif no_change==False:#is changed
            pb.voc.vocXmlWrite(xml_path,xml)            

    #rmdir if is empty
    try:
        os.rmdir(move_jpeg_dir)
    except Exception as e:
        pass
    try:
        os.rmdir(move_anno_dir)
    except Exception as e:
        pass
    try:
        os.rmdir(args.move)
    except Exception as e:
        pass


    #print bad size images
    if len(bad_img_size_list)!=0:
        print('There are {0} images\' size not matched, such as:'.format(len(bad_img_size_list)))
        for img_shape, xml, xml_path in bad_img_size_list[:3]:
            print('{0}  img={1} vs xml=({2}, {3}, {4}).'.format(\
                    os.path.basename(xml_path),\
                    img_shape,\
                    xml.height, xml.width, xml.depth))
        print('...')
    return len(bad_img_size_list)!=0

def countVoc(voc_root_path):
    jpeg_path = os.path.join(voc_root_path,'JPEGImages')
    anno_path = os.path.join(voc_root_path,'Annotations')
    pairs, others_in_jpeg, others_in_anno = pb.scan_pair(jpeg_path,anno_path,'.jpg.jpeg','.xml',True,True)
    with open(os.path.join(voc_root_path, 'countVoc.txt'),'w') as fp:
        fp.write('******************************************************\n')
        fp.write('{0:<40} = {1:>6}\n'.format('effective data', len(pairs)))
        count = 0
        obj_dict = {}
        for img_path,xml_path in pairs:
            for obj in pb.voc.vocXmlRead(xml_path).objs:
                if obj_dict.get(obj.name) is None:
                    obj_dict[obj.name] = 1
                else:
                    obj_dict[obj.name] += 1
                count += 1
        fp.write('{0:<40} = {1:>6}\n'.format('effective bndbox', count))
        fp.write('******************************************************\n')
        label_list = list(obj_dict.keys())
        label_list.sort()
        for label in label_list:
            fp.write('{0:<40} = {1:>6}\n'.format(label, obj_dict[label]))
    return 

def cutVoc(voc_root_path,rate=None):
    jpeg_path = os.path.join(voc_root_path,'JPEGImages')
    anno_path = os.path.join(voc_root_path,'Annotations')
    pairs, others_in_jpeg, others_in_anno = pb.scan_pair(jpeg_path,anno_path,'.jpg.jpeg','.xml',True,True)
    save_root_path = os.path.join(voc_root_path,'cutVoc')
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


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str,required=True, help = 'Where is the VOC data set?')
    parser.add_argument('-q','--quiet',nargs='?',default='NOT_MENTIONED',help='Sure to cover the source file? \"OFF\"')
    parser.add_argument('-f','--fast',nargs='?',default='NOT_MENTIONED',help='Skip checking image size to be faster. \"OFF\"')
    parser.add_argument('-c','--cut',nargs='?',default='NOT_MENTIONED',help = 'Whether cut objects into \"$dir"/cut\", Set value to enlarge bndbox.')
    args = parser.parse_args()

    #quiet mode
    if args.quiet=='NOT_MENTIONED':
        print('You are going to cover the source files. Continue? [Y/N]?')
        str_in = input()
        if str_in !='Y' and str_in !='y':
            sys.exit('user quit')

    have_size_error = check_voc(args.dir, args.fast=='NOT_MENTIONED')
    if have_size_error:
        print('deal with the size errors and check again.')
    else:
        countVoc(args.dir)
        if args.cut != 'NOT_MENTIONED':
            cutVoc(args.dir, args.cut)

    


          