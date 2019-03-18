import sys
sys.path.append('../Common')
from pyBoost import *
import shutil

import cv2

b_list = scanner.list(r'D:\dataset\egohand\egohands_data\b2.txt')

bnd_dict = {}

for one_line in b_list:
    get_list = bnd_dict.get(one_line[0])
    if(get_list==None):
        lll = ['hand']+[int(x) for x in one_line[2:]]
        lll[3] = lll[3]+lll[1]
        lll[4] = lll[4]+lll[2]
        one_obj_dict = vocXmlIO().getDefaultObj()
        one_obj_dict['name'] = 'hand'+one_line[1]
        one_obj_dict['xmin'] = lll[1]
        one_obj_dict['ymin'] = lll[2]
        one_obj_dict['xmax'] = lll[3]
        one_obj_dict['ymax'] = lll[4]
        bnd_dict[one_line[0]] = [one_obj_dict]
    else:
        lll = ['hand']+[int(x) for x in one_line[2:]]
        lll[3] = lll[3]+lll[1]
        lll[4] = lll[4]+lll[2]
        one_obj_dict = vocXmlIO().getDefaultObj()
        one_obj_dict['name'] = 'hand'+one_line[1]
        one_obj_dict['xmin'] = lll[1]
        one_obj_dict['ymin'] = lll[2]
        one_obj_dict['xmax'] = lll[3]
        one_obj_dict['ymax'] = lll[4]
        get_list.append(one_obj_dict)
        bnd_dict[one_line[0]] = get_list

print(len(bnd_dict))



for key,values in  bnd_dict.items():
    # print('D:/dataset/egohand/voc_big_10/JPEGImages/'+key+'.jpg')
    # input()
    parts = key.split('/')

    img_shape = cv2.imread('D:/dataset/egohand/egohands_data/_LABELLED_SAMPLES/'+parts[0]+'/'+parts[1]+'.jpg').shape
    vxml = vocXmlIO()
    vxml.folder = 'JPEGImages'
    vxml.filename = parts[0]+'_'+parts[1]+'.jpg'
    vxml.path = 'D:/dataset/egohand/voc_big_10/JPEGImages/'+parts[0]+'_'+parts[1]+'.jpg'    
    vxml.database = 'Unknown'
    vxml.width = img_shape[1]
    vxml.height = img_shape[0]
    vxml.depth = 3
    vxml.segmented = '0'
    # vxml.objs = values
    vxml.objs = []
    for one_value in values:
        hand_type = one_value['name']
        one_value['name'] = 'hand'
        one_value['xmin'] = max(one_value['xmin'],0)
        one_value['ymin'] = max(one_value['ymin'],0)
        one_value['xmax'] = min(one_value['xmax'],img_shape[1])
        one_value['ymax'] = min(one_value['ymax'],img_shape[0])
        if((hand_type=='hand1' or hand_type=='hand2') and \
            one_value['xmax']-one_value['xmin']>img_shape[1]*0.15 and \
            one_value['ymax']-one_value['ymin']>img_shape[0]*0.15):
            vxml.objs.append(one_value)
        if((hand_type=='hand3' or hand_type=='hand4') and \
            one_value['xmax']-one_value['xmin']>img_shape[1]*0.05 and \
            one_value['ymax']-one_value['ymin']>img_shape[0]*0.05):
            vxml.objs.append(one_value)
        # if(img_shape[1]*0.05<one_value['xmin'] and one_value['xmin']<img_shape[1]*0.95 and \
        #     img_shape[1]*0.05<one_value['xmax'] and one_value['xmax']<img_shape[1]*0.95 and \
        #     img_shape[0]*0.05<one_value['ymin'] and one_value['ymin']<img_shape[0]*0.95 and \
        #     img_shape[0]*0.05<one_value['ymax'] and one_value['ymax']<img_shape[0]*0.95 and \
        #     one_value['xmax']-one_value['xmin']>img_shape[1]*0.05 and \
        #     one_value['ymax']-one_value['ymin']>img_shape[0]*0.05):
        #     vxml.objs.append(one_value)
        # elif(one_value['xmax']-one_value['xmin']>img_shape[1]*0.1 and \
        #     one_value['ymax']-one_value['ymin']>img_shape[0]*0.1):
        #     vxml.objs.append(one_value)
    if(len(vxml.objs)==0):
        continue

    # print(parts)
    # input()
    shutil.copy('D:/dataset/egohand/egohands_data/_LABELLED_SAMPLES/'+parts[0]+'/'+parts[1]+'.jpg',\
        'D:/dataset/egohand/voc_big_10/JPEGImages/'+parts[0]+'_'+parts[1]+'.jpg')




    vxml.write('D:/dataset/egohand/voc_big_10/Annotations/'+parts[0]+'_'+parts[1]+'.xml')

