import sys
import os
__pyBoost_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__pyBoost_root_path)
import pyBoost as pb

import argparse
import tqdm


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str,required=True, help = 'Where is the VOC data set?')
    args = parser.parse_args()

    #read voc data list
    jpeg_path = os.path.join(args.dir,'JPEGImages')
    anno_path = os.path.join(args.dir,'Annotations')
    pairs, others_in_jpeg, others_in_anno = pb.scan_pair(jpeg_path,anno_path,'.jpg.jpeg','.xml',True,True)

    with open(os.path.join(args.dir, 'countVoc.txt'),'w') as fp:
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
