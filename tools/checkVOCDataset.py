import sys
import os
__pyBoost_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__pyBoost_root_path)

import pyBoost as pb


import argparse
from tqdm import tqdm
import json
import shutil


def read(img_path,xml_path,is_check_imgsize):
    if is_check_imgsize:
        return cv2.imread(img_path).shape,pb.voc.vocXmlRead(xml_path),is_check_imgsize
    else:
        return None, pb.voc.vocXmlRead(xml_path),is_check_imgsize

def check(imgshape,xml,xml_path,is_check_imgsize):
    good_imgsize, no_change, no_obj = True, True, True
    if is_check_imgsize:
        good_imgsize = img_shape[0]!=org_xml.height or img_shape[1]!=org_xml.width
        if good_imgsize:
            no_change = pb.voc.adjustBndbox(xml)==0
            no_obj = len(xml.objs)==0
    else:
        no_change = pb.voc.adjustBndbox(xml)==0
        no_obj = len(xml.objs)==0
    return good_imgsize, no_change, no_obj, xml, xml_path

def save_or_move(good_imgsize, no_change, no_obj, xml, xml_path):
    good_size_error, io_error = True, True
    if good_imgsize==False:
        pass
    elif no_change:
        good_size_error, io_error = False,False
    else:
        io_error = pb.voc.vocXmlWrite(xml_path)==True
    return good_size_error, io_error, xml_path

    




if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str,required=True, help = 'Where is the VOC data.')
    parser.add_argument('--move',type=str,help = 'where the other files move to? \"$dir$/others"')
    parser.add_argument('-f','--fast',nargs='?',default='NOT_MENTIONED',help='skip checking image size to be faster, \"OFF\"')
    parser.add_argument('-q','--quiet',nargs='?',default='NOT_MENTIONED',help='sure to cover the source file."OFF"')
    args = parser.parse_args()

    if args.quiet=='NOT_MENTIONED':
        print('You are going to cover the source files. Continue? [Y/N]?')
        str_in = input()
        if str_in !='Y' and str_in !='y':
            sys.exit('user quit')

    jpegPath = os.path.join(args.dir,'JPEGImages')
    annoPath = os.path.join(args.dir,'Annotations')

    if(args.move is not None):
        moveJpgDir = os.path.join(args.move,'JPEGImages')
        moveXmlDir = os.path.join(args.move,'Annotations')
    else:
        moveJpgDir = os.path.join(args.dir,'others','JPEGImages')
        moveXmlDir = os.path.join(args.dir,'others','Annotations')

    pairs, others_in_jpg, others_in_xml = scanner.pair(jpegPath,annoPath,'.jpg.jpeg','.xml',False)

    if(is_check[0]!=0):
        for x in others_in_jpg:
            print('Other file in: JPEGImages/'+x)
            shutil.move(os.path.join(jpegPath,x),moveJpgDir) 
        for x in others_in_xml:
            print('Other file in: Annotations/'+x)
            shutil.move(os.path.join(annoPath,x),moveXmlDir)

    #for i,x in tqdm(enumerate(pairs),ncols=50):
    #    org_xml = vocXmlRead(os.path.join(annoPath,x[1]))
    #    if(is_check[1]!=0):
    #        img_shape = cv2.imread(os.path.join(jpegPath,x[0])).shape
    #        if(img_shape[0]!=org_xml.height or img_shape[1]!=org_xml.width):
    #            print("xml width or height error: "+pairs[i][1])
    #            break
    #    if(is_check[2]!=0):
    #        flag = adjustBndbox(org_xml)
    #        if(len(org_xml.objs)==0):
    #            print('There is no object after adjust bounding box: '+pairs[i][1])
    #            if(os.path.isdir(moveJpgDir) is not True):
    #                os.makedirs(moveJpgDir)
    #            if(os.path.isdir(moveXmlDir) is not True):
    #                os.makedirs(moveXmlDir)  
    #            shutil.move(os.path.join(jpegPath,pairs[i][0]), moveJpgDir)
    #            shutil.move(os.path.join(annoPath,pairs[i][1]), moveXmlDir)
    #        elif(flag == -1):
    #            print('some bounding boxes were removed in: '+pairs[i][1])
    #            break
    #        elif(flag == 1):
    #            #Using vocXmlIo to save may be slower, because it will sleep long white FPS is not smooth
    #            vocXmlWrite(os.path.join(annoPath,x[1]),org_xml)
    #        else:#flag == 0
    #            pass 
