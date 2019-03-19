#import sys
#import os
#__pyBoost_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.append(__pyBoost_root_path)
#print('msg')
#print(__file__)
#print(__pyBoost_root_path)


#import pyBoost as pb

#import argparse
#from tqdm import tqdm
#import json
#import shutil

#def checkfun(param):
#    pairs = param
#    img_shape = cv2.imread(pairs[0]).shape
#    org_xml = vocXmlRead(pairs[1])
#    if(img_shape[0]!=org_xml.height or img_shape[1]!=org_xml.width):
#        print("xml weidth or height error: "+pairs[0])
#        return
#    flag = adjustBndbox(org_xml)
#    if(len(org_xml.objs)==0):
#        print('There is no object after resized: '+pairs[1])
#    if(flag == -1):
#        #print('There is no object after resized: '+pairs[1])
#        return vocXmlWrite(pairs[1],org_xml)
#    elif(flag == 1):
#        #print('Have adjust bndboxes in: '+pairs[1])
#        return vocXmlWrite(pairs[1],org_xml)
#    else:
#        return 

#    #list_to_del = []
#    #for i,one_bndbox in enumerate(org_xml.objs):
        
#    #    #check
#    #    if(one_bndbox.xmin<0):
#    #        print('['+str(i)+'] xmin error: '+str(one_bndbox.xmin)+'  '+pairs[0])
#    #        one_bndbox.xmin = 0
#    #    if(one_bndbox.ymin<0):
#    #        print('['+str(i)+'] ymin error: '+str(one_bndbox.ymin)+'  '+pairs[0])
#    #        one_bndbox.ymin = 0
#    #    if(one_bndbox.xmax>=img_shape[1]):
#    #        print('['+str(i)+'] xmax error: '+str(one_bndbox.xmax)+'  '+str(img_shape[1]-1)+'  '+pairs[0])
#    #        one_bndbox.xmax = img_shape[1]-1
#    #    if(one_bndbox.ymax>=img_shape[0]):
#    #        print('['+str(i)+'] ymax error: '+str(one_bndbox.ymax)+'  '+str(img_shape[0]-1)+'  '+pairs[0])
#    #        one_bndbox.ymax = img_shape[0]-1
#    #    if(one_bndbox.xmin>=one_bndbox.xmax or one_bndbox.ymin>=one_bndbox.ymax):
#    #        print('['+str(i)+'] too small error: '+pairs[0])
#    #        list_to_del.append(i)
#    #for i in reversed(list_to_del):
#    #    temp = org_xml.objs.pop(i)
#    #if(org_xml.objs==[]):
#    #    print('No objs in xml: '+pairs[0])
#    #    #sys.exit()
#    ##vocXmlWrite(pairs[1],org_xml)
#    return


#if __name__=='__main__':

#    parser = argparse.ArgumentParser('move other files, check image size in xml, adjust bndox at edge')
#    parser.add_argument('--dir', type = str,required=True, help = 'Where is the VOC data.')
#    parser.add_argument('--check',type=str,default="[1,1,1]",help='Check what? [scan other files, image size, bndbox at edge], \"[1,1,1]\"')
#    parser.add_argument('--move',type=str,help = 'where the other files move to? \"$dir$/others"')
#    #parser.add_argument('-q','-quiet',help='sure to cover the source file.')
#    args = parser.parse_args()


#    print('You are going to cover the source files. Continue? [Y/N]?')
#    str_in = input()
#    if(str_in !='Y' and str_in !='y'):
#        sys.exit('user quit')

#    is_check = json.loads(args.check)
#    jpegPath = os.path.join(args.dir,'JPEGImages')
#    annoPath = os.path.join(args.dir,'Annotations')

#    if(args.move is not None):
#        moveJpgDir = os.path.join(args.move,'JPEGImages')
#        moveXmlDir = os.path.join(args.move,'Annotations')
#    else:
#        moveJpgDir = os.path.join(args.dir,'others','JPEGImages')
#        moveXmlDir = os.path.join(args.dir,'others','Annotations')

#    pairs, others_in_jpg, others_in_xml = scanner.pair(jpegPath,annoPath,'.jpg.jpeg','.xml',False)

#    if(is_check[0]!=0):
#        for x in others_in_jpg:
#            print('Other file in: JPEGImages/'+x)
#            shutil.move(os.path.join(jpegPath,x),moveJpgDir) 
#        for x in others_in_xml:
#            print('Other file in: Annotations/'+x)
#            shutil.move(os.path.join(annoPath,x),moveXmlDir)

#    #if(is_check[1]!=0):
#    #    p_imgIO = imgIO([os.path.join(jpegPath,x[0]) for x in pairs],False,1000)
#    #else:
#    #    p_imgIO =None
#    #p_xmlIO = vocXmlIO([os.path.join(annoPath,x[1]) for x in pairs],False,1000)

#    for i,x in tqdm(enumerate(pairs),ncols=50):
#        org_xml = vocXmlRead(os.path.join(annoPath,x[1]))
#        #org_xml = p_xmlIO.read()
#        if(is_check[1]!=0):
#            img_shape = cv2.imread(os.path.join(jpegPath,x[0])).shape
#            #img_shape = p_imgIO.read().shape
#            if(img_shape[0]!=org_xml.height or img_shape[1]!=org_xml.width):
#                print("xml width or height error: "+pairs[i][1])
#                break
#        if(is_check[2]!=0):
#            flag = adjustBndbox(org_xml)
#            if(len(org_xml.objs)==0):
#                print('There is no object after adjust bounding box: '+pairs[i][1])
#                if(os.path.isdir(moveJpgDir) is not True):
#                    os.makedirs(moveJpgDir)
#                if(os.path.isdir(moveXmlDir) is not True):
#                    os.makedirs(moveXmlDir)  
#                shutil.move(os.path.join(jpegPath,pairs[i][0]), moveJpgDir)
#                shutil.move(os.path.join(annoPath,pairs[i][1]), moveXmlDir)
#            elif(flag == -1):
#                print('some bounding boxes were removed in: '+pairs[i][1])
#                break
#            elif(flag == 1):
#                #Using vocXmlIo to save may be slower, because it will sleep long white FPS is not smooth
#                vocXmlWrite(os.path.join(annoPath,x[1]),org_xml)
#            else:#flag == 0
#                pass 
