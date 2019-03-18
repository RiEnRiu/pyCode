import cv2
import numpy as np

import argparse
import configparser 
import os
import shutil
import sys


import random
import math
from tqdm import tqdm 

import multiprocessing

from cvAugmentor import *
sys.path.append('../Common')
from pyBoost import *



class AUG_OPERATION:
    def __init__(self):
        self.optName = ''
        self.isParamRandom = False
        self.param = [0,0,0]
        self.outDscrb = []
        self.outNum = 0


class cvAugSolver:

    def __rmParenthesis(self,_input_str):
        index_L_big = _input_str.find('{')
        index_L_mid = _input_str.find('[')
        index_L_sma = _input_str.find('(')

        if(index_L_big == -1 and index_L_mid == -1 and index_L_sma == -1):
            return [_input_str,'']

        index_R_big = _input_str.rfind('}')
        index_R_mid = _input_str.rfind(']')
        index_R_sma = _input_str.rfind(')')

        L_index = 10000
        if(index_L_big!=-1):
            L_index = min(L_index,index_L_big)
        if(index_L_mid!=-1):
            L_index = min(L_index,index_L_mid)
        if(index_L_sma!=-1):
            L_index = min(L_index,index_L_sma)

        R_index = 0
        if(index_R_big!=-1):
            R_index = max(R_index,index_R_big)
        if(index_R_mid!=-1):
            R_index = max(R_index,index_R_mid)
        if(index_R_sma!=-1):
            R_index = max(R_index,index_R_sma)

        return [_input_str[0:L_index],_input_str[(L_index+1):R_index]]

    def __str2data(self,_input_str):
        pre_index = -1
        output = []
        max_i = len(_input_str)-1
        for i,one_char in enumerate(_input_str):
            if(one_char == ' ' or one_char == ',' or one_char == ';'\
                or one_char == '{' or one_char == '[' or one_char == '('\
                or one_char == '}' or one_char == ']' or one_char == ')'):
                output.append(float(_input_str[pre_index+1:i]))
                pre_index = i
            elif(i == max_i):
                output.append(float(_input_str[pre_index+1:]))
        return output
                
    def __toParam(self,_param_str):
        if(_param_str == ''):
            return False,[]
        if(_param_str.find('rand')!=-1):
            str1,str2 = self.__rmParenthesis(_param_str)
            return True,self.__str2data(str2)
        return False,self.__str2data(_param_str)

    def __getStepOpt(self,stepStr):
        lower_str = stepStr.lower()
        opts = []
        opt_list = lower_str.split('+')
        for one_opt_str in opt_list:
            opt = AUG_OPERATION()
            str1,str2 = self.__rmParenthesis(one_opt_str)
            opt.optName = str1
            if(str2!=''):
                opt.isParamRandom,opt.param = self.__toParam(str2)
                if(opt.isParamRandom):
                    opt.outDscrb = [opt.optName+'(RAND)']*int(opt.param[0])
                    opt.outNum = opt.param[0]
                else:
                    for one_param in opt.param:
                        opt.outDscrb.append(opt.optName+'('+str(one_param)+')')
                    opt.outNum = len(opt.param)
            elif(opt.optName!='prestep'):#prestep
                opt.outDscrb.append(opt.optName)
                opt.outNum = 1
            else:#Filp,FilpUD,FilpLR
                opt.outDscrb.append('')
                opt.outNum = 1
            opts.append(opt)
        return opts

    def __getAugConfig(self,config):
        cf = configparser.ConfigParser()
        
        if(cf.read(config) == []):
            return False,{},{}
        
        
        _t_img_process = {}#all string
        read_img_process = cf.items('IMGPROCESS')
        for one_process in read_img_process:
            _t_img_process[one_process[0]] = one_process[1]
        
        _t_aug_config = {}#all string
        read_aug_config = cf.items('AUGCONFIG')
        for one_config in read_aug_config:
            _t_aug_config[one_config[0]] = self.__getStepOpt(one_config[1])

        return True, _t_img_process, _t_aug_config

    def __getReport(self,_t_aug_config):
        _t_opt_gen_num = [[1]]
        _t_step_gen_num = [1]
        _t_acc_step_gen_num = [1]
        _t_sum_acc_step_gen_num = [1]
        for i in range(len(_t_aug_config)):
            step_opt = _t_aug_config['step'+str(i+1)]
            this_step_opt_num = []
            sum_this_step_num = 0
            for one_opt in step_opt:
                this_opt_num = one_opt.outNum
                #if(one_opt.optName == 'prestep'):
                #    this_opt_num = 1
                #elif(one_opt.optName == 'Pyramid'):
                #    this_opt_num = np.array(one_opt.param,np.int).sum()+1
                #elif(one_opt.optName.find('flip')!=-1):
                #    this_opt_num = 1
                #elif(one_opt.isParamRandom):
                #    this_opt_num = int(one_opt.param[0])
                #else:
                #    this_opt_num = len(one_opt.param)
                this_step_opt_num.append(this_opt_num)
                sum_this_step_num+=this_opt_num
            _t_opt_gen_num.append(this_step_opt_num)
            _t_step_gen_num.append(sum_this_step_num)
            _t_acc_step_gen_num.append(_t_acc_step_gen_num[-1]*_t_step_gen_num[-1])
            _t_sum_acc_step_gen_num.append(_t_sum_acc_step_gen_num[-1]+_t_acc_step_gen_num[-1])

        pre_step_outDscrb_list = [[]]
        for i in range(len(_t_aug_config)):
            step_opt = _t_aug_config['step'+str(i+1)]
            this_step_outDscrb_list =[]
            for one_pre_step_outDscrb_list in pre_step_outDscrb_list:
                for one_opt in step_opt:
                    #print(one_opt.outDscrb)
                    for one_outDscrb in one_opt.outDscrb:
                        this_step_outDscrb_list.append(one_pre_step_outDscrb_list+[one_outDscrb])
            pre_step_outDscrb_list = this_step_outDscrb_list

        begin_index_list = [0]+_t_sum_acc_step_gen_num[:-1]
        range_list = [range(int(x),int(y)) for x,y in zip(begin_index_list,_t_sum_acc_step_gen_num)]
        
        return range_list,this_step_outDscrb_list

    def __init__(self,config,first_index):

        self.__first_index = first_index
        #__img_process: step0,can do some image process
        #__aug_config:step1~step_last, read by config
        self.__is_ready ,self.__img_process, self.__aug_config = self.__getAugConfig(config)

        #__step_read_num_range: the index range for reading each step temp image
        #__outDscrb_list: this list to self.report
        self.__step_read_num_range, self.__outDscrb_list = self.__getReport(self.__aug_config)
        
        #print(self.__is_ready)
        #print(self.__acc_step_gen_num, self.__sum_acc_step_gen_num)
        #print(self.__aug_config['step1'][0].outDscrb)
        #print(self.__aug_config['step1'][1].outDscrb)

    def isReady(self):
        return self.__is_ready
        
    def __do_clean_hue(self,img):
        obj = img[:,:,0:3]
        hls = cv2.cvtColor(obj,cv2.COLOR_BGR2HLS)
        mask = img[:,:,3]
        mask_255 = mask>127
        mask_0 = mask<=127
        smooth_obj = cv2.blur(obj,(9,9))
        hls_smooth_obj = cv2.cvtColor(smooth_obj,cv2.COLOR_BGR2HLS)
        hue = hls_smooth_obj[0]*2.0
        bg_hue = hue[mask_0]
        refer_hue =  bg_hue.sum()/bg_hue.size
        hue[hue<refer_hue-45] += 360.0
        coef_a = np.zeros(hue.shape)
        coef_a[np.bitwise_and(refer_hue-15<hue,hue<refer_hue+15)] = 1.0
        line_part1 = np.bitwise_and(refer_hue-45<=hue,hue<=refer_hue-15)
        line_part_coef_a1 = coef_a[line_part1]
        line_part_hue1 = hue[line_part1]
        line_part_coef_a1 = (refer_hue-line_part_hue)/30.0
        line_part2 = np.bitwise_and(refer_hue+15<=hue,hue<=refer_hue+45)
        line_part_coef_a2 = coef_a[line_part2]
        line_part_hue2 = hue[line_part2]
        line_part_coef_a2 = (line_part_hue-refer_hue)/30.0
        light = hls[:,:,1]+0.5*coef_a*hls[:,:,1]
        light[light>255] = 255
        sat = hls[:,:,2]-0.5*coef_a*hls[:,:,2]
        sat[sat<0]=0
        hls[:,:,1] = light
        hls[:,:,2] = sat
        out_obj = img.copy()
        out_obj[:,:,0:3] = cv2.cvtColor(hls,cv2.COLOR_HLS2BGR)
        return out_obj

    def __aug_one_opt(self,img,one_opt):
        if(one_opt.optName == 'flip'):#1
            return cvAugmentor.imFlip(img)
        elif(one_opt.optName == 'flipud'):#2
            return cvAugmentor.imFlipUD(img)     
        elif(one_opt.optName == 'fliplr'):#3
            return cvAugmentor.imFlipLR(img)  
        elif(one_opt.optName == 'prestep'):#4
            return cvAugmentor.imPerstep(img)  

        if(one_opt.isParamRandom):
            true_param = list(np.random.rand(int(one_opt.param[0]))*(one_opt.param[2]-one_opt.param[1])+one_opt.param[1])
        else:
            true_param = one_opt.param
        if(one_opt.optName == 'affine'):#5
            return cvAugmentor.imAffine(img,true_param)
        elif(one_opt.optName == 'affinex'):#6
            return cvAugmentor.imAffineX(img,true_param)
        elif(one_opt.optName == 'affiney'):#7
            return cvAugmentor.imAffineY(img,true_param)
        elif(one_opt.optName == 'crop'):#8
            return cvAugmentor.imCrop(img,true_param)
        elif(one_opt.optName == 'hue'):#9
            return cvAugmentor.imHue(img,true_param)
        elif(one_opt.optName == 'lightness'):#10
            return cvAugmentor.imLightness(img,true_param)
        elif(one_opt.optName == 'noise'):#11
            return cvAugmentor.imNoise(img,true_param)
        elif(one_opt.optName == 'perspective'):#12
            return cvAugmentor.imPerspective(img,true_param)
        elif(one_opt.optName == 'perspectived'):#13
            return cvAugmentor.imPerspectiveD(img,true_param)
        elif(one_opt.optName == 'perspectivedl'):#14
            return cvAugmentor.imPerspectiveDL(img,true_param)
        elif(one_opt.optName == 'perspectivedr'):#15
            return cvAugmentor.imPerspectiveDR(img,true_param)
        elif(one_opt.optName == 'perspectivel'):#16
            return cvAugmentor.imPerspectiveL(img,true_param)
        elif(one_opt.optName == 'perspectiver'):#17
            return cvAugmentor.imPerspectiveR(img,true_param)
        elif(one_opt.optName == 'perspectiveu'):#18
            return cvAugmentor.imPerspectiveU(img,true_param)
        elif(one_opt.optName == 'perspectiveul'):#19
            return cvAugmentor.imPerspectiveUL(img,true_param)
        elif(one_opt.optName == 'perspectiveur'):#20
            return cvAugmentor.imPerspectiveUR(img,true_param)
        elif(one_opt.optName == 'rotate'):#21
            return cvAugmentor.imRotate(img,true_param)   
        elif(one_opt.optName == 'saturation'):#22
            return cvAugmentor.imSaturation(img,true_param)


    def __findBndbox(self,mask):
        colSum = mask.sum(0)
        rowSum = mask.sum(1)
        minw = 0
        maxw = len(colSum)
        minh = 0
        maxh = len(rowSum)
        for i in range(len(colSum)):
            if(colSum[i]!=0):
                minw = i
                break
        for i in range(len(colSum)):
            if(colSum[-i-1]!=0):
                maxw = -i-1
                break
        for i in range(len(rowSum)):
            if(rowSum[i]!=0):
                minh = i
                break
        for i in range(len(rowSum)):
            if(rowSum[-i-1]!=0):
                maxh = -i-1
                break
        return [minw,minh,maxw,maxh]

    def doAug(self,obj_path,mask_path,save_path):
        #load img
        obj = cv2.imread(obj_path,cv2.IMREAD_UNCHANGED)
        img = np.zeros([obj.shape[0],obj.shape[1],4],np.uint8)
        img[:,:,0:3]=obj[:,:,0:3]
        if(mask_path == ''and obj.shape[2]!=4):
            sys.exit('No set mask')
        elif(mask_path!=''):
            mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
            img[:,:,3] = mask
        else:
            img[:,:,3]=obj[:,:,3]

        #make temp folder
        if(os.path.isdir('.AUGTEMP')is not True):
            os.makedirs('.AUGTEMP')

        #img name and label
        _t_obj_path, img_full_name = os.path.split(obj_path)
        label_index = max(_t_obj_path.rfind('\\'),_t_obj_path.rfind('/'))
        label = _t_obj_path[label_index+1:]
        img_name,img_ext = os.path.splitext(img_full_name)

        if(os.path.isdir(os.path.join(save_path,'obj',label))is not True):
            os.makedirs(os.path.join(save_path,'obj',label))
        if(os.path.isdir(os.path.join(save_path,'mask',label))is not True):
            os.makedirs(os.path.join(save_path,'mask',label))

        #temp info
        temp_save_name_front = os.path.join('.AUGTEMP',label+'_'+img_name+'_temp_')
        temp_save_name_back = '.png'
        
        #step0
        bin_mask = self.__img_process.get('bin_mask')
        #min_alpha = self.__img_process.get('min_alpha')
        #clean_hue = self.__img_process.get('clean_hue')
        _mask_to_save = img[:,:,3]
        if(bin_mask == 1):
            _mask_to_save[_mask_to_save>127] = 255
            _mask_to_save[_mask_to_save<=127] = 0

        [minw, minh, maxw , maxh] = self.__findBndbox(_mask_to_save) 

        if(len(self.__aug_config) == 1):
            cv2.imwrite(os.path.join(save_path,'obj',label,img_name+'_aug_'+str(self.__first_index + 0)+'.jpg'),img[minh:maxh,minw:maxw,0:3])            
            cv2.imwrite(os.path.join(save_path,'mask',label,img_name+'_aug_'+str(self.__first_index + 0)+'.png'),_mask_to_save[minh:maxh,minw:maxw])
            return
        cv2.imwrite(temp_save_name_front+str(0)+temp_save_name_back,img[minh:maxh,minw:maxw])

        #step 1~last-1
        temp_save_index = 1
        #0 config is step1, and so on
        for step_i in range(1,len(self.__aug_config)):
            step_opt = self.__aug_config['step'+str(step_i)]
            for temp_img_i in self.__step_read_num_range[step_i-1]:
                temp_img = cv2.imread(temp_save_name_front+str(temp_img_i)+temp_save_name_back,cv2.IMREAD_UNCHANGED)
                for one_opt in step_opt:
                    aug_output = self.__aug_one_opt(temp_img,one_opt)
                    for one_aug_output in aug_output:
                        cv2.imwrite(temp_save_name_front+str(temp_save_index)+temp_save_name_back,one_aug_output)
                        temp_save_index+=1
                os.unlink(temp_save_name_front+str(temp_img_i)+temp_save_name_back)

        #last step
        save_index = self.__first_index
        for temp_img_i in self.__step_read_num_range[-2]:
            temp_img = cv2.imread(temp_save_name_front+str(temp_img_i)+temp_save_name_back,cv2.IMREAD_UNCHANGED)
            for one_opt in step_opt:
                aug_output = self.__aug_one_opt(temp_img,one_opt)
                for one_aug_output in aug_output:
                    cv2.imwrite(os.path.join(save_path,'obj',label,img_name+'_aug_'+str(save_index)+'.jpg'),one_aug_output[:,:,0:3])
                    _mask_to_save = one_aug_output[:,:,3]
                    if(bin_mask == 1):
                        _mask_to_save[_mask_to_save>127] = 255
                        _mask_to_save[_mask_to_save<=127] = 0
                    cv2.imwrite(os.path.join(save_path,'mask',label,img_name+'_aug_'+str(save_index)+'.png'),_mask_to_save)
                    save_index += 1
            os.unlink(temp_save_name_front+str(temp_img_i)+temp_save_name_back)
        return

    def report(self,save_path):
        if(os.path.isdir(save_path) is not True):
            os.makedirs(save_path)
        with open(os.path.join(save_path,'aug_report.txt'),'w') as fd:
            for i,x in enumerate(self.__outDscrb_list):
                fd.write('aug_'+str(self.__first_index+i))
                for opt_str in x:
                    if(opt_str == ''):
                        fd.write(' \"\"')
                    else:
                        fd.write(' '+opt_str)
                fd.write('\n')
        return
    


def aug_process_fun(param):
    augSlvr = param[0]
    augSlvr.doAug(param[1],param[2],param[3])
    return


if __name__=='__main__':
    
    #cvas =  cvAugSolver(r'.\aug_config.ini',10001)
    #cvas.report(r'.\test')
    #input()


    #cf = configparser.ConfigParser()
    #print(cf.read(r'.\aug_config.ini'))
    ##print(cf.items('AUG'))
    #input()

    parser = argparse.ArgumentParser(description='pyFusion')
    #'common cfg'
    parser.add_argument('--obj',type=str, required=True, help='[path] or [list file]')
    parser.add_argument('--mask',type=str, default='', help='[path]')
    parser.add_argument('--config',type=str,default='./aug_config.ini',help='config file.')
    parser.add_argument('--save',type=str, required=True, help='path to save')
    parser.add_argument('--first',default=10001,type=int,help='the fisrt index. default = 10001')
    
    #progrom parser
    parser.add_argument('--process',default=multiprocessing.cpu_count()-1,type=int,help='use how many process.')
    args = parser.parse_args()

    augSlvr = cvAugSolver(r'.\aug_config.ini',args.first)
    if(augSlvr.isReady() is not True):
        sys.exit('read configuration file fail: '+args.config)

    augSlvr.report(args.save)

    label = scanner.folder(args.obj)

    #process list
    aug_list = []
    if(args.mask == ''):
        one_label_obj_list = scanner.file(os.path.join(args.obj,one_label),'.png')
        for x in one_label_obj_list:
            aug_list.append([augSlvr,x,'',args.save])
        if(len(aug_list) == 0):
            sys.exit('Can not scan any obj in : '+args.obj)
    else:
        pairs,other1,other2 = scanner.pair_r(args.obj,args.mask,'.jpg.png','.png',True)
        #one_label_obj_list = scanner.file(os.path.join(args.obj,one_label),'.jpg.png')
        #pairs,others = scanner.pair(os.path.join(args.mask,one_label),one_label_obj_list,'.png')
        #input()
        for x in pairs:
            aug_list.append([augSlvr,x[0],x[1],args.save])
        if(len(aug_list) == 0):
            sys.exit('Can not scan any obj (with mask) in : '+args.obj)

    num_process = 1
    if(args.process<=1 or int(multiprocessing.cpu_count())-1<=1):
        num_process =1
    elif(len(aug_list)<min(int(multiprocessing.cpu_count())-1,args.process)):
        num_process = len(aug_list)
    else:
        num_process = min(int(multiprocessing.cpu_count())-1,args.process)

    if(num_process == 1):
        for one_param in tqdm(aug_list):
            aug_process_fun(one_param)
    else:
        pool = multiprocessing.Pool(num_process) 
        for x in tqdm(pool.imap_unordered(aug_process_fun,aug_list)):
            pass
        pool.close()
        pool.join()  

    #for one_label in label:
    #    if(args.mask == ''):
    #        one_label_obj_list = scanner.file(os.path.join(args.obj,one_label),'.png')
    #        for x in one_label_obj_list:
    #            augSlvr.doAug(x,'',args.save)
    #    else:
    #        one_label_obj_list = scanner.file(os.path.join(args.obj,one_label),'.jpg.png')
    #        pairs,others = scanner.pair(os.path.join(args.mask,one_label),one_label_obj_list,'.png')
    #        for x in pairs:
    #            augSlvr.doAug(x[0],x[1],args.save)
