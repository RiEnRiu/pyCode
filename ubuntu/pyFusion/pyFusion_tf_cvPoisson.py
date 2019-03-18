import cv2
import numpy as np
import argparse
import os
import datetime
import random
import math
from tqdm import tqdm
import json

# import threading
import multiprocessing

#from tf_mix_one import *

import datetime


# global til_here_time
# global pre_til_here_time
#
# til_here_time = datetime.datetime.now()
#
# pre_til_here_time = datetime.datetime.now()



# class ThreadWithReturnValue(threading.Thread):
#    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
#        threading.Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)
#        self._return = None
#    def run(self):
#        if self._target is not None:
#            self._return = self._target(*self._args, **self._kwargs)
#    def join(self):
#        threading.Thread.join(self)
#        return self._return


class pyFusion:
    def __init__(self, opt):

        self.__K = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32)
        self.__Kh = np.array([[0, -1, 1]], np.float32)
        self.__Kv = np.array([[0], [-1], [1]], np.float32)

        

        self.__opt = opt
        self.__date = self.__get_date(self.__opt.date)

        self.__obj_size = self.__read_obj_size(self.__opt.obj_size)

        self.__bg = self.__get_bg_list(self.__opt.bg)
        self.__obj_mask_label = self.__get_obj_mask_label_list(self.__opt.obj, self.__opt.with_mask_file)
        self.__combined = self.__combine_balance(self.__bg, self.__obj_mask_label, self.__opt.num, self.__opt.min,
                                                 self.__opt.max)
        # self.__combined = self.__combine_fix(self.__bg,self.__obj_mask_label,self.__opt.num,self.__opt.min,self.__opt.max)
        if (os.path.isdir('./log') is not True):
            os.makedirs('./log')

        with open('./log/fusion_' + self.__date + '_cfg.json', 'w') as cfg:
            # json.dump(self.__opt,cfg)
            to_json = {'bg': opt.bg, 'obj': opt.obj, 'with_mask_file': opt.with_mask_file, \
                       'save': opt.save, 'num': opt.num, 'min': opt.min, 'max': opt.max, \
                       'date': opt.date, 'first': opt.first, 'Fusion': self.__combined}
            json.dump(to_json, cfg, indent=4)

        self.__p_mix_one_tf_tool = mix_one_tf_tool(0, 1)

        self.__grad_size = 15
        self.__grad = [(x,y) for x in range(self.__grad_size) for y in range(self.__grad_size)]




    def __read_obj_size(self,path):
        if(os.path.isfile(path) is not True):
            print('Can not read list from file: ',path)
            return []
        list_from_file = []
        with open(path) as fd:
            one_line = fd.readline()
            while(one_line):
                one_line_list = one_line.split()
                if(len(one_line_list)==0):
                    pass
                else:
                    list_from_file.append(one_line_list)
                one_line = fd.readline()

        obj_size = {}
        for one_obj_size in list_from_file:
            obj_size[one_obj_size[0]] = float(one_obj_size[1])*float(one_obj_size[2])

        return obj_size


    def __combine_balance(self, bg, obj_mask_label, num, min, max):

        state_sample = []  # [index1,index2]
        one_class_num = []  # [len(state_sample(:))]
        label_dict = {}  # label:index
        class_num = 0
        for i, one_obj_mask_label in enumerate(obj_mask_label):
            state_sample_index = label_dict.get(one_obj_mask_label[2])
            if (type(state_sample_index) == type(None)):
                label_dict[one_obj_mask_label[2]] = class_num
                state_sample.append([i])
                class_num += 1
            else:
                state_sample[state_sample_index].append(i)
        for one_state_sample in state_sample:
            one_class_num.append(len(one_state_sample))

        print(class_num)

        result = []
        one_img_num_range = list(range(min, max + 1))
        for i in range(num):
            one_bg = random.sample(bg, 1)
            one_img_num = random.sample(one_img_num_range, 1)
            one_obj_mask_label = []
            for j in range(one_img_num[0]):
                samlpe_rand_value = random.random()
                selected_value = samlpe_rand_value * class_num
                selected_index = int(selected_value)
                index_in_class = int((selected_value - selected_index) * one_class_num[selected_index])
                one_obj_mask_label.append(obj_mask_label[state_sample[selected_index][index_in_class]])
            result.append(one_bg + one_obj_mask_label)
        return result

    def __combine(self, bg, obj_mask_label, num, min, max):
        result = []
        one_img_num_range = list(range(min, max + 1))
        # for test
        # pick_obj_index = 0
        pick_obj_index = len(obj_mask_label)
        obj_num = len(obj_mask_label)
        for i in range(num):
            if (pick_obj_index < obj_num):
                one_bg = random.sample(bg, 1)
                one_img_num = random.sample(one_img_num_range, 1)
                rest = obj_num - (pick_obj_index + one_img_num[0])
                if (rest > 0):
                    one_obj_mask_label = obj_mask_label[pick_obj_index:pick_obj_index + one_img_num[0]]
                    result.append(one_bg + one_obj_mask_label)
                    pick_obj_index += one_img_num[0]
                else:
                    one_obj_mask_label = obj_mask_label[pick_obj_index:]
                    one_obj_mask_label += random.sample(obj_mask_label, -rest)
                    result.append(one_bg + one_obj_mask_label)
                    pick_obj_index += one_img_num[0]
            else:
                one_bg = random.sample(bg, 1)
                one_img_num = random.sample(one_img_num_range, 1)
                one_obj_mask_label = random.sample(obj_mask_label, one_img_num[0])
                result.append(one_bg + one_obj_mask_label)

        # for i in range(num):
        #    one_bg = random.sample(bg,1)
        #    one_img_num = random.sample(one_img_num_range,1)
        #    one_obj_mask_label = random.sample(obj_mask_label,one_img_num[0])
        #    result.append(one_bg+one_obj_mask_label)

        return result

    def __combine_fix(self, bg, obj_mask_label, num, min, max):
        result = []
        for i in range(5908):
            one_obj_mask_label = obj_mask_label[(i * 6):(i * 6 + 6)]
            one_bg = random.sample(bg, 1)
            result.append(one_bg + one_obj_mask_label)
        return result

    def __get_bg_list(self, bg_parser):
        if (os.path.isdir(bg_parser)):
            return self.__scan_file(bg_parser, '.jpg.png.jpeg.bmp')
        else:
            fd = open(bg_parser)
            if (fd is not True):
                return []
            else:
                temp = fd.readlists()
                result = []
                for one_line in temp:
                    if (one_line != '\n'):
                        result.append(one_line[:-1])
                return result

    def __get_obj_mask_label_list(self, obj_parser, is_with_mask):
        obj = []
        mask = []
        label = []

        if (is_with_mask):
            if (os.path.isfile(obj_parser)):  # read file with mask
                with open(obj_parser) as fd:
                    line_num = 1
                    one_line = fd.readline()
                    while (one_line):
                        one_obj = one_line.split()
                        if (len(one_obj) == 0):
                            pass
                        elif (len(one_obj) == 3):
                            obj.append(one_obj[0])
                            mask.append(one_obj[1])
                            label.append(one_obj[2])
                        else:
                            print('line ' + str(line_num) + ' read fail in file \'' + obj_parser + '\'')
                        one_line = fd.readline()
                        line_num += 1
            elif (os.path.isdir(obj_parser)):  # (scan obj , mask)
                folder_names = self.__scan_folder(obj_parser)
                for folder_name in folder_names:
                    this_folder_obj = self.__scan_file(os.path.join(obj_parser, folder_name, 'obj'), '.jpg.png.jpeg')
                    for one_obj_name in this_folder_obj:
                        obj_path, file_name = os.path.split(one_obj_name)
                        front_name = file_name[0:file_name.rfind('.')]
                        if (os.path.isfile(os.path.join(obj_parser, folder_name, 'mask', front_name + '.png'))):
                            obj.append(one_obj_name)
                            mask.append(os.path.join(obj_parser, folder_name, 'mask', front_name + '.png'))
                            label.append(folder_name)
        else:
            if (os.path.isfile(obj_parser)):  # read file without mask
                with open(obj_parser) as fd:
                    line_num = 1
                    one_line = fd.readline()
                    while (one_line):
                        one_obj = one_line.split()
                        if (len(one_obj) == 0):
                            pass
                        elif (len(one_obj) == 2):
                            obj.append(one_obj[0])
                            mask.append('')
                            label.append(one_obj[1])
                        else:
                            print('line ' + str(line_num) + ' read fail in file \'' + obj_parser + '\'')
                        one_line = fd.readline()
                        line_num += 1
            elif (os.path.isdir(obj_parser)):  # (scan .png)
                folder_names = self.__scan_folder(obj_parser)
                for folder_name in folder_names:
                    this_folder_png_img = self.__scan_file(os.path.join(obj_parser, folder_name), '.png')
                    obj = obj + this_folder_png_img
                    mask = mask + [''] * len(this_folder_png_img)
                    label = label + [folder_name] * len(this_folder_png_img)
        return [[x, y, z] for x, y, z in zip(obj, mask, label)]

    def __scan_folder(self, path):
        folder_list = []
        dirs = os.listdir(path)
        for one_folder in dirs:
            if (os.path.isdir(os.path.join(path, one_folder))):
                folder_list.append(one_folder)
        return folder_list

    def __scan_file(self, path, postfixes=None):
        if (type(postfixes) == type(None)):
            file_list = []
            dirs = os.listdir(path)
            for one_dir in dirs:
                full_dir = os.path.join(path, one_dir)
                if (os.path.isfile(full_dir)):
                    file_list.append(full_dir)
            return file_list
        else:
            file_list = []
            dirs = os.listdir(path)
            for one_dir in dirs:
                full_dir = os.path.join(path, one_dir)
                dot_index = one_dir.rfind('.')
                if (dot_index == -1):
                    ext = ''
                else:
                    ext = one_dir[dot_index:]
                if (os.path.isfile(full_dir) and postfixes.find(ext) != -1):
                    file_list.append(full_dir)
            return file_list

    def __get_date(self, _appointed):
        if (_appointed == 'TODAY' or type(_appointed) != type('')):
            date = str(datetime.datetime.now())
            date = date[0:date.find(' ')]
            return date
        else:
            return _appointed

    def __isContains(self, region, point):
        return region[0][0] <= point[0] and point[0] < region[1][0] and region[0][1] <= point[1] and point[1] < \
               region[1][1]

    def __isIntersect(self, region1, region2):
        if (region1[1][0] <= region2[0][0]):
            return False
        if (region2[1][0] <= region1[0][0]):
            return False
        if (region1[1][1] <= region2[0][1]):
            return False
        if (region2[1][1] <= region1[0][1]):
            return False
        return True

    def __coverRate(self, region1, region2):
        if (self.__isIntersect(region1, region2) is not True):
            return 0.0
        is_contain_tl = self.__isContains(region1, region2[0])
        is_contain_tr = self.__isContains(region1, (region2[1][0], region2[0][1]))
        is_contain_br = self.__isContains(region1, region2[1])
        is_contain_bl = self.__isContains(region1, (region2[0][0], region2[1][1]))
        width1 = region1[1][0] - region1[0][0]
        height1 = region1[1][1] - region1[0][1]
        width2 = region2[1][0] - region2[0][0]
        height2 = region2[1][1] - region2[0][1]
        cover_width = []
        conver_height = []

        # 0000
        if (is_contain_tl == False and is_contain_tr == False and is_contain_br == False and is_contain_bl == False):
            if (region1[0][0] < region2[0][0] and region2[0][0] < region1[1][0]):
                if (region1[1][0] < region2[1][0]):
                    cover_width = region1[1][0] - region2[0][0]
                    cover_height = height1
                else:
                    cover_width = width2
                    cover_height = height1
            elif (region1[0][1] < region2[0][1] and region2[0][1] < region1[1][1]):
                if (region1[1][1] < region2[1][1]):
                    cover_width = width1
                    cover_height = region1[1][1] - region2[0][1]
                else:
                    cover_width = width1
                    cover_height = height2
            else:
                return 1.0
        elif (is_contain_tl and is_contain_br or is_contain_tr and is_contain_bl):
            cover_width = width2
            cover_height = height2
        # 0011
        elif (is_contain_tl == False and is_contain_tr == False and is_contain_br and is_contain_bl):
            cover_width = width2
            cover_height = region2[1][1] - region1[0][0]
            # 1100
        elif (is_contain_tl and is_contain_tr and is_contain_br == False and is_contain_bl == False):
            cover_width = width2
            cover_height = region1[1][1] - region2[0][0]
            # 0110
        elif (is_contain_tl == False and is_contain_tr and is_contain_br and is_contain_bl == False):
            cover_width = region2[1][0] - region1[0][0]
            cover_height = height2
        # 1001
        elif (is_contain_tl and is_contain_tr == False and is_contain_br == False and is_contain_bl):
            cover_width = region1[1][0] - region2[0][0]
            cover_height = height2
            # 0010
        elif (is_contain_tl == False and is_contain_tr == False and is_contain_br and is_contain_bl == False):
            cover_width = (region2[1][0] - region1[0][0])
            cover_height = (region2[1][1] - region1[0][1])
            # 0001
        elif (is_contain_tl == False and is_contain_tr == False and is_contain_br == False and is_contain_bl):
            cover_width = (region1[1][0] - region2[0][0])
            cover_height = (region2[1][1] - region1[0][1])
            # 1000
        elif (is_contain_tl and is_contain_tr == False and is_contain_br == False and is_contain_bl == False):
            cover_width = (region1[1][0] - region2[0][0])
            cover_height = (region1[1][1] - region2[0][1])
            # 0100
        elif (is_contain_tl == False and is_contain_tr and is_contain_br == False and is_contain_bl == False):
            cover_width = (region2[1][0] - region1[0][0])
            cover_height = (region1[1][1] - region2[0][1])
        return cover_width * cover_height / (width1 * height1)

    def __choicePlaceNoCover(self, bg_shape, obj_shape, exist_bdbs):  # shape = (height, width, dtype = tuple)

        d_width = bg_shape[1] - obj_shape[1];
        d_height = bg_shape[0] - obj_shape[0];

        check_time_step = 1000
        while_num = 0
        begin_time = datetime.datetime.now()

        find_OK_rect = False

        while (1):
            minx = random.randint(0, d_width - 1)
            miny = random.randint(0, d_height - 1)
            maxx = minx + obj_shape[1]
            maxy = miny + obj_shape[0]

            while_num += 1

            find_OK_rect = True

            for one_bdb in exist_bdbs:
                if (one_bdb == [(), ()]):
                    continue
                if (self.__isIntersect(one_bdb, [(minx, miny), (maxx, maxy)])):
                    find_OK_rect = False
                    break

            if (find_OK_rect):
                break

            if (while_num == check_time_step):
                end_time = datetime.datetime.now()
                if ((end_time - begin_time).total_seconds() > 60.0):
                    find_OK_rect = False
                    break
                while_num = 0

        if (find_OK_rect is not True):
            return [(), ()]
        else:
            return [(minx, miny), (maxx, maxy)]

    def __choicePlaceAllowCover(self, bg_shape, obj_shape, exist_bdbs, min_cover_rate,
                                max_cover_rate):  # shape = (height, width, dtype = tuple)

        d_width = bg_shape[1] - obj_shape[1];
        d_height = bg_shape[0] - obj_shape[0];

        check_time_step = 1000
        while_num = 0
        begin_time = datetime.datetime.now()

        find_OK_rect = False
        ___count = 1



        

        while (1):
            ___count+=1
            print(___count)
            minx = random.randint(0, d_width - 1)
            miny = random.randint(0, d_height - 1)
            maxx = minx + obj_shape[1]
            maxy = miny + obj_shape[0]

            while_num += 1

            find_OK_rect = True
            for one_bdb in exist_bdbs:
                if (one_bdb == [(), ()]):
                    continue
                #rate = self.__coverRate(one_bdb, [(minx, miny), (maxx, maxy)])
                rate = max(self.__coverRate(one_bdb, [(minx, miny), (maxx, maxy)]),self.__coverRate([(minx, miny), (maxx, maxy)],one_bdb))
                # rate = 0.0
                if rate > max_cover_rate or rate < min_cover_rate:
                    find_OK_rect = False
                    break

            if (find_OK_rect):
                break

            if (while_num == check_time_step):
                
       


                end_time = datetime.datetime.now()
                if ((end_time - begin_time).total_seconds() > 60.0):
                    print('time out')
                    find_OK_rect = False
                    break
                while_num = 0

        if (find_OK_rect is not True):
            return [(), ()]
        else:
            return [(minx, miny), (maxx, maxy)]

    def __choicePlaceAllowCover_fast(self, bg_shape, obj_shape, exist_bdbs, min_cover_rate,
                                max_cover_rate):  # shape = (height, width, dtype = tuple)

        d_width = bg_shape[1] - obj_shape[1];
        d_height = bg_shape[0] - obj_shape[0];
        find_OK_rect = False
        grad_step = [d_width/self.__grad_size,d_height/self.__grad_size]
        begin_time = datetime.datetime.now()
        #times = 1
        while (1):
            random.shuffle(self.__grad)
            for one_grad in self.__grad:
                #times += 1
                #print(times)
#                if(times=100001):
#                    cv2.imshow('')

                minx = int((random.random()+one_grad[0])*grad_step[0])
                miny = int((random.random()+one_grad[1])*grad_step[1])

                maxx = minx + obj_shape[1]
                maxy = miny + obj_shape[0]

                find_OK_rect = True
                for one_bdb in exist_bdbs:
                    if (one_bdb == [(), ()]):
                        continue
                    #rate = self.__coverRate(one_bdb, [(minx, miny), (maxx, maxy)])
                    rate = max(self.__coverRate(one_bdb, [(minx, miny), (maxx, maxy)]),self.__coverRate([(minx, miny), (maxx, maxy)],one_bdb))
                    # rate = 0.0
                    if rate > max_cover_rate or rate < min_cover_rate:
                        find_OK_rect = False
                        break
                if(find_OK_rect):
                    break

            if (find_OK_rect):
                break

            end_time = datetime.datetime.now()
            if ((end_time - begin_time).total_seconds() > 10.0):
                print('time out')
                find_OK_rect = False
                break

        if (find_OK_rect is not True):
            return [(), ()]
        else:
            return [(minx, miny), (maxx, maxy)]


    def __findMaskBDB(self, mask):
        sum_each_col = mask.astype(np.float32).sum(0).astype(np.uint8)
        sum_each_row = mask.astype(np.float32).sum(1).astype(np.uint8)
        col_nonzero_index = np.nonzero(sum_each_col)[0].tolist()
        row_nonzero_index = np.nonzero(sum_each_row)[0].tolist()

        if (col_nonzero_index != [] and row_nonzero_index != []):
            minx = col_nonzero_index[0]
            maxx = col_nonzero_index[-1] + 1  # minx + width
            miny = row_nonzero_index[0]
            maxy = row_nonzero_index[-1] + 1  # minx + height
            return [(minx, miny), (maxx, maxy)]
        else:
            return [(), ()]

    def __get_fit_shape(self, bg_shape, obj_shape, shape_rate):
        rate = math.sqrt((random.randint(0, 1024) / 2048.0 + 0.5) * shape_rate * (bg_shape[0] * bg_shape[1]) / (
                    obj_shape[0] * obj_shape[1]))
        new_shape = [int(obj_shape[0] * rate), int(obj_shape[1] * rate)]
        while (new_shape[1] >= bg_shape[1] or new_shape[0] >= bg_shape[0]):
            new_shape[0] = int(new_shape[0] * 0.8)
            new_shape[1] = int(new_shape[1] * 0.8)
        return (new_shape[0], new_shape[1])

    def __imgrad(self, bgr_image):
        Fh_out = cv2.filter2D(bgr_image, cv2.CV_32F, self.__Kh)
        Fv_out = cv2.filter2D(bgr_image, cv2.CV_32F, self.__Kv)
        return (Fh_out, Fv_out)

    # def __poissonJacobiSam(self, mix_image, X, Y, mask, _itr, _th):
    #
    #     lap_x = cv2.filter2D(X, cv2.CV_32F, np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]], dtype=np.float32))
    #     lap_y = cv2.filter2D(Y, cv2.CV_32F, np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32))
    #
    #     lap = lap_x + lap_y
    #
    #     # lap = np.zeros(X.shape,np.float32)
    #     # lap_rows = lap.shape[0]
    #     # lap_cols = lap.shape[1]
    #     # for i in range(lap_rows):
    #     #     for j in range(lap_cols):
    #     #         lap[i][j] = X[i][(j-1+lap_cols)%lap_cols] + Y[(i-1+lap_rows)%lap_rows][j] - X[i][j] - Y[i][j]
    #     # time_point_3 = datetime.datetime.now()
    #
    #     diff0 = 10000000.0
    #     max_diff = 10000000.0
    #
    #     mix_image_32FC3 = mix_image.astype(np.float32)
    #
    #     clone_mix_image_32FC3 = [mix_image_32FC3.copy(), mix_image_32FC3.copy()]
    #     i_clone = 0
    #     i_pre_clone = 1
    #
    #     # Mask_8UC3_list = []
    #     # for i in range(mask.shape[0]):
    #     #     Mask_8UC3_list.append(list(zip(mask[i],mask[i],mask[i])))
    #     #
    #     # Mask_8UC3 = np.array(Mask_8UC3_list,np.uint8)
    #     # where_Mask_8UC3_zero = (Mask_8UC3==0)
    #     # where_Mask_8UC3_not_zero = (Mask_8UC3!=0)
    #
    #     mask01_c1 = (mask / 255.0).astype(np.float32)
    #     mask01 = cv2.cvtColor(mask01_c1, cv2.COLOR_GRAY2BGR)
    #     invmask01 = (1.0 - mask01).astype(np.float32)
    #
    #     time_point_1 = datetime.datetime.now()
    #     while (_itr > 0):
    #         _itr -= 1
    #
    #         clone_mix_image_32FC3[i_clone] = cv2.filter2D(clone_mix_image_32FC3[i_pre_clone], cv2.CV_32F, self.__K)
    #
    #         # clone_mix_image_32FC3[i_clone] = clone_mix_image_32FC3[i_clone] * mask01 + clone_mix_image_32FC3[i_pre_clone] * invmask01
    #
    #         clone_mix_image_32FC3[i_clone] = (
    #                     ((clone_mix_image_32FC3[i_clone] + lap) * 0.25) * mask01 + clone_mix_image_32FC3[
    #                 i_pre_clone] * invmask01).astype(np.float32)
    #
    #         # clone_mix_image_32FC3[i_clone] = (
    #         #             (clone_mix_image_32FC3[i_clone] * mask01 + clone_mix_image_32FC3[i_pre_clone] * invmask01 + \
    #         #              lap*mask01) * 0.25).astype(np.float32)
    #
    #         # clone_mix_image_32FC3[i_clone][where_Mask_8UC3_not_zero] = ((clone_mix_image_32FC3[i_clone] + lap)*0.25).astype(np.float32)[where_Mask_8UC3_not_zero]
    #         max_diff = (np.abs(clone_mix_image_32FC3[i_clone] - clone_mix_image_32FC3[i_pre_clone])).max()
    #
    #         if (math.fabs(diff0 - max_diff) / diff0 < _th):
    #             break
    #         diff0 = max_diff
    #
    #         if (i_clone == 0):
    #             i_clone = 1
    #             i_pre_clone = 0
    #         else:
    #             i_clone = 0
    #             i_pre_clone = 1
    #
    #     time_point_2 = datetime.datetime.now()
    #
    #     # result = tf_acc(mix_image, mask01_c1, lap, _th, _itr)
    #     result = tf_acc(mix_image, mask01_c1, lap, _th, 1024-_itr)
    #
    #     time_point_3 = datetime.datetime.now()
    #
    #     ########################
    #     # global  til_here_time
    #     # global pre_til_here_time
    #     #
    #     # til_here_time = datetime.datetime.now()
    #     # d_til_here_time = til_here_time - pre_til_here_time + end -begin
    #     # pre_til_here_time = til_here_time
    #
    #     # print(str(d_til_here_time.total_seconds())+'s  '+str((end-begin).total_seconds())+'s    '+str(1024-_itr))
    #     #########################
    #
    #     print(str((time_point_2 - time_point_1).total_seconds()) + '    ' + \
    #           str(            (time_point_3 - time_point_2).total_seconds())+'    '+str(1024-_itr))
    #     # print(str((time_point_2-time_point_1).total_seconds())+'    '+str((time_point_3-time_point_2).total_seconds()))
    #
    #     # print(1024-_itr)
    #     # result = clone_mix_image_32FC3[i_clone].astype(np.uint8)
    #     # result[where_Mask_8UC3_zero]  = mix_image[where_Mask_8UC3_zero]
    #
    #     return result
    #
    #
    #
    #     result[where_Mask_8UC3_zero] = mix_image[where_Mask_8UC3_zero]
    #
    #     return result

    def __mix_one_obj_poisson_sam(self, to_mixed_one_obj, mixed, bdbs):

        time_point_1 = datetime.datetime.now()

        # obj
        obj = to_mixed_one_obj[0]

        # alpha
        alpha = to_mixed_one_obj[1]

        # bin mask
        ret, mask = cv2.threshold(to_mixed_one_obj[1], 1, 255, cv2.THRESH_BINARY)

        # find blinding box
        [tl, br] = self.__findMaskBDB(mask)
        if (tl == ()):
            bdbs.append([(), ()])
            return mixed, bdbs
        # cut valid part
        obj = obj[tl[1]:br[1], tl[0]:br[0], :].copy()
        alpha = alpha[tl[1]:br[1], tl[0]:br[0]].copy()
        mask = mask[tl[1]:br[1], tl[0]:br[0]].copy()

        # resize
        new_shape = self.__get_fit_shape(mixed.shape, obj.shape, 1.0 / 20.0)
        obj = cv2.resize(obj, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(alpha, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)

        # choose place
        rect = self.__choicePlaceNoCover(mixed.shape, obj.shape, bdbs)

        if (rect == [(), ()]):
            bdbs.append([(), ()])
            return mixed, bdbs

        # deal alpha
        alpha01 = (alpha / 255.0).astype(np.float32)
        # print(alpha01[np.logical_and(alpha!=0,alpha!=255)])
        inv_alpha01 = (1.0 - alpha01).astype(np.float32)

        # grad
        bg_roi = mixed[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0], :]
        (bg_roi_grad_x, bg_roi_grad_y) = self.__imgrad(bg_roi)
        (obj_grad_x, obj_grad_y) = self.__imgrad(obj)

        # alpha paste
        X = np.zeros(obj_grad_x.shape, np.float32)
        Y = np.zeros(obj_grad_y.shape, np.float32)
        mix_image = np.zeros(obj.shape, np.uint8)
        for c in range(bg_roi.shape[2]):
            mix_image[:, :, c] = (alpha01 * obj[:, :, c] + inv_alpha01 * bg_roi[:, :, c]).astype(np.uint8)
            X[:, :, c] = (alpha01 * obj_grad_x[:, :, c] + inv_alpha01 * bg_roi_grad_x[:, :, c]).astype(np.float32)
            Y[:, :, c] = (alpha01 * obj_grad_y[:, :, c] + inv_alpha01 * bg_roi_grad_y[:, :, c]).astype(np.float32)

        # cv2.imshow('obj',obj)
        # cv2.imshow('bg_roi',bg_roi)
        # cv2.imshow('mix_image',mix_image)
        # cv2.waitKey(0)

        # deal mask
        where_mask_zero = mask < 1
        where_mask_not_zero = mask >= 1

        # hls
        pre_hls = cv2.cvtColor(mix_image, cv2.COLOR_BGR2HLS)
        time_point_2 = datetime.datetime.now()

        # mixed[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0], :][where_mask_not_zero] = \
        # self.__poissonJacobiSam(mix_image, X, Y, mask, 1024, 0.003)[where_mask_not_zero]

        # print(bg[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0], :].shape)

        # alpha01 = alpha01.reshape([1,alpha01.shape[0],alpha01.shape[1],1])
        # print(alpha.shape)
        mixed[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0], :] = \
            self.__p_mix_one_tf_tool.tf_acc(mixed[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0], :],obj,alpha01,0.003,1024)

        time_point_3 = datetime.datetime.now()
        # print(str((time_point_2-time_point_1).total_seconds())+'    '+str((time_point_3-time_point_2).total_seconds()))

        mixed_hls = cv2.cvtColor(mixed[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0], :], cv2.COLOR_BGR2HLS)
        mixed_hls[:, :, 0] = pre_hls[:, :, 0]
        mixed[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0], :][where_mask_not_zero] = \
        cv2.cvtColor(mixed_hls, cv2.COLOR_HLS2BGR)[where_mask_not_zero]

        # mixed[rect[0][1]:rect[1][1],rect[0][0]:rect[1][0],:][mask!=0] = mix_image.copy()[mask!=0]

        # return
        bdbs.append(rect)
        return mixed, bdbs


    def __mix_one_obj_poisson_sam_fix_obj_rate(self, to_mixed_one_obj, mixed, bdbs, obj_rate):

        #time_point_1 = datetime.datetime.now()

        # obj
        obj = to_mixed_one_obj[0]

        # alpha
        alpha = to_mixed_one_obj[1]

        # bin mask
        ret, mask = cv2.threshold(to_mixed_one_obj[1], 1, 255, cv2.THRESH_BINARY)

        # find blinding box
        [tl, br] = self.__findMaskBDB(mask)
        if (tl == ()):
            bdbs.append([(), ()])
            return mixed, bdbs
        # cut valid part
        obj = obj[tl[1]:br[1], tl[0]:br[0], :].copy()
        alpha = alpha[tl[1]:br[1], tl[0]:br[0]].copy()
        mask = mask[tl[1]:br[1], tl[0]:br[0]].copy()

        # resize
        new_shape = [obj.shape[0]*obj_rate,obj.shape[1]*obj_rate]
        while(new_shape[0]>mixed.shape[0] or new_shape[1]>mixed.shape[1]):
            new_shape[0] *= 0.9
            new_shape[1] *= 0.9
        new_shape[0] = int(new_shape[0])
        new_shape[1] = int(new_shape[1])

        obj = cv2.resize(obj, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(alpha, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)

        # choose place
        #rect = self.__choicePlaceNoCover(mixed.shape, obj.shape, bdbs)
        rect = self.__choicePlaceAllowCover_fast(mixed.shape, obj.shape, bdbs,-0.00001,0.2)
        if (rect == [(), ()]):
            bdbs.append([(), ()])
            #print('Can not find a fit area')
            return mixed, bdbs

        # deal alpha
        alpha01 = (alpha / 255.0).astype(np.float32)
        # print(alpha01[np.logical_and(alpha!=0,alpha!=255)])
        inv_alpha01 = (1.0 - alpha01).astype(np.float32)

        # grad
        bg_roi = mixed[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0], :]
        (bg_roi_grad_x, bg_roi_grad_y) = self.__imgrad(bg_roi)
        (obj_grad_x, obj_grad_y) = self.__imgrad(obj)

        pre_out = mixed.copy()

        # alpha paste
        X = np.zeros(obj_grad_x.shape, np.float32)
        Y = np.zeros(obj_grad_y.shape, np.float32)
        mix_image = np.zeros(obj.shape, np.uint8)
        for c in range(bg_roi.shape[2]):
            mix_image[:, :, c] = (alpha01 * obj[:, :, c] + inv_alpha01 * bg_roi[:, :, c]).astype(np.uint8)
            X[:, :, c] = (alpha01 * obj_grad_x[:, :, c] + inv_alpha01 * bg_roi_grad_x[:, :, c]).astype(np.float32)
            Y[:, :, c] = (alpha01 * obj_grad_y[:, :, c] + inv_alpha01 * bg_roi_grad_y[:, :, c]).astype(np.float32)

        # cv2.imshow('obj',obj)
        # cv2.imshow('bg_roi',bg_roi)
        # cv2.imshow('mix_image',mix_image)
        # cv2.waitKey(0)

        paste_mixed = mixed.copy()
        paste_mixed_float32 = paste_mixed.astype(np.float32)
        stop_value = 10000

        mask[mask>127]=255
        mask[mask<=127]=0

        while(stop_value>1):
            cv_poisson_mix = cv2.seamlessClone(obj,pre_out,mask,(int(rect[0][0]+rect[1][0]/2),int(rect[0][1]+rect[1][1]/2)),cv2.)
            this_out = ((paste_mixed_float32+cv_poisson_mix+pre_out)/3).astype(np.uint8)
            stop_value = np.abs(this_out - pre_out).max()
            pre_out = this_out.copy()
            to_show =  cv2.resize(pre_out,(int(pre_out.shape[1]/3),int(pre_out.shape[0]/3)))
            cv2.imshow('to_show',to_show)
            cv2.waitKey(0)


        # return
        bdbs.append(rect)
        return mixed, bdbs

    def __load_img(self, to_mixed):
        bg_img = cv2.imread(to_mixed[0])
        to_mixed_obj = [[]] * (len(to_mixed) - 1)
        for i in range(1, len(to_mixed)):
            if (to_mixed[i][1] == ''):  # png
                obj_png = cv2.imread(to_mixed[i][0], cv2.IMREAD_UNCHANGED)
                obj = obj_png[:, :, 0:3]
                mask = obj_png[:, :, 3]
            else:  # jpg
                obj = cv2.imread(to_mixed[i][0])
                mask = cv2.imread(to_mixed[i][1], cv2.IMREAD_GRAYSCALE)
            to_mixed_obj[i - 1] = [obj, mask,to_mixed[i][2]]
        return bg_img, to_mixed_obj

    def __mix(self, bg_img, to_mixed_obj):
        bdbs = []
        for i in range(len(to_mixed_obj)):
            bg_img, bdbs = self.__mix_one_obj_poisson_sam(to_mixed_obj[i], bg_img, bdbs)
        return (bg_img, bdbs)

    def __mix_all_area(self, bg_img, to_mixed_obj):
        bdbs = []
        all_obj_area = 0
        for i in range(len(to_mixed_obj)):	
            all_obj_area += to_mixed_obj[i][0].shape[0] * to_mixed_obj[i][0].shape[1]

        area_rate = random.randint(45,55)/100.0
        #print(area_rate)
        obj_rate = math.sqrt(area_rate*bg_img.shape[0]*bg_img.shape[1]/all_obj_area)


        for i in range(len(to_mixed_obj)):
            bg_img, bdbs = self.__mix_one_obj_poisson_sam_fix_obj_rate(to_mixed_obj[i], bg_img, bdbs, obj_rate)
        return (bg_img, bdbs)

    def __mix_all_area_real_size(self, bg_img, to_mixed_obj):
        bdbs = []
        all_obj_area = 0
        pix_area = [1]*len(to_mixed_obj)
        obj_real_size_pix_area = [1]*len(to_mixed_obj)

        relative_pix_area = 0
        for i in range(len(to_mixed_obj)):
            mask_size = to_mixed_obj[i][1][to_mixed_obj[i][1]>10].size
            pix_area[i] =mask_size +to_mixed_obj[i][1][to_mixed_obj[i][1]<=10].size/2
            if(mask_size/to_mixed_obj[i][1].size>relative_pix_area):
                relative_pix_area = pix_area[i]
            
        for i in range(len(to_mixed_obj)):
            #pix_area[i] = to_mixed_obj[i][1][to_mixed_obj[i][1]>10].size+to_mixed_obj[i][1][to_mixed_obj[i][1]<=10].size/2
            #pix_area[i] = to_mixed_obj[i][1].size
            #obj_real_size_pix_area[i] = self.__obj_size[to_mixed_obj[i][2]] / self.__obj_size[to_mixed_obj[0][2]] * pix_area[0]
            obj_real_size_pix_area[i] = self.__obj_size[to_mixed_obj[i][2]] / self.__obj_size[to_mixed_obj[0][2]] * relative_pix_area
            all_obj_area += obj_real_size_pix_area[i]

        area_rate = random.randint(35,45)/100.0
        obj_rate = area_rate*bg_img.shape[0]*bg_img.shape[1]/all_obj_area

        each_obj_rate = [1]*len(to_mixed_obj)
        for i in range(len(to_mixed_obj)):
            each_obj_rate[i] =  math.sqrt(obj_real_size_pix_area[i]*obj_rate/pix_area[i])

        for i in range(len(to_mixed_obj)):
            bg_img, bdbs = self.__mix_one_obj_poisson_sam_fix_obj_rate(to_mixed_obj[i], bg_img, bdbs, each_obj_rate[i])
        return (bg_img, bdbs)


    def __save(self, fusion_img, bdbs, this__combined, save_path, front_name):
        if (cv2.imwrite(save_path + '/JPEGImages/' + front_name + '.jpg', fusion_img) is not True):
            return False
        xml_file = open(save_path + '/Annotations/' + front_name + '.xml', 'w')
        xml_file.write('<?xml version=\"1.0\" encoding=\"utf-8\"?>' + '\n')
        xml_file.write('<annotation>' + '\n')
        xml_file.write('    <folder>JPEGImages</folder>' + '\n')
        xml_file.write('    <filename>' + front_name + '.jpg' + '</filename>' + '\n')
        xml_file.write('    <path>' + save_path + '/JPEGImages/' + front_name + '.jpg' + '</path>' + '\n')
        xml_file.write('    <source>' + '\n')
        xml_file.write('        <database>Unknown</database>' + '\n')
        xml_file.write('    </source>' + '\n')
        xml_file.write('    <size>' + '\n')
        xml_file.write('        <width>' + str(fusion_img.shape[1]) + '</width>' + '\n')
        xml_file.write('        <height>' + str(fusion_img.shape[0]) + '</height>' + '\n')
        xml_file.write('        <depth>' + str(fusion_img.shape[2]) + '</depth>' + '\n')
        xml_file.write('    </size>' + '\n')
        xml_file.write('    <segmented>0</segmented>' + '\n')
        for i in range(len(bdbs)):
            if (bdbs[i] == [(), ()]):
                continue
            xml_file.write('    <object>' + '\n')
            xml_file.write('        <name>' + this__combined[i + 1][2] + '</name>' + '\n')
            xml_file.write('        <pose>Unspecified</pose>' + '\n')
            xml_file.write('        <truncated>0</truncated>' + '\n')
            xml_file.write('        <difficult>0</difficult>' + '\n')
            xml_file.write('        <bndbox>' + '\n')
            xml_file.write('            <xmin>' + str(bdbs[i][0][0]) + '</xmin>' + '\n')
            xml_file.write('            <ymin>' + str(bdbs[i][0][1]) + '</ymin>' + '\n')
            xml_file.write('            <xmax>' + str(bdbs[i][1][0]) + '</xmax>' + '\n')
            xml_file.write('            <ymax>' + str(bdbs[i][1][1]) + '</ymax>' + '\n')
            xml_file.write('        </bndbox>' + '\n')
            xml_file.write('    </object>' + '\n')
        xml_file.write('</annotation>' + '\n')
        xml_file.close()
        return True

    def process_for(self, i):
        (bg_img, to_mixed_obj) = self.__load_img(self.__combined[i])
        #(fusion_img, bdbs) = self.__mix(bg_img, to_mixed_obj)
        (fusion_img, bdbs) = self.__mix_all_area_real_size(bg_img, to_mixed_obj)
        self.__save(fusion_img, bdbs, self.__combined[i], self.__opt.save,
                    'fusion_' + self.__date + '_' + str(self.__opt.first + i))
        return True

    def doFusion(self):
        save_path = self.__opt.save
        if (os.path.isfile(save_path + '/Annotations') or os.path.isfile(save_path + '/JPEGImages')):
            print('Can not create save dir ' + save_path)
            return False
        if (os.path.isdir(save_path + '/Annotations') is not True):
            os.makedirs(save_path + '/Annotations')
        if (os.path.isdir(save_path + '/JPEGImages') is not True):
            os.makedirs(save_path + '/JPEGImages')

        if (type(self.__opt.process) == type(None)):
            num_process = 1
        elif (self.__opt.process <= 1 or int(multiprocessing.cpu_count()) - 1 <= 1):
            num_process = 1
        elif (len(self.__combined) < min(int(multiprocessing.cpu_count()) - 1, self.__opt.process)):
            num_process = len(self.__combined)
        else:
            num_process = min(int(multiprocessing.cpu_count()) - 1, self.__opt.process)

        if (num_process == 1):
            for i in tqdm(range(len(self.__combined))):
                self.process_for(i)
        else:
            pool = multiprocessing.Pool(num_process)
            i_list = range(len(self.__combined))
            for x in tqdm(pool.imap_unordered(self.process_for, i_list)):
                pass
            pool.close()
            pool.join()
        self.__p_mix_one_tf_tool.close_sess()
        return True



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pyFusion')
    # 'common cfg'
    parser.add_argument('--bg', type=str, required=True, help='[path] or [list file]')
    parser.add_argument('--obj', type=str, required=True, help='[path] or [list file]')
    parser.add_argument('--with_mask_file', default=False, type=int,
                        help='True = [scan or check list file with image file as mask], False = [make alpha channal of obj.png as mask]')
    parser.add_argument('--obj_size', default='obj_size.txt', type=str, help='the obj real size')
    parser.add_argument('--save', type=str, required=True, help='path to save')
    # 'fusion parameter'
    parser.add_argument('--num', type=int, required=True, help='numer of out images.')
    parser.add_argument('--min', default=5, type=int, help='min numer of objs in each image')
    parser.add_argument('--max', default=6, type=int, help='mix numer of objs in each image')
    parser.add_argument('--date', default='TODAY', type=str, help='it can be set.For example: \'2018-08-13\'')
    parser.add_argument('--first', default=10001, type=int, help='the fisrt index. For example: \'10001\'')


    # 'fusion restart'
    # parser.add_argument('--rejson',type=str,help='path of json file about Fusion to restart')
    # parser,add_argument('--rebegin',default=10001,type=int,help='restart with it as begin index')
    # parser.add_argument('--reend',default=-1,type=int,help='restart with it as begin index')

    # progrom parser
    parser.add_argument('--process', default=multiprocessing.cpu_count() - 1, type=int, help='use how many process.')
    opt = parser.parse_args()
    if(opt.process!=1):
        opt.process = 1
        print('Warning: It can only use 1 process now.')

    fu = pyFusion(opt)
    fu.doFusion()















