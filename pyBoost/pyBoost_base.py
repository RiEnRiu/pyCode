#-*-coding:utf-8-*-

import os
import threading
import time
import queue
import datetime
import numpy as np
import json
import pickle

def scan_folder(scanned_dir):
    if os.path.isdir(scanned_dir) == False:
        raise ValueError('It is not valid path: '+scanned_dir)
        return []
    return [x for x in os.listdir(scanned_dir) if os.path.isdir(os.path.join(scanned_dir,x))]

def deep_scan_folder(scanned_dir,with_root_dir=True):
    if os.path.isdir(scanned_dir) == False:
        raise ValueError('It is not valid path: '+scanned_dir)
        return []
    folders_name, folders_full_path = [], []
    folders_to_scan = set()
    folders_to_scan.add('')
    folders_scanned = set()
    while len(folders_to_scan)!=0:
        folder = folders_to_scan.pop()
        folder_full_path = os.path.join(scanned_dir,folder)
        folders_scanned.add(folder_full_path)
        contents = [os.path.join(folder,x) for x in os.listdir(folder_full_path)]
        contents_full_path = [os.path.join(scanned_dir,x) for x in contents]
        for c, cf in zip(contents,contents_full_path):
            if os.path.isdir(cf):
                if os.path.islink(cf):
                    to_continue = False
                    for x in folders_scanned:
                        if os.path.samefile(cf,x):
                            to_continue =True
                            break
                    if to_continue:
                        continue
                folders_name.append(c)
                folders_full_path.append(cf)
                folders_to_scan.add(c)

    if with_root_dir==True:
        return folders_full_path
    else:# elif with_root_dir==False:
        return folders_name


    


# this.splitext('123.txt') = ('123','.txt') , os.path.splitext('123.txt') = ('123','.txt')
# this.splitext('.txt')    = ('','.txt')    , os.path.splitext('.txt')    = ('.txt','')
# this.splitext('123.')    = ('123','')     , os.path.splitext('123.')    = ('123','.')
# this.splitext('123')     = ('123','')     , os.path.splitext('123')     = ('123','')
def splitext(file_full_name):
    if file_full_name[-1]=='.':
        return file_full_name,''
    else:
        dot = file_full_name.rfind('.')
        if dot==-1:
            return file_full_name,''
        else:
           return file_full_name[:dot],file_full_name[dot:]

def __get_exts_set(exts):
    if exts is None:
        return None
    exts_list  = ['.'+x for x in exts.split('.')]
    exts_list.pop(0)
    exts_set = set(exts_list)
    if '.' in exts_set:
        exts_set.pop('.')
        exts_set.add('')
    return exts_set

# exts is not case sensitive 
def scan_file(scanned_dir,exts=None,with_root_dir=True,with_ext=True):
    if os.path.isdir(scanned_dir) == False:
        raise ValueError('It is not valid dir: '+scanned_dir)
        return []
    exts_set = __get_exts_set(exts)

    contents = os.listdir(scanned_dir)
    contents_full_path = [os.path.join(scanned_dir,x) for x in contents]
    files_name, files_full_path = [], []
    for c,cf in zip(contents,contents_full_path):
        if  os.path.isfile(cf):
            files_name.append(c)
            files_full_path.append(cf)

    if exts_set is None:
        if with_root_dir==True and with_ext==True:
            return files_full_path
        elif with_root_dir==False and with_ext==True:
            return files_name
        elif with_root_dir==True and with_ext==False:
            return [splitext(x)[0] for x in files_full_path]
        else:# elif with_root_dir==False and with_ext==False:
            return [splitext(x)[0] for x in files_name]
    else:
        if with_root_dir==True and with_ext==True:
            return [x for x in files_full_path if splitext(x)[1] in exts_set]
        elif with_root_dir==False and with_ext==True:
            return [x for x in files_name if splitext(x)[1] in exts_set]
        elif with_root_dir==True and with_ext==False:
            sp_file = [splitext(x) for x in files_full_path]
            return [front for front,ext in sp_file if ext in exts_set]
        else:# elif with_root_dir==False and with_ext==False:
            sp_file = [splitext(x) for x in files_name]
            return [front for front,ext in sp_file if ext in exts_set]

def deep_scan_file(scanned_dir,exts=None,with_root_dir=True,with_ext=True):
    if os.path.isdir(scanned_dir) == False:
        raise ValueError('It is not valid path: '+scanned_dir)
        return []
    exts_set = __get_exts_set(exts)   

    files_name, files_full_path = [], []
    folders_to_scan = set()
    folders_to_scan.add('')
    folders_scanned = set()
    while len(folders_to_scan)!=0:
        folder = folders_to_scan.pop()
        folder_full_path = os.path.join(scanned_dir,folder)
        folders_scanned.add(folder_full_path)
        contents = [os.path.join(folder,x) for x in os.listdir(folder_full_path)]
        contents_full_path = [os.path.join(scanned_dir,x) for x in contents]
        for c, cf in zip(contents,contents_full_path):
            if os.path.isfile(cf):
                files_name.append(c)
                files_full_path.append(cf)
            else: # elif os.path.isdir(cf):
                if os.path.islink(cf):
                    to_continue = False
                    for x in folders_scanned:
                        if os.path.samefile(cf,x):
                            to_continue =True
                            break
                    if to_continue:
                        continue
                folders_to_scan.add(c)
   
    if exts_set is None:
        if with_root_dir==True and with_ext==True:
            return files_full_path
        elif with_root_dir==False and with_ext==True:
            return files_name
        elif with_root_dir==True and with_ext==False:
            return [splitext(x)[0] for x in files_full_path]
        else:# elif with_root_dir==False and with_ext==False:
            return [splitext(x)[0] for x in files_name]
    else:
        if with_root_dir==True and with_ext==True:
            return [x for x in files_full_path if splitext(x)[1] in exts_set]
        elif with_root_dir==False and with_ext==True:
            return [x for x in files_name if splitext(x)[1] in exts_set]
        elif with_root_dir==True and with_ext==False:
            sp_file = [splitext(x) for x in files_full_path]
            return [front for front,ext in sp_file if ext in exts_set]
        else:# elif with_root_dir==False and with_ext==False:
            sp_file = [splitext(x) for x in files_name]
            return [front for front,ext in sp_file if ext in exts_set]



def scan_pair(scanned_dir1,scanned_dir2,exts1,exts2,with_root_dir=True,with_ext=True):
    if exts1 is None or exts2 is None:
        raise ValueError('\"ext1\" and \"ext2\" in funsion \"scan_pair()\" must be set.')
        return [],[],[]

    exts_set1 = __get_exts_set(exts1)
    exts_set2 = __get_exts_set(exts2)

    file1 = scan_file(scanned_dir1,None,False,True)
    sp1 = [splitext(x) for x in file1]
    others1 = []
    for i in range(len(file1)-1,-1,-1):
        if sp1[i][1] not in exts_set1:
            others1.append(file1[i])
            file1.pop(i)
            sp1.pop(i)

    file2 = scan_file(scanned_dir2,None,False,True)
    sp2 = [splitext(x) for x in file2]
    others2 = []
    for i in range(len(file2)-1,-1,-1):
        if sp2[i][1] not in exts_set2:
            others2.append(file2[i])
            file2.pop(i)
            sp2.pop(i)

    front1 = {x[0]:i for i,x in enumerate(sp1)}
    front2 = {x[0]:i for i,x in enumerate(sp2)}

    paired_front = set(front1.keys()) & set(front2.keys())
    paired_index1_set,paired_index2_set = set(),set()
    paired_index = []
    for k in paired_front:
        i1,i2 = front1[k],front2[k]
        paired_index.append((i1, i2))
        paired_index1_set.add(i1)
        paired_index2_set.add(i2)
    others1.extend([f1 for i1,f1 in enumerate(file1) if i1 not in paired_index1_set])
    others2.extend([f2 for i2,f2 in enumerate(file2) if i2 not in paired_index2_set])
    
    if with_root_dir==True and with_ext==True:
        return [[os.path.join(scanned_dir1,file1[i1]),os.path.join(scanned_dir2,file2[i2])] for i1,i2 in paired_index],\
               [os.path.join(scanned_dir1,o1) for o1 in others1],\
               [os.path.join(scanned_dir2,o2) for o2 in others2]
    elif with_root_dir==False and with_ext==True:
        return [[file1[i1],file2[i2]] for i1,i2 in paired_index],\
               others1,\
               others2
    elif with_root_dir==True and with_ext==False:
        return [[os.path.join(scanned_dir1,splitext(file1[i1])[0]), os.path.join(scanned_dir2,splitext(file2[i2])[0])] \
                   for i1,i2 in paired_key],\
               [os.path.join(scanned_dir1,o1) for o1 in others1],\
               [os.path.join(scanned_dir2,o2) for o2 in others2]
    else:# elif with_root==False and with_ext==False:
        return [[splitext(file1[i1])[0],splitext(file2[i2])[0]] for i1,i2 in paired_key],\
               others1,\
               others2

def scan_text(path,sep=None):
    with open(path,'r') as fd:
        ls = fd.readlines()
    sp_line = [s.split(sep) for s in ls]
    return [l for l in sp_line if len(l)!=0]

class FPS:
    def __init__(self, num=30, max_fps = 100000):
        self.__num=num
        self.__time_point = [time.time()]*num
        self.__index = 0
        self.__next_index = 1
        self.__count = 0
        self.__s_time = 1/max_fps
        self._min_s_time = 1/max_fps
        return

    def click(self):
        if self.__count!=self.__num:
            if self.__index!=0:
                self.__time_point[self.__index] = time.time()
                self.__s_time = max((self.__time_point[self.__index] - self.__time_point[0]),self._min_s_time)
            else:
                time0  = time.time()
                self.__s_time = max((time0 - self.__time_point[0]),self._min_s_time)
                self.__time_point[self.__index] = time0
            self.__count += 1
        else:
            self.__time_point[self.__index] = time.time()
            self.__s_time = max((self.__time_point[self.__index] - self.__time_point[self.__next_index]),self._min_s_time)
        self.__index = self.__next_index
        self.__next_index += 1
        if self.__next_index==self.__num:
            self.__next_index = 0
        return

    #with click
    def get(self,click=1):
        if click<=0:
            return self.__count/self.__s_time
        else:
            self.click()
            return self.__count/self.__s_time*click


def makedirs(path):
    if os.path.isdir(path) == False:
        os.makedirs(path)
    return

def cp_dir_tree(src_dir, dst_dir,with_root,level=0):
    dirs = deep_scan_folder(src_dir,False)
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
    for x in dirs:
        p = os.path.join(dst_dir,x)
        if not os.path.isdir(p):
            os.mkdir(p)
    return


def get_time_stamp(start='year',end='second'):
    dd = str(datetime.datetime.now())
    #               1         2     
    #     01234567890123456789012345
    #    '2019-04-19 10:57:37.826145'
    # to '2019-04-19_10-57-37-826145'
    dd = dd.replace(':','-').replace('.','-').replace(' ','_')
    i0 = 0
    i1 = 26
    if start=='year': i0 = 0
    elif start=='month': i0 = 5
    elif start=='day': i0 = 8
    elif start=='hour': i0 = 11
    elif start=='minute': i0 = 14
    elif start=='second': i0 = 17
    elif start=='microsecond': i0 = 20
    else:
        raise ValueError('start = {0} is invalid.'.format(start))
   
    if end=='year': i1 = 4
    elif end=='month': i1 = 7
    elif end=='day': i1 = 10
    elif end=='hour': i1 = 13
    elif end=='minute': i1 = 16
    elif end=='second': i1 = 19
    elif end=='microsecond': i1 = 26
    else:
        raise ValueError('end = {0} is invalid.'.format(end))

    if i0>=i1:
        raise ValueError('start = {0}, end = {1} is invalid.'.format(start,end))

    return dd[i0:i1]

    
 

class productionLineWorker:
    def __init__(self, func, maxsize=0, flow_type = 'FIFO', refresh_HZ = 1000):
        if hasattr(func, '__call__')==False:
            raise TypeError('func must be function class.')
        self.func = func
        self.maxsize = maxsize
        self.flow_type = flow_type # or 'LIFO'
        self.refresh_HZ = refresh_HZ

#TODO make it more useful
class productionLine:
    def __init__(self, workers, maxsize = 0, flow_type='FIFO'):
        # param
        self._output_maxsize = maxsize
        self._flow_type = flow_type

        # workers
        self._workers = workers
        self._num_workers = len(self._workers)

        # data queue        
        _q_list = []
        for i,x in enumerate(self._workers):
            # x = productionLineWorker()
            if x.flow_type=='FIFO':
                _q_list.append(queue.Queue(x.maxsize))
            elif x.flow_type=='LIFO':
                _q_list.append(queue.LifoQueue(x.maxsize))
            else:
                raise TypeError('Unknown type in workers[{0}].flow_type=={1}'.format(i,x.flow_type))
        if flow_type=='FIFO':
            _q_list.append(queue.Queue(maxsize))
        elif x.flow_type=='LIFO':
            _q_list.append(queue.LifoQueue(maxsize))
        else:
            raise TypeError('Unknown type in productionLine.flow_type=={0}'.format(flow_type))
        self._q_list = _q_list

        # flags
        self._have_joined = False
        self._put_mutex = threading.Lock()
        self._to_brk_thread = [False]*self._num_workers
        self._brk_with_join = [True]*self._num_workers

        # lock
        _thread_mutex = []
        for i in range(self._num_workers):
            _thread_mutex.append(threading.Lock())
        self._thread_mutex = _thread_mutex

        # thread
        p_thread = []
        for i in range(self._num_workers):
            p_thread.append(threading.Thread(target=productionLine.thread_function,args=(self,i),daemon=True))
        for p in p_thread:
            p.start()
        self._p_thread = p_thread

    def thread_function(self,worker_index):
        lock = self._thread_mutex[worker_index]
        que = self._q_list[worker_index]
        worker = self._workers[worker_index]
        next_que = self._q_list[worker_index+1]
        while(self._to_brk_thread[worker_index]==False):
            if que.empty():
                time.sleep(1.0/worker.refresh_HZ)
            else:
                lock.acquire()
                data = que.get()
                output = worker.func(*data)
                next_que.put(output)
                lock.release()
        if self._brk_with_join[worker_index]:
            while(que.empty()==False):
                data = que.get()
                output = worker.func(*data)
                next_que.put(output)
        return 
        
    def put(self,*data):
        self._put_mutex.acquire()
        if self._have_joined:
            raise RuntimeError('productionLine has been joined.')
        elif len(self._q_list)==0:
            raise RuntimeError('no worker in productionLine.')
        else:
            self._q_list[0].put(data)
        self._put_mutex.release()
        return 

    def release(self):
        self._put_mutex.acquire()
        self._have_joined = True
        for i in range(self._num_workers):
            self._brk_with_join[i] = False
            self._to_brk_thread[i] = True
            if self._q_list[i+1].empty()==False:
                self._q_list[i+1].get()
            self._p_thread[i].join()
        self._put_mutex.release()
        return 

    def join(self):
        self._put_mutex.acquire()
        self._have_joined = True
        for i in range(self._num_workers):
            self._brk_with_join[i] = True
            self._to_brk_thread[i] = True
            self._p_thread[i].join()
        self._put_mutex.release()
        return 

    def wait_finish(self):
        self._put_mutex.acquire()
        for i in range(len(self._q_list)-1):
            worker = self._workers[i]
            que = self._q_list[i]
            lock = self._thread_mutex[i]
            while(1):
                lock.acquire()
                if que.empty():
                    lock.release()
                    break
                lock.release()
                time.sleep(1.0 / worker.refresh_HZ)
        self._put_mutex.release()
        return

    def get(self):
        if self.empty():
            #TODO bug = len(self._workers) == self._num_workers-1
            time.sleep(1.0 / self._workers[self._num_workers-1].refresh_HZ)
            return False,None
        else:
            return True,self._q_list[self._num_workers].get()

    def empty(self):
        return self._q_list[self._num_workers].empty()

    def full(self):
        return self._q_list[self._num_workers].full()

def str_table(table):
    try:
        rows = len(table)
        cols = len(table[0])
        r = [[str(table[i][j]) for j in range(cols)] for i in range(rows)]
    except Exception as e:
        raise ValueError('Invalid table.')
    if rows==0 or cols==0:
        raise ValueError('Invalid table.')
    max_len = np.array([[len(y) for y in x] for x in r], np.int32)
    max_len = max_len.max(0)
    dividing_line = ['-'*(n+2) for n in max_len]
    dividing_line = '+{0}+'.format('+'.join(dividing_line))
    data = []
    for x in r:
        tmp = []
        for n,y in zip(max_len,x):
            tmp.append(y.center(n+2))
        data.append('|{0}|'.format('|'.join(tmp)))
    rtable = []
    rtable.append(dividing_line)
    rtable.append(data[0])
    rtable.append(dividing_line)
    rtable.extend(data[1:])
    rtable.append(dividing_line)
    return '\n'.join(rtable)

def print_table(table,add_first=False):
    if add_first and len(table)>0 and len(table[0])>0:            
        ntable = [[''] + ['col_{}'.format(i) for i,x in enumerate(table[0])]]
        for i,x in enumerate(table):
            ntable.append(['row_{}'.format(i)]+list(x))
        print(str_table(ntable))
    else:
        print(str_table(table))

def load_json_file(jpath,**kwargs):
    fp = open(jpath,'r')
    r = json.load(fp,**kwargs)
    fp.close()
    return r

def dump_json_file(obj,jpath,indent=4,sort_keys=True,**kwargs):
    fp = open(jpath,'w')
    json.dump(obj,fp,indent=indent,sort_keys=sort_keys,**kwargs)
    fp.close()
    return

def load_pkl_file(ppath,**kwargs):
    fp = open(ppath,'rb')
    r = pickle.load(fp,**kwargs)
    fp.close()
    return r

def dump_pkl_file(obj,pjpath,**kwargs):
    fp = open(pjpath,'wb')
    pickle.dump(obj,fp,**kwargs)
    fp.close()
    return



if __name__=='__main__':
    ##################################################################
    #test module
    ##################################################################
    import sys
    sys.path.append('../')
    import pyBoost as pb
    #import pyBoostBase as pb

    def test_get_time_stemp():
        for i in range(100):
            print(get_time_stamp('year','microsecond'))
            time.sleep(0.1)
        return

    def test_scan():
        path_to_scanner = r'G:\obj_mask_10'
        scanner_file = pb.scan_file(path_to_scanner)
        print('scanner_file = ')
        print(scanner_file)
        print()
        scanner_file_txt = pb.scan_file(path_to_scanner,'.txt',True,False)
        print('scanner_file(\'.txt\') = ')
        print(scanner_file_txt)
        print()
        scanner_file_r_img_full = pb.deep_scan_file(path_to_scanner,'.jpg.png.jpeg')
        print('scanner_file_r_img_full[0:5] = ')
        print(scanner_file_r_img_full[0:5])
        print()
        scanner_folder = pb.scan_folder(path_to_scanner)
        print('scanner_folder = ')
        print(scanner_folder)
        print()
        scanner_list = pb.scan_text(os.path.join(path_to_scanner,'label.txt'))
        print('scanner_list[0:5] = ')
        print(scanner_list[0:5])
        print()


    def test_FPS():
        import time
        p_fps = pb.FPS()
        key=-1
        time.time()
        for x in range(100):
            begin = time.time()
            time.sleep(1/30)
            if x%5==0:
                print('fps = {:.6f}, {:.6f}'.format(p_fps.get(5),1/(time.time()-begin)))
        return    

    def test_dump_load(dir):
        time_stamp = get_time_stamp()
        test_dict = {'time_stamp':time_stamp,'mode':'test','key':2333}
        save_path = os.path.join(dir,'dump_load')
        pb.makedirs(save_path)
        print('source content = ')
        print(test_dict)

        pb.dump_json_file(test_dict,os.path.join(save_path,'test.json'))
        jload = pb.load_json_file(os.path.join(save_path,'test.json'))
        print('json file content = ')
        with open(os.path.join(save_path,'test.json'),'r') as fp:
            print(''.join(fp.readlines()))
        print('json reloaded content = ')
        print(jload)

        pb.dump_pkl_file(test_dict,os.path.join(save_path,'test.pkl'))
        pload = pb.load_pkl_file(os.path.join(save_path,'test.pkl'))
        print('pkl reloaded content = ')
        print(pload)

        
    def test_print_table():
        import cv2
        color_ring = pb.read_color_ring()
        gray = cv2.cvtColor(color_ring,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray,(5,5))
        print('source table:')
        print(img)
        print('print table:')
        pb.print_table(img)
        print('print added first table')
        pb.print_table(img, add_first=True)


    def test_productionLine():
        import cv2
        import numpy as np
        def readimg(img_path,i):
            time.sleep(2/1000)
            return np.zeros([720,1280,3],np.uint8),i
    
        def dealimg(img,i):
            cv2.line(img,(i,0),(i,10000),(0,0,255),3)
            time.sleep(3/1000)
            return img,i

        def saveimg(img,saved_index):
            time.sleep(1/1000)
            return saved_index

        begin = time.time()
        for i in range(720):
            img,_i = readimg('',i)
            img,_ii = dealimg(img,_i)
            saved_index = saveimg(img,_ii)
            #print(i,saved_index)
            if i%72==0:
                print(saved_index)
        end = time.time()
        print('serial:    all result = {0}    total = {1:.4f}s    avg_ms = {2:.4f}'.format(720,end-begin,(end-begin)/720))

        print('')

        begin = time.time()

        workers = [pb.productionLineWorker(readimg,30),\
                   pb.productionLineWorker(dealimg,30),\
                   pb.productionLineWorker(saveimg,30)]
        p_line = pb.productionLine(workers)
        saved_indices = []    
        for i in range(720):
            p_line.put('',i)
            ret,got = p_line.get()
            if ret:
                saved_indices.append(got)
        p_line.wait_finish()
        ret,got = p_line.get()
        while(ret):
            saved_indices.append(got)
            ret,got = p_line.get()
        print('len = '+str(len(saved_indices)))        

        for i,index in enumerate(saved_indices):
            if i%72==0:
                print(index)
        end = time.time()
        print('multithreading:    all result = {0}    total = {1:.4f}s    avg_ms = {2:.4f}'.format(len(saved_indices),end-begin,(end-begin)/len(saved_indices)))
    
        return

    #####################################################################
    save_folder_name = 'pyBoost_test_output'
    #test_get_time_stemp()
    # test_scan()
    # test_dump_load(save_folder_name)
    test_print_table()
    #test_FPS()
    #test_productionLine()

