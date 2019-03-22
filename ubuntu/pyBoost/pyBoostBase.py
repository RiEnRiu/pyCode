import os
import threading
import time
import queue


def scan_folder(path):
    if os.path.isdir(path) == False:
        raise ValueError('It is not valid path: '+path)
        return []
    return [x for x in os.listdir(path) if os.path.isdir(os.path.join(path,x))]


# this.splitext('123.txt') ->  ['123', '.txt'] ,    the same as os.path.splitext()
# this.splitext('.txt') ->  ['', '.txt']       , different from os.path.splitext()
# this.splitext('123.') ->  ['123', '']        , different from os.path.splitext()
# this.splitext('123') ->  ['123', '']         ,    the same as os.path.splitext()
def splitext(fileFullName):
    if fileFullName[-1]=='.':
        return fileFullName,''
    else:
        dot = fileFullName.rfind('.')
        if dot==-1:
            return fileFullName,''
        else:
            return fileFullName[:dot],fileFullName[dot:]

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
def scan_file(path,exts=None,with_root_path=True,with_ext=True):
    if os.path.isdir(path) == False:
        raise ValueError('It is not valid path: '+path)
        return []
    dirs = os.listdir(path)
    all_file_full_name = [x for x in dirs if os.path.isfile(os.path.join(path,x))]
    exts_set = __get_exts_set(exts)

    if exts_set is None:
        if with_root_path==True and with_ext==True:
            return [os.path.join(path,x) for x in all_file_full_name]
        elif with_root_path==False and with_ext==True:
            return all_file_full_name
        elif with_root_path==True and with_ext==False:
            return [os.path.join(path, splitext(x)[0]) for x in all_file_full_name]
        else:# elif with_root_path==False and with_ext==False:
            return [splitext(x)[0] for x in all_file_full_name]
    else:
        sp_file = [splitext(x) for x in all_file_full_name]
        if with_root_path==True and with_ext==True:
            return [os.path.join(path,x) \
                for i,x in enumerate(all_file_full_name)\
                    if sp_file[i][1] in exts_set]
        elif with_root_path==False and with_ext==True:
            return [x \
                for i,x in enumerate(all_file_full_name)\
                    if sp_file[i][1] in exts_set]
        elif with_root_path==True and with_ext==False:
            return [os.path.join(path,x[0]) \
                for x in sp_file\
                    if x[1] in exts_set]
        else:# elif with_root_path==False and with_ext==False:
            return [x[0] \
                for x in sp_file\
                    if x[1] in exts_set]

def __do_scan_file_r(path, output, with_root_path, relative_path = ''):
    true_root = os.path.join(path,relative_path)
    all = os.listdir(true_root)
    for x in all:
        path_x = os.path.join(true_root,x)
        if os.path.isdir(path_x):
            __do_scan_file_r(path,output,with_root_path,os.path.join(relative_path,x))
        elif with_root_path==True:
            output.append(path_x)
        else:
            output.append(os.path.join(relative_path, x))
    return


def scan_file_r(path,exts=None,with_root_path=True,with_ext=True):
    if os.path.isdir(path) == False:
        raise ValueError('It is not valid path: '+path)
        return []
    
    all_file_with_ext = []
    __do_scan_file_r(path,all_file_with_ext, with_root_path)
    exts_set = __get_exts_set(exts)
   
    if exts_set is None:
        if with_ext==True:
            return all_file_with_ext
        else:
            return [splitext(x)[0] for x in all_file_with_ext]
    else:
        sp_file = [splitext(x) for x in all_file_with_ext]

        if with_ext==True:
            return [x for i,x in enumerate(all_file_with_ext)\
                          if sp_file[i][1] in exts_set]
        else:
            return [x[0] for x in sp_file\
                             if x[1] in exts_set]

def scan_pair(path1,path2,exts1,exts2,with_root_path=True,with_ext=True):
    if exts1 is None or exts2 is None:
        raise ValueError('\"ext1\" and \"ext2\" in funsion \"scan_pair()\" must be set.')
        return [],[],[]

    exts_set1 = __get_exts_set(exts1)
    exts_set2 = __get_exts_set(exts2)

    file1 = scan_file(path1,None,False,True)
    sp1 = [splitext(x) for x in file1]
    others1 = []
    for i in range(len(file1)-1,-1,-1):
        if sp1[i][1] not in exts_set1:
            others1.append(file1[i])
            file1.pop(i)
            sp1.pop(i)

    file2 = scan_file(path2,None,False,True)
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
    paired_key = [[front1[k],front2[k]] for k in paired_front]
    others1.extend([file1[i] \
                for i in (set(range(len(file1)))-set(x[0] for x in paired_key))])
    others2.extend([file2[i] \
                for i in (set(range(len(file2)))-set(x[1] for x in paired_key))])
    
    if with_root_path==True and with_ext==True:
        return [[os.path.join(path1,file1[k1]),os.path.join(path2,file2[k2])] \
             for k1,k2 in paired_key],\
               [os.path.join(path1,x) for x in others1],\
               [os.path.join(path2,y) for y in others2]
    elif with_root_path==False and with_ext==True:
        return [[file1[k1],file2[k2]] for k1,k2 in paired_key],\
               others1,\
               others2
    elif with_root_path==True and with_ext==False:
        return [[os.path.join(path1,splitext(file1[k1])[0]),\
                 os.path.join(path2,splitext(file2[k2])[0])] \
             for k1,k2 in paired_key],\
               [os.path.join(path1,x) for x in others1],\
               [os.path.join(path2,y) for y in others2]
    else:# elif with_root==False and with_ext==False:
        return [[splitext(file1[k1])[0],splitext(file2[k2])[0]] \
             for k1,k2 in paired_key],\
               others1,\
               others2

def scan_text(path,sep=None):
    with open(path,'r') as fd:
        ls = fd.readlines()
    if sep is None:
        sp_line = [s.split() for s in ls]
    else:
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

    def waitFinish(self):
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

if __name__=='__main__':
    ##################################################################
    #test module
    ##################################################################
    import sys
    sys.path.append('../')
    import pyBoost as pb
    #import pyBoostBase as pb

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
        scanner_file_r_img_full = pb.scan_file_r(path_to_scanner,'.jpg.png.jpeg')
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
        p_line.waitFinish()
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
    test_scan()
    #test_FPS()
    #test_productionLine()

