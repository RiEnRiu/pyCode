#-*-coding:utf-8-*-
import threading
import time
import queue

class productionLineWorker:
    def __init__(self, func, key, pre_workers_keys=None, next_workers_keys=None, maxsize=0, flow_type = 'FIFO', refresh_HZ = 1000):
        if hasattr(func, '__call__')==False:
            raise TypeError('func must be function class.')
        self.func = func
        self.key = key
        self.pre_workers_keys = pre_workers_keys
        self.next_workers_keys = next_workers_keys
        self.maxsize = maxsize
        self.flow_type = flow_type # or 'LIFO'
        self.refresh_HZ = refresh_HZ

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
            time.sleep(1.0 / self._workers[self._num_workers].refresh_HZ)
            return False,None
        else:
            return True,self._q_list[self._num_workers].get()

    def empty(self):
        return self._q_list[self._num_workers].empty()

    def full(self):
        return self._q_list[self._num_workers].full()

