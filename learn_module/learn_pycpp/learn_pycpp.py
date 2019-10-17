'''
# frist install boost
unzip boost_1_69_0.zip
cd boost_1_69_0
# ./bootstrap.sh --with-libraries=python --with-toolset=gcc --with-python=$HOME/anaconda3/bin/python3 --with-python-version=3.6 --with-python-root=~/anaconda3/lib/python3
./bootstrap.sh --with-libraries=python --with-toolset=gcc 
./b2 cflags='-fPIC' cxxflags='-fPIC' --with-python include="$HOME/anaconda3/include/python3.6m/"
sudo ./b2 install
cd /usr/local/lib
sudo ln -s libboost_python36.so libbost_python3.so
sudo ln -s libboost_python36.a libboost_python3.so
sudo ln -s libboost_numpy36.so libboost_numpy3.so
sudo ln -s libboost_numpy36.a libboost_numpy3.a
sudo vim $HOME/.bashrc
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
'''

'''
# second make .os library
g++ -shared -fPIC -std=c++11 -O3 $HOME/anaconda3/lib/libpython3.6m.a modify_order_by_weight.cpp  -o modify_order_by_weight.so -I$HOME/anaconda3/include/python3.6m -lboost_python36
'''

'''
# third use .os in python
from modify_order_by_weight import modify_order as cpp_modify_order
'''

from py_modify_order_by_weight import modify_order_multicls as py_modify_order
from modify_order_by_weight import modify_order as cpp_modify_order


if __name__=='__main__':

    import time
    goods_weight =\
    {
        "yb-ybcjs-pz-yw-555ml": 581.33,
        "yy-yylght-gz-ht-240ml": 285.0,
        "ksf-ksfmlmc-pz-yw-500ml": 537.33,
        "wlj-wljlc-hz-yw-250ml": 269.33,
        "asm-asmnc-pz-yw-500ml": 544.67,
        "ksf-ksfmlqc-pz-yw-500ml": 532.0,
        "nfsq-nfsqjjydyl-pz-nmw-550ml": 602.0,
        "ty-hzy-pz-gw-500ml": 550.0,
        "nfsq-nfsqyytrs-pz-yw-550ml": 570.67
    }

    '''
    modify_order_multicls(order, in_out_list,\
                         weight_measure, goods_weight, \
                         weight_thresh = 20, max_modify_times = 3,
                         max_in_out_time=30):

    '''

    in_out_list = [{'in_out_track':'IN','hid':1,'time_stamp':1,'cls_90':['hand'],'cls_conf_90':[0.95]}, \
                   {'in_out_track':'OUT','hid':2,'time_stamp':2,'cls_90':['yy-yylght-gz-ht-240ml'],'cls_conf_90':[0.95]}, \
                   {'in_out_track':'OUT','hid':1,'time_stamp':2,'cls_90':['yb-ybcjs-pz-yw-555ml'],'cls_conf_90':[0.95]}, \
                   # {'in_out_track':'IN','hid':1,'time_stamp':3,'cls_90':['yy-yylght-gz-ht-240ml'],'cls_conf_90':[0.95]}, \
                   # {'in_out_track':'IN','hid':1,'time_stamp':4,'cls_90':['hand'],'cls_conf_90':[0.95]}, \
                   {'in_out_track':'OUT','hid':1,'time_stamp':4,'cls_90':['hand'],'cls_conf_90':[0.95]}]
    for d in in_out_list:
        d['cls'] = d['cls_90'][0]
        d['cls_conf'] = d['cls_conf_90'][0]
    weight_measure = [0,\
                -goods_weight["yy-yylght-gz-ht-240ml"]-goods_weight["yb-ybcjs-pz-yw-555ml"], \
                -goods_weight["yb-ybcjs-pz-yw-555ml"],0]

#########################################################################################
    t1 = time.time()
    for i in range(100):
        opt_order, opt_modification, opt_is_friendly = py_modify_order(dict(), in_out_list, \
                                                                weight_measure, goods_weight, \
                                                                20.5, 3, 30)
    print(opt_order)
    print(opt_modification)
    print(opt_is_friendly)
    t2 = time.time()
    print('time = {0}'.format(t2-t1))
    print('')

#########################################################################################
    t3 = time.time()
    for i in range(100):
        opt_order, opt_modification, opt_is_friendly = cpp_modify_order(dict(), in_out_list, \
                                                                weight_measure, goods_weight, \
                                                                20.5, 3, 30)
    print(opt_order)
    print(opt_modification)
    print(opt_is_friendly)
    t4 = time.time()
    print('time = {0}'.format(t4-t3))
    print('')



