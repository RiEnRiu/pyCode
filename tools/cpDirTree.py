#-*-coding:utf-8-*-
import sys
import os
__pyBoost_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__pyBoost_root_path)
import pyBoost as pb

import argparse

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('src_dir',type = str, help = 'source directory.')
    parser.add_argument('dst_dir',type = str, help = 'destination directory.')
    parser.add_argument('-w','--with_root',nargs='?',default='NOT_MENTIONED',help='copy dir with root source dir. \"OFF\"')
    args = parser.parse_args()
    
    pb.cp_dir_tree(args.src_dir,args.dst_dir,not args.with_root=='NOT_MENTIONED')


    
