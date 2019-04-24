#-*-coding:utf-8-*-
import os
import sys
__pyBoost_path = os.path.dirname(__file__)
#print('__init__(pyBoost) path = {0}'.format(__file__))
sys.path.append(__pyBoost_path)
from pyBoost_base import *
import img
import voc
import video
sys.path.remove(__pyBoost_path)
