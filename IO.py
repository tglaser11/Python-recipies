# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 11:15:38 2015

@author: tom
"""

path = '/home/tom/Python/'

import numpy as np
from random import gauss

a = [gauss(1.5, 2) for i in range(1000000)]

import pickle

pkl_file = open(path + 'data.pkl', 'w')

# %time pickle.dump(a,pkl_file)

pickle.dump(a, pkl_file)

pkl_file.close()


pkl_file = open(path + 'data.pkl', 'r')

b = pickle.load(pkl_file)

pkl_file.close()
