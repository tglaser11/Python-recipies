# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 11:34:55 2015

@author: tom
"""

path = '/home/tom/Python/'

import numpy as np

rows = 5000
a = np.random.standard_normal((rows,5))

a.round(4)


import pandas as pd
t = pd.date_range(start='2014/1/1', periods=rows, freq='H')


# write to CSV

csv_file = open(path + 'data.csv', 'w')
header = 'date,no1,no2,no3,no4,no5\n'
csv_file.write(header)

for t_, (no1,no2,no3,no4,no5) in zip (t, a):
    s = '%s,%f,%f,%f,%f,%f\n' % (t_, no1, no2, no3, no4, no5)
    csv_file.write(s)

csv_file.close()

# open file for reading
csv_file = open(path + 'data.csv', 'r')

# read one line at a time
for i in range(5):
    print csv_file.readline()
    
# read all lines

content = csv_file.readlines()
for line in content[:5]:
    print line
    
csv_file.close()
