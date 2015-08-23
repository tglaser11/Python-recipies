__author__ = 'tom'

path = '/home/tom/Python/'

import numpy as np

# create a range of dates
dtimes = np.arange('2015-01-01 10:00:00', '2021-12-31 22:00:00', dtype='datetime64[m]')

# dtype specifies field names and data type for array
dty = np.dtype([('Date', 'datetime64[m]'), ('No1','f'),('No2','f')])
# return an array of zeros with columns of dtype
data = np.zeros(len(dtimes), dtype=dty)

# set Date column field for each row to dtimes values
data['Date'] = dtimes

# create psuedo random numbers for shape -- rows=number of dates & columns=2
a = np.random.standard_normal((len(dtimes), 2)).round(5)

# set data fields in array to newly generated random numbers
data['No1'] = a[:,0]
data['No2'] = a[:,1]

# save to disk
np.save(path + 'array', data)

# load from disk
np.load(path + 'array.npy')