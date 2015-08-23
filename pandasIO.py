__author__ = 'tom'

path = '/home/tom/Python/'

import numpy as np
import pandas as pd

# create psuedo random numbers for shape described by parameters & round numbers to 5 digits
data = np.random.standard_normal((1000000, 5)).round(5)

filename = path + 'numbs'

import sqlite3 as sq3


query = 'CREATE TABLE numbers (No1 real, No2 real, No3 real, No4 real, No5 real)'

con = sq3.Connection(filename + '.db')


con.execute(query)

# insert array data into table
con.executemany('INSERT INTO numbers VALUES (?,?,?,?,?)', data)
con.commit()

# fetch all data and print first two rows
temp = con.execute('SELECT * FROM numbers').fetchall()
print temp[:2]
temp = 0.0

query = 'SELECT * FROM numbers WHERE No1 > 0 AND No2 < 0'
res = np.array(con.execute(query).fetchall()).round(3)

res = res[::100] # every 100th result
import matplotlib.pyplot as plt
plt.plot(res[:,0], res[:,1], 'ro')
plt.grid(True); plt.xlim(-0.5, 4.5); plt.ylim(-4.5, 0.5)
plt.show()


# manipulate data in memory with pandas for raster analytics
import pandas.io.sql as pds
data = pds.read_sql('SELECT * FROM numbers', con)

data[(data['No1'] > 0 ) & (data['No2'] < 0)]
res = data[['No1','No2']][((data['No1'] > 0.5) | (data['No1'] < -0.5)) & ((data['No2'] < -1) | (data['No2'] > 1))]

plt.plot(res.No1, res.No2, 'ro')
plt.grid(True); plt.axis('tight')
plt.show()

# HDFS store allows for fast IO
# Write to HDFS store
h5s = pd.HDFStore(filename + '.h5s', 'w')
h5s['data'] = data
h5s.close()

# Read from HDFS store
h5s = pd.HDFStore(filename + '.h5s', 'r')
temp = h5s['data']
h5s.close()

# compare data to see if same
check = np.allclose(np.array(temp), np.array(data))

temp = 0.0 # zero out temp

# drop table for subsequent runs
con.execute('DROP TABLE numbers')

# CSV store -- most common format for data exchange
# convert using to_csv method
data.to_csv(filename + '.csv')

# chart frequency data using histogram charts
pd.read_csv(filename + '.csv')[['No1', 'No2', 'No3', 'No4']].hist(bins=20)
plt.show()

# Reading and Writing from MSFT Excel spreadsheet

data[:100000].to_excel(filename + '.xlsx')
pd.read_excel(filename + '.xlsx', 'Sheet1').cumsum().plot()
plt.show()