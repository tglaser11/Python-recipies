

path = '/home/tom/Python/'

import sqlite3 as sq3

query = 'CREATE TABLE numbs (Date date, No1 real, No2 real)'

con = sq3.connect(path + 'numbs.db')

# con.execute(query)

# con.commit()


import datetime as dt
import numpy as np

con.execute('INSERT INTO numbs VALUES(?,?,?)', (dt.datetime.now(), 0.12, 7.3))

data = np.random.standard_normal((10000, 2)).round(5)

for row in data:
    con.execute('INSERT INTO numbs VALUES(?,?,?)', (dt.datetime.now(), row[0], row[1]))

con.commit()

pointer = con.execute('SELECT * FROM numbs')

for i in range(5):
    print pointer.fetchone()

con.close()
