# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import pandas.io.data as web

sp500 = web.DataReader('^GSPC', data_source='yahoo', start='1/1/1990', end='12/31/2014')
sp500.info()



sp500['42d'] = np.round(pd.rolling_mean(sp500['Close'], window=42), 2)
sp500['252d'] = np.round(pd.rolling_mean(sp500['Close'], window=252), 2)

sp500[['Close', '42d', '252d']].plot(grid=True, figsize=(8,5))

sp500['42-252'] = sp500['42d'] - sp500['252d']

# SD is signal threshold
SD=50
sp500['Regime']=np.where(sp500['42-252'] > SD, 1, 0)
sp500['Regime']=np.where(sp500['42-252'] < -SD, -1, sp500['Regime'])
sp500['Regime'].value_counts()


sp500['Regime'].plot(lw=1.5)
# plt.ylim([-1.1,1.1])

sp500['Market']=np.log(sp500['Close'] / sp500['Close'].shift(1))
sp500['Strategy']=sp500['Regime'].shift(1)*sp500['Market']
sp500[['Market', 'Strategy']].cumsum().apply(np.exp).plot(grid=True, figsize=(8,5))