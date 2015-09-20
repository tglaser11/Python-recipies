__author__ = 'tom'

import numpy as np
import pandas as pd
import pandas.io.data as web
from sklearn.decomposition import KernelPCA

symbols = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'GS', 'HD', 'IBM',
           'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV',
           'UNH', 'UTX', 'V', 'VZ', 'WMT', 'XOM', '^DJI']

data = pd.DataFrame()
for sym in symbols:
    data[sym] = web.DataReader(sym, data_source='yahoo')['Adj Close']
data = data.dropna()

dji = pd.DataFrame(data.pop('^DJI'))

scale_function = lambda x: (x - x.mean()) / x.std()

pca = KernelPCA().fit(data.apply(scale_function))

pca.lambdas_[:10].round()

get_we = lambda x: x / x.sum()

get_we(pca.lambdas_)[:10]
get_we(pca.lambdas_)[:5].sum()

pca = KernelPCA(n_components=1).fit(data.apply(scale_function))
dji['PCA_1'] = pca.transform(-data)

import matplotlib.pyplot as plt
dji.apply(scale_function).plot(figsize=(20, 10))
plt.show()

