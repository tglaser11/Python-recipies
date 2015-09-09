__author__ = 'tom'


import pandas as pd
import pandas.io.data as web

import scipy.stats as scs
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt


# Create function to provide basic stats on an array of values
def print_statistics(array):
    '''
    Prints selected statistics
    :param array: ndarray
    :return:
    '''
    sta = scs.describe(array)
    print "%14s %15s" % ('statistic', 'value')
    print 30 * "-"
    print "%14s %15.5f" % ('size', sta[0])          # number of data points
    print "%14s %15.5f" % ('min', sta[1][0])        # minimum daily return
    print "%14s %15.5f" % ('max', sta[1][1])        # max daily return
    print "%14s %15.5f" % ('mean', sta[2])          # average daily return
    print "%14s %15.5f" % ('std', np.sqrt(sta[3]))  # average daily volatility
    print "%14s %15.5f" % ('skew', sta[4])
    print "%14s %15.5f" % ('kurtosis', sta[5])      # measure of whether the data is peaked or flat compared to normal distribution


def normality_tests(arr):
    print "Skew of data set %14.3f" % scs.skew(arr)
    print "Skew test p-value %14.3f" % scs.skewtest(arr)[1]
    print "Kurt of data set %14.3f" % scs.kurtosis(arr)
    print "Kurt test p-value %14.3f" % scs.kurtosistest(arr)[1]
    print "Norm test p-value %14.3f" % scs.normaltest(arr)[1]


symbols = ['^GDAXI', '^GSPC', 'YHOO', 'MSFT']

data = pd.DataFrame()
for sym in symbols:
    data[sym] = web.DataReader(sym, data_source='yahoo', start='1/1/2006')['Adj Close']

data = data.dropna()
print(data.info())

(data / data.ix[0] * 100).plot(figsize=(8, 6))
plt.show()

# calculate log returns
log_returns = np.log(data / data.shift(1))

# show returns in distribution
log_returns.hist(bins = 50, figsize=(9,6))
plt.show()

# check basic statistics
for sym in symbols:
    print "\nResults for symbol %s" % sym
    print 30 * "-"
    log_data = np.array(log_returns[sym].dropna())
    print_statistics(log_data)

# Test for normality using qq plot
sm.qqplot(log_returns['^GSPC'].dropna(), line='s')
plt.grid(True)
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')
plt.show()


sm.qqplot(log_returns['MSFT'].dropna(), line='s')
plt.grid(True)
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')
plt.show()


# run formal normality tests
for sym in symbols:
    print "\nResults for symbol %s" % sym
    print 32 * "-"
    log_data = np.array(log_returns[sym].dropna())
    normality_tests(log_data)

# IN GENERAL THESE TESTS INDICATE THAT STOCK MARKET RETURNS ARE NOT NORMALLY DISTRIBUTED