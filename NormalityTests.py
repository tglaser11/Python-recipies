__author__ = 'tom'


import numpy as np
np.random.seed(1000)
import scipy.stats as scs
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt


def gen_paths(S0, r, sigma, T, M, I):
    '''  generates Monte Carlo paths for geometric Brownian motion

    :param S0:      (float) initial stock/index value
    :param r:       (float) constant short rate
    :param sigma:   (float) constant volatility
    :param T:       (float) final time horizon
    :param M:       (int) number of time steps/intervals
    :param I:       (int) number of paths to be simulated
    :return:
        paths : ndarray, shape (M + 1, I) simulated paths given the parameters
    '''

    dt = float(T) / M
    paths = np.zeros((M + 1, I), np.float64)
    paths[0] = S0
    for t in range(1, M + 1):
        rand = np.random.standard_normal(I)
        rand = (rand - rand.mean()) / rand.std()
        # Black Schoels Option pricing model
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand)
    return paths

S0 = 100.
r = 0.05
sigma = 0.2
T = 1.0
M = 50
I = 250000

paths = gen_paths(S0, r, sigma, T, M, I)

plt.plot(paths[:,:20])
plt.grid(True)
plt.xlabel('time steps')
plt.ylabel('index level')
plt.show()

# Create daily log returns
log_returns = np.log(paths[1:] / paths[0:-1])

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
    print "%14s %15.5f" % ('kurtosis', sta[5])

print_statistics(log_returns.flatten())  # flatten() takes a 2D array and makes it 1D


# Plot daily returns and compare to theoretical probability distribution function (pdf)
plt.hist(log_returns.flatten(), bins=70, normed=True, label='frequency')
plt.grid(True)
plt.xlabel('log-return')
plt.ylabel('frequency')
x = np.linspace(plt.axis()[0], plt.axis()[1])
plt.plot(x, scs.norm.pdf(x, loc=r / M, scale=sigma / np.sqrt(M)), 'r', lw=2.0, label='pdf')
plt.legend()
plt.show()

# create a function to test normality
# Skewness test -- should be close to zero for normalized data
# Krutosis test -- test whether the kurtosis of sample is normal (close to zero)
# Normality test -- test for normality
def normality_tests(arr):
    print "Skew of data set %14.3f" % scs.skew(arr)
    print "Skew test p-value %14.3f" % scs.skewtest(arr)[1]
    print "Kurt of data set %14.3f" % scs.kurtosis(arr)
    print "Kurt test p-value %14.3f" % scs.kurtosistest(arr)[1]
    print "Norm test p-value %14.3f" % scs.normaltest(arr)[1]

normality_tests(log_returns.flatten())
