__author__ = 'tom'


import numpy as np
import pandas as pd
import pandas.io.data as web
import matplotlib.pyplot as plt

symbols = ['AAPL', 'MSFT', 'YHOO', 'DB', 'GLD']
noa = len(symbols)

data = pd.DataFrame()
for sym in symbols:
    data[sym] = web.DataReader(sym, data_source='yahoo', end='2014-09-12')['Adj Close']

data.columns = symbols

(data / data.ix[0] * 100).plot(figsize=(8, 5))
plt.show()

# log returns
rets = np.log(data / data.shift(1))

# Mean-Variance in Mean-Variance Portfolio selection refers to the mean and variance of log returns
# annualize log returns (252 trading days)
print(rets.mean() * 252)

# covariance matrix for assets to see which assets daily returns go up/down together
# positive value indicates direct or increasing linear relationship (move in same directions)
# negative value indicates decreasing relationship (move in opposite directions)
# covariance doesn't indicate anything about strength of relationship -- that is correlation
print(rets.cov() * 252)

# come up with 5 random numbers that add up to 1.0
weights = np.random.random(noa)
weights /= np.sum(weights)      # /= is divide and set equal to

# expected portfolio return given random weights
print("Expected Portfolio Annual Return")
print(np.sum(rets.mean() * weights) * 252)

# expected portfolio variance
# print("Expected Variance")
# print(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))

# expected standard deviation (volatility)
print("Expected Portfolio Volatility")
print(np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights))))

# Previous computations based upon single set of random portfolio weightings
# Next section implements Monte Carlo simulation for generating random portfolio weightings

prets = []
pvols = []
for p in range (2500):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    prets.append(np.sum(rets.mean() * weights) * 252)
    pvols.append(np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights))))

prets = np.array(prets)
pvols = np.array(pvols)

plt.figure(figsize=(10, 6))
plt.scatter(pvols, prets, c=prets / pvols, marker='o')
plt.grid(True)
plt.title('Expected Return & Volatility & Sharpe Ratio')
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio (expected return over risk free rate)')
plt.show()


## Portfolio Optimization Section

def statistics(weights):
    ''' Returns portfolio statistics
    :param weights: array-like.  Weights for different securities in portfolio

    :return:
    pret : (float) expected portfolio return
    pvol : (float) expected portfolio volatility
    pret / pvol : (float) Sharpe ratio for rf=0
    '''
    weights = np.array(weights)
    pret = np.sum(rets.mean() * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
    return np.array([pret, pvol, pret / pvol])

import scipy.optimize as sco

def min_func_sharpe(weights):
    return -statistics(weights)[2]

# Constrain all parameters to add up to 1
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) -1 })

# Bound weights to be within 0 to 1
bnds = tuple((0,1) for x in range(noa))

noa * [1. / noa,]

# Optimize portfolio by minimizing Sharpe Ratio
opts = sco.minimize(min_func_sharpe, noa * [1. / noa,], method='SLSQP', bounds=bnds, constraints=cons)
# Print optimal portfolio composition
print(opts['x'].round(3))
# Print expected portfolio return, volatility and Sharpe ratio
print(statistics(opts['x']).round(3))


# Optimize portfolio by minimizing volatility (or variance)
def min_func_variance(weights):
    return statistics(weights)[1] ** 2

optv = sco.minimize(min_func_variance, noa * [1. / noa,], method='SLSQP', bounds=bnds, constraints=cons)
# Print optimal portfolio composition
print(optv['x'].round(3))
# Print expected portfolio return, volatility and Sharpe ratio
print(statistics(optv['x']).round(3))


# Efficient Frontier
# Use algos below to derive all optimal portfolios with higher return than minimum variance portfolios

def min_func_port(weights):
    return statistics(weights)[1]

# iterate over several target return levels to get minimum target volatilities
trets =np.linspace(0.0, 0.25, 50)
tvols = []
for tret in trets:
    cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0] - tret}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    res = sco.minimize(min_func_port, noa * [1. / noa,], method='SLSQP', bounds=bnds, constraints=cons)
    tvols.append(res['fun'])
tvols = np.array(tvols)

plt.figure(figsize=(8, 4))
plt.scatter(pvols, prets, c=prets / pvols, marker='o')      # random portfolio composition
plt.scatter(tvols, trets, c=trets / tvols, marker='x')      # efficient frontier
plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0], 'r*', markersize=15.0)     # portfolio with highest Sharpe ratio
plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0], 'y*', markersize=15.0)     # minimum variance portfolio
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()



