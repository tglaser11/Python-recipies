__author__ = 'tom'

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt


## Example Simulation for pricing Options

S0 = 100            # current stock price
r = 0.05            # constant short rate
sigma = 0.25        # constant volatility (standard deviation of past S daily returns
T = 2.0             # time in years
I = 10000           # number of random draws

# Black-Scholes-Merton calculation
ST1 = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * npr.standard_normal(I))

plt.hist(ST1, bins=50)
plt.xlabel('index level')
plt.ylabel('frequency')
plt.grid(True)

plt.show()



# Standard Normal Random Number Generator
def gen_sn(M, I, anti_paths=True, mo_match=True):
    ''' Function to generate random numbers for simulation

    :param M: (int)
        number of time intervals for discretization
    :param I:
        number of paths to be simulated
    :param anti_paths:
        use of anithetic variates
    :param mo_match:
        use of moment matching
    :return:
    '''

    if anti_paths is True:
        sn = npr.standard_normal((M + 1, I / 2))
        sn = np.concatenate((sn, -sn), axis=1)
    else:
        sn = npr.standard_normal((M + 1, I))
    if mo_match is True:
        sn = (sn - sn.mean()) / sn.std()
    return sn




# The function gbm_mcs_amer implements Least Squares regression for American option valuation
S0 = 100.
r = 0.05
sigma = 0.25
T = 1.0
I = 50000
M = 50


def gbm_mcs_amer(K, option='call'):
    ''' Valuation of American option in Black-Scholes-Merton
    by Monte Carlo simulation by LSM algorithm
    :param K: float; (positive) strike price of the option
    :param option: string; type of the option to be called ('call', 'put')
    :return:C0: float; estimated present value of European call option
    '''
    dt = T / M
    df = np.exp(-r * dt)
    # simulation of index levels
    S = np.zeros((M + 1, I))
    S[0] = S0
    sn = gen_sn(M, I)
    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * sn[t])
    # case-based calculation of payoff
    if option == 'call':
        h = np.maximum(S - K, 0)
    else:
        h = np.maximum(K - S, 0)
    # LSM algorithm
    V = np.copy(h)
    for t in range(M - 1, 0, -1):
        reg = np.polyfit(S[t], V[t + 1] * df, 7)
        C = np.polyval(reg, S[t])
        V[t] = np.where(C > h[t], V[t + 1] * df, h[t])
    # MCS estimator
    C0 = df * 1 / I * np.sum(V[1])
    return C0

# evaluate strike price of 110 for a security trading at roughly 100 (value of S0)
print(gbm_mcs_amer(110., option='call'))
print(gbm_mcs_amer(110., option='put'))


