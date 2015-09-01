__author__ = 'tom'


import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.stats as scs

## Value at Risk

S0 = 100
r = 0.05
sigma = 0.25
T = 30 / 365.       # value at risk over a period of 30 days
I = 10000
ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * npr.standard_normal(I))

# Similate absolute profits and losses relative to value of position today (sorted from severest loss to largest profit)
R_gbm = np.sort(ST - S0)

plt.hist(R_gbm, bins=50)
plt.xlabel('absolute return')
plt.ylabel('frequency')
plt.grid(True)
plt.show()

# Compute and display value at risk (most someone can lose) within a certain confidence level
percs = [0.01, 0.1, 1., 2.5, 5.0, 10.0]             # confidence level is 1 - percs

# scoreatpercentile computes VAR by reading left side of distribution to find where each 'percs' is
# from a cumulative frequency perspective
var = scs.scoreatpercentile(R_gbm, percs)
print "%16s %16s" % ('Confidence Level', 'Value-at-Risk')
print 33 * "-"
for pair in zip(percs, var):
    print "%16.2f %16.3f" % (100 - pair[0], -pair[1])

