__author__ = 'tom'


import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

# Random number generators in one and two dimensions
print(npr.rand(10))
print(npr.rand(5,5))

# Random numbers for an interval 5-10
a = 5.
b = 10.
print(npr.rand(10) * (10-5) + a)

# Various random number generators
sample_size = 500
rn1 = npr.rand(sample_size, 3)
rn2 = npr.randint(0, 10, sample_size)
rn3 = npr.sample(size=sample_size)
a = [0, 25, 50, 75, 100]
rn4 = npr.choice(a, size=sample_size)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))

ax1.hist(rn1, bins=25, stacked=True)
ax1.set_title('rand')
ax1.set_ylabel('frequency')
ax1.grid(True)
ax2.hist(rn2, bins=25)
ax2.set_title('randint')
ax2.grid(True)
ax3.hist(rn3, bins=25)
ax3.set_title('sample')
ax3.set_ylabel('frequency')
ax3.grid(True)
ax4.hist(rn4, bins=25)
ax4.set_title('choice')
ax4.grid(True)

plt.show()

## Generating Random Numbers using distributions.
# A few select functions below from numpy random library
sample_size = 500
rn1 = npr.standard_normal(sample_size)          # standard normal distribution
rn2 = npr.normal(100, 20, sample_size)                       # normal distribution
rn3 = npr.chisquare(df=0.5, size=sample_size)   # chi square distribution
rn4 = npr.poisson(lam=1.0, size=sample_size)    # Poisson distribution

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))

ax1.hist(rn1, bins=25)
ax1.set_title('standard normal')
ax1.set_ylabel('frequency')
ax1.grid(True)
ax2.hist(rn2, bins=25)
ax2.set_title('normal (100,20)')
ax2.grid(True)
ax3.hist(rn3, bins=25)
ax3.set_title('chi square')
ax3.set_ylabel('frequency')
ax3.grid(True)
ax4.hist(rn4, bins=25)
ax4.set_title('Poisson')
ax4.grid(True)

plt.show()