__author__ = 'tom'


import scipy.interpolate as spi
import numpy as np
import matplotlib.pyplot as plt

## Interpolation

x = np.linspace(-2 * np.pi, 2 * np.pi, 25)

def f(x):
    return np.sin(x) + 0.5 * x

# find B-spline representation of 1-D curve and evaluate
ipo = spi.splrep(x, f(x), k=1)
iy = spi.splev(x, ipo)

plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, iy, 'r.', label='interpolation')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

# confirm numerically
print(np.allclose(f(x), iy))
# should = TRUE


## Integration

import scipy.integrate as sci

def f(x):
    return np.sin(x) + 0.5 * x

a = 0.5 # left integral limit
b = 9.5 # right integral limit
x = np.linspace(0, 10)
y = f(x)

from matplotlib.patches import Polygon

fig, ax = plt.subplots(figsize=(7, 5))
plt.plot(x, y, 'b', linewidth=2)
plt.ylim(ymin=0)

# area under the function
# between lower and upper limit
Ix = np.linspace(a,b)
Iy = f(Ix)
verts = [(a, 0)] + list(zip(Ix, Iy)) + [(b, 0)]
poly = Polygon(verts, facecolor='0.7', edgecolor='0.5')
ax.add_patch(poly)

# show integration plot
plt.show()

# Numerical integration
print(sci.fixed_quad(f, a, b)[0])   # Guassian quadrature integration
print(sci.quad(f, a, b)[0])         # adaptive quadrature integration
print(sci.romberg(f, a, b))         # Romberg integration









