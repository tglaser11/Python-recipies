__author__ = 'tom'


import numpy as np
import matplotlib.pyplot as plt


# Approximation example

# Trigonomic Function
def f(x):
    return np.sin(x) + 0.5 * x

x = np.linspace(-2 * np.pi, 2 * np.pi, 50)

# Regression function approximating
reg = np.polyfit(x, f(x), deg=1)
ry = np.polyval(reg,x)

plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, ry, 'r.', label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

# Regression function taking monomials up to 5 as basis function
reg = np.polyfit(x, f(x), deg=5)
ry = np.polyval(reg,x)

plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, ry, 'r.', label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()


print(np.allclose(f(x), ry))  # approximation between function and regression is close but not exact
print(np.sum((f(x) - ry) ** 2) / len(x))  # within .05 mean squared error (MSE)

# Regression on Noisy Data

xn = np.linspace(-2 * np.pi, 2 * np.pi, 50)
xn = xn + 0.15 * np.random.standard_normal(len(xn))
yn = f(xn) + 0.25 * np.random.standard_normal(len(xn))

reg = np.polyfit(xn, yn, 7)
ry = np.polyval(reg,xn)

plt.plot(xn, yn, 'b^', label='f(x)')
plt.plot(xn, ry, 'ro', label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()


#  Regression on Multiple Dimensions
def fm((x,y)):
    return np.sin(x) + 0.25 * x + np.sqrt(y) + 0.05 * y ** 2

x = np.linspace(0, 10, 20)
y = np.linspace(0, 10, 20)
X, Y = np.meshgrid(x , y) # generates 2-d grids out of the 1-d arrays

Z = fm((X, Y))
x = X.flatten()
y = Y.flatten() # yields 1-d arrays from the 2-d grids

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

fig = plt.figure(figsize=(9, 6))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=mpl.cm.coolwarm, linewidth=0.5, antialiased=True)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# set of basis functions for regression
matrix = np.zeros((len(x), 6+1))
matrix[:, 6] = np.sqrt(y)
matrix[:, 5] = np.sin(x)
matrix[:, 4] = y ** 2
matrix[:, 3] = x ** 2
matrix[:, 2] = y
matrix[:, 1] = x
matrix[:, 0] = 1

# print(matrix)

# import least squares regression for multiple dimensions (works for single dim as well)
import statsmodels.api as sm
model = sm.OLS(fm((x,y)), matrix).fit()


print(model.rsquared)