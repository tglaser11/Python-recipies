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