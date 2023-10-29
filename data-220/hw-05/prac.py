import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

a0 = 2
a1 = 3

N = 1000

x = np.random.rand(N)

noise = 0.1 * np.random.randn(N)

y = a0 + a1 * x + noise

# plt.plot(x, y)
# plt.show()


X = np.vstack((np.ones(N), x))
a = la.solve(X.T @ X, X.T @ y)

xs = np.linspace(0, 1, 10)
ys = a[0] + a[1] * xs

plt.plot(xs, ys, 'r', linewidth=2)
plt.scatter(x, y)
plt.show()
