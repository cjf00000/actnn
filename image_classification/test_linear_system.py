import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge

X, y = torch.load('linear_system.pkl')
X = np.array(X)
y = np.array(y)

fig, ax = plt.subplots(4, figsize=(5, 20))
ax[0].plot(X[:, 0], y, '.')
ax[1].plot(X[:, 1], y, '.')
ax[2].plot(X[:, 2], y, '.')
ax[3].plot(X[:, 3], y, '.')
fig.savefig('linear.pdf')

y -= 1.4
clf = Ridge(alpha=1.0, fit_intercept=False)
clf.fit(X, y)
print(clf.coef_)
