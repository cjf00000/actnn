import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge

# X, y = torch.load('linear_system.pkl')
X = torch.load('X.p')
y = torch.load('y.p')
# exit(0)
X = torch.tensor(X)
y = torch.tensor(y)
N = X.shape[0]
# X = torch.cat([X, torch.ones([N, 1])], 1)
F = X.shape[1]
Xy = (X * y.view(-1, 1)).sum(0)

# Bagging
coefs = []
for s in range(3):
    idx = torch.randint(F, [F])
    Xs = X[idx]
    Xys = Xy[idx]
    V = torch.eye(F) * 1 + Xs.t() @ Xs
    coefs.append(V.inverse() @ Xys)

coefs = torch.stack(coefs, 0)
coefs_mean = coefs.mean(0)
coefs_std = coefs.std(0)
alpha = -coefs_mean / coefs_std
m = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
Z = 1 - m.cdf(alpha)
coefs = coefs_mean + m.log_prob(alpha).exp() / Z * coefs_std

# X = np.array(X)
# y = np.array(y)
#
# fig, ax = plt.subplots(4, figsize=(5, 20))
# ax[0].plot(X[:, 0], y, '.')
# ax[1].plot(X[:, 1], y, '.')
# ax[2].plot(X[:, 2], y, '.')
# ax[3].plot(X[:, 3], y, '.')
# fig.savefig('linear.pdf')
#
# y -= 1.4
# clf = Ridge(alpha=1.0, fit_intercept=True)
# clf.fit(X, y)
# print(clf.coef_, clf.intercept_)
# coef0 = clf.coef_

# Xs = {}
# ys = {}
# for i in range(X.shape[0]):
#     X_row = X[i]
#     y_row = y[i]
#     id = str(X_row)
#     if not id in Xs:
#         Xs[id] = []
#         ys[id] = []
#
#     Xs[id].append(X_row)
#     ys[id].append(y_row)
#
# for id in Xs:
#     print(id, np.mean(ys[id]), np.std(ys[id]))



# X, y = torch.load('test_set.pkl')
# X = np.array(X)
# y = np.array(y)
#
# fig, ax = plt.subplots(4, figsize=(5, 20))
# ax[0].plot(X[:, 0], y, '.')
# ax[1].plot(X[:, 1], y, '.')
# ax[2].plot(X[:, 2], y, '.')
# ax[3].plot(X[:, 3], y, '.')
# fig.savefig('linear_test.pdf')
#
# # Partition
# N = X.shape[0]
# perm = np.random.permutation(N)
# X_train = X[perm[:200]]
# y_train = y[perm[:200]]
# X_test = X[perm[200:]]
# y_test = y[perm[200:]]
# clf = Ridge(alpha=1.0, fit_intercept=False)
# clf.fit(X_train, y_train)
# print(clf.coef_, clf.intercept_)
# y_pred = clf.predict(X_test)
# for i in range(X_test.shape[0]):
#     print(X_test[i], y_test[i], y_pred[i])
#
# mse = ((y_test - y_pred)**2).mean()
# print('Test Self MSE = ', mse)
#
# clf.coef_ = coef0
# y_pred = clf.predict(X_test)
# mse = ((y_test - y_pred)**2).mean()
# print('Test Transfer MSE = ', mse)
