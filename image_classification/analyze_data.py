import os
import torch
import numpy as np
import pickle
from sklearn.linear_model import Ridge
import actnn.cpp_extension.calc_precision as ext_calc_precision

if os.path.exists('test_set_flatten.pkl'):
    bits, y, es, isizes, gsizes, losses0, losses1, qerrors, iscales = torch.load('test_set_flatten.pkl')
else:
    bits, y, es, isizes, gsizes, losses0, losses1, qerrors, iscales = torch.load('test_set.pkl')
    bits = torch.stack(bits, 0)                                 # [sample, L]
    y = torch.stack(y)                                          # [sample]
    es = torch.tensor(es)                                       # [sample, batch]
    losses0 = torch.stack(losses0)
    losses1 = torch.stack(losses1)
    qerrors = torch.stack([torch.stack(s) for s in qerrors])    # [sample, batch, L]
    iscales = torch.stack([torch.stack(s) for s in iscales])    # [sample, batch, L]
    isizes = torch.stack([torch.stack(s) for s in isizes])  # [sample, batch, L]
    gsizes = torch.stack([torch.stack(s) for s in gsizes])  # [sample, batch, L]
    torch.save([bits, y, es, isizes, gsizes, losses0, losses1, qerrors, iscales], 'test_set_flatten.pkl')

layer_names, dims = pickle.load(open('layer_names.pkl', 'rb'))
dims = torch.tensor(dims, dtype=torch.int64)
amt_descent = (losses0 - losses1).mean(1)
amt_descent_noisy = losses0[:, :-1] - losses1[:, 1:]

N, B, L = qerrors.shape
num_train = int(N * 0.6)
num_samples = 10000

y_train = y[:num_train]
y_test = y[num_train:]
# y_train = amt_descent[:num_train] * 10000
# y_test = amt_descent[num_train:] * 10000

# Preprocess data
Xbits = 2 ** (-2 * bits.float())
Xq = qerrors.mean(1)
Xq = Xq / (Xq.max(0, keepdims=True)[0] + 1e-7)

# X_train = Xbits[:num_train]
# X_test = Xbits[num_train:]
X = Xq
# X = Xbits

# Group the layers
membership = torch.zeros(L, 4)
for l in range(L):
    if 'conv' in layer_names[l] or 'downsample' in layer_names[l]:
        membership[l, 0] = 1
    elif 'relu' in layer_names[l]:
        membership[l, 1] = 1
    elif 'bn' in layer_names[l]:
        membership[l, 2] = 1
    else:
        membership[l, 3] = 1

layer_idx = torch.arange(1, L+1).float().view(L, 1)
isizes = isizes.mean([0, 1]) / dims
gsizes = gsizes.mean([0, 1]) / dims
isizes = isizes.view(L, 1)
gsizes = gsizes.view(L, 1)

# X = torch.cat([X @ membership, X @ layer_idx, X @ dims.view(-1, 1).float()], 1)
X = torch.cat([X @ membership, X @ layer_idx], 1)
# X = torch.cat([X @ membership], 1)
# X = torch.cat([X @ membership, X @ layer_idx, X @ isizes, X @ gsizes], 1)
# X = torch.cat([X @ membership, X @ layer_idx, X @ layer_idx**2], 1)
# X = X @ membership
# normalize
# X = X / (X.max(0)[0] + 1e-9)

X_train = X[:num_train]
X_test = X[num_train:]

# Use noisy training signals
indices = torch.randint(0, num_train, [num_samples])
s_indices = torch.randint(0, B, [num_samples])
X_train = X_train[indices]
y_train = torch.tensor([es[indices[i]][s_indices[i]] for i in range(num_samples)])

clf = Ridge(alpha=1.0, fit_intercept=True)
clf.fit(X_train, y_train)

b = torch.ones(L, dtype=torch.int32) * 8
C = membership @ torch.tensor(clf.coef_[:4], dtype=torch.float32) + \
    layer_idx.view(-1) * clf.coef_[4] #+ isizes.view(-1) * clf.coef_[5] + gsizes.view(-1) * clf.coef_[6]
# C = torch.tensor(clf.coef_, dtype=torch.float32)
b = ext_calc_precision.calc_precision(b, C, dims, 2*dims.sum())     # TODO how good is this selection algorithm...?
print(b)

print(clf.coef_)
# for l in range(L):
#     print(layer_names[l], C[l], '\t ', b[l].item())
print('Intercept: ', clf.intercept_)

y_pred = clf.predict(X_test)
# print(np.stack([y_test, y_pred], 1))

print('RMSE: ', np.sqrt(((y_test - y_pred)**2).mean()))

# X_best = (2 ** (-2 * b.float())).view(1, -1)
# #X_best = torch.cat([X_best @ membership, X_best @ layer_idx, X_best @ isizes, X_best @ gsizes], 1)
# X_best = torch.cat([X_best @ membership, X_best @ layer_idx], 1)
# y_best = clf.predict(X_best)
# print(y_best)

# # b = torch.ones_like(b) * 2
torch.save(b, 'b.pkl')

# Solve the Ridge problem manually
X_train = torch.cat([X_train, torch.ones([X_train.shape[0], 1])], 1)
V = X_train.t() @ X_train + torch.eye(X_train.shape[1])
b = (y_train.view(-1, 1) * X_train).sum(0)
theta = V.inverse() @ b

