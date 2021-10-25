import os
import torch
import numpy as np
import pickle
from sklearn.linear_model import Ridge, Lasso
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

bits[bits >= 8] = 8
layer_names, dims = pickle.load(open('layer_names.pkl', 'rb'))
dims = torch.tensor(dims, dtype=torch.int64)
amt_descent = (losses0 - losses1).mean(1)
amt_descent_noisy = losses0[:, :-1] - losses1[:, 1:]

deltas = torch.load('deltas.pkl')
isizes, gsizes = torch.load('sizes.pkl')

# delta_scale = deltas[:, :1] + 1e-9
# deltas = deltas / delta_scale   # Preprocessing transformations
delta_scale = deltas.std()
deltas = deltas / delta_scale
gsizes = gsizes * 1e7

N, B, L = qerrors.shape

# Preprocess data
P = torch.zeros(L, 10)
for l in range(L):
    if ('conv' in layer_names[l] or 'downsample' in layer_names[l]) and 'layer' in layer_names[l]:
        P[l, 0] = 1
        P[l, 5] = gsizes[l]
    elif 'relu' in layer_names[l]:
        P[l, 1] = 1
        P[l, 6] = gsizes[l]
    elif 'bn' in layer_names[l]:
        P[l, 2] = 1
        P[l, 7] = gsizes[l]
    elif 'fc' in layer_names[l]:
        P[l, 3] = 1
        P[l, 8] = gsizes[l]
    else:
        P[l, 4] = 1
        P[l, 9] = gsizes[l]

Xbits = torch.zeros(N, L)
for n in range(N):
    for l in range(L):
        Xbits[n, l] = deltas[l, bits[n, l] - 1]

# X = Xbits
X = Xbits @ P

num_train = int(N * 0.6)
num_samples = 10000

y_train = y[:num_train]
y_test = y[num_train:]

X_train = X[:num_train]
X_test = X[num_train:]
Xbits_train = Xbits[:num_train]
Xbits_test = Xbits[num_train:]

# Use noisy training signals
indices = torch.randint(0, num_train, [num_samples])
s_indices = torch.randint(0, B, [num_samples])
X_train = X_train[indices]
Xbits_train = Xbits_train[indices]
y_train = torch.tensor([es[indices[i]][s_indices[i]] for i in range(num_samples)])

ridge_pred = 0.0
lasso_pred = 0.0
for iter in range(10):
    clf = Ridge(alpha=1.0, fit_intercept=False)
    clf.fit(X_train, y_train - lasso_pred)

    C = P @ torch.tensor(clf.coef_, dtype=torch.float32).view(-1, 1)

    # Fit the residual with Lasso
    ridge_pred = torch.tensor(clf.predict(X_train), dtype=torch.float32)
    lasso = Lasso(alpha=1.0, fit_intercept=False)
    lasso.fit(Xbits_train, y_train - ridge_pred)
    # lasso.coef_[0] = -0.6
    lasso_pred = torch.tensor(lasso.predict(Xbits_train), dtype=torch.float32)
    C0 = torch.tensor(lasso.coef_, dtype=torch.float32).view(-1, 1)
    C = C + C0

    print('RMSE ', ((ridge_pred + lasso_pred - y_train)**2).mean().sqrt())

# C = torch.tensor(clf.coef_, dtype=torch.float32)
for l in range(L):
    if 'conv' in layer_names[l] or 'fc' in layer_names[l] or 'downsample' in layer_names[l]:
        print(layer_names[l], C[l].item(), deltas[l,0].item(),
              gsizes[l].item(), (C[l] / gsizes[l]).item())

for l in range(L):
    if 'bn' in layer_names[l]:
        print(layer_names[l], C[l].item(),
              gsizes[l].item(), (C[l] / gsizes[l]).item())

b = torch.ones(L, dtype=torch.int32) * 8
b = ext_calc_precision.calc_precision_table(b, deltas, C, dims, 2*dims.sum())     # TODO how good is this selection algorithm...?
print(b)
torch.save(b, 'b.pkl')

print('Intercept: ', clf.intercept_)

y_pred = (Xbits_test @ C).view(-1)
# print(np.stack([y_test, y_pred], 1))

print('RMSE: ', np.sqrt(((y_test - y_pred)**2).mean()))
