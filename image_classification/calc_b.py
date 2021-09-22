import os
import torch
import numpy as np
import pickle
from sklearn.linear_model import Ridge
import actnn.cpp_extension.calc_precision as ext_calc_precision

C = []
with open('sensitivity.log') as f:
    for line in f:
        C.append(float(line.split()[2]))
layer_names, dims = pickle.load(open('layer_names.pkl', 'rb'))
dims = torch.tensor(dims, dtype=torch.int64)
deltas = torch.load('deltas.pkl')
isizes, gsizes = torch.load('sizes.pkl')

C = torch.tensor(C, dtype=torch.float32)
print(C.sum())
L = 141
b = torch.ones(L, dtype=torch.int32) * 8
# b = ext_calc_precision.calc_precision(b, C, dims, 2*dims.sum())
deltas = deltas / (deltas[:,:1] + 1e-9)
b = ext_calc_precision.calc_precision_table(b, deltas, C, dims, 2*dims.sum())
print(b)
torch.save(b, 'b.pkl')

# Build a low-rank approximation of C...
X = torch.zeros(L, 8)
gsizes = gsizes * 1e7
for l in range(L):
    if 'conv' in layer_names[l] or 'downsample' in layer_names[l]:
        X[l, 0] = 1
        X[l, 4] = gsizes[l]
    elif 'relu' in layer_names[l]:
        X[l, 1] = 1
        X[l, 5] = gsizes[l]
    elif 'bn' in layer_names[l]:
        X[l, 2] = 1
        X[l, 6] = gsizes[l]
    else:
        X[l, 3] = 1
        X[l, 7] = gsizes[l]

# X * bc = C
sample_weight = torch.zeros(L)
for l in range(L):
    sample_weight[l] = deltas[l, b[l]]

# TODO: Is the weight correct?
sample_weight = sample_weight / sample_weight.mean()

clf = Ridge(alpha=1, fit_intercept=False)
clf.fit(X.numpy(), C.numpy(), sample_weight=sample_weight)
print(clf.coef_)
Cp = X @ clf.coef_
C1 = X[:,:4] @ clf.coef_[:4]
C2 = X[:,4:] @ clf.coef_[4:]

for l in range(L):
    if 'conv' in layer_names[l] or 'fc' in layer_names[l]:
        print(layer_names[l], C[l].item(), Cp[l].item(), C1[l].item(), C2[l].item(), gsizes[l].item(), (C[l]/gsizes[l]).item())

for l in range(L):
    if 'bn' in layer_names[l]:
        print(layer_names[l], C[l].item(), Cp[l].item(), C1[l].item(), C2[l].item(), gsizes[l].item(), (C[l]/gsizes[l]).item())

b = ext_calc_precision.calc_precision_table(b, deltas, C, dims, 2*dims.sum())
print(b)
torch.save(b, 'b.pkl')


# for l in range(L):
#     print(layer_names[l], '\t', C[l].item(), '\t', b[l].item())

# Allocate between linear layers
# C1 = []
# D1 = []
# for l in range(L):
#     if 'conv' in layer_names[l] or 'fc' in layer_names[l]:
#         C1.append(C[l])
#         D1.append(dims[l])
#
# C1 = torch.tensor(C1)
# D1 = torch.tensor(D1)
# L1 = C1.shape[0]
# b1 = torch.ones(L1, dtype=torch.int32) * 8
# b1 = ext_calc_precision.calc_precision(b1, C1, D1, 2*D1.sum())
#
# b = torch.ones(L, dtype=torch.int32)
# cnt = 0
# for l in range(L):
#     if 'conv' in layer_names[l] or 'fc' in layer_names[l]:
#         b[l] = b1[cnt]
#         cnt += 1
#     elif 'relu' in layer_names[l]:
#         b[l] = 1
#     else:
#         b[l] = b1[cnt]
# print(b)
# torch.save(b, 'b.pkl')
