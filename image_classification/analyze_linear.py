import torch
import pickle
import actnn.cpp_extension.calc_precision as ext_calc_precision

X, y, deltas, delta_normal, coefs, intercept, P, gsizes = torch.load('linear_system.pkl')
layer_names, dims = pickle.load(open('layer_names.pkl', 'rb'))
dims = torch.tensor(dims, dtype=torch.int64)

C = P @ coefs
C0 = P[:, :5] @ coefs[:5]
C1 = P[:, 5:] @ coefs[5:]
L = 141
b = torch.ones(L, dtype=torch.int32) * 8
b = ext_calc_precision.calc_precision_table(b, deltas / delta_normal, C, dims, 2*dims.sum())

print(b)
torch.save(b, 'b.pkl')

gg = gsizes / gsizes.median()

for l in range(L):
    if 'conv' in layer_names[l] or 'fc' in layer_names[l]:
        print(layer_names[l], b[l].item(), C[l].item(), C0[l].item(), C1[l].item())

for l in range(L):
    if 'bn' in layer_names[l]:
        print(layer_names[l], b[l].item(), C[l].item())
