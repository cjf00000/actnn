import torch
import actnn.cpp_extension.calc_precision as ext_calc_precision

L = 8
dims = torch.tensor([1, 10, 100, 1000, 10000, 100000, 1000000, 1e8], dtype=torch.int64)
bits = torch.ones(L, dtype=torch.int32) * 32
C = torch.ones(L)
total_bits = dims.sum() * 2

bits = ext_calc_precision.calc_precision(bits, C, dims, total_bits)
print(bits)