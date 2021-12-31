import torch
import numpy as np
import random
import actnn.cpp_extension.calc_precision as ext_calc_precision
from sklearn.linear_model import Ridge


# Faster but more memory-consuming AutoPrecision by directly computing 2nd order gradient
class AutoPrecisionFast:
    def __init__(self, schemes, get_grad, bits, dims, max_bits=8):
        self.L = dims.shape[0]
        self.dims = dims
        # Sensitivity for each tensor, tied within each group
        self.C = torch.ones(self.L)

        self.schemes = schemes
        self.get_grad = get_grad

        self.abits = bits
        self.bits = torch.ones(self.L, dtype=torch.int32) * bits
        self.total_bits = bits * dims.sum()
        self.max_bits = max_bits

    def adapt(self):
        # TODO backup the random seed and resume it after adapt
        def get_grad(bits, inject_noise, create_graph):
            torch.manual_seed(0)
            random.seed(0)
            np.random.seed(0)
            # torch.use_deterministic_algorithms(True)
            return self.get_grad(bits, inject_noise, create_graph)

        inject_noise = [False for l in range(self.L)]
        det_grad = get_grad(self.bits, inject_noise, create_graph=False).detach()
        inject_noise = [True for l in range(self.L)]
        grad = get_grad(self.bits, inject_noise, create_graph=True)
        grad_diff = ((grad - det_grad)**2).sum()
        inputs = [scheme.input for scheme in self.schemes]
        grad = torch.autograd.grad(grad_diff, inputs, allow_unused=True)
        for l in range(self.L):
            if grad[l] is None:
                self.C[l] = 0
                print(l, ' is None.')
            else:
                self.C[l] = (grad[l]**2).sum()
            print(self.C[l])

    def refresh_bits(self):
        total_bits = self.total_bits

        self.bits = torch.ones(self.L, dtype=torch.int32) * self.max_bits
        self.bits = ext_calc_precision.calc_precision(self.bits,
                                                      self.C,
                                                      self.dims,
                                                      total_bits)
