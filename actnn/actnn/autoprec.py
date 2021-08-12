import torch
import numpy as np
import actnn.cpp_extension.calc_precision as ext_calc_precision
from sklearn.linear_model import Ridge


# Automatically compute the precision for each tensor
class AutoPrecision:
    """
    Usage diagram:

    In each iteration:
    1. during forward and back propagation, use self.bits to quantize activations
    2. update optimizer parameters
    3. sample the gradient
    4. call iterate(gradient)
    5. call end_epoch
    """
    def __init__(self, bits, groups, dims, warmup_epochs=2,
                 max_bits=8, adaptive=True, reg=1.0, num_trails=5):
        """
        :param bits: average number of bits (Python int)
        :param groups: group id of each tensor (Python list)
        :param dims: dimensionality of each tensor (torch.long)
        :param warmup_epochs: use adaptive sensitivity after certain epochs
        :param max_bits: maximum number of bits per each dim
        :param adaptive: use adaptive sensitivity or not.
                         If False, no gradient is required in iterate()
        :param reg: weight decay for ridge regression
        :param num_trails: number of Bagging trails for computing the coefficient confidence
        """
        self.L = len(groups)
        self.num_groups = np.max(groups) + 1
        self.groups = groups
        self.dims = dims
        # Sensitivity for each tensor, tied within each group
        self.C = torch.ones(self.L)
        self.iter = 0
        self.epoch = 0
        self.adaptive = adaptive

        self.grad_acc = 0
        self.grad_sqr_acc = 0
        self.batch_grad = 0
        self.sample_var = 0

        self.bits = torch.ones(self.L, dtype=torch.int32) * bits
        self.total_bits = bits * dims.sum()
        self.max_bits = max_bits
        self.X = []     # The linear system, epoch_size * num_groups matrix
        self.y = []

        self.warmup_epochs = warmup_epochs
        self.reg = reg
        self.num_trails = num_trails

    def iterate(self, grad):
        """
        Given the sampled gradient vector (gather selected dimensions from the full gradient)
        This procedure will calculate the bits allocation for next iteration, which
        is available in self.bits.

        If grad is not available, simply pass torch.tensor(1.0)
        """
        grad = grad.detach().cpu()

        # Update batch gradient
        # beta1 will converge to 1
        self.grad_acc = self.grad_acc + grad
        # self.grad_sqr_acc = self.grad_sqr_acc + grad**2     # TODO this is useless...

        # Generate the bits allocation
        # if self.adaptive:   # Inject some noise for exploration
        #     C = self.C * (2 * torch.rand_like(self.C) - 1).exp()
        #     total_bits = (self.total_bits * (0.9 + torch.rand([]) * 0.2)).long()
        # else:
        C = self.C
        total_bits = self.total_bits

        self.bits = torch.ones(self.L, dtype=torch.int32) * self.max_bits
        self.bits = ext_calc_precision.calc_precision(self.bits,
                                                      C, self.dims, total_bits)

        if self.adaptive:
            delta1 = (torch.rand(self.L) < 0.1).int() * 8
            delta2 = (torch.rand(self.L) < 0.05).int() * -1
            self.bits = torch.clamp(self.bits + delta1 + delta2, 1, 8)

            # ind = np.random.randint(0, self.L)
            # self.bits[ind] = 8

        # else:
        #     print('Non adaptive...')
        #     for i in range(self.L):
        #         print(C[i], self.bits[i], self.dims[i])

        # Update the underlying linear system
        X_row = [0 for i in range(self.num_groups)]
        for l in range(self.L):
            X_row[self.groups[l]] += 2 ** (-2.0 * self.bits[l])

        y_row = ((grad - self.batch_grad)**2).sum()
        self.X.append(X_row)
        self.y.append(y_row)

        self.iter += 1

    def end_epoch(self):
        self.batch_grad = self.grad_acc / self.iter
        # second_momentum = self.grad_sqr_acc / self.iter
        # self.sample_var = (second_momentum - self.batch_grad ** 2).sum()

        self.epoch += 1
        if self.epoch >= self.warmup_epochs:
            self.update_coef()

        self.X = []
        self.y = []
        self.iter = 0

        self.grad_acc = 0
        self.grad_sqr_acc = 0

    def update_coef(self):
        """
        Update the per-tensor sensitivity by solving the linear system
        """
        torch.save([self.X, self.y], 'linear_system.pkl')
        X = np.array(self.X)
        y = np.array(self.y)

        data_size = X.shape[0]
        coefs = []
        for i in range(self.num_trails):
            clf = Ridge(alpha=self.reg, fit_intercept=True)
            idx = np.random.randint(0, data_size, [data_size])
            clf.fit(X[idx], y[idx])
            coefs.append(clf.coef_)
            print(clf.intercept_)

        print(coefs)
        coefs = np.stack(coefs)
        mean_coef = np.mean(coefs, 0)
        std_coef = np.std(coefs, 0)

        coef = mean_coef + std_coef
        min_coef = np.min(coef)
        print(coef)
        if min_coef < 0:
            print('Warning: negative coefficient detected ', min_coef)
            coef = coef - min_coef + 1e-8

        self.C = torch.tensor(coef, dtype=torch.float32)
        groups = torch.tensor(self.groups, dtype=torch.long)
        self.C = self.C[groups]
