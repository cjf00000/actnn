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
    def __init__(self, bits, groups, dims,
                 momentum=0.999, warmup_iters=1000, update_interval=10, sample_size=1000,
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
        self.C = torch.ones(self.num_groups)
        self.iter = 0
        self.epoch = 0
        self.adaptive = adaptive

        self.batch_grad_ema = 0
        self.beta1 = 0

        self.bits = torch.ones(self.L, dtype=torch.int32) * bits
        self.total_bits = bits * dims.sum()
        self.max_bits = max_bits
        self.X = []     # The linear system, epoch_size * num_groups matrix
        self.y = []

        self.momentum = momentum
        self.warmup_iters = warmup_iters
        self.update_interval = update_interval
        self.sample_size = sample_size
        self.reg = reg
        self.num_trails = num_trails
        self.refresh_bits()

    def refresh_bits(self):
        total_bits = self.total_bits

        groups = torch.tensor(self.groups, dtype=torch.long)
        C = self.C[groups]

        self.bits = torch.ones(self.L, dtype=torch.int32) * self.max_bits
        self.bits = ext_calc_precision.calc_precision(self.bits,
                                                      C, self.dims, total_bits)

        if self.adaptive:
            # b = torch.randint_like(self.bits, 2) * 7 + 1
            # self.bits = b[groups]
            # self.bits = torch.randint_like(self.bits, 8) + 1
        # if self.adaptive:       # TODO control the overall bits
            mask = (torch.rand(self.L) < 0.05).int()
            new_bits = torch.randint_like(self.bits, 2) * 7 + 1
            self.bits = self.bits * (1 - mask) + mask * new_bits
            # print(self.bits)
            # delta1 = (torch.rand(self.L) < 0.1).int() * 8
            # delta2 = (torch.rand(self.L) < 0.05).int() * -1
            # self.bits = torch.clamp(self.bits + delta1 + delta2, 1, 8)

    def generate_ls(self, grad):
        X_row = [0 for i in range(self.num_groups)]
        for l in range(self.L):
            X_row[self.groups[l]] += 2 ** (-2.0 * self.bits[l])

        y_row = ((grad - self.batch_grad_ema / self.beta1)**2).sum()
        return X_row, y_row

    def iterate(self, grad):
        # print(grad)
        """
        Given the sampled gradient vector (gather selected dimensions from the full gradient)
        This procedure will calculate the bits allocation for next iteration, which
        is available in self.bits.

        If grad is not available, simply pass torch.tensor(1.0)
        """
        self.iter += 1
        grad = grad.detach().cpu()

        # Update the underlying linear system
        if self.iter >= self.warmup_iters:
            X_row, y_row = self.generate_ls(grad)
            if y_row < 1e6:
                self.X.append(X_row)
                self.y.append(y_row)
            if len(self.X) > self.sample_size:
                self.X.pop(0)
                self.y.pop(0)

        self.refresh_bits()

        # Update batch gradient
        # beta1 will converge to 1
        self.batch_grad_ema = self.momentum * self.batch_grad_ema + (1 - self.momentum) * grad
        self.beta1 = self.momentum * self.beta1 + (1 - self.momentum)

        if self.iter >= 2 * self.warmup_iters and self.iter % self.update_interval == 0:
            self.update_coef()

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
            # print(clf.intercept_)

        # print(coefs)
        coefs = np.stack(coefs)
        mean_coef = np.mean(coefs, 0)
        std_coef = np.std(coefs, 0)

        # coef = mean_coef + std_coef
        coef = mean_coef
        min_coef = np.min(coef)
        print('Coefficients: ', coef)
        if min_coef < 0:
            print('ActNN Warning: negative coefficient detected ', min_coef)
            coef = coef - min_coef + 1e-8

        self.C = torch.tensor(coef, dtype=torch.float32)
