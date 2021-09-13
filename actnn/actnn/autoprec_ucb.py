import torch
import numpy as np
import actnn.cpp_extension.calc_precision as ext_calc_precision


# Automatically compute the precision for each tensor with linear bandits
class AutoPrecisionUCB:
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
                 momentum=0.999, warmup_iters=1000, sample_size=1000,
                 initial_bits=2, max_bits=8, adaptive=True, reg=1.0, delta=0.5):
        """
        :param bits: average number of bits (Python int)
        :param groups: group id of each tensor (Python list)
        :param dims: dimensionality of each tensor (torch.long)
        :param warmup_iters: burn-in phase to adapt the batch grad
        :param max_bits: maximum number of bits per each dim
        :param adaptive: use adaptive sensitivity or not.
                         If False, no gradient is required in iterate()
        :param reg: weight decay for ridge regression
        :param delta: confidence level for UCB
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
        self.sample_size = sample_size
        self.reg = reg
        self.delta = delta

        self.bits = torch.ones(self.L, dtype=torch.int32) * initial_bits

        self.membership = torch.zeros(self.L, self.num_groups)
        for i in range(self.L):
            self.membership[i, groups[i]] = 1
        # self.membership[self.L, self.num_groups] = 1

    def generate_ls(self, grad):
        X_row = [0 for i in range(self.L)]
        for l in range(self.L):
            X_row[l] += 2 ** (-2.0 * self.bits[l])

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

        # Update batch gradient
        # beta1 will converge to 1
        self.batch_grad_ema = self.momentum * self.batch_grad_ema + (1 - self.momentum) * grad
        self.beta1 = self.momentum * self.beta1 + (1 - self.momentum)

        if self.iter >= 2 * self.warmup_iters:
            self.update_coef()

    def update_coef(self):
        """
        Update the per-tensor sensitivity by solving the linear system
        """
        # torch.save([self.X, self.y], 'linear_system.pkl')
        X = torch.tensor(self.X)
        y = torch.tensor(self.y)

        N, L = X.shape
        G = self.num_groups
        X = X @ self.membership
        V = torch.eye(G, device=X.device) + X.t() @ X
        Vinv = V.inverse().contiguous()

        X = torch.cat([X, torch.ones([N, 1])], 1)   # [N * G+1]
        V = torch.eye(G + 1, device=X.device) + X.t() @ X
        y = (y.view(-1, 1) * X).sum(0)
        theta = V.inverse() @ y

        self.C = theta

        beta = 1.0 + np.sqrt(2 * np.log(1 / self.delta) + G * np.log(1 + N / G))
        if not self.adaptive:
            beta = 0            # Disable exploration

        print(self.C.shape, theta.shape, beta)
        self.bits = torch.ones(self.L, dtype=torch.int32) * self.max_bits
        groups = torch.tensor(self.groups, dtype=torch.int64)
        self.bits = ext_calc_precision.calc_precision_ucb_g(self.bits,
                    theta, beta, Vinv, self.dims, groups, G, self.total_bits)

        min_coef = theta.min()
        print('Coefficients: ', theta)
        if min_coef < 0:
            print('ActNN Warning: negative coefficient detected ', min_coef)
