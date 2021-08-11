import torch
import torch.nn as nn
from actnn import config, QScheme, QBNScheme
from .utils import *
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle
from matplotlib.colors import LogNorm
from copy import deepcopy
import actnn
import actnn.cpp_extension.calc_precision as ext_calc_precision



def get_var(model_and_loss, optimizer, val_loader, num_batches=20, model_state=None):
    num_samples = 3
    # print(QF.num_samples, QF.update_scale, QF.training)
    model_and_loss.train()
    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model.model

    m.set_name()
    weight_names = [layer.layer_name for layer in m.linear_layers]

    data_iter = enumerate(val_loader)
    inputs = []
    targets = []
    indices = []
    config.compress_activation = False
    QScheme.update_scale = True

    def bp(input, target):
        optimizer.zero_grad()
        loss, output = model_and_loss(input, target)
        loss.backward()
        torch.cuda.synchronize()
        grad = {layer.layer_name: layer.weight.grad.detach().cpu() for layer in m.linear_layers}
        return grad #, output

    def save_scale(prefix, index):
        scales = [layer.scheme.scales[index] for layer in m.linear_layers]
        torch.save(scales, prefix + '_scale.pt')

    # First pass
    cnt = 0
    batch_grad = None
    for i, (input, target, index) in tqdm(data_iter):
        QScheme.batch = index
        cnt += 1

        # if i == 0:
        #     print(index)
        #     print('Old scale: ')
        #     scale = m.linear_layers[0].scheme.scales[index]
        #     print(scale)
        #     save_scale('old', index)

        inputs.append(input.clone().cpu())
        targets.append(target.clone().cpu())
        indices.append(index.copy())
        mean_grad = bp(input, target)
        batch_grad = dict_add(batch_grad, mean_grad)

        # if i == 0:
        #     print('New scale: ')
        #     scale = m.linear_layers[0].scheme.scales[index]
        #     print(scale)
        #     save_scale('new', index)
        #
        #     schemes = [layer.scheme for layer in m.linear_layers]
        #     data = [(s.input, s.output, s.grad_input, s.grad_output)
        #             for s in schemes]
        #     weights = [layer.weight for layer in m.linear_layers]
        #     torch.save([data, weights, output, targets], 'data.pt')
        #     exit(0)

        # exit(0)
        if cnt == num_batches:
            break

    num_batches = cnt
    batch_grad = dict_mul(batch_grad, 1.0 / num_batches)
    QScheme.update_scale = False

    print('=======')
    print(m.linear_layers[0].scheme.scales)
    print('=======')

    if model_state is not None:
        model_and_loss.load_model_state(model_state)

    if config.perlayer:
        config.compress_activation = True
        QScheme.batch = indices[0]
        grad = bp(inputs[0].cuda(), targets[0].cuda())
        QScheme.allocate_perlayer()
        QBNScheme.allocate_perlayer()

    for name, module in m.named_modules():
        if hasattr(module, "scheme"):
            print(name, module.scheme.bits)

    total_var = None
    total_error = None
    total_bias = None
    sample_var = None
    for i, input, target, index in tqdm(zip(range(num_batches), inputs, targets, indices)):
        input = input.cuda()
        target = target.cuda()
        QScheme.batch = index
        config.compress_activation = False
        exact_grad = bp(input, target)
        sample_var = dict_add(sample_var, dict_sqr(dict_minus(exact_grad, batch_grad)))

        mean_grad = None
        second_momentum = None
        config.compress_activation = True
        for iter in range(num_samples):
            grad = bp(input, target)

            mean_grad = dict_add(mean_grad, grad)
            total_error = dict_add(total_error, dict_sqr(dict_minus(exact_grad, grad)))
            second_momentum = dict_add(second_momentum, dict_sqr(grad))

        mean_grad = dict_mul(mean_grad, 1.0 / num_samples)
        second_momentum = dict_mul(second_momentum, 1.0 / num_samples)

        grad_bias = dict_sqr(dict_minus(mean_grad, exact_grad))
        total_bias = dict_add(total_bias, grad_bias)

        grad_var = dict_minus(second_momentum, dict_sqr(mean_grad))
        total_var = dict_add(total_var, grad_var)

    total_error = dict_mul(total_error, 1.0 / (num_samples * num_batches))
    total_bias = dict_mul(total_bias, 1.0 / num_batches)
    total_var = dict_mul(total_var, 1.0 / num_batches)

    all_qg = 0
    all_b = 0
    all_s = 0
    for k in total_var:
        g = (batch_grad[k]**2).sum()
        sv = sample_var[k].sum()
        v = total_var[k].sum()
        b = total_bias[k].sum()
        e = total_error[k].sum()
        avg_v = v / total_var[k].numel()

        all_qg += v
        all_b += b
        all_s += sv
        print('{}, grad norm = {}, sample var = {}, bias = {}, var = {}, avg_var = {}, error = {}'.format(k, g, sv, b, v, avg_v, e))

    print('Overall Bias = {}, Var = {}, SampleVar = {}'.format(all_b, all_qg, all_s))


def get_var_during_training(model_and_loss, optimizer, val_loader, num_batches=20):
    num_samples = 3
    # print(QF.num_samples, QF.update_scale, QF.training)
    model_and_loss.train()
    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    m.set_name()
    weight_names = [layer.layer_name for layer in m.linear_layers]

    print('=======')
    print(m.linear_layers[0].scheme.scales)
    print('=======')

    def bp(input, target):
        optimizer.zero_grad()
        loss, output = model_and_loss(input, target)
        loss.backward()
        torch.cuda.synchronize()
        grad = {layer.layer_name: layer.weight.grad.detach().cpu() for layer in m.linear_layers}
        return grad

    QScheme.update_scale = False
    data_iter = enumerate(val_loader)

    total_var = None
    cnt = 0
    for i, (input, target, index) in tqdm(data_iter):
        QScheme.batch = index
        cnt += 1
        if cnt == num_batches:
            break

        mean_grad = None
        second_momentum = None
        for iter in range(num_samples):
            grad = bp(input, target)

            mean_grad = dict_add(mean_grad, grad)
            second_momentum = dict_add(second_momentum, dict_sqr(grad))

        mean_grad = dict_mul(mean_grad, 1.0 / num_samples)
        second_momentum = dict_mul(second_momentum, 1.0 / num_samples)
        grad_var = dict_minus(second_momentum, dict_sqr(mean_grad))
        total_var = dict_add(total_var, grad_var)

    num_batches = cnt
    total_var = dict_mul(total_var, 1.0 / num_batches)

    all_qg = 0
    for k in total_var:
        v = total_var[k].sum()
        avg_v = v / total_var[k].numel()

        all_qg += v
        print('{}, var = {}, avg_var = {}'.format(k, v, avg_v))

    print('Overall Var = {}'.format(all_qg))


def get_var_black_box(model_and_loss, optimizer, val_loader, num_batches=20):
    num_samples = 3
    config.activation_compression_bits = [2]
    config.initial_bits = 2
    config.perlayer = False

    model_and_loss.train()
    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    # m.set_name()
    weight_names = []
    schemes = []
    # m.set_name()
    for name, module in m.model.named_modules():
        if hasattr(module, 'scheme') and isinstance(module.scheme, QScheme):
            schemes.append(module.scheme)
            weight_names.append(name)

    print(weight_names)
    L = len(weight_names)

    def bp(input, target):
        optimizer.zero_grad()
        loss, output = model_and_loss(input, target)
        loss.backward()
        torch.cuda.synchronize()
        grad = []
        for param in m.model.parameters():
            if param.grad is not None:
                grad.append(param.grad.detach().ravel().cpu())

        return torch.cat(grad, 0)

    QScheme.update_scale = False
    data_iter = enumerate(val_loader)

    total_var = None
    cnt = 0
    mean_grad = 0
    second_momentum = 0
    inputs = []
    targets = []
    config.compress_activation = False
    # Compute Sample Var
    for i, (input, target, _) in tqdm(data_iter):
        cnt += 1
        if cnt == num_batches:
            break

        inputs.append(input.clone().cpu())
        targets.append(target.clone().cpu())

        grad = bp(input, target)
        mean_grad = mean_grad + grad
        second_momentum = second_momentum + torch.square(grad)

    num_batches = cnt
    mean_grad = mean_grad / num_batches
    sample_var = second_momentum / cnt - torch.square(mean_grad)
    sample_var = sample_var.sum()

    # Gather samples
    num_var_samples = 1
    # Linear regression
    X = []
    y = []

    config.compress_activation = True
    bp(inputs[0].cuda(), targets[0].cuda())   # Get dim
    w = torch.tensor([scheme.dim for scheme in schemes], dtype=torch.int)
    total_bits = w.sum() * 2

    def get_bits(C):
        b = torch.ones(L, dtype=torch.int32) * 8
        b = ext_calc_precision.calc_precision(b, C, w, total_bits)

        for i in range(L):
            schemes[i].bits = b[i]

        return b

    def add_data():
        X_row = [2 ** (-2.0 * scheme.bits) for scheme in schemes]

        idx = np.random.randint(0, num_batches-1)
        input, target = inputs[idx].cuda(), targets[idx].cuda()

        grad = bp(input, target)
        var = ((grad - mean_grad) ** 2).sum()

        X.append(X_row)
        y.append(var)

    for iter in range(L * num_var_samples * num_batches):
        if iter % 100 == 0:
            print(iter)

        C = (2 * (torch.rand([L]) - 0.5)).exp()
        b = get_bits(C)
        # print(b)
        add_data()

        for scheme in schemes:
            scheme.bits = 2

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32) - sample_var
    print(y)

    # Do Ridge regression...
    from sklearn.linear_model import Ridge
    clf = Ridge(alpha=0.01 * num_batches * num_var_samples, fit_intercept=False)
    clf.fit(X.numpy(), y.numpy())
    print(clf.coef_)

    weights = torch.tensor(clf.coef_, dtype=torch.float32)
    # for i in range(X.shape[0]):
    #     pred = (X[i] * weights).sum()
    #     if i > 0:
    #         print(weight_names[i-1], y[i], pred, (pred - y[i])**2) #, X[i])

    C = weights.abs()
    print(C)
    b = get_bits(C)
    for l in range(L):
        print(weight_names[l], b[l])

    # Compute Quant Var
    quant_var = 0
    overall_var = 0
    for input, target in zip(inputs, targets):
        input = input.cuda()
        target = target.cuda()

        config.compress_activation = False
        exact_grad = bp(input, target)
        config.compress_activation = True
        for iter in range(num_samples):
            grad = bp(input, target)
            quant_var = quant_var + (exact_grad - grad) ** 2
            overall_var = overall_var + (grad - mean_grad) ** 2

    quant_var /= (num_batches * num_samples)
    overall_var /= (num_batches * num_samples)
    print('Sample Var = {}, quant_var = {}, Overall_var = {}'.format(sample_var, quant_var.sum(), overall_var.sum()))
