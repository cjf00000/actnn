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
    config.compress_activation = True
    bp(inputs[0].cuda(), targets[0].cuda())  # Get dim
    scheme_info = {}
    all_schemes = []
    layer_names = []
    for name, module in m.model.named_modules():
        if hasattr(module, 'scheme') and isinstance(module.scheme, QScheme):
            id = str(type(module)) + str(module.scheme.rank)
            if not id in scheme_info:
                scheme_info[id] = []

            info = scheme_info[id]
            info.append(module.scheme)
            all_schemes.append(module.scheme)
            layer_names.append(name)

    weight_names = list(scheme_info.keys())
    weight_names.sort()

    w = torch.tensor([scheme.dim for scheme in all_schemes], dtype=torch.int32)
    schemes = [scheme_info[w] for w in weight_names]

    print(weight_names)
    L = len(all_schemes)
    L0 = len(weight_names)

    num_var_samples = 1
    # Linear regression
    X = []
    y = []
    total_bits = w.sum() * 2

    def get_bits(C):
        b = torch.ones(L, dtype=torch.int32) * 8
        b = ext_calc_precision.calc_precision(b, C, w, total_bits)

        for i in range(L):
            all_schemes[i].bits = b[i]

        return b

    def add_data():
        X_row = [0 for i in range(L0)]
        for l in range(L0):
            for scheme in schemes[l]:
                bits = scheme.bits
                X_row[l] += 2 ** (-2.0 * bits)

        idx = np.random.randint(0, num_batches-1)
        input, target = inputs[idx].cuda(), targets[idx].cuda()

        grad = bp(input, target)
        var = ((grad - mean_grad) ** 2).sum()

        X.append(X_row)
        y.append(var)

    for iter in range(L * num_var_samples):
        if iter % 100 == 0:
            print(iter)

        C = (2 * (torch.rand([L]) - 0.5)).exp()
        b = get_bits(C)
        add_data()

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32) - sample_var
    #print(y)

    # Do Ridge regression...
    from sklearn.linear_model import Ridge
    # Bagging
    num_bagging_samples = 5

    data_size = X.shape[0]
    coefs = []
    for i in range(num_bagging_samples):
        clf = Ridge(alpha=0.01 * num_batches * num_var_samples, fit_intercept=False)
        idx = torch.randint(data_size, [data_size])
        X_sample = X[idx]
        y_sample = y[idx]
        clf.fit(X_sample.numpy(), y_sample.numpy())
        coefs.append(clf.coef_)

    coefs = np.stack(coefs)
    mean_coef = np.mean(coefs, 0)
    std_coef = np.std(coefs, 0)
    print(mean_coef)
    print(std_coef)

    coef = mean_coef + std_coef
    min_coef = np.min(coef)
    if min_coef < 0:
        coef = coef - min_coef + 1e-8
    print(coef)

    weights = torch.tensor(coef, dtype=torch.float32).abs()    # TODO: replace this abs with bagging
    C = torch.zeros(L)
    for l in range(L0):
        for scheme in schemes[l]:
            scheme.coef = weights[l]

    for l in range(L):
        C[l] = all_schemes[l].coef

    b = get_bits(C)
    for l in range(L):
        print(layer_names[l], C[l], b[l])

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


def test_autoprecision(model_and_loss, optimizer, val_loader, num_batches=20):
    config.activation_compression_bits = [2]
    config.initial_bits = 2
    config.perlayer = False

    model_and_loss.train()
    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module.model
    else:
        m = model_and_loss.model.model

    QScheme.update_scale = False

    def bp(input, target):
        optimizer.zero_grad()
        loss, output = model_and_loss(input, target)
        loss.backward()
        torch.cuda.synchronize()
        grad = []
        for param in m.parameters():
            if param.grad is not None:
                grad.append(param.grad.detach().ravel().cpu())

        return loss, torch.cat(grad, 0)

    # Collect groups and dims
    groups = []
    dims = []
    id2group = {}
    schemes = []
    layer_names = []

    data_iter = enumerate(val_loader)
    for i, (input, target, _) in tqdm(data_iter):
        break

    bp(input.cuda(), target.cuda())  # Get dim

    gcnt = 0
    for name, module in m.named_modules():
        if hasattr(module, 'scheme') and isinstance(module.scheme, QScheme):
            id = str(type(module)) + str(module.scheme.rank)
            # id = str(np.random.rand())
            if not id in id2group:
                print(id)
                id2group[id] = gcnt
                gcnt += 1

            groups.append(id2group[id])
            dims.append(module.scheme.dim)
            schemes.append(module.scheme)
            layer_names.append(name)

    L = len(groups)
    # pickle.dump([layer_names, dims], open('layer_names.pkl', 'wb'))
    # exit(0)

    # Test AutoPrecision
    from actnn import AutoPrecision, AutoPrecisionUCB
    dims = torch.tensor(dims, dtype=torch.long)
    # ap = AutoPrecision(2, groups, dims, warmup_iters=150)
    ap = AutoPrecisionUCB(2, groups, dims, warmup_iters=150)

    # Warmup (collect training data)
    cnt = 0
    for epoch in range(3):
        data_iter = enumerate(val_loader)
        for i, (input, target, _) in tqdm(data_iter):
            cnt += 1

            input = input.cuda()
            target = target.cuda()

            for l in range(L):
                schemes[l].bits = ap.bits[l]

            print(ap.bits)

            _, grad = bp(input, target)
            ap.iterate(grad)

    # collect testing data
    X = []
    y = []
    es = []
    bits = []
    qerrors = []    # Quantization error
    iscales = []    # Input scale
    losses0 = []
    losses1 = []
    isizes = []
    gsizes = []
    batch_grad = 0
    data_iter = enumerate(val_loader)
    cnt = 0
    # for i, (input, target, _) in tqdm(data_iter):
    #     cnt += 1
    #     input = input.cuda()
    #     target = target.cuda()
    #     _, grad = bp(input, target)
    #     batch_grad = batch_grad + grad

    # batch_grad = batch_grad / cnt

    b = torch.load('b.pkl')
    print(b)
    for epoch in range(0):
        data_iter = enumerate(val_loader)
        for l in range(L):
            schemes[l].bits = ap.bits[l]
            # schemes[l].bits = b[l]
        print('bits ', ap.bits)
        # print('bits ', b)
        error = 0
        cnt = 0
        errors = []
        qe = []
        isz = []
        gsz = []
        isc = []
        ls0 = []
        ls1 = []
        for i, (input, target, _) in tqdm(data_iter):
            cnt += 1
            input = input.cuda()
            target = target.cuda()
            loss0, grad = bp(input, target)
            e = (grad - batch_grad)**2
            error = error + e
            errors.append(e.sum())
            qe.append(torch.tensor([scheme.delta.item() for scheme in schemes]))
            isc.append(torch.tensor([scheme.ref_delta.item() for scheme in schemes]))
            isz.append(torch.tensor([scheme.isize.item() for scheme in schemes]))
            gsz.append(torch.tensor([scheme.gsize.item() for scheme in schemes]))

            # Backup parameters
            params = [param for param in m.parameters() if param.grad is not None]
            params_bak = [param.clone() for param in params]

            with torch.no_grad():
                # Update parameters
                for param in params:
                    param -= 1e-3 * param.grad

                # Compute loss
                loss1, _ = model_and_loss(input, target)

                # Restore parameters
                for param, param0 in zip(params, params_bak):
                    param.copy_(param0)

            del params_bak
            # print(loss0.item(), loss1.item())
            ls0.append(loss0.item())
            ls1.append(loss1.item())
            # for l in range(L):
            #     print(schemes[l].delta, schemes[l].ref_delta)

        error = error.sum() / cnt

        bits.append(ap.bits)
        ap.iterate(grad)
        y.append(error)
        es.append(errors)
        qerrors.append(qe)
        iscales.append(isc)
        isizes.append(isz)
        gsizes.append(gsz)
        losses0.append(torch.tensor(ls0))
        losses1.append(torch.tensor(ls1))

        print('error ', error)

        if epoch % 10 == 0:
            torch.save([bits, y, es, isizes, gsizes, losses0, losses1, qerrors, iscales], 'test_set.pkl')

    # Compute Quant Var
    cnt = 0
    num_samples = 3
    quant_var = 0
    overall_var = 0
    data_iter = enumerate(val_loader)
    ap.adaptive = False
    for i, (input, target, _) in tqdm(data_iter):
        input = input.cuda()
        target = target.cuda()
        _, grad = bp(input, target)
        ap.iterate(grad)
        break

    for l in range(L):
        # schemes[l].bits = ap.bits[l]
        schemes[l].bits = b[l]
    #
    for l in range(L):
        print(layer_names[l], schemes[l].bits)
    print(ap.C)
    #
    for i, (input, target, _) in tqdm(data_iter):
        cnt += 1
        if cnt == 10:
            break

        input = input.cuda()
        target = target.cuda()

        config.compress_activation = False
        _, exact_grad = bp(input, target)
        config.compress_activation = True
        for iter in range(num_samples):
            _, grad = bp(input, target)
            quant_var = quant_var + (exact_grad - grad) ** 2
            overall_var = overall_var + (grad - batch_grad) ** 2

    quant_var /= (num_batches * num_samples)
    overall_var /= (num_batches * num_samples)

    print('quant_var = {}, Overall_var = {}'
          .format(quant_var.sum(), overall_var.sum()))
