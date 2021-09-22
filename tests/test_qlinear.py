import numpy as np
import torch
from torch.nn import functional as F
from actnn.ops import linear as qlinear

from actnn import config

### When set teh group_size to 64 or 256, the abnormal bahavior of qlinear's grad is differernt.
config.group_size = 64
# config.group_size = 256

### If activation_compression_bits set as 2, the returned gradient is correct.
config.activation_compression_bits = [8]
config.simulate = True
def test_qlinear():
    print('======= test qlinear correctness ======')
    # When the batch size is large, the error occurs. If you set the batch size back to 1000,
    # the returned grad is correct.
    data_np = np.random.randn(100000, 128).astype('float32')
    data = torch.tensor(data_np).to("cuda")
    ce = torch.nn.CrossEntropyLoss().cuda()
    y = torch.empty(100000, dtype=torch.long).random_(4).cuda()
    def test_implementation(func, weight, bias):
        pred = func(data, weight, bias)
        pred = F.relu(pred)
        pred = pred.reshape(pred.shape[0], 4, pred.shape[1] // 4).mean(2)
        loss = ce(pred, y)
        weight.grad = None
        bias.grad = None
        loss.backward()
        return weight.grad.cpu().numpy()

    w = torch.randn((128, 128), requires_grad=True, device='cuda')
    b = torch.randn((128,), requires_grad=True, device='cuda')
    qw = torch.randn((128, 128), requires_grad=True, device='cuda')
    qb = torch.randn((128,), requires_grad=True, device='cuda')
    with torch.no_grad():
        qw.copy_(w)
        qb.copy_(b)
    true_grad = test_implementation(F.linear, w, b)
    grads = []
    for i in range(10):
        grads.append(test_implementation(qlinear.apply, qw, qb))
    grads = np.stack(grads, 0)
    grad_mean = grads.mean(0)
    grad_std = grads.std(0)
    bias = np.linalg.norm(grad_mean - true_grad)
    print('Grad = {}, Bias = {}, Std = {}'.format(np.linalg.norm(true_grad), bias, np.linalg.norm(grad_std)))

if __name__ == '__main__':
    test_qlinear()
