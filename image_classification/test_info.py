import torch

info = torch.load('info.pkl')
gdiffs = [i[0] for i in info]
feats = [i[1] for i in info]
preds = [i[2] for i in info]
targets = [i[3] for i in info]

feat_full = torch.cat(feats, 0)
pred_full = torch.cat(preds, 0)

# mean_feat = feat_full.mean(0, keepdim=True)
# mean_pred = pred_full.mean(0, keepdim=True)

gs = []
for gdiff, feat, pred, target in info:
    y = torch.eye(100)[target]
    p = torch.softmax(pred, 1)
    g = y / p
    gs.append(g)

g_full = torch.cat(gs, 0)
g_mean = g_full.mean(0, keepdim=True)
Xs = []
ys = []
N = len(info)
for i in range(N):
    g_b_mean = gs[i].mean(0)
    g_diff = ((g_b_mean - g_mean)**2).sum(1)

    # feat_diff = ((feat - mean_feat)**2).sum(1).mean()
    # pred_diff = ((pred - mean_pred) ** 2).sum(1).mean()
    # print(gdiff, feat_diff, pred_diff)
    # print(pred_diff / gdiff, gdiff)
    Xs.append(g_diff)
    ys.append(gdiffs[i])

Xs = torch.tensor(Xs)
ys = torch.tensor(ys)
Xs = Xs.view(-1, 1)
from sklearn import linear_model

model = linear_model.LinearRegression()
model.fit(Xs.numpy(), ys.numpy())

beta = (Xs * ys).sum() / (Xs * Xs).sum()
for i in range(Xs.shape[0]):
    print(Xs[i] * beta - ys.mean(), ys[i] - ys.mean())
