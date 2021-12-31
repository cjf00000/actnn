import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys

fig, ax = plt.subplots(figsize=(20, 5))
style = {0: 'x:', 10: '^-', 11: 'x:', 200: 'v-'}
for epoch in [10, 11]:
    with open('sensitivity_{}.log'.format(epoch)) as f:
        info = f.readlines()[-141:]
        ss = []
        for line in info:
            line = line.strip().split()
            sensitivity = float(line[2])
            # if sensitivity < 1:
            ss.append(sensitivity)

    ax.plot(ss, style[epoch], label=str(epoch), alpha=0.5)

ax.set_yscale('log')
ax.legend()
fig.savefig('sensitivity.pdf')