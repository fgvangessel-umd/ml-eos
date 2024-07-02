import os
import numpy as np
from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(14,7))

folders = [r'no_constraints/']
avg = 0
data = []

for folder in folders:
    for i, fname in enumerate(os.listdir(folder)):
        dat = np.loadtxt(folder+fname)
        data.append(dat)

        avg += dat

        ax.plot(dat[:,0], c='navy', linewidth=0.25, alpha=0.25)
        ax.plot(dat[:,1], c='firebrick', linewidth=0.25, alpha=0.25)
        ax.plot(dat[:,2], c='grey', linewidth=0.25, alpha=0.25)

avg = avg/(i+1.)

ax.plot(avg[:,0], c='navy', linewidth=2.)
ax.plot(avg[:,1], c='firebrick', linewidth=2.)
ax.plot(avg[:,2], c='grey', linewidth=2.)

ax.set_yscale('log')
plt.show()
plt.savefig('training.png')
