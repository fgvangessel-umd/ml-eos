import os
import numpy as np
from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(14,7))

folders = [r'no_constraints/', r'positive/']
colors = ['navy', 'firebrick']
avg = 0

x = np.linspace(0, 249, 250, endpoint=True)
print(x)

for folder, color in zip(folders, colors):
    data = np.zeros((10, 250, 3))
    for i, fname in enumerate(os.listdir(folder)):
        dat = np.loadtxt(folder+fname)
        data[i,:,:] = dat

        avg += dat

        #ax.plot(dat[:,0], c=color, linestyle='solid', linewidth=0.25, alpha=0.25)
        #ax.plot(dat[:,1], c=color, linestyle='dotted', linewidth=0.25, alpha=0.25)
        #ax.plot(dat[:,2], c=color, linestyle='dashed', linewidth=0.25, alpha=0.25)

    avg = avg/(i+1.)

    ax.plot(x, data[:, :, 0].mean(axis=0), c=color,  linestyle='solid',linewidth=2.)
    ax.plot(x, data[:, :, 1].mean(axis=0), c=color,  linestyle='dotted',linewidth=2.)
    ax.plot(x, data[:, :, 2].mean(axis=0), c=color,  linestyle='dashed',linewidth=2.)

    #ax.fill_between(x, data[:, :, 0].mean(axis=0) - data[:, :, 0].std(axis=0), \
    #                data[:, :, 0].mean(axis=0) + data[:, :, 0].std(axis=0), \
    #                alpha=0.5, color=color)
    


ax.set_yscale('log')
plt.show()
plt.savefig('training.png')
