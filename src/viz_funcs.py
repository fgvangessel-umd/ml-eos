import os
import torch
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn

def viz_train_loss(error, res_hist, fig_loc):
    '''
    Visualize how loss and residual evolves over training
    loss hist: [nepochs, nsplits]
    res_hist: [nepochs, nsplits, P_res x c2_res x dPdr_res x dPde_res]
    '''
    
    n_epochs = res_hist.shape[0]
    nsplits  = res_hist.shape[1]
    nvars    = res_hist.shape[2]

    split_color = ['k', 'r', 'darkgreen']
    split_linetypes = ['-', '--', ':']
    split_names = ['Train', 'Validation', 'Test']
    var_names = ['Loss (RMSE)', 'P', 'c2', 'dPdr', 'dPde']

    # Create 5 plots (loss and all 4 mean residuals)
    fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(24,20), tight_layout=True, sharex=True)

    for i, (c, l, s) in enumerate(zip(split_color, split_linetypes, split_names)):
        axs[0].plot(np.arange(n_epochs), error[:,i], color=c, linestyle=l, linewidth=3., label=s)
    axs[0].set_ylabel('RMSE', fontsize=20)

    for i, (c, l, s) in enumerate(zip(split_color, split_linetypes, split_names)):
        for j in range(nvars):
            axs[j+1].plot(np.arange(n_epochs), res_hist[:,i,j], color=c, linestyle=l, linewidth=3., label=s)

    for i in range(1, nvars+1):
        axs[i].set_ylabel(var_names[i]+' MAE', fontsize=20)

    axs[-1].set_xlabel('Epochs', fontsize=20)
    axs[-1].legend(fontsize=20)

    for ax in axs:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid()
    
    plt.savefig(fig_loc)
    plt.close()
    
    return None

def viz_eos_data(eos_data, data_train, data_val, data_test, figloc):
    '''
    viz full EOS data surface along with train/validation/test data 
    '''

    nvars   = 4    # Number of eos variables
    nsplits = 4    # Number of data splits (train/validation/test/full)

    split_label = ['Train', 'Validation', 'Test', 'Total']
    var_label = ['Pressure', 'c2', 'dPdr', 'dPde']

    data = np.concatenate((data_train, data_val, data_test), axis=0)

    # Create figure axes
    fig = plt.figure(figsize = (32, 24), tight_layout=True)

    for i, dat in enumerate([data_train, data_val, data_test, data]):
        # Create "Pastel colormap"
        n = dat.shape[0]
        c=0.4
        colors = (1. - c) * plt.get_cmap("RdBu")(np.linspace(0., 1., n)) + c * np.ones((n, 4))
        cmp = ListedColormap(np.flip(colors, axis=0))
        for j in range(nvars):
            iax = nvars*i + j + 1
            ax = fig.add_subplot(4, 4, iax, projection="3d")
            ax.scatter3D(eos_data[:,0], eos_data[:,1], eos_data[:,2+j], c='k', s=10, marker='.', label='Ref. EOS')
            #ax.scatter3D(dat[:, 0], dat[:, 1], dat[:, 2+j], c='r', s=10, label='NN Pred.')
            ax.scatter3D(dat[:,0], dat[:,1], dat[:,2+j], c=dat[:,2], cmap=cmp, s=10,label='NN Pred.')
            ax.view_init(elev=0., azim=270)
    
    '''
    # Plot prediction data
    data = pred_data[0]
    for i in range(nvars):
        ax = fig.add_subplot(4, 4, i+1, projection="3d")
        ax.scatter3D(ref_data[:,0], ref_data[:,1], ref_data[:,i+2], c='k', s=10, marker='.')
        ax.scatter3D(data[:,0], data[:,1], data[:,i+2], c=data[:,2], cmap=cmp, s=10,)
        ax.view_init(elev=0., azim=270)

        ax = fig.add_subplot(4, 4, 12+i+1, projection="3d")
        ax.scatter3D(ref_data[:,0], ref_data[:,1], ref_data[:,i+2], c='k', s=10, marker='.')
        ax.scatter3D(data[:,0], data[:,1], data[:,i+2], c=data[:,2], cmap=cmp, s=10,)
        ax.view_init(elev=0., azim=270)
    '''
    
    # Save fig
    plt.savefig(figloc)
    
    plt.close()
    
    return None

def viz_dist(x, fig_loc):
    '''
    Visualize distribution of inputs/targets
    '''

    ndata = x.shape[1]
    fig, axs = plt.subplots(nrows=1, ncols=ndata, figsize=(4*ndata, 4))
    for i, ax in enumerate(axs):
        ax.hist(x[:,i], bins=40)
    
    # Save fig
    plt.savefig(fig_loc)

    plt.close()

    return None

def viz_pred_dist(eos_data, data_train, data_val, data_test, figloc):
    nvars = 4
    nsplits = 4

    split_label = ['Train', 'Validation', 'Test', 'Total']
    var_label = ['Pressure', 'c2', 'dPdr', 'dPde']

    data = np.concatenate((data_train, data_val, data_test), axis=0)

    fig, axs = plt.subplots(nrows=nsplits, ncols=nvars, figsize=(24, 18))

    for i, dat in enumerate([data_train, data_val, data_test, data]):
        for j in range(nvars):
            axs[i,j].hist(eos_data[:,2+j], bins=40, density=False, label='Ref. EOS')
            axs[i,j].hist(dat[:, 2+j], bins=40, density=False, label='NN Pred.')

        axs[i,0]. set_ylabel(split_label[i], fontsize=20)

    for j in range(nvars):
        axs[0,j].set_title(var_label[j], fontsize=20)

    axs[-1,0].legend(fontsize=20)
    plt.savefig(figloc)
    plt.close()

    return None

def plot_contour(ax, X0, X1, y, xlims, ylims, vlims=[None, None], alpha=0.5, contour_lines=True, contour_labels=True,
                 labels_fs=8, labels_fmt='%d', n_contour_lines=8, contour_color='k', contour_alpha=1,
                 cbar=False, cbar_title='', cmap='RdBu_r', xlabel='', ylabel=''):
    # background surface
    if contour_lines is True:
        contours = ax.contour(X0, X1, y, n_contour_lines, colors=contour_color, alpha=contour_alpha, linestyles='dashed')
        if contour_labels is True:
            _ = ax.clabel(contours, inline=True, fontsize=labels_fs, fmt=labels_fmt)

    y = y[::-1,::-1].T
    y = y[:,::-1]
    y = y[::-1,:]
    mappable = ax.imshow(y, extent=[xlims[0],xlims[1],ylims[0],ylims[1]],
                         origin='lower', cmap=cmap, alpha=alpha, vmin=vlims[0], vmax=vlims[1],
                         aspect=(xlims[1]-xlims[0])/(ylims[1]-ylims[0]))

    if cbar is True:
        cbar = plt.colorbar(mappable=mappable, ax=ax, shrink=0.5)
        cbar.ax.set_ylabel(cbar_title, labelpad=20, rotation=270, fontsize=12)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))

    return mappable

def plot_surface(ax, X0, X1, y, alpha=0.5, cmap='RdBu_r', cbar_format='%.2e', view_angle=[30, -120], xlabel='', ylabel='', zlabel=''):
    mappable = ax.plot_surface(X0, X1, y, alpha=alpha, cmap=cmap)
    cbar = plt.colorbar(mappable=mappable, ax=ax, shrink=0.5, format=cbar_format)
    ax.view_init(view_angle[0], view_angle[1])
    # Set labels for the axes
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_zlabel(zlabel, fontsize=20)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
    ax.ticklabel_format(axis='z', style='sci', scilimits=(-3,3))
    return mappable
