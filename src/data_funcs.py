import os
import sys
import torch
import numpy as np
from pathlib import Path
from scipy.stats import qmc
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn

from utils.printarr import printarr

def gen_eos_data(mtl_params, rho_min, rho_max, ei_min, ei_max, n,device='cuda', data_sample_type='grid', scaler_type='standard'):
    '''
    Function that generates eos data, density, internal energy, pressure, pressure derivatives, and sound velocity (both scaled and unscaled)
    '''
    if data_sample_type=='grid':
        # Generate equispaced grid for each input thermodynamic variable
        rho = np.linspace(rho_min, rho_max, n)
        ei = np.linspace(ei_min, ei_max, n)

        # Generate complete thermo domain density-energy pairs
        E, R = np.meshgrid(ei, rho)
        X = np.vstack((np.ravel(R), np.ravel(E))).T
    elif data_sample_type=='lhs':
        # Generate LHS grid for each input thermodynamic variable
        sampler = qmc.LatinHypercube(d=2)
        sample = sampler.random(n=n**2)
        l_bounds = [rho_min, ei_min]
        u_bounds = [rho_max, ei_max]
        X = qmc.scale(sample, l_bounds, u_bounds)


    # Calculate pressure and sound velocity (squared) for each density-energy pair
    P, c2, dPdr, dPde = jwl(X[:,0], X[:,1], mtl_params)

    # Combine JWL EOS data into reference array. We will always keep the ordering convention
    #  [rho, e, P, c2, dPdr, dPde]    or
    #  inputs = [rho, e]
    #  outputs = [P, c2, dPdr, dPde]
    eos_data = vecs_to_array([X[:,0], X[:,1], P, c2, dPdr, dPde]) 

    ### Scale data and move to device
    if scaler_type == 'standard':
        eos_data_scaled, inputs, targets, mu, sigma = scale_data(eos_data, 'standard')
    elif scaler_type == 'minmax':
        eos_data_scaled, inputs, targets, mu, sigma = scale_data(eos_data, 'minmax')
    elif Path(scaler_type).is_file():
        # Use existing scaling parameters
        # Load mean and std. dev used for training
        dat = np.loadtxt(scaler_type, skiprows=1)
        mu = dat[0,:]
        sigma = dat[1,:]
        eos_data_scaled, inputs, targets = custom_scale_data(eos_data, mu, sigma)
    else:
        sys.exit('Scaler type not supported')
        
    # Create train-validation-test split
    inputs_train, X_test, targets_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=1)
    inputs_val, inputs_test, targets_val, targets_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1) 

    # Move all data to tensors
    inputs_train = torch.tensor(inputs_train, dtype=torch.float64, requires_grad=True).to(device)
    inputs_val   = torch.tensor(inputs_val,   dtype=torch.float64, requires_grad=True).to(device)
    inputs_test  = torch.tensor(inputs_test,  dtype=torch.float64, requires_grad=True).to(device)

    targets_train = torch.tensor(targets_train, dtype=torch.float64, requires_grad=True).to(device)
    targets_val   = torch.tensor(targets_val,   dtype=torch.float64, requires_grad=True).to(device)
    targets_test  = torch.tensor(targets_test,  dtype=torch.float64, requires_grad=True).to(device)

    # Aggregate data
    inputs  = [inputs_train, inputs_val, inputs_test]
    targets = [targets_train, targets_val, targets_test]

    return eos_data, eos_data_scaled, mu, sigma, inputs, targets

def vecs_to_array(vecs):
    ''' Reshape n vectors to single array where each array column is a vector '''
    ncols = len(vecs)
    nrows = len(vecs[0])
    A = np.zeros((nrows, ncols))

    for i, vec in enumerate(vecs):
        A[:,i] = vec

    return A
    
def jwl(r, e, mtl_params):
    ''' Calculate pressure and squared sound velocity given JWL mtl parameters'''
    A, B, R1, R2, w, r0 = mtl_params['A'], mtl_params['B'], mtl_params['R1'], \
                          mtl_params['R2'], mtl_params['omega'], mtl_params['rho0']
    
    # Calculate pressure
    bv = r0/r
    exp1 = np.exp(-R1*bv)
    exp2 = np.exp(-R2*bv)

    P    = A*(1-r*w/r0/R1)*exp1 + B*(1-r*w/r0/R2)*exp2 + e*r*w
    
    # Calculate derivative terms    
    dPdr = e*w - A*w/r0/R1*exp1 - B*w/r0/R2*exp2 +\
           r0/r**2*(A*R1*(1 - r*w/r0/R1)*exp1 + B*R2*(1 - r*w/r0/R2)*exp2)

    dPde = r*w
    
    # Calculate sound velocity
    c2   = dPdr + P/r**2*dPde

    return P, c2, dPdr, dPde

def scale_data(data, scaler_type='standard'):
    ''' Scale data to zero mean / unit variance '''

    if scaler_type == 'standard':
        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        inputs, targets = scaled_data[:,:2], scaled_data[:,2:]
    
        # Save standard deviations for descaling NN derivatives
        mu, sigma = scaler.mean_, scaler.scale_    

    elif scaler_type == 'minmax':
        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        inputs, targets = scaled_data[:,:2], scaled_data[:,2:]

        # Save data range and minimum for descaling NN derivatives
        mu, sigma = scaler.data_min_, scaler.data_range_
    else:
        sys.exit('Scaler type not supported')
        
    return scaled_data, inputs, targets, mu, sigma

def custom_scale_data(data, mu, sigma):
    ''' Scale data using custom scaling parameters '''
    scaled_data = np.zeros(data.shape)
    for i in range(data.shape[1]):
        scaled_data[:,i] = (data[:,i] - mu[i])/sigma[i]

    inputs, targets = scaled_data[:,:2], scaled_data[:,2:]
    return scaled_data, inputs, targets

def unscale_data(data, mu, sigma):
    '''
    De-Scales scaled data. Assumes length of sigma/mu equivalent to 1-dimension of data
    '''
    # Move data from torch tensor back to numpy array
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()

    unscaled_data = np.zeros(data.shape)
    for i in range(data.shape[1]):
        unscaled_data[:,i] = data[:,i]*sigma[i] + mu[i]

    return unscaled_data

def concatenate_data(*xs):
    '''
    concatenate pytorch tensors, e.g. xs = [inputs_train, inputs_val, inputs_test] and concatenate
    into single input tensor
    '''

    x = torch.cat(xs, dim=0)

    return x 

# Create mini-batches
def batch_training_data(xtrain, ytrain, batch_size):
    # Calculate total length of dataset and number of batches
    ntrain = xtrain.size()[0]
    nbatch = ntrain//batch_size+1 # Add extra batch to account for situation when number of training data not divisible by batch size

    # Shuffle training data
    n_feature = xtrain.size()[1]
    n_targets = ytrain.size()[1]

    xtrain    = xtrain.detach().cpu().numpy()
    ytrain    = ytrain.detach().cpu().numpy()
    dat       = np.concatenate((xtrain, ytrain), axis=1)

    np.random.shuffle(dat)

    xtrain = dat[:,:n_feature]
    ytrain = dat[:,n_feature:]

    X = np.zeros((nbatch, batch_size, xtrain.shape[1]))
    y = np.zeros((nbatch, batch_size, ytrain.shape[1]))
    batch_length = []

    # Place shuffled data into minibatch arrays
    for i in range(nbatch):
        if (i+1)*batch_size < ntrain:
            X[i,:,:] = xtrain[i*batch_size:(i+1)*batch_size,:]
            y[i,:,:] = ytrain[i*batch_size:(i+1)*batch_size,:]
            batch_length.append(batch_size)
        else:
            X[i,:ntrain-i*batch_size,:] = xtrain[i*batch_size:,:]
            y[i,:ntrain-i*batch_size,:] = ytrain[i*batch_size:,:]
            batch_length.append(ntrain-i*batch_size)

    # Move to GPU
    X = torch.tensor(X, dtype=torch.float64, requires_grad=True).to('cuda')
    y = torch.tensor(y, dtype=torch.float64, requires_grad=True).to('cuda')

    return X, y, nbatch, batch_length
    
