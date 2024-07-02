import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import time
import torch
import glob
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data_funcs import gen_eos_data, vecs_to_array, jwl, scale_data, unscale_data, batch_training_data
from src.nn_funcs import EOSNeuralNetwork, init_xavier, loss_calculation
from src.viz_funcs import viz_train_loss, viz_pred_dist, viz_eos_data
from src.monitor_funcs import plot_initial_dist, print_progress

from utils.printarr import printarr

# Check CUDA availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    dtype = torch.float32
elif torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.float64
    print(torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    dtype = torch.float64
    print ("MPS or CUDA device not found.")

plt.ioff()

###
### Load YAML config file settings
###
with open('config.yaml') as file:
    config_list        = yaml.load(file, Loader=yaml.FullLoader)
    output_dirs        = config_list['dirs']
    ref_data_params    = config_list['ref_data_params']
    scheduler_params   = config_list['scheduler_params']
    training_params    = config_list['training_params']
    checkpoint_params  = config_list['checkpoint_params']

    chckdir = output_dirs['checkpoint_dir']
    figdir  = output_dirs['figure_dir']
    
###
### Generate EOS Data
###

# Load EOS var bounds and define data grid density
rho_min, rho_max = [float(r) for r in ref_data_params['rho_bounds']]
ei_min, ei_max = [float (ei) for ei in ref_data_params['ei_bounds']]
mtl_params = {}
for (param, value) in zip (ref_data_params['eos_param_symbols'], ref_data_params['eos_params']):
    mtl_params[param] = float(value)
ngrid = int(ref_data_params['ngrid'])
data_sample_type = ref_data_params['sample_type']
scaler_type = ref_data_params['scaler_type']

# Initialize network define optimizer, and set loss function
net = EOSNeuralNetwork(dtype).to(device)
net.apply(init_xavier)
optimizer = torch.optim.Adam(net.parameters(), lr=float(training_params['lr']))

# If checkpoint file exists load and initialize training
chckpt_path = chckdir+'final_model.pt'
my_file = Path(chckpt_path)
if my_file.is_file():
    # Load existing model
    print('Loading Checkpoint Model')
    checkpoint = torch.load(chckpt_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Generate data but use pre-existing scaling parameters for normalization
    eos_data, eos_data_scaled, mu, sigma, inputs, targets =\
        gen_eos_data(mtl_params, rho_min, rho_max, ei_min, ei_max, ngrid,\
                        device=device, data_sample_type=data_sample_type, scaler_type=\
                         chckdir+'scaling.txt', dtype=dtype)

else:
    # Generate reference EOS data and create train/validation/test split
    eos_data, eos_data_scaled, mu, sigma, inputs, targets =\
        gen_eos_data(mtl_params, rho_min, rho_max, ei_min, ei_max, ngrid,\
                        device=device, data_sample_type=data_sample_type, scaler_type=scaler_type, dtype=dtype)

    # Save scaling factors (mu/sigma)
    np.savetxt(chckdir+'scaling.txt', \
               np.concatenate((mu.reshape(1,-1), sigma.reshape(1,-1)), axis=0), \
                header=scaler_type, comments='')

print(" Density   Int. Ener  Pressure     c^2        dPdr       dPde")
print("{}". format("   ".join(f"{x:.2e}" for x in eos_data.min(axis=0))))
print("{}". format("   ".join(f"{x:.2e}" for x in eos_data.max(axis=0))))

X_train, X_val, X_test = inputs[0], inputs[1], inputs[2]
y_train, y_val, y_test = targets[0], targets[1], targets[2]

# Visualize initial data distributions
plot_initial_dist(eos_data, inputs, targets)

# Create mini-batches for batched training
batch_size = training_params['batch_size']
X_batch_train, y_batch_train, nbatch, batch_length = batch_training_data(X_train, y_train, batch_size, device, dtype)

'''
# Create LR scheduler
if scheduler_params['scheduler_name']:
    scheduler = ReduceLROnPlateau(optimizer, 
                                    mode=scheduler_params['mode'], 
                                    factor=float(scheduler_params['factor']), 
                                    patience=int(scheduler_params['patience']), 
                                    min_lr=float(scheduler_params['min_lr']), 
                                    verbose=scheduler_params['verbose'],
                                    )
'''

# Train our models
    
# Set number of epochs
n_epochs = int(training_params['n_epochs'])
    
# Allocate arrays to save train/validation/test loss 
nsplits = 3 #train/val/tes
loss_hist = np.zeros((n_epochs, nsplits))   # RMSE Loss
res_hist = np.zeros((n_epochs, nsplits, 4)) # MAE loss for P, c^2, dp/dr, dP/de

# set loss weights [P, c2, dP/dr, dP/de]
weights = [1.0, 0.0, 1.0, 1.0]

# Set params for checkpointing models
save_error = float(checkpoint_params['save_error'])
save_eps   = float(checkpoint_params['save_eps'])

###
### Train Network
###

t0 = time.time()
for i in range(n_epochs):
    #t0_epoch = time.time()

    # Zero temporary arrays for storing loss
    loss_train = 0.
    MSE_train = np.zeros(4)
    MAR_train = np.zeros(4)

    # Loop over min-batches
    for j in range(nbatch):
        x = X_batch_train[j,:batch_length[j],:]
        y = y_batch_train[j,:batch_length[j],:]
    
        batch_loss_train, batch_preds_train, batch_MSE_train, batch_MAR_train = loss_calculation(net, x, y, mu, sigma, weights)
        loss_train  += float(batch_loss_train)/nbatch
        MSE_train   += batch_MSE_train/nbatch
        MAR_train   += batch_MAR_train/nbatch

        # Perform gradient descent
        optimizer.zero_grad()
        batch_loss_train.backward()
        optimizer.step()

        #if scheduler_params['scheduler_name']:
        #    scheduler.step(loss_val)

    # Calculate validaiton and test loss
    loss_val, preds_val, MSE_val, MAR_val = loss_calculation(net, X_val, y_val, mu, sigma, weights)
    loss_test, preds_test, MSE_test, MAR_test = loss_calculation(net, X_test, y_test, mu, sigma, weights)

    # Save total loss, and individual component residual, values. Everything here operates on scaled data!
    loss_hist[i,:]  = np.array([loss_train, loss_val.detach().cpu().numpy(), loss_test.detach().cpu().numpy()])

    for j, arr in enumerate([MAR_train, MAR_val, MAR_test]):
        res_hist[i,j,:] = arr   # mean absolute residual
    
    '''
    # Output progress to screen
    if (i % int(training_params['n_output_epochs']) == 0) or (i==n_epochs-1):
        # Output training data
        print_progress(loss_hist[i,:], res_hist[i, :, :], i)

        _, preds_train, _, _ = loss_calculation(net, X_train, y_train, mu, sigma, weights)
        # Create scaled data tensor
        scaled_data_train = torch.cat((X_train, preds_train), axis=1).detach().cpu().numpy()
        scaled_data_val = torch.cat((X_val, preds_val), axis=1).detach().cpu().numpy()
        scaled_data_test = torch.cat((X_test, preds_test), axis=1).detach().cpu().numpy()

        # Create unscaled data arrays for viz
        data_train = unscale_data(scaled_data_train, mu, sigma)
        data_val = unscale_data(scaled_data_val, mu, sigma)
        data_test = unscale_data(scaled_data_test, mu, sigma)

        
        # Visualize current prediction distribution
        figloc = figdir+'dist_{0:05}'.format(i)
        viz_pred_dist(eos_data, data_train, data_val, data_test, figloc)

        # Visualize current NN prediction surfaces
        figloc = figdir+'surf_{0:05}'.format(i)
        viz_eos_data(eos_data_scaled, scaled_data_train, scaled_data_val, scaled_data_test, figloc)
        

        data = np.concatenate((data_train, data_val, data_test), axis=0)
        print(" Density   Int. Ener  Pressure     c^2        dPdr       dPde")
        print("{}". format("   ".join(f"{x:.2e}" for x in data.min(axis=0))))
        print("{}". format("   ".join(f"{x:.2e}" for x in data.max(axis=0))))
        print('\n\n')

    '''

    # Checkpoint model
    if (loss_test.detach().cpu().numpy() < save_error - save_eps):

        # Output training data
        print_progress(loss_hist[i,:], res_hist[i, :, :], i)

        '''
        # Create scaled data tensor
        scaled_data_train = torch.cat((X_train, preds_train), axis=1).detach().cpu().numpy()
        scaled_data_val = torch.cat((X_val, preds_val), axis=1).detach().cpu().numpy()
        scaled_data_test = torch.cat((X_test, preds_test), axis=1).detach().cpu().numpy()

        # Create unscaled data arrays for viz
        #data_train = unscale_data(scaled_data_train, mu, sigma)

        # Visualize current prediction distribution
        figloc = 'figures/dist_{0:05}'.format(i)
        viz_pred_dist(eos_data_scaled, scaled_data_train, scaled_data_val, scaled_data_test, figloc)

        # Visualize current NN prediction surfaces
        figloc = 'figures/surf_{0:05}'.format(i)
        viz_eos_data(eos_data_scaled, scaled_data_train, scaled_data_val, scaled_data_test, figloc)
        '''
        
        # Checkpoint current best model
        model_id = chckdir+'model_{0:05}.pt'.format(i)
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, model_id)


        # Decrement save error
        save_error = loss_test.detach().cpu().numpy()
    
    #t1_epoch = time.time()
    #print('Epoch Time: {0}'.format(t1_epoch-t0_epoch))

# Print total training time
t1 = time.time()
print('Training Time: {0}'.format(t1-t0))

# Visualize loss evolution during training
fig_loc = figdir+'train_curves.png'
viz_train_loss(loss_hist, res_hist, fig_loc)

# Save loss data
np.savetxt(chckdir+'loss_hist.txt', loss_hist)
#np.savetxt('checkpoints/res_hist.txt', res_hist)

# Print minimum loss information
imin = np.argmin(loss_hist[:,-1])

print('\n\n\nMinimum Loss:\n\n')

print_progress(loss_hist[imin,:], res_hist[imin, :, :], imin)
