import torch
import yaml
from torch import nn
from utils.printarr import printarr
from src.viz_funcs import viz_eos_data, viz_train_loss, viz_dist
from src.data_funcs import concatenate_data

def print_progress(loss, res, iepoch):
    if (iepoch==0):
        print('\n'*20)
        print('{0: <15}|{1: ^45} | {2: ^45} | {3: ^45} | {4: ^45} | {5: ^45}'.\
                format('', 'Loss (RMSE)', 'Pressure (MAE)', 'c2 (MAE)', 'dPdr (MAE)', 'dPde (MAE)'))

        print('{0: <15}|{1: ^45} | {2: ^45} | {3: ^45} | {4: ^45} | {5: ^45}'.\
                format('', '', '', '', '', ''))

        print('{0: <15}|{1: >15}{2: >15}{3: >15} | {4: >15}{5: >15}{6: >15} | {7: >15}{8: >15}{9: >15} | {10: >15}{11: >15}{12: >15} | {13: >15}{14: >15}{15: >15}'.\
        format('Epoch No.', 'Train', 'Val', 'Test', 'Train', 'Val', 'Test', 'Train', 'Val', 'Test', 'Train', 'Val', 'Test', 'Train', 'Val', 'Test'))
        print('-'*253)

    print('{0:<15}|{1:>15.2e}{2:>15.2e}{3:>15.2e} | {4:>15.2e}{5:>15.2e}{6:>15.2e} | {7:>15.2e}{8:>15.2e}{9:>15.2e} | {10:>15.2e}{11:>15.2e}{12:>15.2e} | {13:>15.2e}{14:>15.2e}{15:>15.2e}'.\
            format(iepoch, loss[0], loss[1], loss[2], res[0,0], res[1,0], res[2,0],\
                            res[0,1], res[1,1], res[2,1], res[0,2], res[1,2], res[2,2], res[0,3], res[1,3], res[2,3]))

def viz_progress(scaled_data, data, eos_data_scaled, loss_hist, res_hist, i):
    
    # Visualize distributions
    with open('config.yaml') as file:
        config_list = yaml.load(file, Loader=yaml.FullLoader)
        output_dirs = config_list['dirs']
        figdir = output_dirs['figure_dir']

    dist_fig_loc = figdir+'pred_dist_scaled_iter_{0:05}.png'.format(i)
    viz_dist(scaled_data[:,2:], dist_fig_loc)

    dist_fig_loc = figdir+'pred_dist_iter_{0:05}.png'.format(i)
    viz_dist(data[:,2:], dist_fig_loc)
    
    # Visualize scaled predicitions
    fig_loc = figdir+'scaled_vars_iter_{0:05}.png'.format(i)
    var_labels = ['Pressure (Scaled)', 'c2 (Scaled)', 'dPdr (Scaled)', 'dPde (Scaled)']
    viz_eos_data(eos_data_scaled, [scaled_data], var_labels, fig_loc)

    '''
    P_fig_loc = 'figures/P_scaled_plot_iter_{0:05}.png'.format(i)
    c2_fig_loc = 'figures/c2_scaled_plot_iter_{0:05}.png'.format(i)
    dPdr_fig_loc = 'figures/dPdr_scaled_plot_iter_{0:05}.png'.format(i)
    dPde_fig_loc = 'figures/dPde_scaled_plot_iter_{0:05}.png'.format(i)

    viz_eos_data(eos_data_scaled[:, [0, 1, 2]], scaled_data[:,[0, 1, 2]], 'Pressure (Scaled)', P_fig_loc)
    viz_eos_data(eos_data_scaled[:, [0, 1, 3]], scaled_data[:,[0, 1, 3]], 'c2 (Scaled)', c2_fig_loc)
    viz_eos_data(eos_data_scaled[:, [0, 1, 4]], scaled_data[:,[0, 1, 4]], 'dPdr (Scaled)', dPdr_fig_loc)
    viz_eos_data(eos_data_scaled[:, [0, 1, 5]], scaled_data[:,[0, 1, 5]], 'dPde (Scaled)', dPde_fig_loc)
    '''

    return None

def plot_initial_dist(eos_data, inputs, targets):

    # Visualize distributions
    with open('config.yaml') as file:
        config_list = yaml.load(file, Loader=yaml.FullLoader)
        output_dirs = config_list['dirs']
        figdir = output_dirs['figure_dir']
    
    # Visualize raw and scaled input and target distributions
    viz_dist(eos_data[:,:2], figdir+'input_dist.png')
    viz_dist(eos_data[:,2:], figdir+'target_dist.png')

    # Train distribution
    viz_dist(inputs[0].detach().cpu().numpy(), figdir+'input_train_dist_scaled.png')
    viz_dist(targets[0].detach().cpu().numpy(), figdir+'target_train_dist_scaled.png')

    # Validation distribution
    viz_dist(inputs[1].detach().cpu().numpy(), figdir+'input_val_dist_scaled.png')
    viz_dist(targets[1].detach().cpu().numpy(), figdir+'target_val_dist_scaled.png')

    # Test distribution
    viz_dist(inputs[2].detach().cpu().numpy(), figdir+'input_test_dist_scaled.png')
    viz_dist(targets[2].detach().cpu().numpy(), figdir+'target_test_dist_scaled.png')

    # Total distribution
    x = concatenate_data(inputs[0], inputs[1], inputs[2])
    y = concatenate_data(targets[0], targets[1], targets[2])
    viz_dist(x.detach().cpu().numpy(), figdir+'input_dist_scaled.png')
    viz_dist(y.detach().cpu().numpy(), figdir+'target_dist_scaled.png')

    return None
