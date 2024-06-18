import torch
import torch.nn as nn
import numpy as np
import sys
import yaml
from printarr import printarr
from matplotlib import pyplot as plt

from data_funcs import gen_eos_data
from nn_funcs import  EOSNeuralNetwork, compute_c2
from viz_funcs import plot_surface, plot_contour

# Load saved NN
model_name=str(sys.argv[1])
ngrid=int(sys.argv[2])
eos=EOSNeuralNetwork()

if torch.cuda.is_available():
    loaded=torch.load(model_name)
    device='cuda'
else:
    loaded=torch.load(model_name,map_location=torch.device('cpu'))
    device='cpu'
eos.load_state_dict(loaded['model_state_dict'])
eos.to(device)


with open('config.yaml') as file:
    config_list        = yaml.load(file, Loader=yaml.FullLoader)
    output_dirs        = config_list['dirs']
    ref_data_params    = config_list['ref_data_params']
    scheduler_params   = config_list['scheduler_params']
    training_params    = config_list['training_params']
    checkpoint_params  = config_list['checkpoint_params']

# ##
# ## Generate EOS Data
# ##

# EOS var bounds
rho_min, rho_max = [float(r) for r in ref_data_params['rho_bounds']]
ei_min, ei_max = [float (ei) for ei in ref_data_params['ei_bounds']]
mtl_params = {}
for (param, value) in zip (ref_data_params['eos_param_symbols'], ref_data_params['eos_params']):
    mtl_params[param] = float(value)

print('Min/Max Density: %4.3e / %4.3e'%(rho_min, rho_max))
print('Min/Max Int. Energy: %4.3e / %4.3e'%(ei_min, ei_max))

# Generate reference EOS data and perform train/validation/test split
eos_data, eos_data_scaled, mu, sigma, inputs, targets =\
            gen_eos_data(mtl_params, rho_min, rho_max, ei_min, ei_max, ngrid,device=device)

# Generate inputs
r    = eos_data[:,0]
e    = eos_data[:,1]
pref = eos_data[:,2]
cref = np.sqrt(eos_data[:,3])

R, E, Pref, Cref = r.reshape(ngrid,ngrid), e.reshape(ngrid,ngrid), pref.reshape(ngrid, ngrid), cref.reshape(ngrid, ngrid)

inputs = np.concatenate((np.ravel(R).reshape(-1,1), np.ravel(E).reshape(-1,1)), axis=1)

# Scale
inputs[:,0] = (inputs[:,0] - mu[0])/sigma[0]
inputs[:,1] = (inputs[:,1] - mu[1])/sigma[1]

# Convert to pt tensor and run prediction
x = torch.tensor(inputs, dtype=torch.float64, requires_grad=True).to(device)
y = eos(x)
derivs = compute_c2(y, x, mu, sigma)

# Convert back to numpy, denormalize, and reshape
outputs = y.detach().cpu().numpy()
c2_scaled = derivs.detach().cpu().numpy()[:,0]
p = outputs*sigma[2] + mu[2]
c2 = c2_scaled*sigma[3] + mu[3]

if np.any(c2<0):
    print('Min/Max Density: %4.3e / %4.3e'%(np.min(r[c2<0]), np.max(r[c2<0])))
    print('Min/Max Int. Energy: %4.3e / %4.3e'%(np.min(e[c2<0]), np.max(e[c2<0])))
    sys.exit('DEBUG')

c  = np.sqrt(c2)

P = p.reshape(ngrid,ngrid)
C = c.reshape(ngrid,ngrid)

# Calculate abs. error and Absolute percent error
P_AE = np.abs(P-Pref)
C_AE = np.abs(C-Cref)
P_PE = np.abs(P-Pref)/Pref*100
C_PE = np.abs(C-Cref)/Cref*100

'''
# Plot pressure & sound velocity predictions

# Create a 3D surface plot
fig = plt.figure(figsize=(30,15))
ax = fig.add_subplot(121, projection='3d')
plot_surface(ax, R, E, P, xlabel='Density[ g/cc]', ylabel='Energy [erg]', zlabel='Pressure [Ba]')
ax = fig.add_subplot(122, projection='3d')
plot_surface(ax, R, E, C, xlabel='Density[ g/cc]', ylabel='Energy [erg]', zlabel='Sound Velocity [cm/sec]',)
plt.show()
plt.savefig('figures/pressure_sound_speed_predictions.png', bbox_inches='tight', dpi=600, pad_inches=1.0)

# Plot pressure and sound velocity abs. error and absolute percent error
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(24,16), tight_layout=True)
plot_contour(axs[0,0], R, E*1e-10, P, n_contour_lines=6, labels_fmt='%.1e', xlims=[rho_min, rho_max],
             ylims=[ei_min*1e-10, ei_max*1e-10], cbar=True, cbar_title='Pressure [Ba]',
             xlabel='Density [g/cc]', ylabel='Energy [$10^{10}$ erg]')
plot_contour(axs[0,1], R, E*1e-10, P_AE, n_contour_lines=4, labels_fmt='%.1e', xlims=[rho_min, rho_max],
             ylims=[ei_min*1e-10, ei_max*1e-10], cbar=True, cbar_title='Pressure Absolute Error [Ba]',
             xlabel='Density [g/cc]', ylabel='Energy [$10^{10}$ erg]')
plot_contour(axs[0,2], R, E*1e-10, P_PE, n_contour_lines=4, labels_fmt='%.1e', xlims=[rho_min, rho_max],
             ylims=[ei_min*1e-10, ei_max*1e-10], cbar=True, cbar_title='Pressure Percent Error',
             xlabel='Density [g/cc]', ylabel='Energy [$10^{10} $erg]')

plot_contour(axs[1,0], R, E*1e-10, C, n_contour_lines=6, labels_fmt='%.1e', xlims=[rho_min, rho_max],
             ylims=[ei_min*1e-10, ei_max*1e-10], cbar=True, cbar_title='Sound Velocity [cm/sec]',
             xlabel='Density [g/cc]', ylabel='Energy [$10^{10}$ erg]')
plot_contour(axs[1,1], R, E*1e-10, C_AE, n_contour_lines=4, labels_fmt='%.1e', xlims=[rho_min, rho_max],
             ylims=[ei_min*1e-10, ei_max*1e-10], cbar=True, cbar_title='Sound Velocity Absolute Error [cm/sec]',
             xlabel='Density [g/cc]', ylabel='Energy [$10^{10} $erg]')
plot_contour(axs[1,2], R, E*1e-10, C_PE, n_contour_lines=4, labels_fmt='%.1e', xlims=[rho_min, rho_max],
             ylims=[ei_min*1e-10, ei_max*1e-10], cbar=True, cbar_title='Sound Velocity Percent Error',
             xlabel='Density [g/cc]', ylabel='Energy [$10^{10} $erg]')

plt.show()
plt.savefig('figures/errors.png', bbox_inches='tight', dpi=600, pad_inches=1.0)
'''

# Calculate errors on fine grid
print('Min. / Max. / Mean Absolute Pressure Error: %5.4e / %5.4e / %5.4e'%(np.min(P_AE), np.max(P_AE),np.mean(P_AE)))
print('Min. / Max. / Mean Absolute Sound Speed Error: %5.4e / %5.4e / %5.4e'%(np.min(C_AE), np.max(C_AE),np.mean(C_AE)))
