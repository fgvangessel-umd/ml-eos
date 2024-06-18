import numpy as np
import torch
from torch import nn
from printarr import printarr

###
### Define NN architecture
###
class EOSNeuralNetwork(nn.Module):
    def __init__(self):
        super(EOSNeuralNetwork, self).__init__()
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(2, 50, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(50, 50, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(50, 50, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(50, 1, dtype=torch.float64),
        )

    def forward(self, x):
        logits = self.linear_tanh_stack(x)
        return logits

# Enable Xavier initialization (https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        
def compute_deriv(x, y):
    ''' Compute derivative of network outputs w.r.t. inputs '''
    dP = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]
    return dP

def compute_c2(pred, inputs, mu, sigma):
    ''' Compute JWL sound velocity '''
    # Make forward prediction
    P = pred[:,0]

    dP = compute_deriv(inputs, P)

    # Parse out individual thermodynamic terms
    P = P
    r = inputs[:,0]
    e = inputs[:,1]

    # Extract derivative terms
    dPdr = dP[:,0]
    dPde = dP[:,1]

    # get std devs and means
    mu_r, mu_e, mu_P, mu_c2, mu_dPdr, mu_dPde = mu[0], mu[1], mu[2], mu[3], mu[4], mu[5]
    s_r, s_e, s_P, s_c2, s_dPdr, s_dPde = sigma[0], sigma[1], sigma[2], sigma[3], sigma[4], sigma[5]

    # Calculate NN predicted sound velocity by de-normalizing (Here normalization has been induced by scaling of r, e, P)
    dPdr = s_P/s_r*dPdr
    dPde = s_P/s_e*dPde
    c2 = dPdr + (P*s_P + mu_P)/(r*s_r + mu_r)**2*dPde

    # Scale derivatives and velocity to align with zero-mean unit variance of raw data
    dPdr_scaled = (dPdr-mu_dPdr)/s_dPdr
    dPde_scaled = (dPde-mu_dPde)/s_dPde
    c2_scaled = (c2 - mu_c2)/s_c2 

    # Concatenate derivative quantities
    derivs = torch.cat((c2_scaled[:, None], dPdr_scaled[:, None], dPde_scaled[:, None]), dim=1)

    return derivs

def calc_residuals(preds, targets):
    ''' 
    Calculate pointwise residuals - recall preds/targets have normalized predicted (actual) P, c2, dPdr, dPde,
    in 1, 2, 3, and 4 respectively
    '''

    P_pred = preds[:,0]
    c2_pred = preds[:,1]
    dPdr_pred = preds[:,2]
    dPde_pred = preds[:,3]

    P_true = targets[:,0]
    c2_true = targets[:,1]
    dPdr_true = targets[:,2]
    dPde_true = targets[:,3]
    
    P_res = P_pred - P_true
    c2_res = c2_pred - c2_true
    dPdr_res = dPdr_pred - dPdr_true
    dPde_res = dPde_pred - dPde_true
    
    return P_res, c2_res, dPdr_res, dPde_res

def custom_loss(preds, targets, weights):
    ''' Calc custom MSE composite loss of pressure and pressure derivatives '''
    # Calculate raw residuals
    P_res, c2_res, dPdr_res, dPde_res = calc_residuals(preds, targets)
    
    # Calculate MSEs
    P_MSE = torch.mean(P_res**2)
    c2_MSE = torch.mean(c2_res**2)
    dPdr_MSE = torch.mean(dPdr_res**2)
    dPde_MSE = torch.mean(dPde_res**2)

    # Compose into loss
    loss = P_MSE*weights[0] + dPdr_MSE*weights[2] + dPde_MSE*weights[3] 

    # Save for monitoring
    MSE = np.array([P_MSE.detach().cpu().numpy(), c2_MSE.detach().cpu().numpy(), dPdr_MSE.detach().cpu().numpy(), dPde_MSE.detach().cpu().numpy()])

    # Calculate Mean Absolute Residual
    P_MAR = torch.mean(torch.abs(P_res)).detach().cpu().numpy()
    c2_MAR = torch.mean(torch.abs(c2_res)).detach().cpu().numpy()
    dPdr_MAR = torch.mean(torch.abs(dPdr_res)).detach().cpu().numpy()
    dPde_MAR = torch.mean(torch.abs(dPde_res)).detach().cpu().numpy()

    # Save for monitoring
    MAR = np.array([P_MAR, c2_MAR, dPdr_MAR, dPde_MAR])
    
    return loss, MSE, MAR

def loss_calculation(net, X, y, mu, sigma, weights):
    '''
    Make NN prediction, calculate associated derivatives, calculate custom loss
    '''
    prediction = net(X)
    derivs = compute_c2(prediction, X, mu, sigma) # Ordering: c2, dPdr, dPde
    preds = torch.cat((prediction, derivs), axis=1)
    loss, MSE, MAR = custom_loss(preds, y, weights)

    return loss, preds, MSE, MAR
