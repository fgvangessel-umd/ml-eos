import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
from scipy import optimize

def unpack_params(params_dict):
    gamma = params_dict['gamma']
    q     = params_dict['q']
    qq    = params_dict['qq']
    b     = params_dict['b']
    P_inf = params_dict['P_inf']
    C_v = params_dict['C_v']
    C_p = params_dict['C_p']
    
    return C_p, C_v, gamma, P_inf, b, q, qq

def nasg_eos_caloric_P(v, e, params_dict):
    '''

    Parameters
    ----------
    v : Specific volume [g/cc]
    e : specific internal energy [erg]
    params_dict: Dictionary of EOS parameters (gamma, P_inf, C_v, b, q, q')

    Returns
    -------
    P: Internal pressure [Ba]
    c2: Sound velocity squared [cm^2/ sec^2]
    
    '''
    
    C_p, C_v, gamma, P_inf, b, q, qq = unpack_params(params_dict)
    
    P = (gamma-1)*(e-q)/(v-b) - gamma*P_inf
    c2 = gamma*v**2*(P+P_inf)/(v-b)
    
    return P, c2

def nasg_eos_caloric_e(v, P, params_dict):
    '''

    Parameters
    ----------
    v : Specific volume [g/cc]
    P : internal pressure [Ba]
    params_dict: Dictionary of EOS parameters (gamma, P_inf, C_v, b, q, q')

    Returns
    -------
    e: specific internal energy [erg]
    c2: Sound velocity squared [cm^2/ sec^2]
    
    '''
    
    C_p, C_v, gamma, P_inf, b, q, qq = unpack_params(params_dict)
    
    e = (P + gamma*P_inf)/(gamma - 1)*(v-b) + q 
    c2 = gamma*(gamma-1)*v**2/(v-b)*((e-q)/(v-b) - P_inf)
    
    return e, c2

def nasg_eos_thermal(P, T, params_dict):
    '''
    

    Parameters
    ----------
    P : Internal Pressure
    T : Temperature
    params_dict : params_dict: Dictionary of EOS parameters (gamma, P_inf, C_v, b, q, q')
    Returns
    -------
    v: specific volume
    h: specific enthalpy
    G: Gibbs free energy
    '''
    
    C_p, C_v, gamma, P_inf, b, q, qq = unpack_params(params_dict)
    
    v = (gamma-1)*C_v*T/(P+P_inf) + b
    h = gamma*C_v*T + b*P + q
    G = (gamma*C_v - qq)*T - C_v*T*np.log(T**gamma/((P+P_inf)**(gamma-1))) + b*P + q
    
    return v, h, G

def equilibrium_terms(params_dict_liquid, params_dict_gas):
    
    C_p_l, C_v_l, gamma_l, P_inf_l, b_l, q_l, qq_l = unpack_params(params_dict_liquid)
    C_p_g, C_v_g, gamma_g, P_inf_g, b_g, q_g, qq_g = unpack_params(params_dict_gas)
    
    dC_p = C_p_l - C_p_g
    dC_v = C_v_l - C_v_g
    dq   = q_l - q_g
    dqq  = qq_l - qq_g
    db   = b_l - b_g
    
    # Define pressure, temperature, GFE equilibrium coefficients
    A = (dC_p + dqq)/(C_p_g - C_v_g)
    B = dq/(C_p_g - C_v_g)
    C = -dC_p/(C_p_g - C_v_g)
    D = (C_p_l - C_v_l)/(C_p_g - C_v_g)
    E = db/(C_p_g - C_v_g)
    
    return A, B, C, D, E

def create_function(T, params_dict_liquid, params_dict_gas):
    '''
    Define a lambda function (function we wish to find the root of) which is 
    parameterized by EOS parameters and the temperature value at wish we want 
    the root
    '''
    
    C_p_l, C_v_l, gamma_l, P_inf_l, b_l, q_l, qq_l = unpack_params(params_dict_liquid)
    C_p_g, C_v_g, gamma_g, P_inf_g, b_g, q_g, qq_g = unpack_params(params_dict_gas)
    
    A, B, C, D, E = equilibrium_terms(params_dict_liquid, params_dict_gas)
    
    #return lambda P: np.log(P + P_inf_g) - A - (B + E*P)/T - C*np.log(T) - D*np.log(P + P_inf_l)
    return lambda P: nasg_eos_thermal(P,T,params_dict_liquid)[-1] - nasg_eos_thermal(P,T,params_dict_gas)[-1]
    
def numerical_derivative(f, x, h=1e-5):
    """
    Compute the numerical derivative of f at x using central difference.
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def secant_method(f, x0, x1, tol=1e-6, max_iter=100):
    """
    Implement the secant method for root finding.
    
    Parameters:
    f : function
        The function for which we are searching for a root
    x0, x1 : float
        Initial guesses
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    
    Returns:
    float
        The estimated root
    """
    for i in range(max_iter):
        fx0 = f(x0)
        fx1 = f(x1)
        print(f"{x0:3.2e}, {x1:3.2e}, {fx0:3.2e}, {fx1:3.2e}")
        if abs(fx1) < tol:
            return x1
        if fx0 == fx1:
            raise ValueError("Division by zero encountered. Choose different initial points.")
        x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        x0, x1 = x1, x_new
    raise RuntimeError(f"Failed to converge after {max_iter} iterations")


# Define reference state
r0 = 957.74
v0 = 1/r0
P0 = 1.0453e5
c0 = 1542

# Define liquid parameter dictionary
params_dict_liquid = {'C_p': 4285, 'C_v': 3610, 'gamma': 1.19, 'P_inf': 7028e5, 'b': 6.61e-4, 'q': -1177788, 'qq': 0.0}
params_dict_gas    = {'C_p': 1401, 'C_v': 955., 'gamma': 1.47, 'P_inf': 0.0000, 'b': 0.00000, 'q': 2077616., 'qq': 14317}

# Select temperature value
T = 500
P = 20.0e5


vl, hl, Gl = nasg_eos_thermal(P, T, params_dict_liquid)
vg, hg, Gg = nasg_eos_thermal(P, T, params_dict_gas)

print(f"Liquid Thermodynamic Values: {vl:3.2e}, {hl:3.2e}")
print(f"Gas Thermodynamic Values   : {vg:3.2e}, {hg:3.2e}")
print(f"Latent Heat of Vaporization: {hg - hl:3.2e}")


# Example usage

'''
P = np.linspace(2e6, 4e6)
vl, hl, Gl = nasg_eos_thermal(P, T, params_dict_liquid)
vg, hg, Gg = nasg_eos_thermal(P, T, params_dict_gas)

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12,6), tight_layout=True)
axs[0,0].plot(P, vl, 'r')
axs[1,0].plot(P, hl, 'r')
axs[2,0].plot(P, Gl, 'r')
axs[0,1].plot(P, vg, 'r')
axs[1,1].plot(P, hg, 'r')
axs[2,1].plot(P, Gg, 'r')

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6), tight_layout=True)
ax.plot(P, f(P), c='k')
ax.plot(P, Gl - Gg, c='r')

P0 = 1e6
P1 = 4.e6

# Secant method
secant_root = secant_method(f, P0, P1)
print(f"The root found using the secant method: {secant_root:3.2e}")
'''

temperature = np.linspace(270, 650, 100)
pressure = np.zeros(temperature.shape)

P0 = 1e2
P1 = 1.e8

# Secant method
for i, T in enumerate(temperature):
    f = create_function(T, params_dict_liquid, params_dict_gas)

    '''
    P = np.linspace(P0, P1)
    vl, hl, Gl = nasg_eos_thermal(P, T, params_dict_liquid)
    vg, hg, Gg = nasg_eos_thermal(P, T, params_dict_gas)

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12,6), tight_layout=True)
    axs[0,0].plot(P, vl, 'r')
    axs[1,0].plot(P, hl, 'r')
    axs[2,0].plot(P, Gl, 'r')
    axs[0,1].plot(P, vg, 'r')
    axs[1,1].plot(P, hg, 'r')
    axs[2,1].plot(P, Gg, 'r')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6), tight_layout=True)
    ax.plot(P, f(P), c='k')
    ax.plot(P, Gl - Gg, c='r')
    '''
    
    root = optimize.root_scalar(f, bracket=[P0, P1], method='brentq')
    pressure[i] = root.root
    
# Calculate liquid and gas quantities at aquilibrium
vl, hl, Gl = nasg_eos_thermal(pressure, temperature, params_dict_liquid)
vg, hg, Gg = nasg_eos_thermal(pressure, temperature, params_dict_gas)
Lv = hg - hl
    
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6), tight_layout=True)
ax.plot(temperature, pressure*1e-5, c='b', linewidth=4.)


fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12,12), tight_layout=True)
axs[0,0].plot(temperature, pressure*1e-5, 'b')
axs[1,0].plot(temperature, hl, 'b')
axs[2,0].plot(temperature, vl, 'b')
axs[0,1].plot(temperature, Lv, 'b')
axs[1,1].plot(temperature, hg, 'b')
axs[2,1].plot(temperature, vg, 'b')

for i in range(3):
    axs[i,0].set_xlim(250, 650)
    axs[i,1].set_xlim(250, 650)
    axs[i,0].grid()
    axs[i,1].grid()
    
axs[0,0].set_ylim(0, 250)
axs[1,0].set_ylim(-5.0e5, 2.5e6)
axs[2,0].set_ylim(0.8e-3, 2.8e-3)

axs[0,1].set_ylim(0, 2.5e6)
axs[1,1].set_ylim(2.1e6, 3.0e6)
axs[2,1].set_ylim(1e-3, 1e3)
axs[2,1].set_yscale('log')


'''
# Define thermodynamic ranges
P = np.linspace(1e-3, 250, 10)*1e5
v = np.linspace(0.9, 1.4, 10)*1e-3

e, c2 = nasg_eos_caloric_e(v, P, params_dict_liquid)

PP, cc2 = nasg_eos_caloric_P(v, e, params_dict)
'''

r0 = 957.74
P0 = 1.0453e5
v0 = 1/r0

e0, c20 = nasg_eos_caloric_e(v0, P0, params_dict_liquid)
c0 = np.sqrt(c20)

print(e0, c20)

print(f"Reference liquid Energy Density: {e0:3.2e}")
print(f"Reference Liquid Sound Velocity: {c0:3.2e}")


