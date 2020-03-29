# %%
"""
# Optimization of DLA structure using inverse design
This file will give you a short tutorial on optimizing a DLA structure.

### Import packages
"""

# %%
import numpy as np
import autograd.numpy as npa
from autograd.scipy.signal import convolve as conv

import sys
sys.path.append('../ceviche')

from ceviche import fdfd_hz, jacobian
from ceviche.optimizers import adam_optimize
from ceviche.constants import C_0

import matplotlib.pylab as plt
from skimage.draw import circle

# %%
"""
### Physical parameters
First of all, we choose the physical parameters.
"""

# %%
wavelength = 2e-6                # wavelength of laser light (m)
beta = 0.5                       # speed of electron / speed of light
omega = 2*np.pi*C_0/wavelength   # angular frequencies (1/s)

# %%
"""
### Simulation cell
Next step is to create our simulation cell.
"""

# %%
dL = 2e-8                # grid size (m)
npml = [20, 0]           # size of perfectly matched layers
gap = 15                 # gap size for electron beam
spc = 40                 # space between design region and end of cell
pillar_width = 50        # maximal size of our structure
Nx = spc*2 + gap + pillar_width*2 # number of grid cells in x direction
Ny = int(beta*wavelength/dL)      # number of grid cells in y direction
epsr_max = 12            # permittivity of dielectric structure
epsr_min = 1             # vacuum permittivity

# %%
"""
Our simulation cell consists of\
space  |  pillar  |  gap  |  pillar  |  space
"""

# %%
design_region = np.zeros((Nx, Ny))
design_region[spc:spc+pillar_width, :] = 1
design_region[-spc-pillar_width:-spc, :] = 1
plt.imshow(design_region.T)
plt.show()

# %%
"""
### Material density and blur function
Instead of directly optimizing the relative permittivity, we will optimize the material density in the design region and convert it afterward.\
This will allow us to blur the structure and thereby increase the feature size.
"""

# %%
blur_radius = 7         # smear structure to avoid sharp edges
gamma = 5e3             # make structure binary (vacuum, structure)

def operator_blur(rho, radius, Ny):
    """Blur operator implemented via two-dimensional convolution """
    rr, cc = circle(radius, radius, radius+1)
    kernel = np.zeros((2*radius+1, 2*radius+1), dtype=np.float)
    kernel[rr, cc] = 1
    kernel=kernel/kernel.sum()
    rho_stack = npa.append(rho, npa.append(rho, rho, axis=1), axis=1)
    conv_rho = conv(rho_stack, kernel, mode='full')[radius:-radius,radius+Ny:2*Ny+radius]
    return conv_rho

def make_rho(rho, design_region, radius, Ny):
    """Helper function for applying the blur to only the design region """
    return rho * (design_region==0).astype(np.float) + operator_blur(rho, radius, Ny) * design_region

def operator_proj(rho, eta, gamma):
    """Density projection """
    return npa.divide(npa.tanh(gamma * eta) + npa.tanh(gamma * (rho - eta)), npa.tanh(gamma * eta) + npa.tanh(gamma * (1 - eta)))

def convert_rho_epsr(rho):
    """ Helper function to convert the material density rho to permittivity eps_r """
    return epsr_min + (epsr_max-epsr_min)*operator_proj(make_rho(rho, design_region, blur_radius, Ny), 0.5, gamma)

rho_init = 0.5 * design_region
eps_r = convert_rho_epsr(rho_init)

# %%
"""
### Source term
Now, let's add a source term in the form of an oscillating magnetic dipole in z-direction.
"""

# %%
source = np.zeros((Nx, Ny))
source[npml[0]+10, :] = 1     # use just this line for single side drive
source[-npml[0]-10-1, :] = 1  # add this for dual side drive
plt.imshow(source.T + design_region.T)
plt.show()

# %%
"""
### Simulation run
We are ready to run the FDFD simulation
"""

# %%
F = fdfd_hz(omega, dL, eps_r, npml)   # create an object that stores our parameters and current structure.
Ex, Ey, Hz = F.solve(source)          # calculate the fields for our source term
plt.imshow(np.real(Hz.T), cmap='RdBu')
plt.show()

# %%
"""
### Objective function
In order to run an optimization, we need to define an objective function, which evaluates the quality of the dielectric structure by measuring the acceleration gradient. For this, we define a probe in the center of the electron channel. By introducing a complex wave, we can account for the movement of the electron.
"""

# %%
probe = np.zeros((Nx, Ny), dtype=np.complex128)
probe[spc+pillar_width+gap//2:spc+pillar_width+gap//2+2,:] = np.exp(-1j * 2*np.pi * np.arange(Ny)/Ny)

def objective(rho):
    """ Objective function measuring the acceleration gradient """
    eps_arr = convert_rho_epsr(design_region*rho.reshape((Nx, Ny)))
    F.eps_r = eps_arr
    Ex, Ey, Hz = F.solve(source)
    
    G = npa.abs(npa.sum(Ey*probe))  # calculate objective value G
    
    # normalize G by maximum field
    E_mag = npa.sqrt(npa.square(npa.abs(Ex)) + npa.square(npa.abs(Ey)))
    material_density = (eps_r - 1) / (epsr_max - 1)
    max_field = npa.max(E_mag * material_density)
    
    return G/max_field

# %%
"""
### Optimization run
Finally, we can actually run the optimization.
"""

# %%
N_opts = 50

objective_jac = jacobian(objective, mode='reverse')
(rho_optimum, loss) = adam_optimize(objective, rho_init.flatten(), objective_jac, Nsteps=N_opts, direction='max', step_size=1e-2)

# %%
"""
### Visualization of result
And show the result.
"""

# %%
eps_r = convert_rho_epsr(design_region * rho_optimum.reshape((Nx, Ny)))
G = objective(rho_optimum)
F.eps_r = eps_r
Ex, Ey, Hz = F.solve(source)          # calculate the fields for our source term

plt.imshow(np.real(Ey.T), cmap='RdBu')
plt.contour(eps_r.T, levels=[(epsr_max+1)/2])
plt.show()

# %%
from mpl_toolkits.axes_grid1 import make_axes_locatable
f, ax = plt.subplots(1, 1, tight_layout=True, figsize=(Nx/40+1,Ny/40+0.5))
max_val = np.max(np.abs(Ey))
plt.contour(eps_r.T, levels=[(epsr_max+1)/2])
im = plt.imshow(np.real(Ey.T*np.exp(1j*3)), cmap='RdBu', aspect='auto', vmin=-max_val, vmax=max_val)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = plt.colorbar(im, cax=cax, orientation='vertical');

# %%
