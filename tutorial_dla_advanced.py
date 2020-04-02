import numpy as np
import autograd.numpy as npa
from autograd.scipy.signal import convolve as conv

import sys, os
sys.path.append('../ceviche-master')

import ceviche
from ceviche import fdfd_hz, jacobian
from ceviche.optimizers import adam_optimize
from ceviche.constants import C_0

import matplotlib.pylab as plt
from skimage.draw import circle

from mpl_toolkits.axes_grid1 import make_axes_locatable
import gdspy as gy
from skimage.measure import find_contours

''' Optimization of DLA structure using inverse design '''
# Do you have any special requests?
foldername = '.\optimization\\test_1' # to save the results
Get_gds = True

# Physical parameters
wavelength = 2e-6                # wavelength of laser light (m)
beta = 0.5                       # speed of electron / speed of light
omega = 2*np.pi*C_0/wavelength   # angular frequencies (1/s)

### Simulation cell
dL = 2e-8                # grid size (m)
npml = [20, 0]           # size of perfectly matched layers
gap = 15                 # gap size for electron beam
spc = 40                 # space between design region and end of cell
pillar_width = 50        # maximal size of our structure
Nx = spc*2 + gap + pillar_width*2 # number of grid cells in x direction
Ny = int(beta*wavelength/dL)      # number of grid cells in y direction
epsr_max = 12            # permittivity of dielectric structure
epsr_min = 1             # vacuum permittivity

design_region = np.zeros((Nx, Ny))
design_region[spc:spc+pillar_width, :] = 1
design_region[-spc-pillar_width:-spc, :] = 1

# Blur function
blur_radius = 7         # smear structure to avoid sharp edges
gamma = 5e2             # make structure binary (vacuum, structure)

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

### Source term
source = np.zeros((Nx, Ny))
source[npml[0]+10, :] = 1     # use just this line for single side drive
source[-npml[0]-10-1, :] = 1  # add this for dual side drive

# Setup simulation
F = fdfd_hz(omega, dL, eps_r, npml)   # create an object that stores our parameters and current structure.
Ex, Ey, Hz = F.solve(source)          # calculate the fields for our source term

# Objective function
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

# Optimization run
N_opts = 50

objective_jac = jacobian(objective, mode='reverse')
(rho_optimum, loss) = adam_optimize(objective, rho_init.flatten(), objective_jac, Nsteps=N_opts, direction='max', step_size=1e-2)

## Visualization of result
eps_r = convert_rho_epsr(design_region * rho_optimum.reshape((Nx, Ny)))
G = objective(rho_optimum)
F.eps_r = eps_r
Ex, Ey, Hz = F.solve(source)          # calculate the fields for our source term


f, ax = plt.subplots(1, 1, tight_layout=True, figsize=(Nx/40+1,Ny/40+0.5))
max_val = np.max(np.abs(Ey))
plt.contour(eps_r.T, levels=[(epsr_max+1)/2])
im = plt.imshow(np.real(Ey.T*np.exp(1j*3)), cmap='RdBu', aspect='auto', vmin=-max_val, vmax=max_val)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = plt.colorbar(im, cax=cax, orientation='vertical');

if Get_gds:
    # create folder if needed
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    
    # how many periods do you want to convert into a gds file
    n_periods = 20
    eps_new = np.concatenate((np.ones((Nx,2)), np.tile(eps_r, (1,n_periods)),
                              np.ones((Nx,2))), axis=1)
    
    # make sure that the gds files are named uniquely
    try:
        gds_number = gds_number + 1
    except NameError: gds_number = 0
    cell = gy.Cell('eps'+str(gds_number))
    
    contour_level = 10  # pick a value between epsr_min and epsr_max
    for points in find_contours(eps_new, contour_level):
        cell.add(gy.Polygon(points).fracture())
    
    # write gds file
    gy.write_gds(foldername+'\\gds_file.gds', [cell.name], unit=dL)
    
