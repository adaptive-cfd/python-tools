#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:27:43 2024

@author: engels
"""

import bcolors
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import finite_differences
import wabbit_tools
import sys, os
import inifile_tools


def ACM_operator(c0, C_eta, nu, dx, mask=None, U=1.0, order="4th-classical"):
    """
    Assembles the 1D ACM operator in a matrix. 

    Parameters
    ----------
    c0 : float
        Speed of sound.
    C_eta : float
        Penalization parameter.
    nu : float
        Viscosity.
    dx : float
        Lattice spacing.
    mask : array, optional
        Mask function. The default is None.
    U : float, optional
        Advection velocity used in the nonlinear term. Note that when c0 is relatively small, the nonlinear term can become important in the 
        operator eigenvalues, which may lead to choosing a non-stable RKC scheme. If c0 is small (say, near or below 10), then passing a value
        for U may improve the eigenvalue prediction for the nonlinear problem. The default is 1.0 (because the meanflow often is unity, but note that local velocity
                                                                                                   may be significantly larger)
    Returns
    -------
    operator : matrix 

    """
    import numpy as np
    if mask is None:
        nx = 256
    else:
        nx = mask.shape[0]

    L = 1
    x = np.linspace(0.0, L, nx)

    if order == "FD_2nd_central":
        D1 = finite_differences.Dper(nx, dx, [-0.5, 0.0, 0.5])
        D2 = finite_differences.Dper(nx, dx**2, [1.0, -2.0, 1.0])
    elif order == "FD_4th_central":
        D1 = finite_differences.Dper(nx, dx, [1/12, -2/3, 0, +2/3, -1/12] )
        D2 = finite_differences.Dper(nx, dx**2, [-1.0/12.0, 4.0/3.0, -5.0/2.0, 4.0/3.0, -1.0/12.0] )
    elif order == "FD_6th_central":
        D1 = finite_differences.Dper(nx, dx, [-1.0/60.0, 3.0/20.0, -3.0/4.0, 0.0, 3.0/4.0, -3.0/20.0, 1.0/60.0] )
        D2 = finite_differences.Dper(nx, dx**2, [ 1.0/90.0, -3.0/20.0, 3.0/2.0, -49.0/18.0, 3.0/2.0, -3.0/20.0, 1.0/90.0] )
    else:
        raise ValueError("Order not set..")

    if mask is None:
        mask = finite_differences.smoothstep(x, 0.15, 0.0)

    operator = np.zeros( [2*nx, 2*nx] )
    operator[0:nx, 0:nx] = -U*D1 + nu*D2 - np.diag(mask/C_eta) # constant transport (model for NL term), diffusion, penalization
    operator[0:nx, nx:2*nx] = -D1 # pressure gradient
    operator[nx:2*nx,0:nx] = -D1 * c0**2 # divergence

    return operator


def sponge_mask(x, L_sponge, typ):
    if typ == 'soft':
        m1 = finite_differences.smoothstep(x, L_sponge/2.0, L_sponge/2.0)
        m2 = finite_differences.smoothstep(-(x-x[-1]), L_sponge/2.0, L_sponge/2.0)
        y = m1 + m2

    if typ == 'hard':
        m1 = finite_differences.smoothstep(x, L_sponge, 0.)
        m2 = finite_differences.smoothstep(-(x-x[-1]), L_sponge, 0.)
        y = m1 + m2

    return(y)

#------------------------------------------------------------------------------
# PARAMETERS
#------------------------------------------------------------------------------

inifile = sys.argv[1]
if not os.path.isfile(inifile):
    print("ERROR: I did not find any inifile :(")

c0    = inifile_tools.get_ini_parameter(inifile, 'ACM-new', 'c_0')
nu    = inifile_tools.get_ini_parameter(inifile, 'ACM-new', 'nu')
Bs    = inifile_tools.get_ini_parameter(inifile, 'Blocks', 'number_block_nodes')
Jmax  = inifile_tools.get_ini_parameter(inifile, 'Blocks', 'max_treelevel')
dim   = inifile_tools.get_ini_parameter(inifile, 'Domain', 'dim')
C_eta = inifile_tools.get_ini_parameter(inifile, 'VPM', 'C_eta')
CFL   = inifile_tools.get_ini_parameter(inifile, 'Time', 'CFL')
CFL_eta = inifile_tools.get_ini_parameter(inifile, 'Time', 'CFL_eta')
CFL_nu  = inifile_tools.get_ini_parameter(inifile, 'Time', 'CFL_nu', dtype=float, default=0.10)
L       = inifile_tools.get_ini_parameter(inifile, 'Domain', 'domain_size', vector=True)[0]
order   = inifile_tools.get_ini_parameter(inifile, 'Discretization', 'order_discretization', dtype=str)




# this will be the time step used in the simulation
dx     = 2**(-Jmax)*L/Bs
dt_set = np.min( [CFL * dx / c0, CFL_eta*C_eta] )
print("dt_selected: dt_CFL=%e dt_CFLeta=%e used=%e" % (CFL * dx / c0, CFL_eta*C_eta, dt_set))
K_eta  = np.sqrt(nu*C_eta)/dx



# warn if the time step is still determined by C_eta
if np.abs( dt_set  - CFL_eta*C_eta ) <= 1e-6:    
    print("\n\n\n%sWARNING ! POSSIBLE ERROR IN INIFILE%s" % (bcolors.WARNING, bcolors.ENDC))
    print("""The time step dt when using a traditional RK4 scheme must be smaller than C_eta, the penalization
    constant. This is often a severe restriction. This script chooses the best RKC (note C instead of 4) 
    scheme to perform the time integration, and dt may be larger than the penalization parameter C_eta. 
    However, in your INIFILE the restriction dt <= CFL_eta * C_eta is active, and the constant is 
    set to %sCFL_eta=%2.2f%s. Therefore the time step is STILL chosen very small !!! 
    %sThis is probably not intended -> change CFL_eta=99999 or any other large value!%s
         """ % (bcolors.FAIL, CFL_eta, bcolors.ENDC, bcolors.WARNING, bcolors.ENDC))
    choice = input("Enter y to change this automatically, or any other letter to keep it as is...")
    
    if choice == 'y':
        print('Updating as per your wish...')
        inifile_tools.replace_ini_value(inifile, 'Time', 'CFL_eta', '99999')
        # reload newly set value
        CFL_eta = inifile_tools.get_ini_parameter(inifile, 'Time', 'CFL_eta')
        
    print("\n\n")
    
if CFL_nu < 1.0:    
    print("\n\n\n%sWARNING ! POSSIBLE ERROR IN INIFILE%s" % (bcolors.WARNING, bcolors.ENDC))
    print("""The constant CFL_nu (which determines the time step due to the viscous operator)
    is currently set to a value < 1.0. It should be set to a large value, to avoid that WABBIT
    sets too small time steps -> change CFL_nu=99999 or any other large value!""")
    choice = input("Enter y to change this automatically, or any other letter to keep it as is...")
    
    if choice == 'y':
        print('Updating as per your wish...')
        inifile_tools.replace_ini_value(inifile, 'Time', 'CFL_nu', '99999')
        # reload newly set value
        CFL_nu = inifile_tools.get_ini_parameter(inifile, 'Time', 'CFL_nu', dtype=float, default=0.10)
        
    print("\n\n")
    

# cost of a reference RK4 simulation
CFL4   = 1.0
dt4    = min([0.094*dx**2/nu, 0.99*C_eta, CFL4*dx/c0])
cost4  = np.round(4.0 * 1.0 / dt4)


print(";-------------------")
print("; Using RK4, the cost would be %i NRHS/T dt=%e" % (cost4, dt4) )

#%% obtain eigenvalues of the discrete 1D acm operator

# define some (arbitrary) mask function
x        = np.linspace( start=0.0, stop=1.0, num=512, endpoint=False)
L_sponge = 0.35
mask     = sponge_mask(x, L_sponge, 'hard')

# find best RKC scheme given CFL, but using 1D operator
o = ACM_operator(c0, C_eta, nu, dx, mask, order=order)
eigenvalues, dummy = np.linalg.eig(o)

#%% scale the 1D eigenvalues to 3D and find the best RKC scheme to integrate it

# For the low Re-flyers (and only for those the RKC scheme makes sense), the transport 
# (nonlinear + pressure waves) are relatively slow and can almost be neglected. The remaining
# terms are the diffusion (laplace), and the penalization. Both discrete operators are positive
# semi-definite, and the penalization is even a diagonal matrix.
# Now, given that the penalization parameter C_\eta is linked to the viscosity \nu, we only 
# have to look at the coupling constant, which is K_\eta (or its square).
# If K_\eta is large, say 3.0, then we recover the eigenvalue ratio of the discrete 
#        laplacian, which is simply the dimension D. Hence, we need to multiply the 
#        \lambda_1D by 3 in the 3D case.
# If K_\eta is small, say 0.1, then the penalization term dominates, and as this
#         is a diagonal matrix, its eigenvalue is simply -1/C_eta and independent 
#         of the dimension D.
# There is no direct analytical way to determine the eigenvalues of the sum of both
# matrices, if they are both significant. We determined them numerically, and determined
# as a function of K_\eta, how the real part needs to be scaled from the 1D to the 2D/3D case.
# Those factors are listed below and interpolated to scale the 1D eigenvalues appropriately. 
if dim == 3:
    Scale_vct = [1.0923, 1.0923, 1.3211, 1.5941, 1.8467, 2.0553, 2.2197, 2.3478, 2.4481, 2.5278, 2.5923, 2.6455, 2.6900, 2.7276, 2.7594, 2.7864, 2.8094, 2.8291, 2.8460, 2.8606, 3.0000, 3.0000]
    Keta_vct  = [0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000, 0.9000, 1.0000, 1.1000, 1.2000, 1.3000, 1.4000, 1.5000, 1.6000, 1.7000, 1.8000, 1.9000, 2.0000, 100.00]
elif dim == 2:
    Scale_vct = [1.0501, 1.0501, 1.1741, 1.3212, 1.4560, 1.5661, 1.6514, 1.7167, 1.7665, 1.8049, 1.8348, 1.8585, 1.8774, 1.8928, 1.9054, 1.9158, 1.9245, 1.9319, 1.9382, 1.9437, 2.0000, 2.0000]
    Keta_vct  = [0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000, 0.9000, 1.0000, 1.1000, 1.2000, 1.3000, 1.4000, 1.5000, 1.6000, 1.7000, 1.8000, 1.9000, 2.0000, 100.00] 
    
safety_factor = 1.1 # should be > 1
scaling_real = np.interp(K_eta, Keta_vct, Scale_vct) * safety_factor
print("eigenvalue scaling is: %f" % (scaling_real))

# scale real parts
eigenvalues.real *= scaling_real
# the scaling of the imaginary part is more or less useless: it's just a small safety margin
# to account for the nonlinear terms etc. Even multiplying by 10 or more did not alter the RKC
# scheme at all for the low Reynolds numbers.
eigenvalues.imag *= 3.0

print(";-------------------")
print( "; C0=%f C_eta=%e K_eta=%f Bs=%i" % (c0, C_eta, K_eta, Bs) )
print( "; dx=%e CFL=%f jmax=%i nu=%e" % (dx, CFL, Jmax, nu ) )

safety = False
plot = True
s_best, eps_best = finite_differences.select_RKC_scheme(eigenvalues, dt_set, plot=plot, safety=safety, eps_min=2.0)


if safety:
    for i in range(5):
        print('ATTENTION safety=True, selecting scheme with s_min+1')
        
if plot:
    plt.savefig('RKC.pdf')
    plt.savefig('RKC.png')

print( "; s=%i eps=%2.2f, Cost RK4=%i RKC=%i %s speed-up-factor (RKC vs RK4)=%2.1f %s" % (s_best, eps_best, cost4, s_best/dt_set, bcolors.OKGREEN, cost4/(s_best/dt_set), bcolors.ENDC) )
print("\n\n")

#%% last step: write the scheme we found in the PARAMS file:
mu, mu_tilde, nu, gamma_tilde, c, eps = finite_differences.RKC_coefficients(s_best, eps_best)


def print_array(a):    
    # NOTE: the coefficients are padded with an first element due to pythons
    # 0-based indexing. This element is NAN for safety. It is cut here, as in
    # fortran, we use 1-based indexing
    s = ""
    for i in a[1:-1]:
        s += "%e, " % (i)
    s += '%e' % (a[-1])
    return s
    
inifile_tools.replace_ini_value(inifile, 'Time', 'RKC_mu', print_array(mu))
inifile_tools.replace_ini_value(inifile, 'Time', 'RKC_mu_tilde', print_array(mu_tilde))
inifile_tools.replace_ini_value(inifile, 'Time', 'RKC_nu', print_array(nu))
inifile_tools.replace_ini_value(inifile, 'Time', 'RKC_gamma_tilde', print_array(gamma_tilde))
inifile_tools.replace_ini_value(inifile, 'Time', 'RKC_c', print_array(c))
inifile_tools.replace_ini_value(inifile, 'Time', 's', "%i" % (s_best))