#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:27:43 2024

@author: engels
"""


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
C_eta = inifile_tools.get_ini_parameter(inifile, 'VPM', 'C_eta')
CFL   = inifile_tools.get_ini_parameter(inifile, 'Time', 'CFL')
CFL_eta = inifile_tools.get_ini_parameter(inifile, 'Time', 'CFL_eta')
L     = inifile_tools.get_ini_parameter(inifile, 'Domain', 'domain_size', vector=True)[0]
order = inifile_tools.get_ini_parameter(inifile, 'Discretization', 'order_discretization', dtype=str)




# this will be the time step used in the simulation
dx     = 2**(-Jmax)*L/Bs
dt_set = np.min( [CFL * dx / c0, CFL_eta*C_eta] )
print("dt_selected: dt_CFL=%e dt_CFLeta=%e used=%e" % (CFL * dx / c0, CFL_eta*C_eta, dt_set))
K_eta  = np.sqrt(nu*C_eta)/dx

# warn if the time step is still determined by C_eta
if np.abs( dt_set  - CFL_eta*C_eta ) <= 1e-6:
    import bcolors
    
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


# reference RK4 simulation
CFL4   = 1.0
dt4    = min([0.094*dx**2/nu, 0.99*C_eta, CFL4*dx/c0])
cost4  = np.round(4.0 * 1.0 / dt4)
print("; RK4 COST would be %i NRHS/T dt=%e" % (cost4, dt4) )

#------------------------------------------------------------------------------
# FIND SCHEME
#------------------------------------------------------------------------------

# define some (arbitrary) mask function
x        = np.linspace( start=0.0, stop=1.0, num=512, endpoint=False)
L_sponge = 0.35
mask     = sponge_mask(x, L_sponge, 'hard')

# find best RKC scheme given CFL, but using 1D operator
o = ACM_operator(c0, C_eta, nu, dx, mask, order=order)
eigenvalues, dummy = np.linalg.eig(o)

# see 2023 note on eigenvalues of 3D operator (CFD1 lecture)    
w = eigenvalues
# NOTE: /home/engels/Documents/Research/Teaching/CFD1_2022/VL13/spectrum_1D_vs_3D-2.eps
# The actual scaling is found to be 3*real(eigenvalues) and sqrt(3)*imag(eigenvalues) in the case
# of no penalization. Empirically, we find a scaling 1.10*real and 1.74*imag, but of course
# without the nonlinear term. Therefore, multiplying all eigenvalues by 3 is a conservative estimate.
# w *= 3.0 # 2023, for 3D simulation
# w *= np.sqrt(3)
w.imag *= np.sqrt(3.0)
w.real *= 1.15

print(";-------------------")
print( "; C0=%f C_eta=%e K_eta=%f Bs=%i" % (c0, C_eta, K_eta, Bs) )
print( "; dx=%e CFL=%f jmax=%i nu=%e" % (dx, CFL, Jmax, nu ) )

safety = False
plot = True
s_best, eps_best = finite_differences.select_RKC_scheme(w, dt_set, plot=plot, safety=safety, eps_min=2.0)

if safety:
    for i in range(5):
        print('ATTENTION safety=True, selecting scheme with s_min+1')
        
if plot:
    plt.savefig('RKC.pdf')
    plt.savefig('RKC.png')

print( "; s=%i eps=%2.2f, Cost RK4=%i RKC=%i factor=%2.1f" % (s_best, eps_best, cost4, s_best/dt_set, cost4/(s_best/dt_set)) )
    
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