#!/usr/bin/env python3

import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import inifile_tools
import insect_tools
import bcolors
import glob
import datetime
import argparse
import shutil
import matplotlib.pyplot as plt
import subprocess

if shutil.which('latex'): latex = True
else: latex = False

# this is a peculiar oddity for IRENE, she spits out some runtime warnings....
np.seterr(invalid='ignore')

c_g = bcolors.OKGREEN
c_b = bcolors.OKBLUE
c_e = bcolors.ENDC

# helper class to nicely convert exponential numbers instead of seeing the e
def exp_to_latex(exp_number, depth=2):
    base, exponent = f"{exp_number:.{depth}e}".split('e')
    return f"{base}\\cdot10^{{{int(exponent)}}}"

parser = argparse.ArgumentParser(description="Create a quick report from a wabbit simulation")
parser.add_argument("-d", "--directory", nargs='?', const='./',
                    help="directory of h5 files, if not ./")
parser.add_argument("-i", "--ini", default="./",
                    help="ini file or directory of ini file, if not ./")
group = parser.add_mutually_exclusive_group()
group.add_argument("-n", "--first-n-time-steps", nargs='?', type=int, const=None, default=None,
                    help="Use only the first N time steps")
group.add_argument("-m", "--last-m-time-steps", nargs='?', type=int, const=None, default=None,
                    help="Use only the last M time steps")
parser.add_argument("-p", "--paramsfile", type=str,
                    help="""Parameter (*.ini) file for the wabbit run, required to determine
                    how much time is left in the simulation. If not specified, we
                    try to find an appropriate one in the directory""")

parser.add_argument("--show", action="store_true",
                    help="""Show plots directly in new windows.""")
parser.add_argument("--png", action="store_true",
                    help="""Plot figures as images.""")
parser.add_argument("--pdf", action="store_true",
                    help="""Plot figures as pdf.""")
parser.add_argument("--report", action="store_true",
                    help="""Create one report of all figures.""")
parser.add_argument("-v", "--verbose", action="store_true", help="Print more diagnostic output.")
args = parser.parse_args()

verbose = args.verbose

if verbose:
    print("----------------------------------------")
    print(" Create a quick report for a wabbit cimulation")
    print(" usage: wabbit-report.py -d ./ -i ./ --pdf")
    print("----------------------------------------")

if not (args.show or args.png or args.pdf):
    parser.error("At least one of --show, --png, or --pdf must be specified.")

#------------------------------------------------------------------------------
# directory of simulation
#------------------------------------------------------------------------------
if args.directory is None:
    # default is working directory
    dir = './'
else:
    dir = args.directory

#------------------------------------------------------------------------------
# look for inifile
#------------------------------------------------------------------------------
if os.path.isfile(args.ini):
    right_inifile = inifile_tools.exists_ini_parameter( args.ini, "Time", "time_max" )
    if not right_inifile: parser.error("Ini file provided does not point to a valid ini file.")
    inifile = args.ini
elif os.path.isdir(args.ini):
    list_inifiles = glob.glob( os.path.join(args.ini,'*.ini'))

    if len(list_inifiles) == 0:
        raise ValueError(f"We did not find any ini file in the directory {args.ini} . Are you sure you are at the right place?")

    for inifile in list_inifiles:
        right_inifile = inifile_tools.exists_ini_parameter( inifile, "Time", "time_max" )
        if right_inifile: break

    if not right_inifile:
        raise ValueError("We did not find an inifile which tells us what the target time is.")
else: parser.error("Ini location does not point to anything that exists")



# create a large dictionary of all files
t_values = {'cfl':[], 'div':[], 'dt':[], 'e_kin':[], 'enstrophy':[], 'eps_norm':[], 'forces':[], \
         'forces_rk':[], 'helicity':[], 'kinematics':[], 'krylov_err':[], 'mask_volume':[], 'meanflow':[], 'penal_power':[], \
         'performance':[], 'thresholding':[], 'umag':[], 'u_residual':[], 'scalar_integral':[], 'turbulent_statistics':[]}

for i_name in t_values:
    t_file = os.path.join(dir,f'{i_name}.t')
    try:
        t_values[i_name] = insect_tools.load_t_file(t_file, verbose=verbose)
    except:
        if verbose:
            print(f"File {i_name}.t was not found and is skipped")

# # if we consider only a few time steps, we discard the others:
# if args.first_n_time_steps is not None:
#     d = t_values['performance'][0:args.first_n_time_steps+1, :]

# # if we consider only a few time steps, we discard the others:
# if args.last_m_time_steps is not None:
#     d = t_values['performance'][-args.last_m_time_steps:, :]

# extract infos from ini-file
# figure out how many RHS evaluatins we do per time step
method = inifile_tools.get_ini_parameter( inifile, 'Time', 'time_step_method', str, default="RungeKuttaGeneric")

# default is one (even though that might be wrong...)
nrhs = 1

if method == "RungeKuttaGeneric" or method == "RungeKuttaGeneric-FSI":
    # this is not always true, but most of the time (butcher_tableau)
    nrhs = 4.0
elif method == "RungeKuttaChebychev":
    nrhs = inifile_tools.get_ini_parameter( inifile, 'Time', 's', float)
    
if nrhs == 1:
    print("\n\n\n%sWe assume 1 rhs eval per time step, but that is likely not correct.%s\n\n\n" % (bcolors.FAIL, bcolors.ENDC))
    
# if we perform more than one dt on the same grid, this must be taken into account as well
N_dt_per_grid = inifile_tools.get_ini_parameter( inifile, 'Blocks', 'N_dt_per_grid', float, default=1.0)
nrhs *= N_dt_per_grid


T = inifile_tools.get_ini_parameter( inifile, 'Time', 'time_max', float)
bs = inifile_tools.get_ini_parameter( inifile, 'Blocks', 'number_block_nodes', int, vector=True)
dim = inifile_tools.get_ini_parameter( inifile, 'Domain', 'dim', int)
vpm = inifile_tools.get_ini_parameter( inifile, 'VPM', 'penalization', bool, default=False)

if len(bs) == 1: npoints = bs**dim
else: npoints = np.prod(bs)

# how long did this run run already (hours)
runtime = sum(t_values['performance'][:,2])/3600

# tstart = t_values['performance'][0,0]
# tnow   = t_values['performance'][-1,0]
# nt_now = int(t_values['performance'][-1,1]-t_values['performance'][0,1])
# nt     = min( args.latest_time_steps, t_values['performance'].shape[0] )

if not latex: time_label = "time"
else: time_label = "time $t$"

fig_grid = []  # for adequate positioning

# print info on performance - blocks and timing
if isinstance(t_values['performance'], np.ndarray):
    fig_perf = plt.figure(0, figsize=(8.27, 11.69)) # this is din A4 size
    fig_perf.suptitle('Performance', fontsize=16, y=(11.69-1)/11.69)
    if not latex: label_now = "Number of blocks"
    else: label_now = "Number of blocks $N_b$"
    plt.subplot(2,2,1)    
    plt.semilogy( t_values['performance'][:,0], t_values['performance'][:,3], label="RHS")
    plt.semilogy( t_values['performance'][:,0], t_values['performance'][:,4], label="Grid")
    plt.legend()
    plt.grid(True)
    plt.ylabel(label_now)
    plt.xlabel(time_label)
    plt.xlim(t_values['performance'][0,0], t_values['performance'][-1,0])
    if not latex: label_now = ["JMin", "JMax"]
    else: label_now = [r"$J_{Min}$", r"$J_{Max}$"]
    plt.subplot(2,2,2)    
    plt.plot( t_values['performance'][:,0], t_values['performance'][:,5], label=label_now[0])
    plt.plot( t_values['performance'][:,0], t_values['performance'][:,6], label=label_now[1])
    plt.legend()
    plt.grid(True)
    plt.ylabel('Level of grid after adaption')
    plt.xlabel(time_label)
    plt.xlim(t_values['performance'][0,0], t_values['performance'][-1,0])

    if not latex: label_now = "cost [CPUs / N / Nrhs]"
    else: label_now = r"cost [$CPUs / N / N_{RHS}$]"   
    plt.subplot(2,2,3) 
    c = plt.scatter( t_values['performance'][:,0], t_values['performance'][:,2]*t_values['performance'][:,7] / (t_values['performance'][:,3]*npoints*nrhs), s=4)
    if latex: c.set_rasterized(True)  # backend has troubles with scatter plots, so lets skip them
    plt.gca().set_yscale('log')
    plt.grid(True)
    plt.ylabel(label_now)
    plt.xlabel(time_label)
    plt.xlim(t_values['performance'][0,0], t_values['performance'][-1,0])
 
    plt.subplot(2,2,4)
    # if we have less then 3 hours max walltime then I plot everything in minutes, elsewise in hours, chosen arbitrarily
    if (np.sum(t_values['performance'][:,2]) / 3600 < 3): divide = 60
    else: divide = 3600
    plt.plot( t_values['performance'][:,0], np.cumsum(t_values['performance'][:,2]) / divide)
    plt.grid(True)
    if divide == 60: plt.ylabel("Walltime in minutes")
    else: plt.ylabel("Walltime in hours")
    plt.xlabel(time_label)
    plt.xlim(t_values['performance'][0,0], t_values['performance'][-1,0])
    fig_grid.append([2,2])  # for adequate positioning


# print enstrophy and energy
# check if they exist
plot_conservational = [isinstance(t_values['e_kin'], np.ndarray), isinstance(t_values['enstrophy'], np.ndarray), isinstance(t_values['helicity'], np.ndarray), isinstance(t_values['div'], np.ndarray)]
fig_cons = plt.figure(1, figsize=(8.27, 11.69)) # this is din A4 size
fig_cons.suptitle('Conservation of global quantities', fontsize=16, y=(11.69-1)/11.69)
if plot_conservational[0]:
    plt.subplot(sum(plot_conservational),2,1)
    plt.semilogy(t_values['e_kin'][:,0], t_values['e_kin'][:,1])
    plt.ylabel("Kinetic energy")
    plt.xlabel(time_label)
    plt.grid(True)
    plt.xlim(t_values['e_kin'][0,0], t_values['e_kin'][-1,0])
    plt.subplot(sum(plot_conservational),2,2)
    plt.semilogy(t_values['e_kin'][:,0], np.abs(t_values['e_kin'][:,1]/t_values['e_kin'][0,1] - 1))
    # plt.semilogy(t_values['e_kin'][:,0], 1 - t_values['e_kin'][:,1]/t_values['e_kin'][0,1])
    if not latex: plt.ylabel("Rel. Kin. Energy E(t)/E(0) - 1")
    else: plt.ylabel("Rel. Kin. Energy $E(t)/E_0 - 1$")
    plt.xlabel(time_label)
    plt.grid(True)
    plt.xlim(t_values['e_kin'][0,0], t_values['e_kin'][-1,0])
if plot_conservational[1]:
    plt.subplot(sum(plot_conservational),2,1+2*plot_conservational[0])
    plt.semilogy(t_values['enstrophy'][:,0], t_values['enstrophy'][:,1])
    plt.ylabel("Enstrophy")
    plt.xlabel(time_label)
    plt.grid(True)
    plt.xlim(t_values['enstrophy'][0,0], t_values['enstrophy'][-1,0])
    plt.subplot(sum(plot_conservational),2,2+2*plot_conservational[0])
    plt.semilogy(t_values['enstrophy'][:,0], np.abs(t_values['enstrophy'][:,1]/t_values['enstrophy'][0,1] - 1))
    # plt.semilogy(t_values['enstrophy'][:,0], 1 - t_values['enstrophy'][:,1]/t_values['enstrophy'][0,1])
    if not latex: plt.ylabel("Rel. Enstrophy W(t)/W(0) - 1")
    else: plt.ylabel("Rel. Enstrophy $W(t)/W_0 - 1$")
    plt.xlabel(time_label)
    plt.grid(True)
    plt.xlim(t_values['enstrophy'][0,0], t_values['enstrophy'][-1,0])
if plot_conservational[2]:
    ax1 = plt.subplot(sum(plot_conservational),1,1+plot_conservational[0]+plot_conservational[1])
    plt.plot(t_values['helicity'][:,0], t_values['helicity'][:,1])
    plt.ylabel("Helicity")
    plt.xlabel(time_label)
    plt.grid(True)
    plt.xlim(t_values['helicity'][0,0], t_values['helicity'][-1,0])
if plot_conservational[3]:
    plt.subplot(sum(plot_conservational),1,1+plot_conservational[0]+plot_conservational[1]+plot_conservational[2])
    plt.plot(t_values['div'][:,0], t_values['div'][:,1], label='Max Div')
    plt.plot(t_values['div'][:,0], t_values['div'][:,2], label='Min Div')
    plt.ylabel("Divergence")
    plt.xlabel(time_label)
    plt.grid(True); plt.legend()
    plt.xlim(t_values['div'][0,0], t_values['div'][-1,0])
if any(plot_conservational): fig_grid.append([2,sum(plot_conservational)])  # for adequate positioning

# print thresholding and eps norm
# check if they exist
plot_thresholding = [isinstance(t_values['thresholding'], np.ndarray), isinstance(t_values['eps_norm'], np.ndarray)]
fig_thresh = plt.figure(2, figsize=(8.27, 11.69)) # this is din A4 size
fig_thresh.suptitle('Thresholding used for grid adaption', fontsize=16, y=(11.69-1)/11.69)
if plot_thresholding[0]:
    plt.subplot(sum(plot_thresholding),1,1)
    for i_vals in range(t_values['thresholding'].shape[1]-1):
        plt.semilogy(t_values['thresholding'][:,0], t_values['thresholding'][:,i_vals+1], label=f'Var {i_vals}')
    plt.legend()
    plt.ylabel("Thresholding from CVS")
    plt.xlabel(time_label)
    plt.grid(True)
    plt.xlim(t_values['thresholding'][0,0], t_values['thresholding'][-1,0])
if plot_thresholding[1]:
    plt.subplot(sum(plot_thresholding),1,1+plot_thresholding[0])
    for i_vals in range(t_values['eps_norm'].shape[1]-2):
        plt.semilogy(t_values['eps_norm'][:,0], t_values['eps_norm'][:,i_vals+1] * t_values['eps_norm'][:,-1], label=f'Var {i_vals}')
    plt.legend()
    if not latex: plt.ylabel(f"Normalized thresholding with eps0={t_values['eps_norm'][0,-1]:.2e}")
    else: plt.ylabel(f"Normalized thresholding with ${r'\e'}psilon_0={exp_to_latex(t_values['eps_norm'][0,-1])}$")
    plt.xlabel(time_label)
    plt.grid(True)
    plt.xlim(t_values['eps_norm'][0,0], t_values['eps_norm'][-1,0])
if any(plot_thresholding): fig_grid.append([2,sum(plot_thresholding)])  # for adequate positioning

# print mean and max values
plot_meanmax_flow = [isinstance(t_values['umag'], np.ndarray), isinstance(t_values['meanflow'], np.ndarray)]
fig_meanmax = plt.figure(3, figsize=(8.27, 11.69)) # this is din A4 size
fig_meanmax.suptitle('Mean and maximum value of the fluid velocity', fontsize=16, y=(11.69-1)/11.69)
if plot_meanmax_flow[0]:
    plt.subplot(1+plot_meanmax_flow[0],1,1)
    plt.plot(t_values['umag'][:,0], t_values['umag'][:,1], label=f'Max Mag u' if not latex else r"Max Mag $\vec{u}$")
    plt.legend()
    plt.ylabel("Velocity")
    plt.xlabel(time_label)
    plt.grid(True)
    plt.xlim(t_values['umag'][0,0], t_values['umag'][-1,0])
    plt.subplot(sum(plot_meanmax_flow),1,2)
    plt.semilogy(t_values['umag'][:,0], t_values['umag'][:,2], label='c0' if not latex else "$c_0$")
    plt.semilogy(t_values['umag'][:,0], t_values['umag'][:,4], label='uC' if not latex else "$u_C$")
    plt.legend()
    plt.ylabel("ACM characteristic velocities")
    plt.xlabel(time_label)
    plt.grid(True)
    plt.xlim(t_values['umag'][0,0], t_values['umag'][-1,0])
if plot_meanmax_flow[1]:
    plt.subplot(1+plot_meanmax_flow[0],1,1)
    plt.plot(t_values['meanflow'][:,0], np.sqrt(np.sum(t_values['meanflow'][:,1:]**2, axis=1)), label=f'Mean Mag u' if not latex else r"Mean Mag $\vec{u}$")
    var_names = ["Mean ux", "Mean uy", "Mean uz"] if not latex else ["Mean $u_x$", "Mean $u_y$", "Mean $u_z$"]
    for i_vals in range(dim):
        plt.plot(t_values['meanflow'][:,0], t_values['meanflow'][:,i_vals+1], label=var_names[i_vals])
    plt.legend()
    plt.ylabel("Velocity")
    plt.xlabel(time_label)
    plt.grid(True)
    plt.xlim(t_values['meanflow'][0,0], t_values['meanflow'][-1,0])
if any(plot_meanmax_flow): fig_grid.append([sum(plot_meanmax_flow),1])  # for adequate positioning

# print forces for VPM
plot_forces = isinstance(t_values['forces'], np.ndarray) and vpm
fig_forces = plt.figure(4, figsize=(8.27, 11.69)) # this is din A4 size
fig_forces.suptitle('Forces acting on the immersed body', fontsize=16, y=(11.69-1)/11.69)
if plot_forces:
    forces_names = ["Fx", "Fy", "Fz"] if not latex else ["$F_x$", "$F_y$", "$F_z$"]
    for i_dim in range(dim):
        plt.subplot(dim,1,i_dim+1)
        plt.plot(t_values['forces'][:,0], t_values['forces'][:,i_dim], label=f"Force {forces_names[i_dim]}")
        plt.ylabel(f"Force {forces_names[i_dim]}")
        plt.xlabel(time_label)
        plt.grid(True)
        plt.xlim(t_values['forces'][0,0], t_values['forces'][-1,0])
    fig_grid.append([dim,1])    # for adequate positioning

# print mask volume for VPM
plot_maskvolume = isinstance(t_values['mask_volume'], np.ndarray) and vpm
fig_maskvolume = plt.figure(5, figsize=(8.27, 11.69)) # this is din A4 size
fig_maskvolume.suptitle('Volume of the immersed body and sponge layer', fontsize=16, y=(11.69-1)/11.69)
if plot_maskvolume:
    label_now = ["Mask volume over time", "Sponge volume over time"]
    for i_plot in range(2):
        plt.subplot(2,1,i_plot+1)
        plt.plot(t_values['mask_volume'][:,0], t_values['mask_volume'][:,i_plot+1])
        plt.ylabel("Volume"); plt.xlabel(time_label); plt.grid(True)
        plt.title(label_now[i_plot])
        plt.xlim(t_values['mask_volume'][0,0], t_values['mask_volume'][-1,0])
    fig_grid.append([2,1])  # for adequate positioning

# print scalar integral
plot_scalar = isinstance(t_values['scalar_integral'], np.ndarray)
fig_scalar = plt.figure(5, figsize=(8.27, 11.69)) # this is din A4 size
fig_scalar.suptitle('Scalar norms', fontsize=16, y=(11.69-1)/11.69)
if plot_scalar:
    if not latex: label_now = ["Scalar integral (L1-norm) over time", "Max scalar (Linfty-norm) over time"]
    else: label_now = ["Scalar integral ($L_1$-norm) over time", r"Max scalar ($L_\infty$-norm) over time"]
    for i_plot in range(2):
        plt.subplot(2,1,i_plot+1)
        plt.plot(t_values['scalar_integral'][:,0], t_values['scalar_integral'][:,i_plot+2])
        plt.ylabel("Norm"); plt.xlabel(time_label); plt.grid(True)
        plt.title(label_now[i_plot])
        plt.xlim(t_values['scalar_integral'][0,0], t_values['scalar_integral'][-1,0])
    fig_grid.append([dim,1])  # for adequate positioning

# print turbulent statistics
plot_turbulent_statistics = isinstance(t_values['turbulent_statistics'], np.ndarray)
fig_turbulent_statistics = plt.figure(6, figsize=(8.27, 11.69)) # this is din A4 size
fig_turbulent_statistics.suptitle('Turbulent statistics', fontsize=16, y=(11.69-1)/11.69)
if plot_turbulent_statistics:
    if not latex: label_now = ["Dissipation", "Energy", "U_RMS", "Kolmogorov length", "Kolmogorov time", "Kolmogorov velocity", "Taylor micro-scale", "Taylor Reynolds number"]
    else: label_now = [r"Dissipation $\epsilon$", "Energy $E$", "$U_{RMS}$", r"Kolmogorov length $l_\eta$", r"Kolmogorov time $\tau_\eta$", r"Kolmogorov velocity $u_\eta$", r"Taylor micro-scale $\lambda$", r"Taylor Reynolds number $R_\lambda$"]
    for i_plot in range(len(label_now)):
        plt.subplot(len(label_now)//2+(len(label_now)//2 != len(label_now)/2),2,i_plot+1)
        plt.semilogy(t_values['turbulent_statistics'][:,0], t_values['turbulent_statistics'][:,i_plot+1])
        plt.ylabel(label_now[i_plot]); plt.xlabel(time_label); plt.grid(True)
        # plt.title(label_now[i_plot])
        plt.xlim(t_values['turbulent_statistics'][0,0], t_values['turbulent_statistics'][-1,0])
    fig_grid.append([len(label_now)//2+(len(label_now)//2 != len(label_now)/2),2])  # for adequate positioning


# dict with all names for saving the figures and their figure handle together with how many p
figure_names = {}
if isinstance(t_values['performance'], np.ndarray) : figure_names['performance'] = fig_perf
if np.any(plot_conservational): figure_names['conservational'] = fig_cons
if np.any(plot_thresholding): figure_names['thresholding'] = fig_thresh
if np.any(plot_meanmax_flow): figure_names['ACM_flow_meanmax'] = fig_meanmax
if np.any(plot_forces): figure_names['VPM_forces'] = fig_forces
if np.any(plot_maskvolume): figure_names['VPM_maskvolume'] = fig_maskvolume
if np.any(plot_scalar): figure_names['scalar_integral'] = fig_scalar
if np.any(plot_turbulent_statistics): figure_names['turbulent_statistics'] = fig_turbulent_statistics

# create images of each figure
if args.png:
    for i_name in figure_names:
        plt.figure(figure_names[i_name])
        plt.tight_layout(pad=0.15)  # Adjust subplots to fit into figure area
        plt.savefig( os.path.join(dir,f'{i_name}.png'))

# save each figure as a figure in DinA4 format
if args.pdf:
    # pdf size is fitted to fill a page with appropriate margins, so that we can include it in a report
    font_size = plt.rcParams['font.size']
    pad_in_inches = 0.75 # top, bottom, left, right for DinA4
    pad_frac = pad_in_inches * plt.gcf().get_dpi() / font_size  # Convert inches to pixel to fraction of font size

    for i_n, i_name in enumerate(figure_names):
        plt.figure(figure_names[i_name])
        plt.tight_layout(pad=0)  # Adjust subplots to fit into figure area
        # plt.subplots_adjust(wspace=0.5)
        # hspace is adapted to line height from average figure height
        plt.subplots_adjust(wspace=0.5, hspace=5*font_size/72 / (8.14/fig_grid[i_n][0]), left=(pad_in_inches+1)/8.27, right=(8.27-pad_in_inches-0.3)/8.27, \
            top=(11.69-1.8)/11.69, bottom=(pad_in_inches+1)/11.69)
        if not latex:
            plt.savefig( os.path.join(dir,f'{i_name}.pdf'), backend="pgf")
        else:
            plt.savefig( os.path.join(dir,f'{i_name}.pdf'), backend="pgf")

# create a report by concatenating all pdf pages
if args.report:
    # we assume the order of figure_names.keys is how we want to sort our report as well
    command = f"pdfunite {'.pdf '.join(figure_names.keys())}.pdf report.pdf"
    subprocess.run(command, check=True, shell=True)

    

if args.show:
    plt.show()