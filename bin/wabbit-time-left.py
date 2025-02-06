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

if shutil.which('latex'): latex = True
else: latex = False

# this is a peculiar oddity for IRENE, she spits out some runtime warnings....
np.seterr(invalid='ignore')

c_g = bcolors.OKGREEN
c_b = bcolors.OKBLUE
c_e = bcolors.ENDC


parser = argparse.ArgumentParser(description="Remaining walltime estimator for wabbit")
parser.add_argument("-d", "--directory", nargs='?', const='./',
                    help="directory of h5 files, if not ./")
parser.add_argument("-l", "--latest-time-steps", nargs='?', type=int, const=None, default=20,
                    help="Steps to use for current latest estimate, defaults to 20")
group = parser.add_mutually_exclusive_group()
group.add_argument("-n", "--first-n-time-steps", nargs='?', type=int, const=None, default=None,
                    help="Use only the first N time steps")
group.add_argument("-m", "--last-m-time-steps", nargs='?', type=int, const=None, default=None,
                    help="Use only the last M time steps")
parser.add_argument("-s", "--script-output", action="store_true",
                    help="""When running in a script, it may be useful to just print the number of
                    remaining hours to STDOUT (and not any fancy messages)""")
parser.add_argument("-p", "--paramsfile", type=str,
                    help="""Parameter (*.ini) file for the wabbit run, required to determine
                    how much time is left in the simulation. If not specified, we
                    try to find an appropriate one in the directory""")
parser.add_argument("-g", "--plot", action="store_true",
                    help="""Plot a figure.""")
parser.add_argument("--plot-procs", action="store_true",
                    help="""Additionally plot the amount of processors over the run.""")
args = parser.parse_args()

verbose = not args.script_output

if verbose:
    print("----------------------------------------")
    print(" Remaining walltime estimator for wabbit")
    print(" usage: wabbit-time-left.py --directory ./ --paramsfile PARAMS.ini")
    print("----------------------------------------")


#------------------------------------------------------------------------------
# directory of simulation
#------------------------------------------------------------------------------
if args.directory is None:
    # default is working directory
    dir = './'
else:
    dir = args.directory


#------------------------------------------------------------------------------
# end time of simulation
#------------------------------------------------------------------------------
# look for the ini file, this gives us the information at what time the run is done
if args.paramsfile is None:
    l = glob.glob( os.path.join(dir,'*.ini'))

    if len(l) == 0:
        raise ValueError(f"We did not find any ini file in the directory {dir} . Are you sure you are at the right place?")

    right_inifile = False
    i = 0

    while right_inifile != True:
        inifile = l[i]
        right_inifile = inifile_tools.exists_ini_parameter( inifile, "Time", "time_max" )
        i += 1

    if not right_inifile:
        raise ValueError("We did not find an inifile which tells us what the target time is.")

else:
    inifile = args.paramsfile

if verbose:
    print("We found and extract the final time in the simulation from: "+inifile)





# load the data file
d = insect_tools.load_t_file(os.path.join(dir,'performance.t'), verbose=verbose)

# if we consider only a few time steps, we discard the others:
if args.first_n_time_steps is not None:
    d = d[0:args.first_n_time_steps+1, :]

# if we consider only a few time steps, we discard the others:
if args.last_m_time_steps is not None:
    d = d[-args.last_m_time_steps:, :]




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


if len(bs) == 1:
    npoints = bs**dim
else:
    npoints = np.prod(bs)


# how long did this run run already (hours)
runtime = sum(d[:,2])/3600

tstart = d[0,0]
tnow   = d[-1,0]
nt_now = int(d[-1,1]-d[0,1])
nt     = min( args.latest_time_steps, d.shape[0] )


# compute mean cost per grid point per RHS evaluation, avg over all time steps
mean_cost = np.mean( d[:,2]*d[:,7] / (d[:,3]*npoints) ) / nrhs
std_cost  = np.std(  d[:,2]*d[:,7] / (d[:,3]*npoints) ) / nrhs
max_cost  = np.max(  d[:,2]*d[:,7] / (d[:,3]*npoints) ) / nrhs
min_cost  = np.min(  d[:,2]*d[:,7] / (d[:,3]*npoints) ) / nrhs


# compute mean cost per grid point per time step, avg over last nt time steps
mean_cost2 = np.mean( d[-nt:,2]*d[-nt:,7] / (d[-nt:,3]*npoints) ) / nrhs
std_cost2  = np.std ( d[-nt:,2]*d[-nt:,7] / (d[-nt:,3]*npoints) ) / nrhs
max_cost2  = np.max ( d[-nt:,2]*d[-nt:,7] / (d[-nt:,3]*npoints) ) / nrhs
min_cost2  = np.min ( d[-nt:,2]*d[-nt:,7] / (d[-nt:,3]*npoints) ) / nrhs



cpuh_now = int( np.sum(d[:,2]*d[:,7])/3600 )

# this is a recent file (>20/12/2018) it contains the number of procs in every line
ncpu_now = d[-1,7]

# we can weight past time steps by the current number of CPUS in order to improve
# the estimate how much time remains. We assume, of course, perfect scaling with #CPU
d[:,2] *= d[:,7] / ncpu_now

# avg walltime in seconds for this run to do one time step
twall_avg = np.mean( d[:,2] )

# avg time step until now (note: this is really time steps, but if more than one time step
# is performed on the grid before adaptation, the walltime is per multiple time steps)
# d2 = insect_tools.load_t_file(dir + 'dt.t', verbose=False)
# dt = np.mean( d2[:,1] )
dts = d[1:,0]-d[:-1,0]
dts[ dts < 0.0 ] = np.nan
dt = np.nanmean(dts)


# how many time steps are left
nt_left = (T-tnow) / dt

# this is what we have to wait still
time_left = round(nt_left * twall_avg)
# this is when its finished
eta = datetime.datetime.now()+ datetime.timedelta(seconds=time_left)

# remaining cpu time
cpuh_left = int(ncpu_now*time_left/3600)

# cost in CPUh / T
abs_mean_cost  = (np.mean(d[:,2]*d[:,7])/3600.0) * (1.0 / dt)
abs_mean_cost2 = (np.mean(d[-nt:,2]*d[-nt:,7])/3600.0) * (1.0 / dt)

# second estimate
dt2 = ( d[-1,0]-d[-nt,0] ) / nt
time_left2 = round(np.mean( d[-nt:,2] ) * (T-d[-1,0]) / (dt2) )
eta2 = datetime.datetime.now()+ datetime.timedelta(seconds=time_left2)
cpuh_left2 = int(ncpu_now*time_left2/3600)

if verbose:
    print(f"Right now, running on       {c_b}{int(ncpu_now):d}{c_e} CPUS")
    print(f"Already consumed:           {c_g}{cpuh_now}{c_e} CPUh")
    print(f"Runtime:                    {c_g}{runtime:4.1f}{c_e} h")
    print(f"RHS evals per time step:    {c_g}{int(nrhs)}{c_e}")
    print(f"mean dt total:              {c_g}{dt:e}{c_e}")
    print(f"mean dt now:                {c_b}{dt2:e}{c_e}")
    print("----------------------------------------------------------")
    print("cost [CPUs / N / Nrhs]")
    print("     mean        max         min                 std")
    print("     ---------   ---------   -----------------   -----------------")
    print(f"ALL: {c_g}{mean_cost:8.3e}   {max_cost:8.3e}   {min_cost:8.3e} ({100.0*min_cost/mean_cost:2.1f}%)   {std_cost:8.3e} ({100.0*std_cost/mean_cost:2.1f}%) {c_e}[CPUs / N / Nrhs]")
    print(f"NOW: {c_b}{mean_cost2:8.3e}   {max_cost2:8.3e}   {min_cost2:8.3e} ({100.0*min_cost2/mean_cost2:2.1f}%)   {std_cost2:8.3e} ({100.0*std_cost2/mean_cost2:2.1f}%) {c_e}[CPUs / N / Nrhs]")
    print("-----------------")
    print(f"Absolute cost [all   {' '*int(np.log10(nt))} time steps]: {c_g}{abs_mean_cost:7.1f}{c_e} CPUh/T")
    print(f"Absolute cost [last {nt} time steps]: {c_b}{abs_mean_cost2:7.1f}{c_e} CPUh/T")
    print("-----------------")
    print(f"Absolute Nb [mean over all   {' '*int(np.log10(nt))} time steps]: {c_g}{np.mean(d[:,3]):10.1f}{c_e}")
    print(f"Absolute Nb [mean over last {nt} time steps]: {c_b}{np.mean(d[-nt:,3]):10.1f}{c_e}")
    print("-----------------")
    print(f"blocks-per-rank [mean over all   {' '*int(np.log10(nt))} time steps] (rhs): {c_g}{np.mean(d[:,3]/d[:,7]):7.1f}{c_e}")
    print(f"blocks-per-rank [mean over last {nt} time steps] (rhs): {c_b}{np.mean(d[-nt:,3]/d[-nt:,7]):7.1f}{c_e}")
    print(f"Time to reach: T= {c_g}{T:10.5f}{c_e}")
    print(f"Current time:  t= {c_g}{d[-1,0]:10.5f}{c_e} (it={c_g}{nt_now}{c_e})")
    if d[-1,0] != T:
        try: cpuh_l_len = int(max(np.log10(cpuh_left),np.log10(cpuh_left2)))
        except: cpuh_l_len = 0
        try: cpuh_t_len = int(max(np.log10(cpuh_left+cpuh_now),np.log10(cpuh_left2+cpuh_now)))
        except: cpuh_t_len = 0
        print(f"Remaining time [based on all   {' '*int(np.log10(nt))} time steps]: {c_g}{str(datetime.timedelta(seconds=time_left))}{c_e}   [{cpuh_left:{cpuh_l_len+1}d} CPUH left] = [{cpuh_left+cpuh_now:{cpuh_t_len+1}d} CPUH total]")
        print(f"Remaining time [based on last {nt} time steps]: {c_b}{str(datetime.timedelta(seconds=time_left2))}{c_e}   [{cpuh_left2:{cpuh_l_len+1}d} CPUH left] = [{cpuh_left2+cpuh_now:{cpuh_t_len+1}d} CPUH total]")
        print(f"ETA [based on all   {' '*int(np.log10(nt))} time steps]: {c_g}{eta.strftime('%Y-%m-%d %H:%M:%S')}{c_e}")
        print(f"ETA [based on last {nt} time steps]: {c_b}{eta2.strftime('%Y-%m-%d %H:%M:%S')}{c_e}")
    else:
        print(f"Run is finished.")

if not verbose:
    # when the -s option is active, just print the number of remaining hours
#    print( '%3.1f' % (time_left/3600.0) )
    print( f'{time_left/3600.0:3.1f} {cpuh_left} {cpuh_left+cpuh_now}')
    
    
if args.plot:
    import matplotlib.pyplot as plt

    n_plots = 4 + args.plot_procs
    
    plt.figure(figsize=(8.27, 11.69)) # this is din A4 size
    a = 1.0
    # plt.gcf().set_size_inches( [a*10.0, a*15] ) # default 6.4, 4.8
    
    if not latex: label_now = "Nb"
    else: label_now = "$N_b$"
    plt.subplot(np.ceil(n_plots/2).astype(int),2,1)    
    plt.plot( d[:,0], d[:,3], label=label_now)
    plt.grid(True)
    plt.ylabel(label_now)
    plt.xlabel('time')
    
    if not latex: label_now = "cost [CPUs / N / Nrhs]"
    else: label_now = r"cost [$CPUs / N / N_{RHS}$]"   
    plt.subplot(np.ceil(n_plots/2).astype(int),2,2) 
    c = plt.scatter( d[:,0], d[:,2]*d[:,7] / (d[:,3]*npoints*nrhs), s=4)
    if latex: c.set_rasterized(True)  # backend has troubles with scatter plots, so lets skip them
    plt.gca().set_yscale('log')
    plt.grid(True)
    plt.ylabel(label_now)
    plt.xlabel('time')
    
    plt.subplot(np.ceil(n_plots/2).astype(int),2,3)  
    if not latex: label_now = "Nb/Ncpu"
    else: label_now = "$N_b / N_{CPU}$"     
    plt.plot( d[:,0], d[:,3]/d[:,7])
    plt.grid(True)
    plt.ylabel(label_now)
    plt.xlabel('time')
    
    plt.subplot(np.ceil(n_plots/2).astype(int),2,4)    
    Nb_per_rank  = np.round( d[:,3]/d[:,7])
    cost         = d[:,2]*d[:,7] / (d[:,3]*npoints)       
    Nb_per_rank2 = np.arange( start=np.min(Nb_per_rank), stop=np.max(Nb_per_rank)+1, dtype=float )
    cost_avg     = np.zeros( Nb_per_rank2.shape )
    cost_std     = np.zeros( Nb_per_rank2.shape )
    
    for i in range(len(Nb_per_rank2)):
        cost_avg[i] = np.mean( cost[Nb_per_rank==Nb_per_rank2[i]] ) / nrhs
        cost_std[i] = np.std( cost[Nb_per_rank==Nb_per_rank2[i]] ) / nrhs
    
    if not latex: label_now = "Nb/Ncpu"
    else: label_now = "$N_b / N_{CPU}$"  
    c = plt.scatter(  d[:,3]/d[:,7], cost / nrhs, s=1, color='k', label=label_now)
    insect_tools.plot_errorbar_fill_between( Nb_per_rank2, cost_avg, cost_std )
    plt.gca().set_yscale('log')
    plt.xlabel(label_now)
    if not latex: label_now = "cost [CPUs / N / Nrhs]"
    else: label_now = "cost [CPUs / $N$ / $N_{rhs}$]"   
    plt.ylabel(label_now)
    plt.grid(True)
    if latex: c.set_rasterized(True)  # backend has troubles with scatter plots, so lets skip them
    
    if args.plot_procs:
        if not latex: label_now = "#CPU"
        else: label_now = "\#CPU"   
        plt.subplot(np.ceil(n_plots/2).astype(int),2,5)    
        plt.plot( d[:,0], d[:,7], label=label_now )
        plt.xlabel('time')
        plt.ylabel(label_now)
        plt.grid(True)
    
    # plt.show()

    plt.tight_layout(pad=0.15)  # Adjust subplots to fit into figure area
    
    plt.savefig( os.path.join(dir,'info_performance.png'))

    # pdf size is fitted to fill a page with appropriate margins, so that we can include it in a report
    font_size = plt.rcParams['font.size']
    pad_in_inches = 0.75 # top, bottom, left, right for DinA4
    pad_frac = pad_in_inches * plt.gcf().get_dpi() / font_size  # Convert inches to pixel to fraction of font size
    plt.tight_layout(pad=pad_frac)  # Adjust subplots to fit into figure area
    plt.subplots_adjust(wspace=0.5)
    if not latex:
        plt.savefig( os.path.join(dir,'info_performance.pdf'), backend="pgf")
    else:
        plt.savefig( os.path.join(dir,'info_performance.pdf'), backend="pgf")

