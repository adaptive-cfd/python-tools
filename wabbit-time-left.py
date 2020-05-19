#!/usr/bin/env python3

import numpy as np
import wabbit_tools
import insect_tools
import glob
import datetime
import os
import argparse

class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'



parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", nargs='?', const='./',
                    help="directory of h5 files, if not ./")
parser.add_argument("-n", "--first-n-time-steps", nargs='?', type=int, const=None, default=None,
                    help="Use only the first N time steps")
parser.add_argument("-s", "--script-output", action="store_true",
                    help="""When running in a script, it may be useful to just print the number of
                    remaining hours to STDOUT (and not any fancy messages)""")
parser.add_argument("-p", "--paramsfile", type=str,
                    help="""Parameter (*.ini) file for the wabbit run, required to determine
                    how much time is left in the simulation. If not specified, we
                    try to find an appropriate one in the directory""")
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


if dir[-1] != '/':
    dir = dir+'/'


#------------------------------------------------------------------------------
# end time of simulation
#------------------------------------------------------------------------------
# look for the ini file, this gives us the information at what time the run is done
if args.paramsfile is None:
    l = glob.glob( dir+'*.ini' )

    right_inifile = False
    i = 0

    while right_inifile != True:
        inifile = l[i]
        right_inifile = wabbit_tools.exists_ini_parameter( inifile, "Time", "time_max" )
        i += 1

    if not right_inifile:
        raise ValueError("We did not find an inifile which tells us what the target time is.")

else:
    inifile = args.paramsfile

if verbose:
    print("We found and extract the final time in the simulation from: "+inifile)





# load the data file
d = insect_tools.load_t_file(dir + 'performance.t', verbose=verbose)

# if we consider only a few time steps, we discard the others:
if args.first_n_time_steps is not None:
    d = d[0:args.first_n_time_steps+1, :]

tstart = d[0,0]
tnow = d[-1,0]
# how many time steps did we already do?
nt_now = d.shape[0]
nt = min( 20, nt_now )


# figure out how many RHS evaluatins we do per time step
method = wabbit_tools.get_ini_parameter( inifile, 'Time', 'time_step_method', str)

# default is one (even though that might be wrong...)
nrhs = 1

if method == "RungeKuttaGeneric":
    # this is not always true, but most of the time (butcher_tableau)
    nrhs = 4.0
elif method == "RungeKuttaChebychev":
    nrhs = wabbit_tools.get_ini_parameter( inifile, 'Time', 's', float)
    
# if we perform more than one dt on the same grid, this must be taken into account as well
nrhs *= wabbit_tools.get_ini_parameter( inifile, 'Blocks', 'N_dt_per_grid', float, default=1.0)


T = wabbit_tools.get_ini_parameter( inifile, 'Time', 'time_max', float)
bs = wabbit_tools.get_ini_parameter( inifile, 'Blocks', 'number_block_nodes', int, vector=True)
dim = wabbit_tools.get_ini_parameter( inifile, 'Domain', 'dim', int)


if len(bs) == 1:
    npoints = bs**dim
else:
    npoints = np.product(bs)


# how long did this run run already
runtime = sum(d[:,2])/3600


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

# avg walltime in seconds for this run
twall_avg = np.mean( d[:,2] )

# avg time step until now
dt = (tnow-tstart) / nt_now

# how many time steps are left
nt_left = (T-tnow) / dt

# this is what we have to wait still
time_left = round(nt_left * twall_avg)

# remaining cpu time
cpuh_left = int(ncpu_now*time_left/3600)

if verbose:
    print("Right now, running on %s%i%s CPUS" % (bcolors.OKGREEN, ncpu_now, bcolors.ENDC))
    print("Already consumed %s%i%s CPUh" % (bcolors.OKGREEN, cpuh_now, bcolors.ENDC))
    print("Runtime %s%2.1f%s h" % (bcolors.OKGREEN, runtime, bcolors.ENDC))
    print("RHS evals per time step: %s%i%s" % (bcolors.OKGREEN, int(nrhs), bcolors.ENDC))
    print("----------------------------------------------------------")
    print("cost [CPUs / N / Nrhs]")
    print("    mean       max        min                std")
    print("    ---------  ---------  -------------      -----------------")
    print("ALL: %s%8.3e  %8.3e  %8.3e (%2.1f%%)  %8.3e (%2.1f%%) %s[CPUs / N / Nrhs]" % (bcolors.OKGREEN, mean_cost, max_cost,
          min_cost, 100.0*min_cost/mean_cost,
          std_cost, 100.0*std_cost/mean_cost, bcolors.ENDC))
    print("NOW: %s%8.3e  %8.3e  %8.3e (%2.1f%%)  %8.3e (%2.1f%%) %s[CPUs / N / Nrhs]" % (bcolors.OKGREEN, mean_cost2, max_cost2,
          min_cost2, 100.0*min_cost/mean_cost2,
          std_cost2, 100.0*std_cost/mean_cost2, bcolors.ENDC))
    print("-----------------")
    print("blocks-per-rank [mean over entire run ] (rhs) %s%i%s" % (bcolors.OKGREEN, np.mean(d[:,3]/d[:,7]), bcolors.ENDC))
    print("blocks-per-rank [mean over last it=%i ] (rhs) %s%i%s" % (nt, bcolors.OKGREEN, np.mean(d[-nt:,3]/d[-nt:,7]), bcolors.ENDC))
    print("Time to reach: T=%s%2.3f%s" % (bcolors.OKGREEN, T, bcolors.ENDC) )
    print("Current time: t=%s%2.3f%s (it=%s%i%s)" % (bcolors.OKGREEN, d[-1,0], bcolors.ENDC, bcolors.OKGREEN, nt_now, bcolors.ENDC) )
    print("%s%s%s   [%i CPUH left] = [%i CPUH total] (remaining time based on all past time steps)"  %
          (bcolors.OKGREEN, str(datetime.timedelta(seconds=time_left)), bcolors.ENDC, cpuh_left, cpuh_left+cpuh_now) )

    # second estimate

    dt = ( d[-1,0]-d[-nt,0] ) / nt
    time_left = round(np.mean( d[-nt:,2] ) * (T-d[-1,0]) / (dt) )
    cpuh_left = int(ncpu_now*time_left/3600)
    print("%s%s%s   [%i CPUH left] = [%i CPUH total] (remaining time based on last %i time steps)"
          % (bcolors.OKGREEN, str(datetime.timedelta(seconds=time_left)), bcolors.ENDC, cpuh_left, cpuh_left+cpuh_now, nt ) )

if not verbose:
    # when the -s option is active, just print the number of remaining hours
#    print( '%3.1f' % (time_left/3600.0) )
    print( '%3.1f h %i CPUh (=%i CPUh total)' % (time_left/3600.0, cpuh_left, cpuh_left+cpuh_now) )