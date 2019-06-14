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



if args.directory is None:
    # default is working directory
    dir = './'
else:
    dir = args.directory


if dir[-1] is not '/':
    dir = dir+'/'


# look for the ini file, this gives us the information at what time the run is done
if args.paramsfile is None:
    l = glob.glob( dir+'*.ini' )
    inifile = l[0]
else:
    inifile = args.paramsfile



if os.path.isfile(dir + 'timesteps_info.t'):
    # load the data file
    d = insect_tools.load_t_file(dir + 'timesteps_info.t', verbose=verbose)
    tstart = d[0,0]
    tnow = d[-1,0]

    if verbose:
        print("We found and extract the final time in the simulation from: "+inifile)

    T = wabbit_tools.get_ini_parameter( inifile, 'Time', 'time_max', float)
    bs = wabbit_tools.get_ini_parameter( inifile, 'Blocks', 'number_block_nodes', int)
    dim = wabbit_tools.get_ini_parameter( inifile, 'Domain', 'dim', int)

    # old files lack the information about the number of CPU
    ncpu_now = 0
    cpuh_now = 0
    mean_cost = 0
    runtime = sum(d[:,1])/3600

    if d.shape[1] >= 7:
        # compute mean cost per grid point per time step
        mean_cost = np.mean( d[:,1]*d[:,6] / (8.0*d[:,3]*(bs-1)**dim ) )

        cpuh_now = int( np.sum(d[:,1]*d[:,6])/3600 )
        # this is a recent file (>20/12/2018) it contains the number of procs in every line
        ncpu_now = d[-1,6]
        # we can weight past time steps by the current number of CPUS in order to improve
        # the estimate how much time remains. We assume, of course, perfect scaling with #CPU
        d[:,1] *= d[:,6] / ncpu_now

    # how many time steps did we already do?
    nt_now = d.shape[0]

    # avg walltime second for this run
    twall_avg = np.mean( d[:,1] )

    # avg time step until now
    dt = (tnow-tstart) / nt_now

    # how many time steps are left
    nt_left = (T-tnow) / dt

    # this is what we have to wait still
    time_left = round(nt_left * twall_avg)

    if verbose:
        print("Right now, running on %s%i%s CPUS" % (bcolors.OKGREEN, ncpu_now, bcolors.ENDC))
        print("Already consumed %s%i%s CPUh" % (bcolors.OKGREEN, cpuh_now, bcolors.ENDC))
        print("Runtime %s%2.1f%s h" % (bcolors.OKGREEN, runtime, bcolors.ENDC))
        print("mean cost %s%8.3e%s [CPUs / N / Nt]" % (bcolors.OKGREEN, mean_cost, bcolors.ENDC))
        print("Time to reach: T=%2.3f. Now: we did nt=%i to reach T=%2.1e" % (T, nt_now, d[-1,0]) )
        print( "%s%s%s   [%i CPUH] (remaining time based on all past time steps)"  %
              (bcolors.OKGREEN, str(datetime.timedelta(seconds=time_left)), bcolors.ENDC, int(ncpu_now*time_left/3600)) )

        # second estimate
        nt = min( 20, nt_now )
        dt = ( d[-1,0]-d[-nt,0] ) / nt
        time_left = round(np.mean( d[-nt:,1] ) * (T-d[-1,0]) / (dt) )
        print("%s%s%s   [%i CPUH] (remaining time based on last %i time steps)"
              % (bcolors.OKGREEN, str(datetime.timedelta(seconds=time_left)), bcolors.ENDC, int(ncpu_now*time_left/3600), nt ) )

    if not verbose:
        # when the -s option is active, just print the number of remaining hours
        print( '%3.1f' % (time_left/3600.0) )


elif os.path.isfile(dir + 'performance.t'):
    # load the data file
    d = insect_tools.load_t_file(dir + 'performance.t', verbose=verbose)
    tstart = d[0,0]
    tnow = d[-1,0]

    if verbose:
        print("We found and extract the final time in the simulation from: "+inifile)

    T = wabbit_tools.get_ini_parameter( inifile, 'Time', 'time_max', float)
    bs = wabbit_tools.get_ini_parameter( inifile, 'Blocks', 'number_block_nodes', int)
    dim = wabbit_tools.get_ini_parameter( inifile, 'Domain', 'dim', int)

    # old files lack the information about the number of CPU
    ncpu_now = 0
    cpuh_now = 0
    mean_cost = 0
    runtime = sum(d[:,2])/3600

    # compute mean cost per grid point per time step
    mean_cost = np.mean( d[:,2]*d[:,7] / (8.0*d[:,3]*(bs-1)**dim ) )

    cpuh_now = int( np.sum(d[:,2]*d[:,7])/3600 )
    # this is a recent file (>20/12/2018) it contains the number of procs in every line
    ncpu_now = d[-1,7]
    # we can weight past time steps by the current number of CPUS in order to improve
    # the estimate how much time remains. We assume, of course, perfect scaling with #CPU
    d[:,2] *= d[:,7] / ncpu_now

    # how many time steps did we already do?
    nt_now = d.shape[0]

    # avg walltime in seconds for this run
    twall_avg = np.mean( d[:,2] )

    # avg time step until now
    dt = (tnow-tstart) / nt_now

    # how many time steps are left
    nt_left = (T-tnow) / dt

    # this is what we have to wait still
    time_left = round(nt_left * twall_avg)

    if verbose:
        nt = min( 20, nt_now )


        print("Right now, running on %s%i%s CPUS" % (bcolors.OKGREEN, ncpu_now, bcolors.ENDC))
        print("Already consumed %s%i%s CPUh" % (bcolors.OKGREEN, cpuh_now, bcolors.ENDC))
        print("Runtime %s%2.1f%s h" % (bcolors.OKGREEN, runtime, bcolors.ENDC))
        print("mean cost %s%8.3e%s [CPUs / N / Nt]" % (bcolors.OKGREEN, mean_cost, bcolors.ENDC))
        print("mean blocks-per-rank (rhs) %s%i%s" % (bcolors.OKGREEN, np.mean(d[:,3]/d[:,7]), bcolors.ENDC))
        print("now  blocks-per-rank (rhs) %s%i%s" % (bcolors.OKGREEN, np.mean(d[-nt:,3]/d[-nt:,7]), bcolors.ENDC))
        print("Time to reach: T=%s%2.3f%s" % (bcolors.OKGREEN, T, bcolors.ENDC) )
        print("Now: t=%s%2.3f%s (it=%s%i%s)" % (bcolors.OKGREEN, d[-1,0], bcolors.ENDC, bcolors.OKGREEN, nt_now, bcolors.ENDC) )
        print("%s%s%s   [%i CPUH] (remaining time based on all past time steps)"  %
              (bcolors.OKGREEN, str(datetime.timedelta(seconds=time_left)), bcolors.ENDC, int(ncpu_now*time_left/3600)) )

        # second estimate

        dt = ( d[-1,0]-d[-nt,0] ) / nt
        time_left = round(np.mean( d[-nt:,2] ) * (T-d[-1,0]) / (dt) )
        print("%s%s%s   [%i CPUH] (remaining time based on last %i time steps)"
              % (bcolors.OKGREEN, str(datetime.timedelta(seconds=time_left)), bcolors.ENDC, int(ncpu_now*time_left/3600), nt ) )

    if not verbose:
        # when the -s option is active, just print the number of remaining hours
        print( '%3.1f' % (time_left/3600.0) )