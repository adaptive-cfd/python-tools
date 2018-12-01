#!/usr/bin/env python3

import numpy as np
import wabbit_tools
import insect_tools
import glob
import configparser
import datetime
import os
import sys
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


if not os.path.isfile(dir + 'timesteps_info.t'):
    raise ValueError("The file timesteps_info.t has not been found here.")

# load the data file
d = insect_tools.load_t_file(dir + 'timesteps_info.t', verbose=verbose)

if verbose:
    print("We found and extract the final time in the simulation from: "+inifile)

T = wabbit_tools.get_ini_parameter( inifile, 'Time', 'time_max', float)

# how many time steps did we already do?
nt_now = d.shape[0]

# avg CPU second for this run
tcpu_avg = np.mean( d[:,1] )

# avg time step until now
dt = d[-1,0] / nt_now

# how many time steps are left
nt_left = (T-d[-1,0]) / dt

# this is what we have to wait still
time_left = round(nt_left * tcpu_avg)

if verbose:
    print("Time to reach: T=%e. Now: we did nt=%i to reach T=%e and the remaing time is: %s%s%s"
          % (T, nt_now, d[-1,0], bcolors.OKGREEN, str(datetime.timedelta(seconds=time_left)), bcolors.ENDC ) )

if verbose:
    nt = min( 20, nt_now )
    dt = ( d[-1,0]-d[-nt,0] ) / nt
    time_left = np.mean( d[-nt:,1] ) * (T-d[-1,0]) / (dt)
    print("Based on last %i time steps, the remaing time is: %s%s%s"
          % (nt, bcolors.OKGREEN, str(datetime.timedelta(seconds=time_left)), bcolors.ENDC ) )

if not verbose:
    # when the -s option is active, just print the number of remaining hours
    print(time_left/3600.0)
