#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 18:15:19 2018

@author: engels
"""

import sys, os
sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import wabbit_ini_tools
import insect_tools
import glob
import argparse

#------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("paramsfile", type=str,
                    help="""Parameter (*.ini) file for the wabbit run, required to determine
                    how much time is left in the simulation. If not specified, we
                    try to find an appropriate one in the directory""")
parser.add_argument("-d", "--directory", nargs='?', const='./',
                    help="directory of h5 files, if not ./")
args = parser.parse_args()


#------------------------------------------------------------------------------

if args.directory is None:
    # default is working directory
    dir = './'
else:
    dir = args.directory

if dir[-1] != '/':
    dir = dir+'/'

#------------------------------------------------------------------------------
# look for the ini file, this gives us the information at what time the run is done
if args.paramsfile is None:
    l = glob.glob( dir+'*.ini' )

    right_inifile = False
    i = 0

    while right_inifile != True:
        inifile = l[i]
        right_inifile = wabbit_ini_tools.exists_ini_parameter( inifile, "Time", "time_max" )
        i += 1
else:
    inifile = args.paramsfile

#------------------------------------------------------------------------------

if not os.path.isfile(dir + 'performance.t'):
    raise ValueError("The file performance.t has not been found here.")

# load the data file
d = insect_tools.load_t_file(dir + 'performance.t', verbose=False)

# final time to reach according to paramsfile
T = wabbit_ini_tools.get_ini_parameter( inifile, 'Time', 'time_max', float)

if d[-1,0] >= 0.99*T:
    # run is done
    print('0   (The simulation is done!)')
    sys.exit(0)
else:
    print('1   (The simulation is not yet complete!)')
    sys.exit(1)