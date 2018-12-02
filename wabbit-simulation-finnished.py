#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 18:15:19 2018

@author: engels
"""

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

if dir[-1] is not '/':
    dir = dir+'/'

#------------------------------------------------------------------------------
# look for the ini file, this gives us the information at what time the run is done
if args.paramsfile is None:
    l = glob.glob( dir+'*.ini' )
    inifile = l[0]
else:
    inifile = args.paramsfile

#------------------------------------------------------------------------------
if not os.path.isfile(dir + 'timesteps_info.t'):
    raise ValueError("The file timesteps_info.t has not been found here.")

# load the data file
d = insect_tools.load_t_file(dir + 'timesteps_info.t', verbose=False)

# final time to reach according to paramsfile
T = wabbit_tools.get_ini_parameter( inifile, 'Time', 'time_max', float)

if d[-1,0] >= 0.99*T:
    # run is done
    print('0   (The simulation is done!)')
    sys.exit(0)
else:
    print('1   (The simulation is not yet complete!)')
    sys.exit(1)