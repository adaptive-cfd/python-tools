#!/usr/bin/env python3

import sys, os
sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import wabbit_ini_tools


class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

# fetch jobfile from call
inifile = sys.argv[1]

if not os.path.isfile(inifile):
        print(bcolors.FAIL + "ERROR: I did not find any inifile :(" + bcolors.ENDC)
else:
        print("We found and check the INI file: "+inifile)
        wabbit_ini_tools.check_parameters_for_stupid_errors( inifile )
