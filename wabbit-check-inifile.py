#!/usr/bin/env python3

import numpy as np
import wabbit_tools
import insect_tools
import glob
import configparser
import datetime
import sys

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

print("We found and check the INI file: "+inifile)
wabbit_tools.check_parameters_for_stupid_errors( inifile )
