#!/usr/bin/env python3

import sys, os
sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import inifile_tools
import bcolors

# fetch jobfile from call
inifile = sys.argv[1]

if not os.path.isfile(inifile):
        print(bcolors.FAIL + "ERROR: I did not find any inifile :(" + bcolors.ENDC)
else:
        print("We found and check the INI file: "+inifile)
        inifile_tools.check_parameters_for_stupid_errors( inifile )
