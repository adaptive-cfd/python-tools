#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import inifile_tools

print("----------------------------------------")
print(" wabbit: resume simulation")
print(" wabbit-resume-backup.py [inifile] ")
print(" wabbit-resume-backup.py suzuki.ini")
print("----------------------------------------")

if len(sys.argv) > 1:
    inifile = sys.argv[1]
    if not os.path.isfile(inifile):
        raise ValueError("no inifile found")
else:
    inifile = inifile_tools.find_WABBIT_main_inifile('./')

if os.path.isfile(inifile):
    print( "inifile is " + inifile )
    inifile_tools.prepare_resuming_backup( inifile )
else:
    raise ValueError("no inifile found")