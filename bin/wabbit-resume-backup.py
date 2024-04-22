#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import wabbit_ini_tools

print("----------------------------------------")
print(" wabbit: resume simulation")
print(" wabbit-resume-backup.py [inifile] ")
print(" wabbit-resume-backup.py suzuki.ini")
print("----------------------------------------")


inifile = sys.argv[1]
if not os.path.isfile(inifile):
        print("ERROR: I did not find any inifile :(")
else:
    print( "inifile is " + inifile )
    wabbit_ini_tools.prepare_resuming_backup( inifile )