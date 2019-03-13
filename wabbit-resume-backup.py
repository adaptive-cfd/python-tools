#!/usr/bin/env python3
import numpy as np
import wabbit_tools
import sys

print("----------------------------------------")
print(" wabbit: resume simulation")
print(" wabbit-resume-backup.py [inifile] ")
print(" wabbit-resume-backup.py suzuki.ini")
print("----------------------------------------")


inifile = sys.argv[1]
print( "inifile is " + inifile )

wabbit_tools.prepare_resuming_backup( inifile )