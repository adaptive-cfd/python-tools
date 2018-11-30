#!/usr/bin/env python3
import numpy as np
import wabbit_tools
import sys

print("----------------------------------------")
print(" wabbit: resume simulation")
print(" wabbit-resume-backup.py [inifile] [statevector-prefixes]")
print(" wabbit-resume-backup.py suzuki.ini \"ux uy uz p\"")
print("----------------------------------------")


inifile = sys.argv[1]
print( "inifile is " + inifile )


statevector_prefixes= sys.argv[2].split()
print( statevector_prefixes )

wabbit_tools.prepare_resuming_backup( inifile, statevector_prefixes)