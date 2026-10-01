#!/usr/bin/env python3
import sys, os, argparse
sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import inifile_tools

parser = argparse.ArgumentParser(
    description="wabbit: resume simulation")
parser.add_argument("inifile", nargs="?", default=None, help="WABBIT inifile to resume")
args = parser.parse_args()

print("----------------------------------------")
print(" wabbit: resume simulation")
print("----------------------------------------")

if args.inifile is not None:
    inifile = args.inifile
    if not os.path.isfile(inifile):
        raise ValueError("no inifile found")
else:
    inifile = inifile_tools.find_WABBIT_main_inifile('./')

if os.path.isfile(inifile):
    print( "inifile is " + inifile )
    inifile_tools.prepare_resuming_backup( inifile )
else:
    raise ValueError("no inifile found")
