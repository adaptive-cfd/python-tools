#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 13:53:26 2020

@author: engels
"""

import glob
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import insect_tools
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", nargs='?', const='./',
                    help="directory of h5 files, if not ./")
parser.add_argument("-o", "--output", type=str,
                    help="""output *.csv value""")
args = parser.parse_args()

if args.directory is None:
    args.directory ='./'
    
if args.output is None:
    args.output = "merge.csv"

#%%
files = sorted(glob.glob( args.directory+"/*.csv" ))

print('Found %i csv files' % (len(files)))

# unfortunately extracting wabbit isosurfaces results in cluttered CSV files
d = []
for file in files:
    d.append( np.loadtxt(file, delimiter=',', skiprows=1) )
    
    
d = np.vstack(d)

insect_tools.write_csv_file( args.directory+"/"+args.output, d, header=None, sep=';')