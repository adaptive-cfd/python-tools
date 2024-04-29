#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:46:24 2019

@author: engels
"""

import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import insect_tools

print("----------------------------------------")
print(" jobfile2perf: [infile] [infile:timesteps_info.t] [outfile]")
print("----------------------------------------")



fname = sys.argv[1]

# open file, erase existing
fout = open( sys.argv[3], 'w' )

d = insect_tools.load_t_file( sys.argv[2] )

with open(fname) as f:
    for line in f:
        if "RUN: it=" in line:
            line = line.replace("RUN: it=","")
            line = line.replace("time=","")
            line = line.replace("t_cpu=","")
            line = line.replace("Nb=(","")
            line = line.replace("/","")
            line = line.replace(") Jmin=","")
            line = line.replace("Jmax=","")
            line = line.replace("   "," ")
            line = line.replace("  "," ")

            data = np.asarray(line.split(), float)

            it = int(data[0])

            string = '%e %i %e %i %i %i %i %i\n' % (data[1], it, data[2], int(data[3]),
                                                    int(data[4]), int(data[5]), int(data[6]), int(d[it-1,-1]) )

            fout.write( string )


fout.close()