#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 15:42:59 2018

In the current directory, this script reads the simulation data (timesteps_info.t)
and produces a plot which is stored to a *.png file (without being shown on the display
as its intended use is the command line)

@author: engels
"""

# plot without display: (put before all other modules, esp. insect_tools wabbit_tools)
import matplotlib as mpl
mpl.use('Agg')

import sys, os
sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import insect_tools
import matplotlib.pyplot as plt
import argparse

plt.close('all')

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", nargs='?', const='.',
                    help="directory of simulation files, if not ./")
args = parser.parse_args()


if args.directory is None:
    # default is working directory
    root = './'
else:
    root = args.directory


try:
    d = insect_tools.load_t_file( os.path.join(root,'performance.t') )
    ncpu = d[:,7]
    tcpu = d[:,2]
except:
    d = insect_tools.load_t_file( os.path.join(root,'timesteps_info.t') )
    tcpu = d[:,1]
    if d.shape[1] == 7:
        # this is a recent file (>20/12/2018) it contains the number of procs in every line
        ncpu = d[:,6]
    else:
        # this is an older run, and we can only try to fetch the number of mpiranks from other data
        # unfortunately, we will not be able to tell if the number of procs has been changed during the run.
        e = insect_tools.load_t_file( os.path.join(root,'blocks_per_mpirank_rhs.t'))
        ncpu = e.shape[1]-5

time = d[:, 0]
NblocksRHS = d[:, 3]

blocks_per_rank = NblocksRHS / ncpu
tcpu_per_block = ncpu*tcpu / NblocksRHS
tcpu = ncpu*tcpu

plt.figure()
plt.semilogy(time, tcpu_per_block, '.')
plt.xlabel('$t/T$')
plt.ylabel('$t_{CPU}/N_{b}$ (s)')
plt.savefig(os.path.join(root,'performance_vs_time.png'))


plt.figure()
plt.loglog( blocks_per_rank, tcpu_per_block, '.' )
plt.xlabel('$N_b/N_{CPU}$')
plt.ylabel('$t_{CPU}/N_{b}$ (s)')
plt.savefig(os.path.join(root,'performance_vs_blocks-per-rank.png'))



plt.figure()
plt.semilogy(time, tcpu, '-' )
plt.xlabel('$t/T$')
plt.ylabel('$t_{CPU}$ (s)')
plt.savefig(os.path.join(root,'performance_abs_vs_time.png'))



plt.figure()
plt.semilogy(time, NblocksRHS, '-' )
plt.xlabel('$t/T$')
plt.ylabel('$N_b$')
plt.savefig(os.path.join(root,'blocks_vs_time.png'))