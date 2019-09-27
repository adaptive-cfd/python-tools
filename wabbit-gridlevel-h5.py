#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:06:35 2018

@author: engels
"""


import wabbit_tools
import sys
import h5py
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--infile", nargs='?', const='./',
                    help="what is your hdf5 file to be converted")
parser.add_argument("-o", "--outfile", nargs='?', const='./',
                    help="output file name")
parser.add_argument("-m", "--mpirank", action="store_true",
                    help="""Return a wabbit-type HDF5 file with data values equal to MPIRANK""")
parser.add_argument("-l", "--level", action="store_true",
                    help="""Return a wabbit-type HDF5 file with data values equal to REFINEMENT LEVEL""")
args = parser.parse_args()

file = args.infile




time, x0, dx, box, data, treecode = wabbit_tools.read_wabbit_hdf5( file )


if args.mpirank:
    fid = h5py.File(file,'r')

    # read procs array from file
    b = fid['procs'][:]
    procs = np.array(b, dtype=float)
    fid.close()


    if len(data.shape) == 4:
        N = treecode.shape[0]
        for i in range(N):
            data[i,:,:,:] = float( procs[i] )

    elif len(data.shape) == 3:

        N = treecode.shape[0]
        for i in range(N):
            data[i,:,:] = float( procs[i] )


elif args.level:
    data = wabbit_tools.overwrite_block_data_with_level(treecode, data)



wabbit_tools.write_wabbit_hdf5( args.outfile, time, x0, dx, box, data, treecode)