#!/usr/bin/env python3
"""
Created on 24-09-18 by JB

This script takes in a matlab file and converts it to a wabbit file


*****************
changed on 25-10-01 by EG

to use this skript on fields with non-isotropic resolution
minor adjustments were made

incase of different block sizes use e.g.: 
    mat2hdf.py *.mat --bs 48 128 48 --max-level 2

"""


import sys, os, numpy as np
sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import wabbit_tools
import scipy.io
import argparse, matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Converts an image file to a wabbit input file.")
parser.add_argument('MATLAB', type=str, help='Input matlab file')
parser.add_argument('-o', '--output', type=str, help='Output location of H5 file, if not provided it is saved under same name as input.', default=None)
parser.add_argument('--bs',type=int,nargs='+',metavar='N',default=[16, 16, 16],
                    help='Block size: 2 ints (Bx By) or 3 ints (Bx By Bz). Default: 16 16 16')
parser.add_argument('--domain-size', type=float, nargs='+', metavar='L',default=[1.0, 1.0, 1.0],
                    help='Domain-length: 2 ints (Lx Ly) or 3 ints (Lx Ly Lz). Default: 1 1 1')
parser.add_argument('--max-level', type=int, help='Maxmium level for treecodes, defaults to 9', default=9)
args = parser.parse_args()

data = scipy.io.loadmat(args.MATLAB)

#-------------------------------------
# check and normalize block size input 
# (2D: BX BY -> BX BY 1, or 3D: BX BY BZ)
if len(args.bs) == 2:
    args.bs = [args.bs[0], args.bs[1], 1]
elif len(args.bs) == 3:
    pass
else:
    parser.error("--bs expects 2 or 3 integers (BX BY [BZ])")
    
#-------------------------------------
# check and normalize domain size input 
# (2D: Lx Ly -> Lx Ly 1, or 3D: Lx Ly Lz)
if len(args.domain_size) == 2:
    ds2 = [args.domain_size[0], args.domain_size[1], 1.0]
    ds3 = None
elif len(args.domain_size) == 3:
    ds2 = [args.domain_size[0], args.domain_size[1], 1.0]
    ds3 = args.domain_size
else:
    parser.error("--domain-size expects 2 or 3 numbers. (Lx Ly [Lz])")
    
    
#-------------------------------------
# extract all fields:
# we assume here that all elements in this matlab file correspond to 
# fields because I am too lazy to write checks to exclude specific fields
fields, keys = [], []
for key, value in data.items():
    if not key.startswith('__'):  # Skip meta entries
        fields.append(np.array(value))
        keys.append(key)

#-------------------------------------
# determine output filename (use input name with .h5 if -o not given)
if args.output == None:
    new_name = args.MATLAB.replace(args.MATLAB[args.MATLAB.rfind("."):], ".h5")
else:
    if args.output.endswith("h5"):
        new_name = args.output
    else:
        new_name = args.output + ".h5"

#-------------------------------------
# loop over all fields and write a wabbit file
for i_f, i_k in zip(fields, keys):
    # create wabbit file
    w_obj = wabbit_tools.WabbitHDF5file()
    
    if len(i_f.shape) == 2:
        # 2D: use BX, BY and force BZ=1
        bs_vec = [args.bs[0], args.bs[1], 1]
        w_obj.fill_from_matrix(i_f, bs=bs_vec, dim=2, max_level=args.max_level, domain_size=ds2)
    else:
        # 3D: use BX, BY, BZ as given/normalized
        bs_vec = [args.bs[0], args.bs[1], args.bs[2]]
        w_obj.fill_from_matrix(i_f, bs=bs_vec, dim=3, max_level=args.max_level, domain_size=ds3)

    pos_dot = new_name.rfind(".")
    new_name_now = new_name[:pos_dot] + "_" + i_k + new_name[pos_dot:]
    w_obj.write(new_name_now)
    
    
    