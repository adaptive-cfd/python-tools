#!/usr/bin/env python3
"""
Created on 24-09-18 by JB

This script takes in a matlab file and converts it to a wabbit file

"""
import sys, os, numpy as np
sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import wabbit_tools
import scipy.io
import argparse, matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Converts an image file to a wabbit input file.")
parser.add_argument('MATLAB', type=str, help='Input matlab file')
parser.add_argument('-o', '--output', type=str, help='Output location of H5 file, if not provided it is saved under same name as input.', default=None)
parser.add_argument('--bs', type=int, help='Block size, defaults to 16.', default=16)
parser.add_argument('--max-level', type=int, help='Maxmium level for treecodes, defaults to 9', default=9)
args = parser.parse_args()

data = scipy.io.loadmat(args.MATLAB)

# extract all fields
fields, keys = [], []
for key, value in data.items():
    if not key.startswith('__'):  # Skip meta entries
        fields.append(np.array(value))
        keys.append(key)

# we assume here that all elements in this matlab file correspond to fields because I am too lazy to write checks to exclude specific fields

# look at new name
if args.output == None:
    new_name = args.MATLAB.replace(args.MATLAB[args.MATLAB.rfind("."):], ".h5")
else:
    if args.output.endswith("h5"):
        new_name = args.output
    else:
        new_name = args.output + ".h5"

# loop over all fields and make a wabbit file out of them
for i_f, i_k in zip(fields, keys):
    # create wabbit file
    w_obj = wabbit_tools.WabbitHDF5file()
    if len(i_f.shape) == 2:
        w_obj.fill_from_matrix(i_f, [args.bs, args.bs, 1], dim=2, max_level=args.max_level)
    else:
        w_obj.fill_from_matrix(i_f, [args.bs, args.bs, args.bs], dim=3, max_level=args.max_level)

    pos_dot = new_name.rfind(".")
    new_name_now = new_name[:pos_dot] + "_" + i_k + new_name[pos_dot:]
    w_obj.write(new_name_now)