#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 24-08-14 by JB

For some simulations we need very specific handcrafted grids.
This script takes care of this by taking a text input and converting it into a functionable WABBIT grid.

The first line identifies this as a WABBIT-grid file while the second defines the maximum depth.
Afterwards, information on the block position and lvl follow.

An example:
WABBIT-grid file v1.0
level=2
  22
1 22
22  
221 

The left bottom of a file is the origin. Blocks with lower level only need there bottom-left-most entry to be present,
the rest can be arbitrary values except other numbers.
Obviously, this only works in 2D. Additionally, no gradedness check is done for the file.

"""
import sys, os, numpy as np
sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import wabbit_tools

import argparse, matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Converts a WABBIT-grid text-file to a wabbit input file.")
parser.add_argument('TEXT', type=str, help='Input Wabbit-grid text-file')
parser.add_argument('-o', '--output', type=str, help='Output location of H5 file, if not provided it is saved under same name as input.', default=None)
parser.add_argument('--bs', type=int, help='Block size of grid, default is 16.', default=16)
parser.add_argument('--max-level', type=int, help='Maxmium level for treecode of data, default is 9', default=9)
parser.add_argument('--domain-size', type=float, help='Size of computational domain, default is 1', default=1)
parser.add_argument('--time', type=float, help='Current time of WABBIT-file, default is 0.0', default=0.0)
parser.add_argument('--iteration', type=float, help='Current iteration of WABBIT-file, default is 0', default=0)

args = parser.parse_args()

LATEST_VERSION = "1.0"

with open(args.TEXT) as text_file:
    # read header and identify if this is indeed a WABBIT-grid file
    header = text_file.readline()

    if not header.startswith("WABBIT-grid file"):
        print("ERROR: File does not seem to be a WABBIT-grid file!")
        print("Did you ensure the first line starts with \"WABBIT-grid file\" followed by the version?")
        print(f"Latest version is \"v{LATEST_VERSION} \"")
        sys.exit(1)
    
    version = float(header.split(" ")[-1].replace("v",""))
    # possible version changes can be implemented here
    
    print(f"Reading in WABBIT-grid file with version v{version}")

    # read in next line where we should identify the level
    level_line = text_file.readline()
    if not level_line.startswith("level="):
        print("I was not able to identify Jmax of the grid. Is the line following the header \"level=[Jmax]\"?")
        print("I need this to identify how much I have to read in!")
        sys.exit(1)
    Jmax = int(level_line.replace("level=",""))
    print(f"Grid has maximum level Jmax = {Jmax}")

    grid_lines = text_file.readlines()

    if len(grid_lines) != 2**Jmax:
        print(f"File seems to have {len(grid_lines)} /= {2**Jmax} grid-lines. Please correct that!")
        sys.exit(1)

# get lines to correct length by concatenating spaces or cutting of trailing parts
for i_l in range(len(grid_lines)):
    if len(grid_lines[i_l]) < 2**Jmax:
        grid_lines[i_l] = grid_lines[i_l].ljust(2**Jmax)
    else:
        grid_lines[i_l] = grid_lines[i_l][:2**Jmax]
# we need to invert y of the grid as the origin is at the bottom
grid_lines = grid_lines[::-1]

# now we are ready to identify blocks
# we go level by level and check if a possible block exists (there treecode position is set at this level)
# then we fill the level and treecode array, that is all we need after all!
level = []
treecode = []

for i_level in np.arange(Jmax)+1:
    for ix in np.arange(0,2**Jmax,2**(Jmax-i_level)):
        for iy in np.arange(0,2**Jmax,2**(Jmax-i_level)):
            if grid_lines[iy][ix] == str(i_level):
                level.append(i_level)
                treecode_now = wabbit_tools.tc_encoding([ix+1, iy+1],level=Jmax, max_level=args.max_level, dim=2)
                treecode.append(treecode_now)
                # print(f"Found block lvl {i_level} : {ix} - {iy} with TC= {treecode_now}")
level = np.array(level)
treecode = np.array(treecode)
number_blocks = len(level)

# create blocks array of zeros, +1 as it is redundant format
blocks = np.zeros([number_blocks, args.bs+1, args.bs+1])

# now we can create the wabbit file
w_obj = wabbit_tools.WabbitHDF5file()
w_obj.fill_vars(args.domain_size*np.array([1,1]), blocks, treecode, level, time=args.time, iteration=args.iteration, max_level=args.max_level)

# and finally write it
if args.output == None:
    new_name = args.TEXT.replace(args.TEXT[args.TEXT.rfind("."):], ".h5")
else:
    if args.output.endswith(".h5"):
        new_name = args.output
    else:
        new_name = args.output + ".h5"
w_obj.write(new_name)