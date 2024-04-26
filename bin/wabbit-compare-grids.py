#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 17:02:39 2018
Edited on Fri Apr 12 13:39 2024

@author: engels, JB

Wrapper for compare grid
"""

import sys, os
sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import wabbit_tools

if not len(sys.argv) == 3:
    print(f"ERROR: Wrong number of inputs - {len(sys.argv)-1}, usage: wabbit-compare-grids.py [FILE1] [FILE2]")
if not (os.path.isfile(sys.argv[1]) and sys.argv[1].endswith(".h5")):
    print("ERROR: File1 not suitable, are you sure it exists and is a .h5 file?")
if not (os.path.isfile(sys.argv[2]) and sys.argv[2].endswith(".h5")):
    print("ERROR: File2 not suitable, are you sure it exists and is a .h5 file?")

w_obj1 = wabbit_tools.WabbitHDF5file()
w_obj1.read(sys.argv[1], read_var='meta', verbose=False)
w_obj2 = wabbit_tools.WabbitHDF5file()
w_obj2.read(sys.argv[2], read_var='meta', verbose=False)

print(w_obj1.compareGrid(w_obj2))