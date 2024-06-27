#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 14:19:24 2023
Edited on Fri Apr 12 13:39 2024

@author: engels, JB

Wrapper for isClose
"""

import sys, os
sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import wabbit_tools

# for now only in development option: Write difference as field
write_diff = False

if not len(sys.argv) == 3:
    print(f"ERROR: Wrong number of inputs - {len(sys.argv)-1}, usage: wabbit-compare-grids.py [FILE1] [FILE2]")
    sys.exit(1)
if not (os.path.isfile(sys.argv[1]) and sys.argv[1].endswith(".h5")):
    print("ERROR: File1 not suitable, are you sure it exists and is a .h5 file?")
    sys.exit(1)
if not (os.path.isfile(sys.argv[2]) and sys.argv[2].endswith(".h5")):
    print("ERROR: File2 not suitable, are you sure it exists and is a .h5 file?")
    sys.exit(1)

file1 = sys.argv[1]
file2 = sys.argv[2]

print("*****************************************************")
print("Comparing wabbit HDF5 files \nfile1 =   %s \nfile2 =   %s" % (file1, file2))

w_obj1 = wabbit_tools.WabbitHDF5file()
w_obj1.read(file1)
w_obj2 = wabbit_tools.WabbitHDF5file()
w_obj2.read(file2)

bool_similar = w_obj1.isClose(w_obj2, verbose=True)

if not bool_similar and write_diff:
    path1 = os.path.split(file1)
    w_obj_diff = w_obj1 - w_obj2
    w_obj_new = (w_obj1 * 0) + w_obj2  # sneaky way to interpolate w_obj2 grid onto w_obj_new
    w_obj_diff.write(os.path.join(path1[0], "diff-" + path1[1]))
    w_obj_new.write(os.path.join(path1[0], "new-" + path1[1]))
        
#------------------------------------------------------------------------------
# return error code
sys.exit(not bool_similar)