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

import argparse

#------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Compare to WABBIT files.")
parser.add_argument('FILE1', type=str, help='First file')
parser.add_argument('FILE2', type=str, help='Second file')
parser.add_argument('--write-diff', action='store_true', help='Write differences to a new file with prefix "diff-" at location of first file.')
parser.add_argument('--write-int', action='store_true', help='Write interpolation of second file onto first files grid with prefix "new-" at location of first file ')

args = parser.parse_args()

#------------------------------------------------------------------------------

file1 = args.FILE1
file2 = args.FILE2

if os.path.isfile(file1) and os.path.isfile(file2):

    print("*****************************************************")
    print("Comparing wabbit HDF5 files \nfile1 =   %s \nfile2 =   %s" % (file1, file2))

    w_obj1 = wabbit_tools.WabbitHDF5file()
    w_obj1.read(file1)
    w_obj2 = wabbit_tools.WabbitHDF5file()
    w_obj2.read(file2)

    bool_similar = w_obj1.isClose(w_obj2, verbose=True)

    if not bool_similar and args.write_diff or args.write_int:
        path1 = os.path.split(file1)
        w_obj_diff = w_obj1 - w_obj2
        w_obj_new = (w_obj1 * 0) + w_obj2  # sneaky way to interpolate w_obj2 grid onto w_obj_new
        if args.write_diff:
            w_obj_diff.write(os.path.join(path1[0], "diff-" + path1[1]))
        if args.write_int:
            w_obj_new.write(os.path.join(path1[0], "new-" + path1[1]))

else:
    bool_similar = False
    print("Files not found, please check the file locations again!")
        
#------------------------------------------------------------------------------
# return error code
sys.exit(not bool_similar)