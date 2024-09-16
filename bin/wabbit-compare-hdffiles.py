#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 14:19:24 2023
Edited on Fri Apr 12 13:39 2024

@author: engels, JB

Wrapper for isClose
"""

import sys, os, argparse, glob
sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import wabbit_tools

#------------------------------------------------------------------------------

def compare_files(file1, file2, args):
    print("Comparing wabbit HDF5 files \nfile1 =   %s \nfile2 =   %s" % (file1, file2))

    w_obj1 = wabbit_tools.WabbitHDF5file()
    w_obj1.read(file1)
    w_obj2 = wabbit_tools.WabbitHDF5file()
    w_obj2.read(file2)

    bool_similar = w_obj1.isClose(w_obj2, verbose=True)

    if (not bool_similar or args.write_all) and (args.write_diff or args.write_int):
        path1 = os.path.split(file1)
        if args.write_diff:
            w_obj_diff = w_obj1 - w_obj2
            w_obj_diff.write(os.path.join(path1[0], "diff-" + path1[1]))
        if args.write_int:
            w_obj_new = (w_obj1 * 0) + w_obj2  # sneaky way to interpolate w_obj2 grid onto w_obj_new
            w_obj_new.write(os.path.join(path1[0], "new-" + path1[1]))

    return bool_similar
        
#------------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compare two WABBIT files or two directories with WABBIT files.")
    parser.add_argument('PATH1', type=str, help='First file or directory.')
    parser.add_argument('PATH2', type=str, help='Second file or directory.')
    parser.add_argument('--write-all', action='store_true', help='Usually only files which differ are written with write-diff or write-int. With this flag they are written nonetheless.')
    parser.add_argument('--write-diff', action='store_true', help='Write differences to a new file with prefix "diff-" at location of first path.')
    parser.add_argument('--write-int', action='store_true', help='Write interpolation of second file onto first files grid with prefix "new-" at location of first path.')

    args = parser.parse_args()

    #------------------------------------------------------------------------------

    print("*****************************************************")

    path1 = args.PATH1
    path2 = args.PATH2

    # if we input files then they are simply compared against each other
    if os.path.isfile(path1) and os.path.isfile(path2):
        bool_similar = compare_files(path1, path2, args)

    # if we input two directories then files with same name are compared against each other, nice for comparing test results
    elif os.path.isdir(path1) and os.path.isdir(path2):
        print("Comparing two directories. Trying to compare each .h5 file in first path with file with same name in second path.")
        bool_similar = True

        # get all available files in the directories
        files1 = sorted(glob.glob(os.path.join(path1, "*.h5")))
        files2 = sorted(glob.glob(os.path.join(path2, "*.h5")))

        for i_f1 in files1:
            # extract file name
            name = os.path.split(i_f1)[1]
            i_f2 = os.path.join(path2, name)
            # check if second folder has a file with similar name, if so then compare
            if i_f2 in files2:
                i_bool = compare_files(i_f1, i_f2, args)
                bool_similar = bool_similar and i_bool
            else:
                print(f"File found in path1 but not in path2: {i_f1} - {i_f2}")
                bool_similar = False
    else:
        bool_similar = False
        print("Files or directories not found, please check the path locations again!")

    # return error code
    sys.exit(not bool_similar)