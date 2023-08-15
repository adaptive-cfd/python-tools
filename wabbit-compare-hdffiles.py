#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 14:19:24 2023

@author: engels
"""

import wabbit_tools
import numpy as np
import sys
import os

file1 = sys.argv[1]
file2 = sys.argv[2]

# file1 = "/home/engels/dev/WABBIT777/3vort/filter_tests/CDF22bio_Jmax7_c10_eps5.0000e-05_original/vor_000012500000.h5"
# file2 = "/home/engels/dev/WABBIT777/3vort/filter_tests/CDF22bio_Jmax7_c10_eps5.0000e-05_original/vor_000015000000.h5"

print("*****************************************************")
print("Comparing wabbit HDF5 files \nfile1 =   %s \nfile2 =   %s" % (file1, file2))

if not os.path.isfile(file1) or not os.path.isfile(file2):
    print("ERROR: at least one file not found...can't compare")
    sys.exit(1)
    


time1, x01, dx1, box1, data1, treecode1 = wabbit_tools.read_wabbit_hdf5( file1 )
time2, x02, dx2, box2, data2, treecode2 = wabbit_tools.read_wabbit_hdf5( file2 )


grid_similarity = wabbit_tools.compare_two_grids(treecode1, treecode2)

print("Grid similarity=%f" % (grid_similarity))

#------------------------------------------------------------------------------
# compute some keyvalues to compare the files
#------------------------------------------------------------------------------
def keyvalues(x0, dx, data):
    max1, min1 = np.max(data), np.min(data)
    mean1, squares1 = 0.0, 0.0
    
    # loop over all blocks
    for i in range( data.shape[0] ):
        if len(data.shape) == 3: ## 2D
            mean1 = mean1 + np.product(dx[i,:]) * np.sum(data[i,:,:])
            squares1 = squares1 + np.product(dx[i,:]) * np.sum(data[i,:,:]**2)
        else: ## 3D
            mean1 = mean1 + np.product(dx[i,:]) * np.sum(data[i,:,:,:])
            squares1 = squares1 + np.product(dx[i,:]) * np.sum(data[i,:,:,:]**2)
        
    return(max1, min1, mean1, squares1 )

max1, min1, mean1, squares1 = keyvalues(x01, dx1, data1)
max2, min2, mean2, squares2 = keyvalues(x02, dx2, data2)


#------------------------------------------------------------------------------
# compute L2 norm of difference, but only if the grids are identical
#------------------------------------------------------------------------------
diff_L2 = 0.0
norm_L2 = 0.0
error_L2 = np.nan

if abs(grid_similarity - 1.0) <= 1.0e-11:    
    # this algorithm is SLOW (N^2)
    for i in range(treecode1.shape[0]):
        # we look for this tree code in the second array
        code1 = treecode1[i,:]
        
        # normalization is norm of data1
        if len(data1.shape) == 3: ## 2D
            norm_L2 = norm_L2 + np.linalg.norm( np.ndarray.flatten(data1[i,:,:]) )
        else: ## 3D
            norm_L2 = norm_L2 + np.linalg.norm( np.ndarray.flatten(data1[i,:,:,:]) )
    
        # L2 difference
        for j in range(treecode2.shape[0]):
            code2 = treecode2[j,:]
            if np.linalg.norm( code2-code1 ) < 1.0e-13:
                # found code1 in the second array
                if len(data1.shape) == 3: ## 2D
                    diff_L2 = diff_L2 + np.linalg.norm( np.ndarray.flatten(data1[i,:,:]-data2[j,:,:]) )
                else:
                    diff_L2 = diff_L2 + np.linalg.norm( np.ndarray.flatten(data1[i,:,:,:]-data2[j,:,:,:]) )
                
        if norm_L2 >= 1.0e-10:
            # relative error
            error_L2 = diff_L2 / norm_L2
        else:
            # absolute error
            error_L2 = diff_L2
        
        
print("max=%2.5e min=%2.5e mean=%2.5e, squares=%2.5e, L2_error=%2.5e" % (max1, min1, mean1, squares1, error_L2))
print("max=%2.5e min=%2.5e mean=%2.5e, squares=%2.5e" % (max2, min2, mean2, squares2))
        
#------------------------------------------------------------------------------
# return error code
if (abs(grid_similarity - 1.0) <= 1.0e-11) and (error_L2 <= 1.0e-13):
    print("GREAT: files are indeed identical.")
    sys.exit(0)
else:
    print("ERROR: files are not identical.")
    sys.exit(1)