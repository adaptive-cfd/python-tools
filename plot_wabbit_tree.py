#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 14:16:20 2022

@author: engels
"""

import numpy as np
import matplotlib.pyplot as plt
import finite_differences
import insect_tools
import wabbit_tools


file = '/home/engels/dev/WABBIT3-org/for-YP_3vort-parameters/CDF44_NdtPerGrid/vor_000020000000.h5'
# file = '/home/engels/dev/WABBIT3-org/CiCP_CDF40_simulations/bumblebee_Jmax8_CF40/loadbalancing_figure/mask_000002300000.h5'

wabbit_tools.plot_wabbit_file(file, gridonly=True, gridonly_coloring='level')

time, x0, dx, box, data, treecode=wabbit_tools.read_wabbit_hdf5(file)

dim = dx.shape[1]

# #%% add 0th column that will contain the level of the block
# # this is the 1st sorting criterion
# treecode2 = np.zeros( (treecode.shape[0],treecode.shape[1]+1) )
# treecode2[:,1:] = treecode

# for i in range(treecode2.shape[0]):
#     level = wabbit_tools.treecode_level(treecode[i,:])
#     treecode2[i,0] = level


# #%% sort treecodes
# keys = []
# for i in range(treecode2.shape[1]):
#     keys.append(  treecode2[:,treecode2.shape[1]-1 - i] )
        
# ii = np.lexsort(keys=tuple(keys))

# treecode3 = treecode2[ ii, :].copy()

if dim == 3:
    raise "This is 3D data and we cannot yet handle it."

plt.figure()
# for i in range( 1000 ):
for i in range( treecode.shape[0] ):    
    x1, y1 = 0.0, 0.0
    
    for j in range( treecode.shape[1] ):  
        if (treecode[i,j] >= 0.0):
            dJ = 0.25*2**(-2*(j+2))#1.0/(6*(j+1))
            print(dJ)
            x2 = x1 + dJ*(treecode[i,j]-1.5)
            y2 = y1 + 2**-(j+1)
            
            plt.plot( [x1,x2], [y1,y2], 'ko-', mfc='w')
                
            
            
            x1,y1 = x2,y2
    plt.plot( [x2], [y2], 'ro-')
   
    
plt.gcf().set_size_inches( [20.0, 10.0] )