#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 14:16:20 2022
Edited on Fri Apr 12 13:39 2024

@author: engels, JB

This function displays an actual tree formulation for the grid. Use it from the command line with a h5 file or hardcode your own files
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import wabbit_tools

def plot_wabbit_tree(file):
    w_obj = wabbit_tools.WabbitHDF5file()
    w_obj.read(file, read_var="meta")
    wabbit_tools.plot_wabbit_file(w_obj, gridonly=True, gridonly_coloring='level')

    dim = w_obj.dim

    plt.figure()
    # for i in range( 1000 ):
    for i_b in range( w_obj.total_number_blocks ):    
        x1, y1 = 0.0, 0.0

        tc_now = w_obj.block_treecode_num[i_b]
        lvl_now = w_obj.level[i_b]
        
        for j_l in range( lvl_now ):  
            digit_now = wabbit_tools.tc_get_digit_at_level(tc_now, j_l, w_obj.max_level, w_obj.dim)

            dJ = 2.0**-w_obj.dim * 2.0**(-w_obj.dim*(j_l+2))#1.0/(6*(j+1))
            x2 = x1 + dJ*(digit_now-(1.5 + 2*(w_obj.dim==3)))
            y2 = y1 + 2**-(j_l+1)
            
            plt.plot( [x1,x2], [y1,y2], 'ko-', mfc='w')
            x1,y1 = x2,y2
        plt.plot( [x2], [y2], 'ro-')
    
        
    plt.gcf().set_size_inches( [20.0, 10.0] )
    plt.show()

if __name__ == "__main__":

    if len(sys.argv) == 2:
        file = sys.argv[1]
    else:
        file = '../../WABBIT/TESTING/conv/blob_2D_adaptive_CDF62/phi_000000250000.h5'
        # file = '../../WABBIT/TESTING/conv/blob_3D_adaptive_CDF40/phi_000000051549.h5'

    plot_wabbit_tree(file)