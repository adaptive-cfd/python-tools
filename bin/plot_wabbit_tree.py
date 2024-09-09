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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--angular", action="store_true",
                    help="Plot in angular plot")
parser.add_argument("-i", "--input", type=str, default=None,
                    help="Input .h5 file to be plotted")
parser.add_argument("-o", "--output", type=str, default=None,
                    help="output PDF file")
args = parser.parse_args()



def plot_wabbit_tree(file, plot_grid=True):
    w_obj = wabbit_tools.WabbitHDF5file()
    w_obj.read(file, read_var="meta")

    if plot_grid:
        wabbit_tools.plot_wabbit_file(w_obj, gridonly=True, gridonly_coloring='level')

    dim = w_obj.dim

    plt.figure()
    # for i in range( 1000 ):
    for i_b in range( w_obj.total_number_blocks ):    
        x1, y1 = 0.0, 0.0

        tc_now = w_obj.block_treecode_num[i_b]
        lvl_now = w_obj.level[i_b]
        
        for j_l in range( lvl_now ):  
            digit_now = wabbit_tools.tc_get_digit_at_level(tc_now, j_l+1, w_obj.max_level, w_obj.dim)

            dJ = 2.0**-w_obj.dim * 2.0**(-w_obj.dim*(j_l+2))#1.0/(6*(j+1))
            x2 = x1 + dJ*(digit_now-(1.5 + 2*(w_obj.dim==3)))
            y2 = y1 + 2**-(j_l+1)
            
            plt.plot( [x1,x2], [y1,y2], 'ko-', mfc='w')
            x1,y1 = x2,y2
        plt.plot( [x2], [y2], 'ro-')
    
        
    plt.gcf().set_size_inches( [20.0, 10.0] )
    plt.show()


def plot_wabbit_tree_angular(file, plot_grid=True):
    w_obj = wabbit_tools.WabbitHDF5file()
    w_obj.read(file, read_var="meta")

    if plot_grid:
        wabbit_tools.plot_wabbit_file(w_obj, gridonly=True, gridonly_coloring='level')

    dim = w_obj.dim

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # for i in range( 1000 ):
    for i_b in range( w_obj.total_number_blocks ):    
        r1, t1 = 0.0, 0.0

        tc_now = w_obj.block_treecode_num[i_b]
        lvl_now = w_obj.level[i_b]
        
        for j_l in range( lvl_now ):  
            digit_now = wabbit_tools.tc_get_digit_at_level(tc_now, j_l+1, w_obj.max_level, w_obj.dim)

            # r2 = r1 + 2**-(j_l+1)
            r2 = r1 + 1
            t2 = t1 + (digit_now - (2**dim-1)/2) * 2*np.pi / (2**(dim*(j_l+1)))
            
            plt.plot( [t1,t2], [r1,r2], 'ko-', mfc='w', linewidth=0.5)
            r1,t1 = r2,t2
        plt.plot( [t2], [r2], 'ro-', linewidth=0.5)
    
    # plt.grid(False)

    ax.set_yticks( np.arange(lvl_now+1))
    ax.set_xticks( np.arange(2**dim) * 2*np.pi / 2**dim)
        
    plt.gcf().set_size_inches( [20.0, 10.0] )
    plt.show()


if __name__ == "__main__":

    if args.input is not None:
        file = args.input
    else:
        file = '../WABBIT/TESTING/conv/blob_2D_adaptive_CDF62/phi_000000250000.h5'
        # file = '../WABBIT/TESTING/conv/blob_3D_adaptive_CDF40/phi_000000051549.h5'

    if not args.angular:
        plot_wabbit_tree(file, plot_grid=False)
    else:
        plot_wabbit_tree_angular(file, plot_grid=False)
        
    if args.output is None:
        args.output = file.replace('.h5', '.pdf')
        
    plt.savefig(args.output)