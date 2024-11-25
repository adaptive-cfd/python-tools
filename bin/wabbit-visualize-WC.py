#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 24-08-19 by JB

Visualize the wavelet coefficients of a wavelet decomposed state in Mallat form.
This should be applied on fields, which were written with saveHDF5_wavelet_decomposed_tree.
Additionally, it works best if files are equidistant. 

"""
import sys, os, numpy as np
sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import wabbit_tools

import argparse, matplotlib.pyplot as plt, matplotlib, scipy.io

#------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Visualize wavelet coefficients of a wavelet decomposed state in Mallat form.")
parser.add_argument('FILE', type=str, help='Input file')
parser.add_argument('-l', '--level', type=str, help='Level to display, choose 1-JmaxActive or \'all\'.', default="all")
parser.add_argument('-f', '--full', action="store_true", help='If true: Display all WC from lower levels and only have SC of lowest level. If false: each level have only it\'s own SC and WC')
parser.add_argument('--no-sc', action="store_true", help='SC messing up the colorbar? Use this and they will be ignored')
parser.add_argument('-d', '--display', action='store_true', help='Display plots.')
group_save = parser.add_mutually_exclusive_group()
group_save.add_argument('-s', '--save', action='store_true', help='Save plots at position of original file as png and pdf files.')
group_save.add_argument('--save-png', action='store_true', help='Save plots at position of original file as png files.')
group_save.add_argument('--save-pdf', action='store_true', help='Save plots at position of original file as pdf files.')
parser.add_argument('--L2', action='store_true', help='Renorm WC into L2 norm to match matlab')
parser.add_argument('--plot-log', action='store_true', help='Plots with logarithmic colorbar.')
parser.add_argument('--plot-col-sym', action='store_true', help='Plot the colors symmetric around zero.')
group_lim = parser.add_mutually_exclusive_group()
group_lim.add_argument('--plot-lim-WC', action='store_true', help='Set maxima after wavelet coefficients of this level only.')
group_lim.add_argument('--plot-lim', type=float, help='Set min/max of plots', default=-1)
parser.add_argument('--plot-keys', action='store_true', help='Plots key values (max, min, std) for different levels.')
parser.add_argument('--plot-hist', action='store_true', help='Plots histogram for different levels.')
parser.add_argument('--matlab', action='store_true', help="Output file as matlab matrix file as well.")

args = parser.parse_args()

#------------------------------------------------------------------------------

w_obj = wabbit_tools.WabbitHDF5file()
w_obj.read(args.FILE)


level_min, level_max = w_obj.get_max_min_level()
if args.level == 'all':
    level = level_max
else:
    level = int(args.level)

# size of decomposed patches
bs_WD = np.array(w_obj.block_size[:w_obj.dim]-1)//2

field_l = []
for i_level in np.arange(level)+1:
    field_l.append(np.zeros([2**i_level*(w_obj.block_size[0]-1)]*w_obj.dim))

# loop over each point as wc or sc and copy them into corresponding positions (multiple for low-level sc)
for i_b in range(w_obj.total_number_blocks):
    i_level = w_obj.level[i_b]

    # skip too high levels
    if i_level > level: continue

    # get coordinates of this block on this level, used for shifting values, for full representation we need the representation on higher levels too
    ixyz = [0] * (level+1)
    for j_level in np.arange(i_level, level+1):
        ixyz[j_level] = wabbit_tools.tc_decoding(w_obj.block_treecode_num[i_b], level=j_level, dim=w_obj.dim, max_level=w_obj.max_level)
        # yes, here again we somehow need to interchange X and Y because the treecode is defined differently
        ixyz[j_level][0], ixyz[j_level][1] = ixyz[j_level][1], ixyz[j_level][0]

    # loop over each point
    for ix in range(w_obj.block_size[0]-1):
        for iy in range(w_obj.block_size[1]-1):
            # for dim=2 this loop only goes once with iz=0
            for iz in range(w_obj.block_size[2]-1 * (w_obj.dim==3)):
                if w_obj.dim == 2:
                    value = w_obj.blocks[i_b, ix, iy]
                else:
                    value = w_obj.blocks[i_b, ix, iy, iz]
                
                # renorm with L2 norm if wanted
                if np.any(np.array([ix%2, iy%2, iz%2]) != 0) and args.L2:
                    # renorm level shift
                    value /= np.power(2,(i_level-level_max-1)*w_obj.dim/2)
                    # renorm what component we have (if in one direction or cross direction)
                    value /= 2**np.sum([ix%2==1, iy%2==1, iz%2==1])
                
                # we want to write in our own level or higher
                for j_level in np.arange(i_level, level+1):

                    # SC: write only in own level, for full write in higher level if on lowest level
                    if np.all(np.array([ix%2, iy%2, iz%2]) == 0):
                        if args.no_sc: continue
                        if not args.full and not i_level == j_level: continue
                        if args.full and not i_level == level_min: continue
                        is_wc = False
                    # WC: write only in own level, for full always write in higher level
                    else:
                        if not args.full and not i_level == j_level: continue
                        is_wc = True

                    # WC, write on level and higher levels for full representation
                    coord = (np.array(ixyz[i_level])-1)*bs_WD + (np.array([ix,iy,iz])//2)[:w_obj.dim]
                    coord += (np.array([ix%2==1, iy%2==1, iz%2==1])*2**(i_level-1)*(w_obj.block_size-1))[:w_obj.dim]
                    if w_obj.dim == 2:
                        field_l[j_level-1][coord[0]][coord[1]] = value
                    else:
                        field_l[j_level-1][coord[0]][coord[1]][coord[2]] = value

# output some measurements
val_max, val_min, val_std, val_L2 = [0] * (level+1), [0] * (level+1), [0] * (level+1), [0] * (level+1)
for i_level in np.arange(level)+1:
    s_f = field_l[i_level-1].shape
    val_max[i_level] = [np.max(field_l[i_level-1][:s_f[0]//2,:s_f[1]//2]), np.max(field_l[i_level-1][s_f[0]//2:,:s_f[1]//2]), \
               np.max(field_l[i_level-1][:s_f[0]//2,s_f[1]//2:]), np.max(field_l[i_level-1][s_f[0]//2:,s_f[1]//2:])]
    val_min[i_level] = [np.min(field_l[i_level-1][:s_f[0]//2,:s_f[1]//2]), np.min(field_l[i_level-1][s_f[0]//2:,:s_f[1]//2]), \
               np.min(field_l[i_level-1][:s_f[0]//2,s_f[1]//2:]), np.min(field_l[i_level-1][s_f[0]//2:,s_f[1]//2:])]
    val_std[i_level] = [np.std(field_l[i_level-1][:s_f[0]//2,:s_f[1]//2]), np.std(field_l[i_level-1][s_f[0]//2:,:s_f[1]//2]), \
               np.std(field_l[i_level-1][:s_f[0]//2,s_f[1]//2:]), np.std(field_l[i_level-1][s_f[0]//2:,s_f[1]//2:])]
    val_L2[i_level] = [np.mean(field_l[i_level-1][:s_f[0]//2,:s_f[1]//2]**2), np.mean(field_l[i_level-1][s_f[0]//2:,:s_f[1]//2]**2), \
               np.mean(field_l[i_level-1][:s_f[0]//2,s_f[1]//2:]**2), np.mean(field_l[i_level-1][s_f[0]//2:,s_f[1]//2:]**2)]
    print(f"L{i_level:02d} - printing values for SC - WX - WY - WXY")
    print(f"    Max: {val_max[i_level][0]:9g} - {val_max[i_level][1]:9g} - {val_max[i_level][2]:9g} - {val_max[i_level][3]:9g}, total: {np.max(val_max[i_level][:]):9g}")
    print(f"    Min: {val_min[i_level][0]:9g} - {val_min[i_level][1]:9g} - {val_min[i_level][2]:9g} - {val_min[i_level][3]:9g}, total: {np.min(val_min[i_level][:]):9g}")
    print(f"    Std: {val_std[i_level][0]:9g} - {val_std[i_level][1]:9g} - {val_std[i_level][2]:9g} - {val_std[i_level][3]:9g}, total: {np.std(field_l[i_level-1][:,:]):9g}")
    print(f"    L2:  { val_L2[i_level][0]:9g} - { val_L2[i_level][1]:9g} - { val_L2[i_level][2]:9g} - { val_L2[i_level][3]:9g}, total: {np.sum(val_L2[i_level][:]):9g}")

if args.plot_keys:
    fig = plt.figure(100, figsize=[7,5])

    plt.plot(np.arange(level)+1, np.array(val_max[1:])[:,1], "o-", label="WX")
    plt.plot(np.arange(level)+1, np.array(val_max[1:])[:,2], "o-", label="WY")
    plt.plot(np.arange(level)+1, np.array(val_max[1:])[:,3], "o-", label="WXY")
    plt.legend()
    plt.xlabel("Level"); plt.ylabel("Max"); plt.title("Max values")
    plt.tight_layout(pad=0.15)
    fig = plt.figure(101, figsize=[7,5])
    plt.plot(np.arange(level)+1, np.array(val_min[1:])[:,1], "o-", label="WX")
    plt.plot(np.arange(level)+1, np.array(val_min[1:])[:,2], "o-", label="WY")
    plt.plot(np.arange(level)+1, np.array(val_min[1:])[:,1], "o-", label="WXY")
    plt.legend()
    plt.xlabel("Level"); plt.ylabel("Min"); plt.title("Min values")
    plt.tight_layout(pad=0.15)
    fig = plt.figure(102, figsize=[7,5])
    plt.plot(np.arange(level)+1, np.array(val_std[1:])[:,1], "o-", label="WX")
    plt.plot(np.arange(level)+1, np.array(val_std[1:])[:,2], "o-", label="WY")
    plt.plot(np.arange(level)+1, np.array(val_std[1:])[:,3], "o-", label="WXY")
    plt.legend()
    plt.xlabel("Level"); plt.ylabel("Std"); plt.title("Standard deviation")
    plt.tight_layout(pad=0.15)
    fig = plt.figure(103, figsize=[7,5])
    plt.plot(np.arange(level)+1, np.array(val_L2[1:])[:,1], "o-", label="WX")
    plt.plot(np.arange(level)+1, np.array(val_L2[1:])[:,2], "o-", label="WY")
    plt.plot(np.arange(level)+1, np.array(val_L2[1:])[:,3], "o-", label="WXY")
    plt.legend()
    plt.xlabel("Level"); plt.ylabel("$L_2$"); plt.title("$L_2$-Norm")
    plt.tight_layout(pad=0.15)

    names, fig_num = ["Max", "Min", "Std", "L2"], [100, 101, 102, 103]
    for i_fig in range(4):
        fig = plt.figure(fig_num[i_fig], figsize=[7,5])
        if args.save or args.save_png:
            plt.savefig( args.FILE.replace('.h5',f'-{names[i_fig]}.png'), dpi=200, transparent=True )
        if args.save or args.save_pdf:
            plt.savefig( args.FILE.replace('.h5',f'-{names[i_fig]}.pdf') )

if args.plot_hist:
    for i_level in np.arange(level)+1:
        bins = int(min(bs_WD[0]*2 * 2**(i_level), 500))
        fig = plt.figure(200 + i_level, figsize=[7,5])
        s_f = field_l[i_level-1].shape
        # plt.hist(field_l[i_level-1][:s_f[0]//2,:s_f[1]//2].flatten(), bins=bins, label='SC', density=True, histtype='step')
        plt.hist(field_l[i_level-1][s_f[0]//2:,:s_f[1]//2].flatten(), bins=bins, label='WX', density=True, histtype='step')
        plt.hist(field_l[i_level-1][:s_f[0]//2,s_f[1]//2:].flatten(), bins=bins, label='WY', density=True, histtype='step')
        plt.hist(field_l[i_level-1][s_f[0]//2:,s_f[1]//2:].flatten(), bins=bins, label='WXY', density=True, histtype='step')
        plt.legend()
        plt.xlim(-50, 50)
        if i_level == level: plt.ylim(0, 0.06)
        else: plt.ylim(0, 0.04)
        plt.xlabel("WC value"); plt.ylabel("Incidence"); plt.title(f"Histogram on level {i_level}")
        plt.grid(True)
        plt.tight_layout(pad=0.15)

        if args.save or args.save_png:
            plt.savefig( args.FILE.replace('.h5',f'-hist-L{i_level}.png'), dpi=200)
            # plt.savefig( args.FILE.replace('.h5',f'-hist-L{i_level}.png'), dpi=200, transparent=True )
        if args.save or args.save_pdf:
            plt.savefig( args.FILE.replace('.h5',f'-hist-L{i_level}.pdf') )

# plot
for i_level in np.arange(level)+1:
    fig = plt.figure(i_level, figsize=[7,5])

    if not args.plot_log:
        if args.plot_col_sym:
            if args.plot_lim_WC:
                vmax = np.max([-np.min(val_min[i_level][1:]), np.max(val_max[i_level][1:])])
            elif args.plot_lim != -1:
                vmax = abs(args.plot_lim)
            else:
                vmax = np.max([-np.min(val_min[i_level]), np.max(val_max[i_level])])
            plt.imshow(field_l[i_level-1], origin='lower', vmin=-vmax, vmax=vmax, cmap="RdBu")
        else:
            if args.plot_lim_WC:
                vmaxmin = [np.min(val_min[i_level][1:]), np.max(val_max[i_level][1:])]
            elif args.plot_lim != -1:
                vmaxmin = [-abs(args.plot_lim), abs(args.plot_lim)]
            else:
                vmaxmin = [np.min(val_min[i_level]), np.max(val_max[i_level])]
            plt.imshow(field_l[i_level-1], origin='lower', vmin=vmaxmin[0], vmax=vmaxmin[1])
    else:
        plt.imshow(abs(field_l[i_level-1]), origin='lower', norm=matplotlib.colors.LogNorm())

    lines_start = level_min
    if not args.full: lines_start = i_level
    for j_level in np.arange(lines_start, i_level+1):
        plt.vlines(2**(j_level-1)*(w_obj.block_size[0]-1)-0.5, -0.5, 2**j_level*(w_obj.block_size[0]-1)-0.5, colors='k')
        plt.hlines(2**(j_level-1)*(w_obj.block_size[0]-1)-0.5, -0.5, 2**j_level*(w_obj.block_size[0]-1)-0.5, colors='k')

    plt.colorbar()
    plt.tight_layout(pad=0.15)

    if args.save or args.save_png:
        plt.savefig( args.FILE.replace('.h5',f'-WC-L{i_level}.png'), dpi=200, transparent=True )
    if args.save or args.save_pdf:
        plt.savefig( args.FILE.replace('.h5',f'-WC-L{i_level}.pdf') )

if args.matlab:
    mat_name = args.FILE.replace(args.FILE[args.FILE.rfind("."):], ".mat")
    field_s = {}
    for i_level in np.arange(level)+1:
        field_s[f'L{i_level}'] = field_l[i_level-1]
    scipy.io.savemat(mat_name, field_s)

if args.display:
    plt.show()