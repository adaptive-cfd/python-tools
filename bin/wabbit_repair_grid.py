#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 24-08-13 by JB

This file attemps to repair the grid information of a file.
For a file to have all necessary information for dealing with them it needs the block values, treecode information and level information.
If one of them is intact the blocks cannot be placed correctly (with the correct values).

If parts are missing, we can try to recreate the information from redundant infos.
Treecode:
    1) coords_origin together with the domain size can give the treecode information in order to fill missing parts.
Level:
    1) coords_spacing together with the domain size can give the level information in order to fill missing parts.
    2) There is also a second option if that is not sufficient: We can walk from a treecode in any direction and find the closest block,
       the distance gives the size that this block occupies

"""
import sys, os, numpy as np
sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import wabbit_tools

import argparse, matplotlib.pyplot as plt

#------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Performs simple check if a wabbit state file is corrupted or not. Can also output some visualizations.")
parser.add_argument('FILE', type=str, help='Input file')

args = parser.parse_args()

#------------------------------------------------------------------------------

# load in file
w_obj = wabbit_tools.WabbitHDF5file()
w_obj.read(args.FILE)

# firstly, check if block data has any obvious erros
print("1) Checking if any block data is completely missing by checking sum(blocks[i_b]).")
has_0 = np.sum(np.sum(w_obj.blocks, axis=tuple(np.arange(1, w_obj.dim+1))) == 0)
if has_0 > 0:
    print(f"   {has_0} blocks have sum(blocks[i_b, :])=0. It should be 0 and therefore the file is irreperable.")
    sys.exit(1)
else:
    print(f"   0 Blocks have sum(blocks[i_b, :])=0. It should be 0 so that check passed.")


# second: check if treecode is consistent
print("2) Checking if we need to repair the treecodes.")
has_0 = np.sum(w_obj.block_treecode_num == 0)
if has_0 == 1:
    print(f"   {has_0} blocks have treecode=0. It should be 1 so the treecode array is intact.")
elif has_0 > 1:
    print(f"   {has_0} blocks have treecode=0. It should be 1 so the treecode array needs to be repaired.")
    
    # check if we can actually repair
    TC_origin = np.concatenate([w_obj.block_treecode_num[:, None], w_obj.coords_origin], axis=1)
    has_0 = np.sum(np.sum(TC_origin, axis=1) == 0)
    if has_0 > 1:
        print(f"   {has_0} blocks have treecode=0 and origin=0. It should be 1 so the treecode is irreperable.")
        sys.exit(1)
    else:
        print(f"   {has_0} blocks have treecode=0 and origin=0. It should be 1 so we can repair the treecode.")

    rep_blocks = 0
    for i_b in range(w_obj.total_number_blocks):
        if w_obj.block_treecode_num[i_b] == 0:
            w_obj.block_treecode_num[i_b] = wabbit_tools.origin2treecode(w_obj.coords_origin[i_b], w_obj.max_level, w_obj.dim, w_obj.domain_size)
            rep_blocks += 1
    print(f"   Repaired treecodes for {rep_blocks} blocks from coords_origin.")
    rep_blocks = 0
    for i_b in range(w_obj.total_number_blocks):
        if np.sum(w_obj.coords_origin[i_b]) == 0:
            w_obj.coords_origin[i_b] = wabbit_tools.treecode2origin(w_obj.block_treecode_num[i_b], w_obj.max_level, w_obj.dim, w_obj.domain_size)
            rep_blocks += 1
    if rep_blocks != 1: print(f"   Repaired coords_origin for {rep_blocks} blocks from treecodes.")


# third: check if level is consistent
print("3) Checking if we need to repair the levels.")
has_0 = np.sum(w_obj.level == 0)
if has_0 == 0:
    print(f"   {has_0} blocks have level=0. It should be 0 so the level array is intact.")
elif has_0 > 0:
    print(f"   {has_0} blocks have level=0. It should be 0 so the level array needs to be repaired.")

    # check if we can repair using spacing
    level_spacing = np.concatenate([w_obj.level[:, None], w_obj.coords_spacing], axis=1)
    has_0 = np.sum(np.sum(level_spacing, axis=1) == 0)
    if has_0 == 0:
        print(f"   {has_0} blocks have level=0 and spacing=0. It should be 0 so we can repair the level with the spacing.")
        rep_blocks = 0
        for i_b in range(w_obj.total_number_blocks):
            if w_obj.level[i_b] == 0:
                w_obj.level[i_b] = wabbit_tools.spacing2level(w_obj.coords_spacing[i_b], w_obj.block_size, w_obj.domain_size)
                rep_blocks += 1
        print(f"   Repaired level for {rep_blocks} blocks from coords_spacing.")

    else:
        print(f"   {has_0} blocks have level=0 and spacing=0. It should be 0 so not all levels can be repaired using the spacing.")
        rep_blocks = 0
        for i_b in range(w_obj.total_number_blocks):
            if w_obj.level[i_b] == 0 and np.all(w_obj.coords_spacing[i_b] != 0):
                w_obj.level[i_b] = wabbit_tools.spacing2level(w_obj.coords_spacing[i_b], w_obj.block_size, w_obj.domain_size)
                rep_blocks += 1
        print(f"   Repaired level for {rep_blocks} blocks from coords_spacing.")
        print(f"   Luckily, we can repair all other levels using the treecode array.")
        rep_blocks = 0
        for i_b in range(w_obj.total_number_blocks):
            if w_obj.level[i_b] == 0:
                w_obj.level[i_b] = wabbit_tools.level_from_treecode(w_obj.block_treecode_num[i_b], w_obj.block_treecode_num, max_level=w_obj.max_level, dim=w_obj.dim)
                rep_blocks += 1
        print(f"   Repaired level for {rep_blocks} blocks from treecode array.")

    rep_blocks = 0
    for i_b in range(w_obj.total_number_blocks):
        if np.sum(w_obj.coords_spacing[i_b]) == 0:
            w_obj.coords_spacing[i_b] = wabbit_tools.level2spacing(w_obj.level[i_b], w_obj.dim, w_obj.block_size, w_obj.domain_size)
            rep_blocks += 1
    if rep_blocks != 1: print(f"   Repaired coords_spacing for {rep_blocks} blocks from level.")

print("Data should be repaired now. However, consider that your data might still be corrupted. Saving new data ..")
w_obj.write(os.path.join(os.path.split(args.FILE)[0], f"repaired-{os.path.split(args.FILE)[1]}"))