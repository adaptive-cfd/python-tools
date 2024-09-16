#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 24-08-12 by JB

check if file is corrupt or not
1: check for uniqueness of entries
2: check for arrays that should have monotonously ascending entries
3: check if arrays have entry 0
4: check if redundant arrays coincide

"""
import sys, os, numpy as np
sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import wabbit_tools

import argparse, matplotlib.pyplot as plt

#------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Performs simple check if a wabbit state file is corrupted or not. Can also output some visualizations.")
parser.add_argument('FILE', type=str, help='Input file')
parser.add_argument('--plot-monotone', action='store_true', help='Plot monotonity of lgt_ids and procs array.')
parser.add_argument('--plot-zero', action='store_true', help='Plot where arrays have elements or sum of entries = 0.')
parser.add_argument('--plot-blocks-sum', action='store_true', help='Plot sum of blocks over block index.')
parser.add_argument('-d', '--display', action='store_true', help='Display plots.')
parser.add_argument('-s', '--save', action='store_true', help='Save plots at position of original file.')

args = parser.parse_args()

#------------------------------------------------------------------------------


w_obj = wabbit_tools.WabbitHDF5file()
w_obj.read(args.FILE)


is_valid = True
# 1: check if some arrays are unique. This should be the case for:
#    - lgt_id
#    - pairs of [treecode, level]
is_unique = len(w_obj.lgt_ids) == len(np.unique(w_obj.lgt_ids)) ; is_valid = is_valid and is_unique
if not is_unique:
    print(f"Blocks are not unique. Found {len(w_obj.lgt_ids) - len(np.unique(w_obj.lgt_ids))} blocks that doubles another lgt_id.")
paired = np.core.records.fromarrays([w_obj.level, w_obj.block_treecode_num])
is_unique = len(paired) == len(np.unique(paired)) ; is_valid = is_valid and is_unique
if not is_unique:
    print(f"Grid is not unique. Found {len(paired) - len(np.unique(paired))} blocks that double another [treecode, level] pair.")

# 2: check if some arras are monotonly ascending. In the way that wabbit writes the files this should be the case for:
#    - lgt_ids
#    - procs
is_monotone = np.all(w_obj.lgt_ids[1:] > w_obj.lgt_ids[:-1]) ; is_valid = is_valid and is_monotone
if not is_monotone:
    print(f"Blocks have not monotonly ascending lgt_ids. Found {np.sum(w_obj.lgt_ids[1:] <= w_obj.lgt_ids[:-1])} blocks that have lower or equal ids as preceding block.")
is_monotone = np.all(w_obj.procs[1:] >= w_obj.procs[:-1])  ; is_valid = is_valid and is_monotone
if not is_monotone:
    print(f"Blocks have not monotonly ascending procs. Found {np.sum(w_obj.procs[1:] < w_obj.procs[:-1])} blocks that have lower proc as preceding block.")

# 3: check if arrays have zero entries which they should not have or only have 1
#     - lgt_ids (0)
#     - treecode (1)
#     - level (0)
#     - blocks (0)
#     - origin (1)
#     - spacing (0)
has_0 = np.sum(w_obj.lgt_ids == 0) ; is_valid = is_valid and has_0 <= 0
if has_0 > 0:
    print(f"{has_0} blocks have lgt_id=0. It should be 0.")
has_0 = np.sum(w_obj.block_treecode_num == 0) ; is_valid = is_valid and has_0 <= 1
if has_0 > 1:
    print(f"{has_0} blocks have treecode=0. It should be 1.")
has_0 = np.sum(w_obj.level == 0) ; is_valid = is_valid and has_0 <= 0
if has_0 > 0:
    print(f"{has_0} blocks have level=0. It should be 0.")
has_0 = np.sum(np.sum(w_obj.blocks, axis=tuple(np.arange(1, w_obj.dim+1))) == 0) ; is_valid = is_valid and has_0 <= 0
if has_0 > 0:
    print(f"{has_0} blocks have sum(blocks[:])=0. It should be 0.")
has_0 = np.sum(np.sum(w_obj.coords_origin, axis=1) == 0) ; is_valid = is_valid and has_0 <= 1
if has_0 > 1:
    print(f"{has_0} blocks have sum(origin)=0. It should be 1.")
has_0 = np.sum(np.sum(w_obj.coords_spacing, axis=1) == 0) ; is_valid = is_valid and has_0 <= 0
if has_0 > 0:
    print(f"{has_0} blocks have sum(spacing)=0. It should be 0.")


# 4: check if redundant arrays coincide, these are:
#     - treecode and coords_origin
#     - level and coords_spacing
i_corr1, i_corr2 = 0, 0
for i_b in range(w_obj.total_number_blocks):
    tc_from_origin = wabbit_tools.origin2treecode(w_obj.coords_origin[i_b], max_level=w_obj.max_level, dim=w_obj.dim, domain_size=w_obj.domain_size)
    if tc_from_origin != w_obj.block_treecode_num[i_b]: i_corr1 += 1
    level_from_spacing = wabbit_tools.spacing2level(w_obj.coords_spacing[i_b], block_size=w_obj.block_size, domain_size=w_obj.domain_size)
    if level_from_spacing != w_obj.level[i_b]: i_corr2 += 1
if i_corr1 > 0:
    print(f"{i_corr1} blocks do not have treecode and coords_origin that mean the same thing.")
if i_corr2 > 0:
    print(f"{i_corr2} blocks do not have level and coords_spacing that mean the same thing.")
is_valid = is_valid and (i_corr1 == 0) and (i_corr2 == 0)

# return result on screen
print("-"*80)
if is_valid:
    print("   File passed all obvious checks. That's a good indication that it is not corrupt, but still things could have gone wrong. :)")
else:
    print("   File did not pass some obvious checks. It is very likely that it is corrupt. :(")
print("-"*80)




if args.plot_monotone:
    plt.figure(1, figsize=[7,5])
    ax1 = plt.gca()

    ax2 = ax1.twinx()
    ax1.plot(w_obj.lgt_ids, color='g', label="lgt_ID")
    ax2.plot(w_obj.procs, color='b', label="Proc")

    ax1.set_xlabel('Block Index')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.set_ylabel('lgt_ID', color='g')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.set_ylabel('Proc', color='b')
    plt.title(f'Monotonely increasing arrays of blocks for {args.FILE}')

    plt.tight_layout(pad=0.15)
    if args.save:
        plt.savefig( args.FILE.replace('.h5','-integrity-monotone.pdf'))


if args.plot_zero:
    plt.figure(2, figsize=[7,5])
    zeros_list = [
        np.where(w_obj.lgt_ids == 0),
        np.where(w_obj.block_treecode_num == 0),
        np.where(w_obj.level == 0),
        np.where(w_obj.procs == 0),
        np.where(np.sum(w_obj.blocks, axis=tuple(np.arange(1, w_obj.dim+1))) == 0),
        np.where(np.sum(w_obj.coords_origin, axis=1) == 0),
        np.where(np.sum(w_obj.coords_spacing, axis=1) == 0)
    ]
    array_names = ["lgt_IDs", "Treecode", "Level", "Procs", "Blocks", "Origin", "Spacing"]
    for i_p, i_zeros in enumerate(zeros_list[::-1]):
        if len(i_zeros[0]) > 0:
            plt.barh(y=i_p+0.5, width=1.1, left=i_zeros[0]-0.05)

    plt.xlabel('Block index')
    plt.xlim(0, w_obj.total_number_blocks)
    plt.ylabel('Array name')
    plt.ylim(0, len(zeros_list))
    plt.yticks(np.arange(len(zeros_list))+0.5, labels=array_names[::-1])
    plt.title(f'Zero values of blocks in different arrays for {args.FILE}')

    plt.tight_layout(pad=0.15)
    if args.save:
        plt.savefig( args.FILE.replace('.h5','-integrity-zeros.pdf'))



if args.plot_blocks_sum:
    plt.figure(3, figsize=[7,5])
    blocks_sum = np.sum(w_obj.blocks, axis=tuple(np.arange(1, w_obj.dim+1)))
    plt.plot(blocks_sum)

    plt.xlabel('Block index')
    plt.ylabel('Sum of block data entry')
    plt.title(f'Sum of all data values of each block for {args.FILE}')

    plt.tight_layout(pad=0.15)
    if args.save:
        plt.savefig( args.FILE.replace('.h5','-integrity-blocks-sum.pdf'))


if args.display:
    plt.show()