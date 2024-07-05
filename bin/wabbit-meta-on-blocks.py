#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# this file creates a new .h5 file from a given file where the point values are a meta entry of the block
# this might seem unnecessary with the hypertreegrid file, but is useful to study meta entries on isosurfaces (especially the level)

import sys, os, argparse, glob
sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import wabbit_tools

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", default="./", type=str, help="Input of file or directory, defaults to ./")
parser.add_argument("-l", "--level", action="store_true", help="Write level file")
parser.add_argument("--treecode", action="store_true", help="Write treecode file")
parser.add_argument("-r", "--refinement-status", action="store_true", help="Write refinement_status file")
parser.add_argument("--coords-origin", action="store_true", help="Write coords_origin file")
parser.add_argument("--coords-spacing", action="store_true", help="Write coords_spacing file")
parser.add_argument("-p", "--procs", action="store_true", help="Write procs file")
parser.add_argument("--lgt-ids", action="store_true", help="Write lgt_ids file")
parser.add_argument("-v", "--verbose", action="store_true", help="Increased output to console")
args = parser.parse_args()

# check input
if os.path.isdir(args.input):
    h5_files = glob.glob(os.path.join(args.input, "*.h5"))
elif os.path.isfile(args.input) and args.input.endswith(".h5"):
    h5_files = [args.input]
else:
    print(f"ERROR: No suitable h5 files found at location {args.input}")
    sys.exit(1)

for i_file in h5_files:
    w_obj = wabbit_tools.WabbitHDF5file()
    w_obj.read(i_file, verbose=args.verbose)

    # prepare meta variables
    meta_check = []
    if args.level: meta_check.append(["level", w_obj.level])
    if args.treecode: meta_check.append(["treecode", w_obj.block_treecode_num])
    if args.refinement_status: meta_check.append(["refinement_status", w_obj.refinement_status])
    if args.coords_origin: meta_check.append(["coords_origin", w_obj.coords_origin])
    if args.coords_spacing: meta_check.append(["coords_spacing", w_obj.coords_spacing])
    if args.procs: meta_check.append(["procs", w_obj.procs])
    if args.lgt_ids: meta_check.append(["lgt_ids", w_obj.lgt_ids])

    # for every meta variable that is selected, replace the block values and then save the file
    for i_check in meta_check:
        for i_block in range(w_obj.total_number_blocks):
            i_meta = i_check[1][i_block]
            w_obj.blocks[i_block, :] = i_meta
        file_write = i_file.replace(w_obj.var_from_filename(), i_check[0])
        if not file_write.endswith(".h5"): file_write += ".h5"
        w_obj.write(file_write, verbose=args.verbose)