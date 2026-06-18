#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply arithmetic operations to WABBIT HDF5 files.

Examples:
    wabbit-operation.py add phi1.h5 phi2.h5 -o phi_sum.h5
    wabbit-operation.py divide phi1.h5 --value 2.0 -o phi_half.h5
"""

import argparse
import copy
import os
import shutil
import sys

import numpy as np

sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import wabbit_tools


def normalize_operation(operation):
    op_map = {
        "add": "add",
        "plus": "add",
        "+": "add",
        "subtract": "subtract",
        "sub": "subtract",
        "minus": "subtract",
        "-": "subtract",
        "multiply": "multiply",
        "mul": "multiply",
        "mult": "multiply",
        "times": "multiply",
        "*": "multiply",
        "divide": "divide",
        "div": "divide",
        "/": "divide",
        "min": "min",
        "max": "max",
        "avg": "avg",
        "average": "avg",
        "mean": "avg",
    }
    return op_map.get(operation.lower())


def apply_block_operation(block1, block2, operation):
    if operation == "add":
        return block1 + block2
    if operation == "subtract":
        return block1 - block2
    if operation == "multiply":
        return block1 * block2
    if operation == "divide":
        return block1 / block2
    if operation == "min":
        return np.minimum(block1, block2)
    if operation == "max":
        return np.maximum(block1, block2)
    if operation == "avg":
        return 0.5 * (block1 + block2)
    raise ValueError(f"Unknown operation: {operation}")


def apply_operation_objects(lhs, rhs, operation):
    new_obj = copy.deepcopy(lhs)
    equal_grid = lhs.compareGrid(rhs)
    equal_attr = lhs.compareAttr(rhs)
    if not equal_attr:
        print("ERROR: Attributes are not equal.")
        return None

    grid_interpolator = None
    if not equal_grid:
        print("WARNING: Grids are not equal, operation interpolates non-consistent blocks. This might take a while.")
        grid_interpolator = rhs.create_interpolator()

    for i_blocks in range(new_obj.total_number_blocks):
        i_other = rhs.get_block_id(lhs.block_treecode_num[i_blocks], lhs.level[i_blocks])
        if i_other != -1:
            other_block = rhs.blocks[i_other, :]
        else:
            other_block = rhs.interpolate_block(
                new_obj.blocks[i_blocks, :],
                new_obj.coords_origin[i_blocks],
                new_obj.coords_spacing[i_blocks],
                grid_interpolator,
            )
        new_obj.blocks[i_blocks, :] = apply_block_operation(new_obj.blocks[i_blocks, :], other_block, operation)

    return new_obj


def apply_operation(lhs, rhs, operation):
    if isinstance(rhs, wabbit_tools.WabbitHDF5file):
        return apply_operation_objects(lhs, rhs, operation)
    if isinstance(rhs, (int, float, np.integer, np.floating)):
        new_obj = copy.deepcopy(lhs)
        new_obj.blocks[:] = apply_block_operation(new_obj.blocks[:], rhs, operation)
        return new_obj
    raise ValueError(f"Unsupported second operand type: {type(rhs)}")


def process_streaming(file1, file2, scalar_value, operation, output_file):
    lhs_meta = wabbit_tools.WabbitHDF5file()
    lhs_meta.read(file1, read_var="meta")

    rhs_meta = None
    if file2 is not None:
        rhs_meta = wabbit_tools.WabbitHDF5file()
        rhs_meta.read(file2, read_var="meta")
        if not lhs_meta.compareAttr(rhs_meta):
            print("ERROR: Streaming mode requires matching grid attributes.")
            return False
        if not lhs_meta.compareGrid(rhs_meta):
            print("ERROR: Streaming mode requires matching grid blocks. Use non-stream mode for interpolation-based operations.")
            return False

    shutil.copyfile(file1, output_file)
    out_meta = wabbit_tools.WabbitHDF5file()
    out_meta.read(output_file, read_var="meta")

    total = int(out_meta.total_number_blocks)
    for i_b in range(total):
        b1 = lhs_meta.block_read(i_b, file=file1)
        if file2 is not None:
            j_b = rhs_meta.get_block_id(lhs_meta.block_treecode_num[i_b], lhs_meta.level[i_b])
            if j_b == -1:
                print(f"ERROR: Could not find matching block for index {i_b} in second file.")
                return False
            b2 = rhs_meta.block_read(j_b, file=file2)
        else:
            b2 = scalar_value

        b_out = apply_block_operation(b1, b2, operation)
        out_meta.block_write(i_b, b_out, file=output_file)

        if (i_b + 1) % max(1, total // 20) == 0 or i_b + 1 == total:
            print(f"Streaming progress: {i_b + 1}/{total}")

    return True


def default_output_name(input_file, operation):
    directory, filename = os.path.split(input_file)
    stem, _ = os.path.splitext(filename)
    return os.path.join(directory, f"{stem}-{operation}.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply add/subtract/multiply/divide/min/max/avg on WABBIT files and write a new file."
    )
    parser.add_argument("OPERATION", type=str, help='Operation to apply: add, subtract, multiply, divide, min, max, avg (aliases supported).')
    parser.add_argument("FILE1", type=str, help="First input WABBIT file (.h5).")
    parser.add_argument("FILE2", type=str, nargs="?", default=None, help="Optional second input WABBIT file (.h5).")
    parser.add_argument("--value", type=float, default=None, help="Scalar value used as second operand when FILE2 is not provided.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output WABBIT file. Defaults to <FILE1>-<operation>.h5")
    parser.add_argument("--stream", action="store_true", help="Use memory-efficient block-by-block streaming mode (requires matching grids for FILE2).")

    args = parser.parse_args()

    operation = normalize_operation(args.OPERATION)
    if operation is None:
        print(f"Unknown operation '{args.OPERATION}'.")
        print("Choose one of: add, subtract, multiply, divide, min, max, avg (aliases are accepted).")
        sys.exit(1)

    if not os.path.isfile(args.FILE1):
        print(f"FILE1 does not exist: {args.FILE1}")
        sys.exit(1)

    if args.FILE2 is not None and args.value is not None:
        print("Provide either FILE2 or --value, not both.")
        sys.exit(1)

    if args.FILE2 is None and args.value is None:
        print("Provide a second file (FILE2) or a scalar with --value.")
        sys.exit(1)

    if args.FILE2 is not None and not os.path.isfile(args.FILE2):
        print(f"FILE2 does not exist: {args.FILE2}")
        sys.exit(1)

    if operation == "divide" and args.value == 0:
        print("Division by zero is not allowed for scalar operation.")
        sys.exit(1)

    output_file = args.output if args.output is not None else default_output_name(args.FILE1, operation)

    if args.stream:
        ok = process_streaming(args.FILE1, args.FILE2, args.value, operation, output_file)
        if not ok:
            sys.exit(1)
        print(f"Saved result to: {output_file}")
        sys.exit(0)

    lhs = wabbit_tools.WabbitHDF5file()
    lhs.read(args.FILE1)

    rhs = args.value
    if args.FILE2 is not None:
        rhs = wabbit_tools.WabbitHDF5file()
        rhs.read(args.FILE2)

    result = apply_operation(lhs, rhs, operation)
    if result is None:
        print("Operation failed due to incompatible file attributes or grid metadata.")
        sys.exit(1)

    result.write(output_file)
    print(f"Saved result to: {output_file}")