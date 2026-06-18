#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wabbit-interpolate-data.py

Interpolate a WABBIT HDF5 data field onto a (possibly different) WABBIT grid.

The two input files may differ in block-size, refinement level distribution, and
domain size.  For target grid points that fall outside the source domain the
output is set to a user-supplied fill value (default: 0.0).

Usage
-----
    wabbit-interpolate-data.py DATA_FILE GRID_FILE [options]

Positional arguments
--------------------
DATA_FILE   HDF5 file that contains the field values to be interpolated.
GRID_FILE   HDF5 file whose block/treecode structure defines the output grid.

Optional arguments
------------------
-o / --output      Output filename.  Default: VAR-interpolated_TIME.h5
-f / --fill-value  Value written for points outside the source domain. Default: 0.0
-s / --shift       Origin shift vector (grid → data). Default: 0 ... 0
-j / --jobs        Number of parallel worker processes (joblib). Omit for sequential mode.

Notes
-----
- Time, iteration, block-size and treecode structure are taken from GRID_FILE.
- The script hard-fails if DATA_FILE and GRID_FILE have different spatial dimensions.
- Interpolation is bilinear (2-D) / trilinear (3-D) via
  scipy.interpolate.RegularGridInterpolator.
- PERFORMANCE WARNING: The interpolation is a pure-Python point-wise algorithm and
  can be very slow for large 3-D grids.
"""

import sys
import os
import time
import argparse
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# make sure the parent package (wabbit_tools) is importable when the script is
# called directly from the repository
sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import wabbit_tools

# ──────────────────────────────────────────────────────────────────────────────

def _build_one_interpolator(i_block, origin, end, block_data, method):
    """
    Construct a single RegularGridInterpolator.  Module-level for picklability.
    Returns (i_block, interpolator) so results can be reassembled in order.
    """
    x_coords = [
        np.linspace(origin[d], end[d], block_data.shape[d])
        for d in range(len(origin))
    ]
    interp = RegularGridInterpolator(
        x_coords,
        block_data,
        method=method,
        bounds_error=False,
        fill_value=np.nan,
    )
    return i_block, interp


def build_interpolators(data_obj, fill_value, method="linear", jobs=None, verbose=False):
    """
    Build one RegularGridInterpolator per block of *data_obj*.

    Points outside a block's bounding box return NaN (not fill_value) so that
    we can safely hand the evaluation to the correct block later.  fill_value
    is applied in the main loop after all blocks have been tried.

    Parameters
    ----------
    jobs : int or None
        If set, build interpolators in parallel using joblib (must already be
        imported by the caller).  None → sequential.

    Returns
    -------
    interpolators : list[RegularGridInterpolator]  (ordered by block index)
    data_end      : ndarray, shape (N_src, dim)
                    Physical end-coordinates of each source block.
    """
    Bs_data  = np.array(data_obj.blocks.shape[1:])           # (Bz, By, Bx) Python size
    data_end = data_obj.coords_origin + data_obj.coords_spacing * (Bs_data - 1)

    origins = data_obj.coords_origin   # (N, dim)
    ends    = data_end                 # (N, dim)

    if jobs is not None:
        from joblib import Parallel, delayed
        if verbose:
            print(f"  Building per-block interpolators in parallel with {jobs} workers for {data_obj.total_number_blocks} blocks …")
        raw = Parallel(n_jobs=jobs, verbose=10 if verbose else 0)(
            delayed(_build_one_interpolator)(
                i, origins[i], ends[i], data_obj.blocks[i], method
            )
            for i in range(data_obj.total_number_blocks)
        )
        # results may arrive out of order depending on scheduler
        interpolators = [None] * data_obj.total_number_blocks
        for i_block, interp in raw:
            interpolators[i_block] = interp
    else:
        interpolators = []
        for i_block in range(data_obj.total_number_blocks):
            _, interp = _build_one_interpolator(
                i_block, origins[i_block], ends[i_block],
                data_obj.blocks[i_block], method
            )
            interpolators.append(interp)

    return interpolators, data_end


def _process_block(i_tgt, origin_tgt, spacing_tgt, dim, Bs_grid,
                   data_origins, data_end_arr, interpolators, fill_value, shift):
    """
    Interpolate a single target block.  Module-level so joblib can pickle it.

    Returns (i_tgt, result_flat, n_fill_block).
    """
    axes = [
        origin_tgt[d] + np.arange(Bs_grid[d]) * spacing_tgt[d]
        for d in range(dim)
    ]

    if dim == 2:
        gz, gy = np.meshgrid(axes[0], axes[1], indexing="ij")
        query_pts = np.stack([gz.ravel(), gy.ravel()], axis=1)
    else:
        gz, gy, gx = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
        query_pts = np.stack([gz.ravel(), gy.ravel(), gx.ravel()], axis=1)

    query_pts = query_pts + shift

    N_pts = query_pts.shape[0]
    result = np.full(N_pts, np.nan)

    eps = 1e-12
    for i_src in range(len(interpolators)):
        lo = data_origins[i_src]
        hi = data_end_arr[i_src]

        in_src = np.ones(N_pts, dtype=bool)
        for d in range(dim):
            in_src &= (query_pts[:, d] >= lo[d] - eps)
            in_src &= (query_pts[:, d] <= hi[d] + eps)

        if not np.any(in_src):
            continue

        vals = interpolators[i_src](query_pts[in_src])
        valid = ~np.isnan(vals)
        idx   = np.where(in_src)[0][valid]
        result[idx] = vals[valid]

    still_nan = np.isnan(result)
    n_fill_block = int(np.sum(still_nan))
    result[still_nan] = fill_value

    return i_tgt, result, n_fill_block


def interpolate_onto_grid(data_obj, grid_obj, fill_value, verbose=False, method="linear", shift=None, jobs=None):
    """
    For every block in *grid_obj* evaluate the field from *data_obj* at all
    interior grid points.

    Parameters
    ----------
    shift : array-like, shape (dim,), optional
        Translation vector from the grid origin to the data origin, in the
        same coordinate order as coords_origin (Z, Y[, X]).
        A grid query point at position p is looked up at data position p + shift.
        Default: zero vector (origins coincide).

    Returns
    -------
    new_blocks : ndarray  –  same shape as grid_obj.blocks, filled with
                             interpolated values (or fill_value where no source
                             data exists).
    n_fill     : int      –  total number of points that were set to fill_value.
    jobs       : int or None
        Number of parallel worker processes (passed to joblib.Parallel n_jobs).
        None (default) → sequential execution, no joblib dependency required.
    """
    if shift is None:
        shift = np.zeros(grid_obj.dim)
    shift = np.asarray(shift, dtype=float)

    if verbose:
        print(f"  Building per-block interpolators for source file (method={method}) …")
        print(f"  Origin shift (grid → data): {shift}")

    interpolators, data_end = build_interpolators(data_obj, fill_value, method=method, jobs=jobs, verbose=verbose)

    # Allocate output array, pre-filled with fill_value.
    new_blocks = np.full(grid_obj.blocks.shape, fill_value, dtype=grid_obj.blocks.dtype)

    Bs_grid = np.array(grid_obj.blocks.shape[1:])   # (Bz, By, Bx) Python size
    n_fill = 0
    n_total = 0

    # source block bounding boxes (already computed in build_interpolators, but
    # we need them again for the fast pre-filter below)
    Bs_data = np.array(data_obj.blocks.shape[1:])
    data_end_arr = data_obj.coords_origin + data_obj.coords_spacing * (Bs_data - 1)

    # ── Common arguments for _process_block ───────────────────────────────────
    common = dict(
        dim          = grid_obj.dim,
        Bs_grid      = Bs_grid,
        data_origins = data_obj.coords_origin,
        data_end_arr = data_end_arr,
        interpolators= interpolators,
        fill_value   = fill_value,
        shift        = shift,
    )

    if jobs is not None:
        # ── Parallel branch ───────────────────────────────────────────────────
        try:
            from joblib import Parallel, delayed
        except ImportError:
            print("ERROR: joblib is required for parallel execution (-j/--jobs) but is not installed.")
            print("       Install it with:  pip install joblib")
            print("       Or run without -j to use sequential mode.")
            sys.exit(1)

        if verbose:
            print(f"  Interpolating in parallel with {jobs} workers for {grid_obj.total_number_blocks} blocks …")

        results = Parallel(n_jobs=jobs, verbose=10 if verbose else 0)(
            delayed(_process_block)(
                i_tgt,
                grid_obj.coords_origin[i_tgt],
                grid_obj.coords_spacing[i_tgt],
                **common,
            )
            for i_tgt in range(grid_obj.total_number_blocks)
        )

        for i_tgt, result_flat, n_fill_block in results:
            new_blocks[i_tgt, :] = result_flat.reshape(Bs_grid)
            n_fill  += n_fill_block
            n_total += int(result_flat.size)

    else:
        # ── Sequential branch ──────────────────────────────────────────────────
        start_time = time.time()
        total = grid_obj.total_number_blocks
        for i_tgt in range(total):
            # ETA: same formula as hdf2vtkhdf
            elapsed  = time.time() - start_time
            rem_time = (total - i_tgt) * elapsed / (i_tgt + 1e-4 * (i_tgt == 0))
            hours,   rem     = divmod(rem_time, 3600)
            minutes, seconds = divmod(rem, 60)

            if verbose and i_tgt % 50 == 0:
                print(
                    f"  Block {i_tgt+1:>{len(str(total))}}/{total}  "
                    f"({100*(i_tgt+1)//total:3d}%)  "
                    f"ETA: {int(hours)}h {int(minutes):02d}m {seconds:04.1f}s",
                    flush=True,
                )

            origin_tgt  = grid_obj.coords_origin[i_tgt]
            spacing_tgt = grid_obj.coords_spacing[i_tgt]

            i_tgt_out, result_flat, n_fill_block = _process_block(
                i_tgt, origin_tgt, spacing_tgt, **common
            )

            new_blocks[i_tgt, :] = result_flat.reshape(Bs_grid)
            n_fill  += n_fill_block
            n_total += int(result_flat.size)

    if verbose:
        print(f"  Done.  Total points: {n_total}, fill-value points: {n_fill}")

    return new_blocks, n_fill


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate a WABBIT HDF5 data field (DATA_FILE) onto the grid "
            "defined by GRID_FILE.  The two files may differ in block-size, "
            "refinement level, and domain size.  Points outside the source "
            "domain are set to FILL_VALUE."
        )
    )
    parser.add_argument(
        "DATA_FILE",
        type=str,
        help="HDF5 file containing the source field values.",
    )
    parser.add_argument(
        "GRID_FILE",
        type=str,
        help="HDF5 file whose block/treecode structure defines the output grid.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help=(
            "Output filename.  "
            "Default: VAR-interpolated_TIME.h5  (inserts '-interpolated' "
            "before the last underscore, or before the extension if none)."
        ),
    )
    parser.add_argument(
        "-m", "--method",
        type=str,
        default="linear",
        dest="method",
        choices=["linear", "nearest", "slinear", "cubic", "quintic", "pchip"],
        help=(
            "Interpolation method passed to scipy RegularGridInterpolator.  "
            "Choices: linear (default), nearest, slinear, cubic, quintic, pchip.  "
            "Higher-order methods (cubic, quintic) are more accurate but slower "
            "and may overshoot; pchip is monotone cubic with no overshoots."
        ),
    )
    parser.add_argument(
        "-f", "--fill-value",
        type=float,
        default=0.0,
        dest="fill_value",
        help=(
            "Value assigned to output points that lie outside the source domain.  "
            "Default: 0.0"
        ),
    )
    parser.add_argument(
        "-s", "--shift",
        type=float,
        nargs="+",
        default=None,
        dest="shift",
        metavar="VAL",
        help=(
            "Origin shift vector from the grid origin to the data origin, given as "
            "space-separated floats in Z Y [X] order (matching coords_origin storage order).  "
            "A grid query point p is looked up at data position p + shift.  "
            "Example: --shift 0.5 0.0 for a 2-D case where the data starts 0.5 units "
            "above the grid.  Default: all zeros (origins coincide)."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Print detailed progress information.",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=None,
        dest="jobs",
        metavar="N",
        help=(
            "Number of parallel worker processes (requires joblib).  "
            "Use -1 to use all available cores.  "
            "Omit (default) to run sequentially without any joblib dependency."
        ),
    )

    args = parser.parse_args()

    # ── Derive default output filename ────────────────────────────────────────
    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.DATA_FILE))[0]
        # Insert '-interpolated' before the last underscore (VAR_TIME → VAR-interpolated_TIME).
        # If there is no underscore, append it directly before the extension.
        last_us = stem.rfind("_")
        if last_us != -1:
            out_stem = stem[:last_us] + "-interpolated" + stem[last_us:]
        else:
            out_stem = stem + "-interpolated"
        args.output = os.path.join(
            os.path.dirname(args.DATA_FILE) or ".",
            out_stem + ".h5",
        )

    # ── Read input files ──────────────────────────────────────────────────────
    print(f"Reading data  file: {args.DATA_FILE}")
    data_obj = wabbit_tools.WabbitHDF5file()
    data_obj.read(args.DATA_FILE, verbose=args.verbose)

    print(f"Reading grid  file: {args.GRID_FILE}")
    grid_obj = wabbit_tools.WabbitHDF5file()
    grid_obj.read(args.GRID_FILE, verbose=args.verbose)

    # ── Validate ──────────────────────────────────────────────────────────────
    dim = grid_obj.dim  # both must be equal, checked below

    if data_obj.dim != grid_obj.dim:
        print(
            f"ERROR: Spatial dimension mismatch — "
            f"DATA_FILE has dim={data_obj.dim}, "
            f"GRID_FILE has dim={grid_obj.dim}.  Aborting."
        )
        sys.exit(1)

    # Parse / validate shift
    if args.shift is None:
        shift = np.zeros(dim)
    else:
        shift = np.asarray(args.shift, dtype=float)
        if len(shift) != dim:
            print(
                f"ERROR: --shift expects {dim} values for a {dim}-D field, "
                f"but {len(shift)} were given.  Aborting."
            )
            sys.exit(1)

    if not np.allclose(data_obj.domain_size[:data_obj.dim],
                       grid_obj.domain_size[:grid_obj.dim]):
        print(
            f"INFO: Domain sizes differ — "
            f"data domain = {data_obj.domain_size[:data_obj.dim]}, "
            f"grid domain = {grid_obj.domain_size[:grid_obj.dim]}.  "
            f"Points outside the source domain will be set to fill_value={args.fill_value}."
        )

    print(
        f"\nSource : {data_obj.total_number_blocks} blocks, "
        f"BS={data_obj.block_size[:data_obj.dim]}, "
        f"domain={data_obj.domain_size[:data_obj.dim]}"
    )
    print(
        f"Target : {grid_obj.total_number_blocks} blocks, "
        f"BS={grid_obj.block_size[:grid_obj.dim]}, "
        f"domain={grid_obj.domain_size[:grid_obj.dim]}"
    )
    print(f"Interpolation method       : {args.method}")
    print(f"Origin shift (grid → data) : {shift}")
    if args.jobs is not None:
        print(f"Parallel workers           : {args.jobs}")
    print(f"Fill value for out-of-domain points: {args.fill_value}")

    # Warn about potentially long runtime for large 3-D grids
    if grid_obj.dim == 3:
        print(
            "\nWARNING: Interpolation on large 3-D grids can take a very long time. Be patient and hold on..\n"
        )
    else: print()

    # ── Interpolate ───────────────────────────────────────────────────────────
    new_blocks, n_fill = interpolate_onto_grid(
        data_obj, grid_obj, args.fill_value, args.verbose,
        method=args.method, shift=shift, jobs=args.jobs
    )

    # ── Assemble output: grid structure + interpolated values ─────────────────
    # All metadata (time, iteration, treecodes, domain/block size) come from
    # grid_obj; only the block values are replaced.
    grid_obj.blocks = new_blocks

    # ── Write output ──────────────────────────────────────────────────────────
    print(f"\nWriting output: {args.output}")
    grid_obj.write(args.output, verbose=args.verbose)

    n_total = grid_obj.total_number_blocks * int(np.prod(grid_obj.blocks.shape[1:]))
    print(
        f"Done.  "
        f"{n_total - n_fill}/{n_total} points interpolated, "
        f"{n_fill} points set to fill_value={args.fill_value}."
    )


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
