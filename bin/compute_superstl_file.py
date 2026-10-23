#!/usr/bin/env python3
"""
Create a WABBIT-compatible ASCII superSTL (.superstl) from an STL file.

Each output row contains 30 scientific-notation values, in this exact order:
  1:9    triangle vertices (x1,y1,z1,x2,y2,z2,x3,y3,z3), after x/scale + x0
  10:12  facet normal as stored in the STL
  13:21  angle-weighted vertex pseudonormals at vertices 1, 2, 3
  22:30  edge pseudonormals on edges (1,2), (2,3), (3,1)

The pseudonormal conventions correspond to the definitions from

[J. A. Baerentzen, H. Aanaes, Signed distance computation using the
angle weighted pseudonormal, IEEE Trans. Vis. Comput. Graph. 11
(2005) 243–253. doi:10.1109/tvcg.2005.49.]

The normals in the STL file are unchanged - use external tools such as blender
or meshlab to ensure the normals are pointing towards the fluid. 

Usage:
  ./superstl.py --stl input.stl -o output.sstl --scale 100 --x0 2,3,4

--scale is WABBIT's divisor (coordinates are divided by it), not a multiplier.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict

import numpy as np

_EPS = 1.0e-30


def unit(v: np.ndarray) -> np.ndarray:
    """Return a unit vector, or zero if its magnitude is numerically zero."""
    v = np.asarray(v, dtype=np.float64)
    length = float(np.linalg.norm(v))
    if not np.isfinite(length) or length <= _EPS:
        return np.zeros(3, dtype=np.float64)
    return v / length


def read_stl(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read binary or ASCII STL, preserving triangle order and stored normals.

    Returns
    -------
    triangles : (ntri, 3, 3) float32 array
        Three vertices per triangle.
    facet_normals : (ntri, 3) float32 array
        Normals supplied in each STL facet record, like WABBIT's reader.
    """
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 84:
        is_binary = False
    else:
        ntri_header = struct.unpack_from("<I", data, 80)[0]
        is_binary = len(data) == 84 + 50 * ntri_header

    if is_binary:
        ntri = struct.unpack_from("<I", data, 80)[0]
        triangles = np.empty((ntri, 3, 3), dtype=np.float32)
        normals = np.empty((ntri, 3), dtype=np.float32)
        offset = 84
        for i in range(ntri):
            # Binary STL facet: normal (3f), vertices (9f), attribute byte count (H).
            values = struct.unpack_from("<12fH", data, offset)
            normals[i] = values[0:3]
            triangles[i] = np.asarray(values[3:12], dtype=np.float32).reshape(3, 3)
            offset += 50
        return triangles, normals

    # ASCII STL: parse each facet normal and its three vertex records. The
    # numeric expression handles decimals, signs, and scientific notation.
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is neither a valid binary nor UTF-8 ASCII STL") from exc

    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?"
    facet_re = re.compile(
        rf"facet\s+normal\s+({number})\s+({number})\s+({number})(.*?)endfacet",
        flags=re.IGNORECASE | re.DOTALL,
    )
    vertex_re = re.compile(
        rf"vertex\s+({number})\s+({number})\s+({number})",
        flags=re.IGNORECASE,
    )
    tris: list[list[list[float]]] = []
    norms: list[list[float]] = []
    for facet in facet_re.finditer(text):
        normal = [float(x.replace("D", "E").replace("d", "e"))
                  for x in facet.group(1, 2, 3)]
        vertices = [
            [float(x.replace("D", "E").replace("d", "e")) for x in match]
            for match in vertex_re.findall(facet.group(4))
        ]
        if len(vertices) != 3:
            raise ValueError(
                f"Malformed ASCII STL: expected 3 vertices per facet, found {len(vertices)}"
            )
        tris.append(vertices)
        norms.append(normal)

    if not tris:
        raise ValueError(f"Could not parse any STL facets in {path}")
    return np.asarray(tris, dtype=np.float32), np.asarray(norms, dtype=np.float32)


def corner_angle(tri: np.ndarray, corner: int) -> float:
    """Interior angle at one triangle corner in radians."""
    a = tri[(corner + 1) % 3].astype(np.float64) - tri[corner].astype(np.float64)
    b = tri[(corner + 2) % 3].astype(np.float64) - tri[corner].astype(np.float64)
    return float(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))


def coord_key(v: np.ndarray) -> tuple[float, float, float]:
    """Coordinate key matching WABBIT's exact shared-point comparisons."""
    # Input and normalized point arrays are float32, so duplicate STL vertices
    # represented by the same stored coordinates form identical keys.
    return tuple(float(x) for x in v)  # type: ignore[return-value]


def compute_superstl(
    triangles: np.ndarray,
    facet_normals: np.ndarray,
) -> np.ndarray:
    """Build an (ntri, 30) WABBIT-format superSTL array.

    Facet normals in the output are preserved from the input STL. Unitized
    versions of those normals are used when accumulating pseudonormals, just
    as in WABBIT's Fortran implementation.
    """
    ntri = len(triangles)
    if triangles.shape != (ntri, 3, 3) or facet_normals.shape != (ntri, 3):
        raise ValueError("Invalid internal STL array shape")

    # Use unit face normals for pseudonormal sums, but preserve the unmodified
    # STL normals in output columns 10:12.
    unit_faces = np.asarray([unit(n) for n in facet_normals], dtype=np.float64)

    # Coordinate -> every triangle-corner occurrence. A corner occurrence is
    # (triangle index, local vertex index), not just an incident triangle,
    # matching the three separate incidence counts in WABBIT's vertexnormal.
    vertex_incidence: DefaultDict[tuple[float, float, float], list[tuple[int, int]]] = defaultdict(list)
    edge_incidence: DefaultDict[
        tuple[tuple[float, float, float], tuple[float, float, float]], list[int]
    ] = defaultdict(list)

    for fi, tri in enumerate(triangles):
        keys = [coord_key(v) for v in tri]
        for vi, key in enumerate(keys):
            vertex_incidence[key].append((fi, vi))
        for a, b in ((0, 1), (1, 2), (2, 0)):
            ka, kb = keys[a], keys[b]
            edge_key = (ka, kb) if ka <= kb else (kb, ka)
            edge_incidence[edge_key].append(fi)

    # Calculate one pseudonormal per distinct coordinate, then reuse it for
    # every triangle corner having that coordinate.
    vertex_normal_by_key: dict[tuple[float, float, float], np.ndarray] = {}
    for key, occurrences in vertex_incidence.items():
        if len(occurrences) == 1:
            # WABBIT special case: an isolated point belonging to one triangle
            # gets an in-plane, outward-pointing bisector rather than its face
            # normal. This is important for open surfaces / isolated triangles.
            fi, vi = occurrences[0]
            tri = triangles[fi].astype(np.float64)
            others = [tri[j] - tri[vi] for j in range(3) if j != vi]
            vertex_normal_by_key[key] = unit(-(others[0] + others[1]))
        else:
            accum = np.zeros(3, dtype=np.float64)
            for fi, vi in occurrences:
                accum += corner_angle(triangles[fi], vi) * unit_faces[fi]
            vertex_normal_by_key[key] = unit(accum)

    # Per-triangle edge normals. For a shared edge WABBIT uses the normalized
    # sum of the two unit incident face normals. At a boundary edge it uses a
    # normal in the triangle plane, perpendicular to the edge and pointing
    # away from the triangle's third vertex.
    def edge_pseudonormal(fi: int, a: int, b: int) -> np.ndarray:
        tri = triangles[fi].astype(np.float64)
        p1, p2 = tri[a], tri[b]
        k1, k2 = coord_key(triangles[fi, a]), coord_key(triangles[fi, b])
        edge_key = (k1, k2) if k1 <= k2 else (k2, k1)

        other_face = next((j for j in edge_incidence[edge_key] if j != fi), None)
        if other_face is not None:
            return unit(unit_faces[fi] + unit_faces[other_face])

        # Boundary edge: choose the in-plane perpendicular pointing outward.
        n_tmp = unit(np.cross(p1 - p2, facet_normals[fi].astype(np.float64)))
        third = tri[next(j for j in range(3) if j not in (a, b))]
        if np.dot(third - p1, n_tmp) > 0.0:
            n_tmp = unit(np.cross(p2 - p1, facet_normals[fi].astype(np.float64)))
        return n_tmp

    # Layout exactly matches superstl(i, 1:30) in WABBIT.
    out = np.empty((ntri, 30), dtype=np.float64)
    for fi, tri in enumerate(triangles):
        out[fi, 0:9] = tri.astype(np.float64).reshape(9)
        out[fi, 9:12] = facet_normals[fi]

        for vi in range(3):
            out[fi, 12 + 3 * vi:15 + 3 * vi] = vertex_normal_by_key[
                coord_key(triangles[fi, vi])
            ]

        # Edge order is (vertex 1,2), (vertex 2,3), (vertex 3,1).
        for edge_slot, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
            out[fi, 21 + 3 * edge_slot:24 + 3 * edge_slot] = edge_pseudonormal(fi, a, b)

    return out


def parse_x0(values: list[str]) -> np.ndarray:
    """Accept --x0 'x,y,z' or --x0 x y z."""
    parts: list[str] = []
    for value in values:
        parts.extend(part for part in value.split(",") if part.strip())
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--x0 requires three values, e.g. --x0 2,3,4")
    try:
        xyz = np.asarray([float(v) for v in parts], dtype=np.float64)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--x0 values must be numbers") from exc
    if not np.all(np.isfinite(xyz)):
        raise argparse.ArgumentTypeError("--x0 values must be finite")
    return xyz


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a WABBIT-compatible ASCII superSTL (30 columns per triangle)."
    )
    parser.add_argument("--stl", required=True, help="input STL file (binary or ASCII)")
    parser.add_argument("-o", "--o", dest="output", default="dummy.sstl",
                        help="output ASCII superSTL file (default: dummy.sstl)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="WABBIT isometric scale divisor; coordinates are divided by this (default: 1)")
    parser.add_argument("--x0", nargs="+", default=["0,0,0"], metavar="X0",
                        help="origin shift as 'x,y,z' or as three values (default: 0,0,0)")
    args = parser.parse_args(argv)

    if not np.isfinite(args.scale) or args.scale == 0.0:
        parser.error("--scale must be a finite, non-zero number")
    try:
        x0 = parse_x0(args.x0)
        triangles, facet_normals = read_stl(args.stl)

        # WABBIT normalize_stl_file does: triangles = triangles / scale,
        # followed by triangles(component,:) += origin(component). Its point
        # storage is kind=4, so the transformed coordinates are rounded to f32.
        triangles = (triangles.astype(np.float64) / args.scale + x0).astype(np.float32)
        superstl = compute_superstl(triangles, facet_normals)

        output_path = Path(args.output)
        with output_path.open("w", encoding="ascii", newline="\n") as f:
            # Equivalent to Fortran format (30(es15.8,1x)).
            for row in superstl:
                f.write("".join(f"{value:15.8E} " for value in row).rstrip() + "\n")

    except (OSError, ValueError, struct.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    mins = triangles.reshape(-1, 3).min(axis=0)
    maxs = triangles.reshape(-1, 3).max(axis=0)
    print(f"Input STL:       {args.stl}")
    print(f"Output superSTL: {args.output}")
    print(f"Triangles:       {len(triangles)}")
    print(f"Scale divisor:   {args.scale:g}")
    print(f"Origin shift:    {x0[0]:g}, {x0[1]:g}, {x0[2]:g}")
    print(f"Bounds after normalization: min={mins}, max={maxs}")
    print("Wrote 30 columns per triangle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
