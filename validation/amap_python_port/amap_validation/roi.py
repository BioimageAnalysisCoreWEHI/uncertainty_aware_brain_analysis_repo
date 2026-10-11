"""Minimal ImageJ .roi reader (polygon/freehand/traced and composite ShapeRois) for the rater outlines."""
from __future__ import annotations

import struct

import numpy as np


def read_roi(path):
    """Return (list of (N,2) [col, row] vertex arrays in TIFF pixel coordinates, (left, top, right, bottom))."""
    with open(path, "rb") as f:
        d = f.read()
    if d[:4] != b"Iout":
        raise ValueError(f"{path}: not an ImageJ ROI")
    typ = d[6]
    top, left, bottom, right, n = struct.unpack(">hhhhH", d[8:18])
    shape_size = struct.unpack(">i", d[36:40])[0]
    bbox = (left, top, right, bottom)
    if typ in (0, 7, 8) and n > 2:
        xs = np.frombuffer(d[64:64 + 2 * n], ">i2").astype(float)
        ys = np.frombuffer(d[64 + 2 * n:64 + 4 * n], ">i2").astype(float)
        return [np.c_[left + xs, top + ys]], bbox
    if typ == 1 and shape_size > 0:
        arr = np.frombuffer(d[64:64 + 4 * shape_size], ">f4").astype(float).tolist()
        paths, cur, i = [], [], 0
        while i < len(arr):
            seg = int(arr[i])
            i += 1
            if seg == 0:  # moveTo
                if len(cur) > 1:
                    paths.append(cur)
                cur = [(arr[i], arr[i + 1])]
                i += 2
            elif seg == 1:  # lineTo
                cur.append((arr[i], arr[i + 1]))
                i += 2
            elif seg == 2:  # quadTo (keep end point)
                cur.append((arr[i + 2], arr[i + 3]))
                i += 4
            elif seg == 3:  # cubicTo (keep end point)
                cur.append((arr[i + 4], arr[i + 5]))
                i += 6
            elif seg == 4:  # close
                if len(cur) > 1:
                    paths.append(cur)
                cur = []
            else:
                break
        if len(cur) > 1:
            paths.append(cur)
        paths = [np.array(p) for p in paths]
        allp = np.vstack(paths)
        absolute = allp[:, 0].min() >= left - 2 and allp[:, 0].max() <= right + 2 and allp[:, 1].min() >= top - 2
        if not absolute:
            paths = [p + [left, top] for p in paths]
        return paths, bbox
    return [], bbox
