"""Minimal single-file NIfTI-1 (.nii / .nii.gz) reader/writer, enough for this pipeline (no nibabel dependency).

Arrays use NIfTI index order (dim1, dim2, dim3, ...), i.e. the same indexing as MATLAB's load_nii ``.img``.
``save_sparse`` / ``load_sparse`` stream one z-plane at a time so full brain grids (~1 G voxels) never have to be
held densely in memory.
"""
from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np

_DTYPES = {2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8", 256: "i1", 512: "u2", 768: "u4"}
_CODES = {np.dtype(v).str[1:]: k for k, v in _DTYPES.items()}


def _open(path, mode="rb", level=1):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode, compresslevel=level) if "w" in mode else gzip.open(path, mode)
    return open(path, mode)


def read_header(path) -> dict:
    with _open(path) as f:
        h = f.read(352)
    if len(h) < 348:
        raise ValueError(f"{path}: too short for a NIfTI-1 header")
    e = "<" if struct.unpack("<i", h[:4])[0] == 348 else ">"
    if struct.unpack(e + "i", h[:4])[0] != 348:
        raise ValueError(f"{path}: not a NIfTI-1 file")
    dim = struct.unpack(e + "8h", h[40:56])
    datatype = struct.unpack(e + "h", h[70:72])[0]
    if datatype not in _DTYPES:
        raise ValueError(f"{path}: unsupported datatype {datatype}")
    srow = np.array([struct.unpack(e + "4f", h[280 + 16 * i:296 + 16 * i]) for i in range(3)])
    return dict(
        endian=e,
        shape=tuple(int(d) for d in dim[1:1 + dim[0]]),
        dtype=np.dtype(e + _DTYPES[datatype]),
        pixdim=struct.unpack(e + "8f", h[76:108]),
        vox_offset=int(struct.unpack(e + "f", h[108:112])[0]),
        scl_slope=struct.unpack(e + "f", h[112:116])[0],
        scl_inter=struct.unpack(e + "f", h[116:120])[0],
        qform_code=struct.unpack(e + "h", h[252:254])[0],
        sform_code=struct.unpack(e + "h", h[254:256])[0],
        sform=np.vstack([srow, [0, 0, 0, 1]]),
    )


def _apply_scaling(data, hdr):
    slope = hdr["scl_slope"]
    if slope not in (0.0, 1.0) or hdr["scl_inter"] != 0.0:
        data = data * (slope if slope else 1.0) + hdr["scl_inter"]
    return data


def load(path, mmap: bool = False) -> np.ndarray:
    """Voxel data (unscaled unless scl_slope is set), shape = header dims, NIfTI/MATLAB index order."""
    hdr = read_header(path)
    shape = hdr["shape"]
    if mmap:
        if str(path).endswith(".gz"):
            raise ValueError("memory mapping is not possible for .nii.gz")
        data = np.memmap(path, dtype=hdr["dtype"], mode="r", offset=hdr["vox_offset"], shape=shape, order="F")
    else:
        n = int(np.prod(shape))
        with _open(path) as f:
            f.seek(hdr["vox_offset"])
            data = np.frombuffer(f.read(n * hdr["dtype"].itemsize), dtype=hdr["dtype"]).reshape(shape, order="F")
    return _apply_scaling(data, hdr)


def _header_bytes(shape, dtype: np.dtype, affine: np.ndarray, sform_code: int, qform_code: int) -> bytes:
    code = _CODES.get(np.dtype(dtype).str[1:])
    if code is None:
        raise ValueError(f"unsupported dtype {dtype}")
    if len(shape) > 7:
        raise ValueError("NIfTI supports at most 7 dimensions")
    dims = [len(shape)] + list(shape) + [1] * (7 - len(shape))
    zooms = np.linalg.norm(affine[:3, :3], axis=0)
    pixdim = [1.0] + list(zooms) + [1.0] * 4
    h = bytearray(352)
    struct.pack_into("<i", h, 0, 348)
    struct.pack_into("<8h", h, 40, *dims)
    struct.pack_into("<hh", h, 70, code, np.dtype(dtype).itemsize * 8)
    struct.pack_into("<8f", h, 76, *pixdim)
    struct.pack_into("<f", h, 108, 352.0)
    struct.pack_into("<f", h, 112, 0.0)
    struct.pack_into("<B", h, 123, 2)  # xyzt_units: mm
    struct.pack_into("<hh", h, 252, qform_code, sform_code)
    for i in range(3):
        struct.pack_into("<4f", h, 280 + 16 * i, *affine[i])
    h[344:348] = b"n+1\0"
    return bytes(h)


def save(path, data: np.ndarray, affine: np.ndarray, sform_code: int = 1, qform_code: int = 0):
    """Write little-endian NIfTI-1 (gzip if the name ends in .gz) with the given sform."""
    data = np.asarray(data)
    with _open(path, "wb") as f:
        f.write(_header_bytes(data.shape, data.dtype, affine, sform_code, qform_code))
        f.write(np.asfortranarray(data).astype(data.dtype.newbyteorder("<"), copy=False).tobytes(order="F"))


def save_sparse(path, shape, dtype, voxels, values, affine: np.ndarray, sform_code: int = 1,
                qform_code: int = 0, level: int = 1):
    """Write a 3-D volume given only its nonzero voxels, streaming one z-plane at a time.

    ``voxels`` (N, 3) are 0-based (x, y, z) indices; ``values`` (N,) the voxel values. Byte-identical to
    ``save`` of the equivalent dense array.
    """
    nx, ny, nz = shape
    dtype = np.dtype(dtype).newbyteorder("<")
    voxels = np.asarray(voxels, int).reshape(-1, 3)
    values = np.asarray(values).astype(dtype)
    if len(voxels) and (voxels.min() < 0 or np.any(voxels.max(0) >= np.array(shape))):
        raise ValueError("voxel index outside the volume")
    order = np.argsort(voxels[:, 2], kind="stable")
    voxels, values = voxels[order], values[order]
    bounds = np.searchsorted(voxels[:, 2], np.arange(nz + 1))
    zero = np.zeros((nx, ny), dtype).tobytes(order="F")
    with _open(path, "wb", level) as f:
        f.write(_header_bytes(shape, dtype, affine, sform_code, qform_code))
        for z in range(nz):
            a, b = bounds[z], bounds[z + 1]
            if a == b:
                f.write(zero)
                continue
            plane = np.zeros((nx, ny), dtype)
            plane[voxels[a:b, 0], voxels[a:b, 1]] = values[a:b]
            f.write(plane.tobytes(order="F"))


def load_sparse(path):
    """Nonzero voxels (N, 3) and values of a 3-D NIfTI, read one z-plane at a time."""
    hdr = read_header(path)
    nx, ny, nz = hdr["shape"][:3]
    item = hdr["dtype"].itemsize
    vox, vals = [], []
    with _open(path) as f:
        f.seek(hdr["vox_offset"])
        for z in range(nz):
            plane = np.frombuffer(f.read(nx * ny * item), dtype=hdr["dtype"]).reshape((nx, ny), order="F")
            if plane.any():
                x, y = np.nonzero(plane)
                vox.append(np.stack([x, y, np.full_like(x, z)], axis=1))
                vals.append(plane[x, y])
    if not vox:
        return np.zeros((0, 3), int), np.zeros(0, hdr["dtype"])
    return np.vstack(vox), _apply_scaling(np.concatenate(vals), hdr)
