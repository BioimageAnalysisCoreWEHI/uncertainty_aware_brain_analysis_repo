"""Manual-segmentation masks as the MATLAB pipeline sees them.

Arrays follow MATLAB orientation: ``imread(tif)'`` gives shape (width, height), i.e. index [x, y] with
x = TIFF column and y = TIFF row (userRater.m:103-104, segAnalyser.m:153-154).

Two different downsamplings are used by the original code and both are emulated here:
* STAPLE input: ``imresize(uint8(img>0), 0.25, 'nearest')`` (userRater.m:106)
* scoring:      ``imresize(logical, 0.25)`` = bicubic with antialiasing (segAnalyser.m:156)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import tifffile


def _box(x):
    return ((-0.5 <= x) & (x < 0.5)).astype(float)


def _cubic(x):
    ax = np.abs(x)
    ax2, ax3 = ax ** 2, ax ** 3
    return ((1.5 * ax3 - 2.5 * ax2 + 1) * (ax <= 1)
            + (-0.5 * ax3 + 2.5 * ax2 - 4 * ax + 2) * ((1 < ax) & (ax <= 2)))


def matlab_output_length(in_length: int, scale: float) -> int:
    return int(math.ceil(scale * in_length))


def contributions(in_length: int, out_length: int, scale: float, method: str, antialiasing: bool):
    """Port of imresize's contributions(): weights (out, P) and 0-based input indices (out, P)."""
    kernel, width = (_box, 1.0) if method == "nearest" else (_cubic, 4.0)
    if scale < 1 and antialiasing:
        h = lambda x: scale * kernel(scale * x)  # noqa: E731
        width = width / scale
    else:
        h = kernel
    x = np.arange(1, out_length + 1, dtype=float)[:, None]
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - width / 2)
    P = int(math.ceil(width)) + 2
    idx = left + np.arange(P)[None, :]
    w = h(u - idx)
    w = w / w.sum(axis=1, keepdims=True)
    # aux = [1:n, n:-1:1]; indices = aux(mod(indices-1, 2n) + 1)   (mirror boundary)
    m = np.mod(idx - 1, 2 * in_length).astype(int)
    idx0 = np.where(m < in_length, m, 2 * in_length - 1 - m)
    keep = np.any(w != 0, axis=0)
    return w[:, keep], idx0[:, keep]


def nearest_indices(in_length: int, scale: float) -> np.ndarray:
    """0-based input index sampled for each output element by imresize(..., 'nearest')."""
    out = matlab_output_length(in_length, scale)
    w, idx = contributions(in_length, out, scale, "nearest", antialiasing=False)
    return idx[np.arange(out), np.argmax(w, axis=1)]


def _weight_matrix(in_length: int, scale: float) -> sp.csr_matrix:
    out = matlab_output_length(in_length, scale)
    w, idx = contributions(in_length, out, scale, "bicubic", antialiasing=True)
    rows = np.repeat(np.arange(out), w.shape[1])
    m = sp.coo_matrix((w.ravel(), (rows, idx.ravel())), shape=(out, in_length))
    return m.tocsr()  # duplicate (mirrored) indices are summed, as in imresizemex


def _round_uint8(a):
    return np.clip(np.floor(a + 0.5), 0, 255)


def imresize_logical_bicubic(mask_xy: np.ndarray, scale: float) -> np.ndarray:
    """imresize(logical, scale) with defaults: uint8(255*A), bicubic+antialias per dim, round, then >128."""
    wx = _weight_matrix(mask_xy.shape[0], scale)
    wy = _weight_matrix(mask_xy.shape[1], scale)
    a = sp.csr_matrix(mask_xy.astype(np.float64) * 255.0)
    b1 = (wx @ a).tocsr()  # dimension order: sort(scale) -> dim 1 first for equal scales
    b1.data = _round_uint8(b1.data)
    b1.eliminate_zeros()
    b2 = (b1 @ wy.T).toarray()
    return _round_uint8(b2) > 128


def imresize_nearest(mask_xy: np.ndarray, scale: float) -> np.ndarray:
    ix = nearest_indices(mask_xy.shape[0], scale)
    iy = nearest_indices(mask_xy.shape[1], scale)
    return mask_xy[np.ix_(ix, iy)]


def canvas_size(paths) -> tuple[int, int]:
    """(maxX, maxY) = max TIFF width/height over all segmentations (userRater.m:27-31)."""
    max_x = max_y = 0
    for p in paths:
        with tifffile.TiffFile(p) as tf:
            h, w = tf.pages[0].shape[:2]
        max_x, max_y = max(max_x, w), max(max_y, h)
    return max_x, max_y


def read_mask_xy(path, canvas_xy: tuple[int, int]) -> np.ndarray:
    """Binary mask in MATLAB [x, y] orientation, top-left padded to the canvas (resizeSegImgsToMax)."""
    img = tifffile.imread(path)
    if img.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D mask, got {img.shape}")
    h, w = img.shape
    if w > canvas_xy[0] or h > canvas_xy[1]:
        raise ValueError(f"{path}: {w}x{h} exceeds canvas {canvas_xy}")
    out = np.zeros(canvas_xy, dtype=bool)
    out[:w, :h] = (img > 0).T
    return out


def downsampled_masks(path, canvas_xy: tuple[int, int], scale: float):
    """(staple_input uint8 [x,y], scoring_mask bool [x,y]) for one segmentation TIFF."""
    m = read_mask_xy(path, canvas_xy)
    return imresize_nearest(m, scale).astype(np.uint8), imresize_logical_bicubic(m, scale)


def _cache_file(cache_dir: Path, tif_path: str) -> Path:
    p = Path(tif_path)
    return cache_dir / p.parent.name / (p.name[:-4] + ".npz")


def cached_masks(tif_path: str, canvas_xy, scale: float, cache_dir: Path):
    """Downsampled masks, cached as bit-packed .npz next to the outputs (never in the repo)."""
    f = _cache_file(cache_dir, tif_path)
    small = (matlab_output_length(canvas_xy[0], scale), matlab_output_length(canvas_xy[1], scale))
    if f.exists():
        z = np.load(f)
        if tuple(z["shape"]) == small and float(z["scale"]) == scale:
            n = small[0] * small[1]
            near = np.unpackbits(z["nearest"])[:n].reshape(small).astype(np.uint8)
            bic = np.unpackbits(z["bicubic"])[:n].reshape(small).astype(bool)
            return near, bic
    near, bic = downsampled_masks(tif_path, canvas_xy, scale)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, nearest=np.packbits(near.astype(bool)), bicubic=np.packbits(bic),
                        shape=np.array(small), scale=np.array(scale))
    return near, bic
