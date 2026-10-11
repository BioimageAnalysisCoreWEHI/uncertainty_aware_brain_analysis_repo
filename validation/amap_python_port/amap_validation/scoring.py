"""Dice / Hausdorff scoring as in helpers/diceScore.m, helpers/hausdorffFromImages.m, helpers/padToLarger.m."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def pad_to_larger(a: np.ndarray, b: np.ndarray):
    """Zero-pad two 2-D arrays to a common size, top-left aligned."""
    shape = (max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1]))
    ra = np.zeros(shape, dtype=float)
    rb = np.zeros(shape, dtype=float)
    ra[: a.shape[0], : a.shape[1]] = a
    rb[: b.shape[0], : b.shape[1]] = b
    return ra, rb


def dice(seg1, seg2) -> float:
    s1, s2 = np.asarray(seg1) > 0, np.asarray(seg2) > 0
    if s1.shape != s2.shape:
        raise ValueError("seg1 and seg2 must be of equal size")
    denom = s1.sum() + s2.sum()
    return float(2 * np.logical_and(s1, s2).sum() / denom) if denom else float("nan")


def bwperim8(mask) -> np.ndarray:
    """bwperim(BW, 8): foreground pixels with a background pixel in their 8-neighbourhood (zero border)."""
    m = np.asarray(mask) > 0
    return m & ~ndi.binary_erosion(m, structure=np.ones((3, 3), bool), border_value=0)


def hausdorff(seg1, seg2) -> float:
    """Symmetric Hausdorff distance (pixels) between 8-connected perimeters; NaN if either is empty."""
    s1, s2 = np.asarray(seg1) > 0, np.asarray(seg2) > 0
    if s1.shape != s2.shape:
        raise ValueError("seg1 and seg2 must be of equal size")
    p1 = np.argwhere(bwperim8(s1))
    p2 = np.argwhere(bwperim8(s2))
    if len(p1) == 0 or len(p2) == 0:
        return float("nan")
    d12 = cKDTree(p2).query(p1)[0].max()
    d21 = cKDTree(p1).query(p2)[0].max()
    return float(max(d12, d21))
