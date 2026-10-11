"""Mapping between the STAPLE canvas / rater-stack plane and the downscaled brains (downscaledBrains/*.nii).

Derived from the forward path the MATLAB code uses to compare aMAP with STAPLE (pavelAtlas.makeAtlasSegStack,
pavelAtlas.m:52-104, called from segAnalyser.m:178-184):

* reference grid (rater-stack frame at the analysis scale): 1-based index (i1, J, K1), world (mm) =
  (i1 * px, J * 0.015, K1 * px) with px = 0.001/scale (save_nii_hdr.m:15-28 sform convention)
* floating slab = downscaled brain restricted to zRange along nii dim 2; world = 1-based slab index * 0.0125
* reg_resample maps reference world -> floating world: f = M r (NiftyReg affine convention)
* output permuted [1 3 2] and flipped along dim 2, then compared top-left with the STAPLE canvas, so
  STAPLE pixel (i1, j1) corresponds to reference (I=i1, K=Hs+1-j1) with Hs = round(H*scale)
* rater slice number s corresponds to reference plane J = s (inferred; the code takes the max Dice over J)

STAPLE indices here are 0-based (i0, j0) into the MATLAB-oriented [x, y] array; nii indices are 0-based
(x, y, z) into the NIfTI array as stored (dim1, dim2, dim3).
"""
from __future__ import annotations

import numpy as np

from .config import StackGeometry

NII_VOXEL_MM = 0.0125
STACK_STEP_MM = 0.015


def matlab_round(x):
    return np.floor(np.asarray(x, dtype=float) + 0.5)


class StackMapping:
    def __init__(self, geom: StackGeometry, scale: float = 0.25):
        self.geom = geom
        self.scale = scale
        self.px_mm = 0.001 / scale
        self.hs = int(matlab_round(geom.height * scale))
        self.ws = int(matlab_round(geom.width * scale))
        self.rot = geom.matrix[:3, :3]
        self.trans = geom.matrix[:3, 3]
        self.rot_inv = np.linalg.inv(self.rot)
        self.z0 = geom.z_range[0]

    # --- STAPLE canvas <-> reference (stack) world -------------------------------------------------
    def staple_to_ref_world(self, i0, j0, plane):
        i1 = np.asarray(i0, dtype=float) + 1
        j1 = np.asarray(j0, dtype=float) + 1
        i1, j1, plane = np.broadcast_arrays(i1, j1, np.asarray(plane, dtype=float))
        return np.stack([self.px_mm * i1, STACK_STEP_MM * plane, self.px_mm * (self.hs + 1 - j1)], axis=-1)

    def ref_world_to_staple(self, r):
        r = np.asarray(r, dtype=float)
        i1 = r[..., 0] / self.px_mm
        plane = r[..., 1] / STACK_STEP_MM
        j1 = self.hs + 1 - r[..., 2] / self.px_mm
        return i1 - 1, j1 - 1, plane

    # --- reference world <-> downscaled brain voxel -------------------------------------------------
    def ref_world_to_nii(self, r):
        f = np.asarray(r, dtype=float) @ self.rot.T + self.trans
        idx1 = f / NII_VOXEL_MM
        idx1[..., 1] += self.z0 - 1
        return idx1 - 1

    def nii_to_ref_world(self, v):
        idx1 = np.asarray(v, dtype=float) + 1
        idx1[..., 1] -= self.z0 - 1
        f = idx1 * NII_VOXEL_MM
        return (f - self.trans) @ self.rot_inv.T

    # --- composites ---------------------------------------------------------------------------------
    def staple_to_nii(self, i0, j0, plane):
        """0-based STAPLE pixel (continuous allowed) + 1-based stack plane -> 0-based nii (x, y, z)."""
        return self.ref_world_to_nii(self.staple_to_ref_world(i0, j0, plane))

    def nii_to_staple(self, v):
        """0-based nii (x, y, z) -> (i0, j0, plane) in STAPLE pixels / 1-based stack plane."""
        return self.ref_world_to_staple(self.nii_to_ref_world(v))

    # --- full-resolution rater pixels (TIFF / ROI coordinates, 1 um) -------------------------------------
    def tiff_to_staple(self, col, row):
        """TIFF pixel centre (0-based col/row) -> continuous STAPLE pixel, imresize centre convention."""
        f = 1.0 / self.scale
        return (np.asarray(col, float) + 0.5) / f - 0.5, (np.asarray(row, float) + 0.5) / f - 0.5

    def staple_to_tiff(self, i0, j0):
        f = 1.0 / self.scale
        return (np.asarray(i0, float) + 0.5) * f - 0.5, (np.asarray(j0, float) + 0.5) * f - 0.5

    def tiff_to_nii(self, col, row, slice_num):
        return self.staple_to_nii(*self.tiff_to_staple(col, row), slice_num)

    def nii_to_tiff(self, v):
        i0, j0, plane = self.nii_to_staple(v)
        c, r = self.staple_to_tiff(i0, j0)
        return c, r, plane


    # --- stack plane as a surface in the brain grid ----------------------------------------------------
    def _plane_coeffs(self):
        """J(x, y, z) = g . (x, y, z) + h for 0-based nii voxel coordinates (the map is affine)."""
        e = np.vstack([np.zeros(3), np.eye(3)])
        J = self.nii_to_staple(e)[2]
        return J[1:] - J[0], J[0]

    def plane_y(self, x, z, plane):
        """0-based nii y (continuous) where stack plane ``plane`` crosses voxel column (x, z)."""
        g, h = self._plane_coeffs()
        return (plane - h - g[0] * np.asarray(x, float) - g[2] * np.asarray(z, float)) / g[1]


def canvas_to_volume(canvas_xy: np.ndarray, mapping: StackMapping, plane: float, nii_shape, margin: int = 3):
    """Place a 2-D canvas image (MATLAB [x, y], 4 um) on stack plane ``plane`` of the brain grid.

    One voxel per (x, z) column is chosen: the one whose centre is nearest the (slightly tilted) plane, i.e.
    y = round(plane_y(x, z)). That voxel takes the canvas value at the nearest pixel to its centre
    (nearest neighbour, like reg_resample -inter 0). Only columns covering the canvas' nonzero bounding box
    (plus ``margin`` voxels) are evaluated.
    Returns (voxels (N, 3) int, values (N,)) for the nonzero voxels.
    """
    nx, ny, nz = nii_shape
    nzp = np.argwhere(canvas_xy != 0)
    if len(nzp) == 0:
        return np.zeros((0, 3), int), np.zeros(0, canvas_xy.dtype)
    lo, hi = nzp.min(0), nzp.max(0)
    corners = np.array([[a, b] for a in (lo[0] - 1, hi[0] + 1) for b in (lo[1] - 1, hi[1] + 1)], float)
    v = mapping.staple_to_nii(corners[:, 0], corners[:, 1], plane)
    x0 = max(0, int(np.floor(v[:, 0].min())) - margin)
    x1 = min(nx - 1, int(np.ceil(v[:, 0].max())) + margin)
    z0 = max(0, int(np.floor(v[:, 2].min())) - margin)
    z1 = min(nz - 1, int(np.ceil(v[:, 2].max())) + margin)
    if x0 > x1 or z0 > z1:
        return np.zeros((0, 3), int), np.zeros(0, canvas_xy.dtype)
    xs, zs = np.meshgrid(np.arange(x0, x1 + 1), np.arange(z0, z1 + 1), indexing="ij")
    ys = matlab_round(mapping.plane_y(xs, zs, plane)).astype(int)
    vox = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)
    vox = vox[(vox[:, 1] >= 0) & (vox[:, 1] < ny)]
    i0, j0, _ = mapping.nii_to_staple(vox.astype(float))
    ii, jj = matlab_round(i0).astype(int), matlab_round(j0).astype(int)
    ok = (ii >= 0) & (ii < canvas_xy.shape[0]) & (jj >= 0) & (jj < canvas_xy.shape[1])
    vals = np.zeros(len(vox), canvas_xy.dtype)
    vals[ok] = canvas_xy[ii[ok], jj[ok]]
    keep = vals != 0
    return vox[keep], vals[keep]


def stack_plane_from_volume(voxels, values, mapping: StackMapping, plane: float, canvas_shape):
    """Emulate the code's forward path for one stack plane: brain grid -> 4 um canvas [x, y].

    pavelAtlas.makeAtlasSegStack resamples the brain-space image into the reference (stack) grid with
    reg_resample -inter 0 (nearest), then permute([1 3 2]) + flip(.,2); segAnalyser pads it top-left against the
    STAPLE canvas. So canvas pixel (i0, j0) takes the voxel nearest to staple_to_nii(i0, j0, plane), for pixels inside
    the Ws x Hs reference grid, and 0 elsewhere. ``voxels``/``values`` describe a sparse brain-space image.
    """
    out = np.zeros(canvas_shape, dtype=np.asarray(values).dtype)
    voxels = np.asarray(voxels, int)
    if len(voxels) == 0:
        return out
    lo, hi = voxels.min(0) - 1, voxels.max(0) + 1
    box = np.zeros(hi - lo + 1, dtype=out.dtype)
    box[tuple((voxels - lo).T)] = values
    # candidate canvas pixels: map the voxel box corners back to the canvas
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)
    i0, j0, _ = mapping.nii_to_staple(corners)
    ia = max(0, int(np.floor(i0.min())) - 2)
    ib = min(min(canvas_shape[0], mapping.ws) - 1, int(np.ceil(i0.max())) + 2)
    ja = max(0, int(np.floor(j0.min())) - 2)
    jb = min(min(canvas_shape[1], mapping.hs) - 1, int(np.ceil(j0.max())) + 2)
    if ia > ib or ja > jb:
        return out
    ii, jj = np.meshgrid(np.arange(ia, ib + 1), np.arange(ja, jb + 1), indexing="ij")
    v = matlab_round(mapping.staple_to_nii(ii, jj, plane)).astype(int) - lo
    inside = np.all((v >= 0) & (v < np.array(box.shape)), axis=-1)
    out[ii[inside], jj[inside]] = box[tuple(v[inside].T)]
    return out
