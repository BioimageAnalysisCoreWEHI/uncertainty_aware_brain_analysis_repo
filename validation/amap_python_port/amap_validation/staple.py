"""Binary label fusion replacing NiftySeg ``seg_LabFusion -STAPLE -unc`` and ``-SBA`` (userRater.m:156-160).

STAPLE follows Warfield et al. 2004 for one foreground label. NiftySeg's internals (initialisation, prior,
convergence test, log format) are not in the repository; the choices below are documented defaults:

* ``unc=True`` ("only consider non-consensus voxels to calculate statistics"): voxels on which all raters
  agree are excluded from estimation and keep the unanimous label.
* prior P(truth=1): mean rater decision over the estimation voxels, held fixed.
* initial sensitivity/specificity 0.95; stop when max parameter change < ``eps`` (NiftySeg default 1e-5).
* hard output ("integer segmented image", no -outProb): posterior > 0.5.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass
class StapleResult:
    labels: np.ndarray          # uint8, same shape as a single input mask
    posterior: np.ndarray       # float32 P(truth=1); unanimous voxels get exactly 0 or 1 when unc=True
    sensitivity: np.ndarray     # p per rater
    specificity: np.ndarray     # q per rater
    prior: float
    iterations: int
    converged: bool


def _pattern_table(decisions: np.ndarray):
    """Unique decision vectors (n_patterns, R) with voxel counts and the per-voxel inverse index."""
    n_raters, n_vox = decisions.shape[0], decisions[0].size
    if n_raters > 62:
        raise ValueError("more than 62 raters not supported")
    key = np.zeros(n_vox, dtype=np.int64)
    for r in range(n_raters):
        key |= decisions[r].reshape(-1).astype(np.int64) << r
    uniq, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    bits = ((uniq[:, None] >> np.arange(n_raters)[None, :]) & 1).astype(float)
    return bits, counts.astype(float), inverse


def staple(masks, unc: bool = True, eps: float = 1e-5, max_iter: int = 10000,
           init: float = 0.95, prior: float | None = None) -> StapleResult:
    """STAPLE on a list/array of binary masks of identical shape (one per segmentation).

    ``prior`` fixes P(truth=1); by default it is the mean decision over the estimation voxels.
    """
    masks = np.asarray(masks)
    if masks.ndim < 3:
        raise ValueError("expected a stack of masks (n_raters, ...)")
    d = masks > 0
    shape = d.shape[1:]
    bits, counts, inverse = _pattern_table(d)
    n_raters = d.shape[0]

    unanimous = (bits.sum(1) == 0) | (bits.sum(1) == n_raters)
    est = ~unanimous if unc else np.ones(len(bits), bool)
    if not est.any():  # everybody agrees everywhere
        post_pat = bits[:, 0].copy()
        return StapleResult(post_pat[inverse].reshape(shape).astype(np.uint8),
                            post_pat[inverse].reshape(shape).astype(np.float32),
                            np.ones(n_raters), np.ones(n_raters), float("nan"), 0, True)

    D, c = bits[est], counts[est]
    if prior is None:
        prior = float((D.mean(1) * c).sum() / c.sum())
    p = np.full(n_raters, init)
    q = np.full(n_raters, init)
    tiny = 1e-12
    converged = False
    for it in range(1, max_iter + 1):
        lp1 = np.log(max(prior, tiny)) + D @ np.log(np.clip(p, tiny, 1)) + (1 - D) @ np.log(np.clip(1 - p, tiny, 1))
        lp0 = np.log(max(1 - prior, tiny)) + (1 - D) @ np.log(np.clip(q, tiny, 1)) + D @ np.log(np.clip(1 - q, tiny, 1))
        w = 1.0 / (1.0 + np.exp(np.clip(lp0 - lp1, -700, 700)))
        cw, cnw = c * w, c * (1 - w)
        p_new = (cw @ D) / max(cw.sum(), tiny)
        q_new = (cnw @ (1 - D)) / max(cnw.sum(), tiny)
        delta = max(np.abs(p_new - p).max(), np.abs(q_new - q).max())
        p, q = p_new, q_new
        if delta < eps:
            converged = True
            break

    post_pat = bits.mean(1).copy()  # unanimous patterns: exactly 0 or 1
    post_pat[est] = w
    labels = (post_pat > 0.5).astype(np.uint8)
    return StapleResult(labels[inverse].reshape(shape), post_pat[inverse].reshape(shape).astype(np.float32),
                        p, q, prior, it, converged)


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    """Signed Euclidean distance to the object boundary: negative inside, positive outside."""
    if not mask.any():
        return np.full(mask.shape, np.inf)
    if mask.all():
        return np.full(mask.shape, -np.inf)
    outside = ndi.distance_transform_edt(~mask)
    inside = ndi.distance_transform_edt(mask)
    return np.where(mask, -(inside - 0.5), outside - 0.5)


def sba(masks, margin: int = 64) -> np.ndarray:
    """Shape-based averaging (Rohlfing & Maurer 2007) for one label: mean signed distance < 0.

    Computed on the bounding box of the union plus a margin; outside it the mean distance is positive.
    """
    masks = np.asarray(masks) > 0
    union = masks.any(0)
    out = np.zeros(masks.shape[1:], np.uint8)
    if not union.any():
        return out
    lo = [max(0, int(np.min(a)) - margin) for a in np.nonzero(union)]
    hi = [min(s, int(np.max(a)) + margin + 1) for a, s in zip(np.nonzero(union), union.shape)]
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    acc = np.zeros([b - a for a, b in zip(lo, hi)])
    for m in masks:
        dist = _signed_distance(m[sl])
        acc += np.where(np.isfinite(dist), dist, np.sign(dist) * 1e6)
    out[sl] = (acc / len(masks) < 0).astype(np.uint8)
    return out


def valid_slices(slice_nums, mode: str = "", arg=None):
    """z-window selection of userRater.create4dNii (userRater.m:51-90)."""
    s = np.asarray(slice_nums, dtype=int)
    if not mode:
        return np.arange(s.min(), s.max() + 1)
    if mode == "range":
        rng = int(arg)
        freq = np.bincount(s, minlength=s.max() + 1)[1:]  # histcounts over 1..max
        if rng >= len(freq):
            return np.arange(s.min(), s.max() + 1)
        window = np.array([freq[j:j + rng].sum() for j in range(len(freq) - rng + 1)])
        start = int(np.argmax(window)) + 1
        return np.arange(start, start + rng)
    if mode == "quantiles":
        q = matlab_quantile(s, arg)
        return np.arange(int(np.floor(q[0])), int(np.floor(q[1])) + 1)
    raise ValueError("mode must be '', 'range' or 'quantiles'")


def matlab_quantile(x, probs):
    """MATLAB quantile(): sorted data at positions (i-0.5)/n, linear interpolation, clamped."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    out = []
    for pr in np.atleast_1d(probs):
        pos = n * pr + 0.5  # 1-based
        if pos < 1:
            out.append(x[0])
        elif pos >= n:
            out.append(x[-1])
        else:
            lo = int(np.floor(pos))
            out.append(x[lo - 1] + (pos - lo) * (x[lo] - x[lo - 1]))
    return np.array(out)
