"""Untuned starting point for the global-QC thresholds (tissue IoU, grad_p95).

Rule (no dataset-specific constants): on the raw-input parameter sweep,
  IoU start  = median(IoU) - 3 * MAD(IoU)
  grad start = median(grad_p95) + 3 * MAD(grad_p95), over the registrations passing the IoU start,
with MAD scaled by 1.4826 (consistent with the SD for normal data). k = 3 is the "very conservative" choice of
Leys et al. (2013, J Exp Soc Psychol 49:764), i.e. it removes only gross outliers. The median and MAD tolerate up
to 50 % failed registrations (Tukey's quartiles only ~25 %), which matters when a whole family of settings fails.
The same thresholds are applied to preprocessed input, so pass rates show whether preprocessing helps.

The starting point is meant to be adjusted by inspection (qc_viewer/index.html) or derived from blinded ratings
(qc_viewer/blind.html + 16_thresholds_from_labels.py).

CLI:  python qc_start.py <qc_results.csv> [--k 3]
      (pipeline format: columns iou, grad_p95; one row per registration of the raw sweep)
"""
from __future__ import annotations

import sys

import numpy as np

K = 3.0
MAD_SCALE = 1.4826


def robust(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    m = float(np.median(x))
    return m, MAD_SCALE * float(np.median(np.abs(x - m)))


def start_thresholds(iou, grad, k: float = K):
    """-> (IoU threshold, grad_p95 threshold); a registration passes if IoU > t_iou and grad_p95 < t_grad."""
    iou, grad = np.asarray(iou, float), np.asarray(grad, float)
    mi, si = robust(iou)
    t_iou = mi - k * si
    mg, sg = robust(grad[iou > t_iou])
    return round(t_iou, 4), round(mg + k * sg, 4)


if __name__ == "__main__":
    import pandas as pd

    if len(sys.argv) < 2:
        sys.exit(__doc__)
    k = float(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else K
    d = pd.read_csv(sys.argv[1])
    ti, tg = start_thresholds(d.iou, d.grad_p95, k)
    n = int(((d.iou > ti) & (d.grad_p95 < tg)).sum())
    print(f"starting point (median -/+ {k:g} MAD, {len(d)} registrations): IoU > {ti}, grad_p95 < {tg}; "
          f"{n} pass ({n / len(d):.0%})")
    print(f"\nqc:\n  iou_threshold: {ti}\n  grad_p95_threshold: {tg}")
