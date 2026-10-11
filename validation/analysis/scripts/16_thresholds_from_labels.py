"""Global-QC thresholds from blinded ratings (CSV exported by qc_viewer/blind.html).

For each rater (and for the registrations on which all raters agree):
  suggested thresholds = the (IoU, grad_p95) pair that best separates rated-pass from rated-fail registrations
  (maximum balanced accuracy = mean of the fraction of rated-pass kept and rated-fail removed; candidates are the
  rated registrations' own values; ties go to the most lenient IoU, then the most lenient grad_p95).
  "unsure" ratings are excluded. The same rule is previewed in blind.html.
Also reported: the same accuracy figures for the starting point (median -/+ 3 MAD, qc_start.py, computed on the
rated registrations' dataset if the full QC table is given with --qc) and Cohen's kappa between each pair of
raters on the registrations both rated (pass / fail / unsure, and pass vs fail only).

Usage: python 16_thresholds_from_labels.py ratings_A.csv [ratings_B.csv ...] [--qc qc_results.csv] [--out DIR]
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qc_start import start_thresholds

MIN_FAIL = 5  # below this many rated failures the suggested thresholds are flagged as unreliable


def suggest(iou, grad, is_pass):
    iou, grad, p = np.asarray(iou, float), np.asarray(grad, float), np.asarray(is_pass, bool)
    if p.all() or not p.any():
        return None
    ci = np.concatenate([[-np.inf], np.unique(iou)])
    cg = np.concatenate([np.unique(grad), [np.inf]])
    best = None
    for ti in ci:  # ascending: first maximum = most lenient IoU; for equal IoU the largest grad wins below
        ok_i = iou > ti
        for tg in cg:
            ok = ok_i & (grad < tg)
            ba = 0.5 * (ok[p].mean() + (~ok[~p]).mean())
            if best is None or ba > best[2] + 1e-12 or (abs(ba - best[2]) <= 1e-12 and ti == best[0] and tg > best[1]):
                best = (ti, tg, ba)
    return best


def evaluate(d, ti, tg):
    ok = (d.iou > ti) & (d.grad_p95 < tg)
    p = d.label == "pass"
    return dict(iou_thr=ti, grad_thr=tg, rated_pass_kept=float(ok[p].mean()),
                rated_fail_removed=float((~ok[~p]).mean()) if (~p).any() else np.nan)


def kappa(a, b):
    cats = sorted(set(a) | set(b))
    a, b = np.asarray(a), np.asarray(b)
    po = float((a == b).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def main(argv):
    files = [f for f in argv if f.endswith(".csv") and (argv.index(f) == 0 or argv[argv.index(f) - 1] != "--qc")]
    qc = pd.read_csv(argv[argv.index("--qc") + 1]) if "--qc" in argv else None
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else Path.cwd()
    lab = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    rows = []
    groups = [(r, g) for r, g in lab.groupby("rater")]
    if lab.rater.nunique() > 1:  # registrations rated by every rater, with identical pass/fail ratings
        w = lab.pivot_table(index="id", columns="rater", values="label", aggfunc="first").dropna()
        agree = w[w.nunique(axis=1) == 1].iloc[:, 0].rename("label")
        g = lab.drop_duplicates("id").set_index("id").drop(columns="label").join(agree, how="inner").reset_index()
        groups.append(("all raters agree", g))
    for name, g in groups:
        for key, d in g.groupby("key"):
            d = d[d.label.isin(["pass", "fail"])]
            base = dict(rater=name, key=key, n_rated=len(d), n_pass=int((d.label == "pass").sum()),
                        n_fail=int((d.label == "fail").sum()))
            s = suggest(d.iou, d.grad_p95, d.label == "pass")
            if s is not None:
                rows.append(dict(base, thresholds="suggested by ratings", balanced_accuracy=s[2],
                                 note="fewer than 5 rated failures: unreliable" if base["n_fail"] < MIN_FAIL else "",
                                 **evaluate(d, s[0], s[1])))
            if qc is not None:
                ti, tg = start_thresholds(qc.iou, qc.grad_p95)
                rows.append(dict(base, thresholds="starting point (median -/+ 3 MAD)", **evaluate(d, ti, tg)))
    res = pd.DataFrame(rows)
    kap = []
    for r1, r2 in itertools.combinations(sorted(lab.rater.unique()), 2):
        m = lab[lab.rater == r1][["id", "label"]].merge(lab[lab.rater == r2][["id", "label"]], on="id")
        pf = m[m.label_x.isin(["pass", "fail"]) & m.label_y.isin(["pass", "fail"])]
        kap.append(dict(rater_1=r1, rater_2=r2, n_common=len(m), kappa_3cat=kappa(m.label_x, m.label_y),
                        n_pass_fail=len(pf), kappa_pass_fail=kappa(pf.label_x, pf.label_y),
                        percent_agreement=float((m.label_x == m.label_y).mean()) if len(m) else np.nan))
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "qclabels_thresholds.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(res.round(4).to_string(index=False))
    if kap:
        pd.DataFrame(kap).to_csv(out / "qclabels_agreement.csv", index=False)
        print(pd.DataFrame(kap).round(3).to_string(index=False))
    print(f"written to {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
