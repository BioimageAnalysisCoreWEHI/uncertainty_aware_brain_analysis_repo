# -*- coding: utf-8 -*-
"""
17_signal_vs_agreement.py

Does agreement with the published registration depend on how much marker signal a region contains, beyond region
size? Independent LSFM, raw input, final QC set, threshold 0.5. Read-only on all inputs.

Per brain x region (6 x 16):
  * dice_3d                    consensus vs published region (results/fig8_region_masks)
  * volume_mm3                 published region volume
  * rel_mean                   PRIMARY brightness measure, fixed before looking at the result: mean image intensity in
                               the published region / median tissue intensity of that brain (tissue = voxels above the
                               image median; same normalisation as Fig. S6C)
  * frac_bright, rel_mean_nbhd secondary (reported, not used to choose anything): fraction of region voxels above the
                               brain's tissue 99th percentile; mean relative intensity in the region plus a 100 um shell
  * pairwise_agreement         (SNc only exists in midbrain_inspection.csv; not computed here)
Per brain: intensity histograms (relative to median tissue) of whole-brain tissue, CP, and the midbrain neighbourhood
(published SNc + VTA dilated by 200 um).

Statistics (descriptive; brain x region cases are not independent): Spearman rho of Dice with log volume and with log
brightness, rho between the two predictors, and an OLS fit of logit(Dice) on standardised log volume and log brightness
(with brain fixed effects) to show each predictor's association adjusted for the other.

Output: results/signal_vs_agreement/signal_vs_agreement.csv, histograms.csv, stats.csv
"""
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage, stats

from gt_utils import ROOT, SAMPLES, VOXEL_MM3, region_table, image_path, read_tiff

OUT = ROOT / "results" / "signal_vs_agreement"
MASKS = ROOT / "results" / "fig8_region_masks"
BINS = np.logspace(np.log10(0.2), np.log10(60), 121)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    regions = region_table().acronym.tolist()
    man = pd.read_csv(MASKS / "manifest.csv").set_index(["sample_id", "region"])
    rows, hist = [], []
    for s in SAMPLES:
        img = read_tiff(image_path(s)).astype(np.float32)
        tissue = img > np.median(img)
        med = float(np.median(img[tissue]))
        p99 = float(np.percentile(img[tissue], 99))
        rel = img / med
        pub = {r: tifffile.imread(MASKS / s / f"{r}__published_registration.tif") != 0 for r in regions}
        for r in regions:
            m = pub[r]
            nb = ndimage.binary_dilation(m, iterations=4)                      # + 100 um shell
            rows.append(dict(sample_id=s, region=r, dice_3d=man.loc[(s, r), "dice_3d"],
                             volume_mm3=m.sum() * VOXEL_MM3, rel_mean=float(rel[m].mean()),
                             frac_bright=float((img[m] >= p99).mean()), rel_mean_nbhd=float(rel[nb].mean()),
                             rel_p95=float(np.percentile(rel[m], 95))))
        mb = ndimage.binary_dilation(pub["SNc"] | pub["VTA"], iterations=8)
        for name, sel in (("whole-brain tissue", tissue), ("CP", pub["CP"]), ("midbrain (SNc + VTA + 200 um)", mb)):
            h, _ = np.histogram(rel[sel], bins=BINS, density=True)
            hist += [dict(sample_id=s, compartment=name, bin_lo=lo, bin_hi=hi, density=v)
                     for lo, hi, v in zip(BINS[:-1], BINS[1:], h)]
        print(f"done {s}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "signal_vs_agreement.csv", index=False)
    pd.DataFrame(hist).to_csv(OUT / "histograms.csv", index=False)

    lv, lb = np.log10(d.volume_mm3), np.log10(d.rel_mean)
    st = [dict(test="Spearman rho, Dice vs log volume", value=stats.spearmanr(d.dice_3d, lv).statistic),
          dict(test="Spearman rho, Dice vs log brightness (primary)", value=stats.spearmanr(d.dice_3d, lb).statistic),
          dict(test="Spearman rho, log volume vs log brightness", value=stats.spearmanr(lv, lb).statistic),
          dict(test="Spearman rho, Dice vs fraction bright (secondary)",
               value=stats.spearmanr(d.dice_3d, d.frac_bright).statistic),
          dict(test="Spearman rho, Dice vs neighbourhood brightness (secondary)",
               value=stats.spearmanr(d.dice_3d, np.log10(d.rel_mean_nbhd)).statistic)]
    # partial Spearman: rank-residualise
    def resid(y, x):
        y, x = stats.rankdata(y), stats.rankdata(x)
        b = np.polyfit(x, y, 1)
        return y - np.polyval(b, x)
    st.append(dict(test="partial Spearman, Dice vs brightness | volume",
                   value=stats.pearsonr(resid(d.dice_3d, lv), resid(lb, lv)).statistic))
    st.append(dict(test="partial Spearman, Dice vs volume | brightness",
                   value=stats.pearsonr(resid(d.dice_3d, lb), resid(lv, lb)).statistic))
    # OLS on logit(Dice) with brain fixed effects
    y = np.log(np.clip(d.dice_3d, 0.01, 0.99) / (1 - np.clip(d.dice_3d, 0.01, 0.99)))
    z = lambda v: (v - v.mean()) / v.std()  # noqa: E731
    X = np.column_stack([np.ones(len(d)), z(lv), z(lb)] +
                        [(d.sample_id == s).astype(float) for s in SAMPLES[1:]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    sigma2 = res @ res / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))
    st += [dict(test="OLS logit(Dice): coef per SD log volume", value=beta[1], se=se[1]),
           dict(test="OLS logit(Dice): coef per SD log brightness", value=beta[2], se=se[2]),
           dict(test="OLS logit(Dice): R2 (with brain fixed effects)", value=r2)]
    st = pd.DataFrame(st)
    st.to_csv(OUT / "stats.csv", index=False)
    pd.set_option("display.width", 220)
    print(st.round(3).to_string(index=False))
    print(d.groupby("region")[["dice_3d", "volume_mm3", "rel_mean", "frac_bright", "rel_mean_nbhd"]].median()
          .sort_values("dice_3d").round(3).to_string())


if __name__ == "__main__":
    main()
