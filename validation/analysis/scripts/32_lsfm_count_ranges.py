"""Original LSFM SNpc counts: how much single registrations vary vs how much the consensus varies across QC rules.

For each of the 10 sample x input combinations:
  1. range of SNpc cell counts over all 48 single registrations of the sweep (no QC), from the paper's per-registration
     table (dataset_qc_comparison/original_lsfm_full_final_results_with_ssim.csv, {raw,preprocessed}_snpc_CellCount);
  2. range of the 0.5-consensus count over the eight QC rules, excluding rules that leave no registrations
     (QC_REGIME_FULLRES_AXIAL_MASKS/counts.csv, script 24);
  3. change of the consensus count under the untuned starting point (median ± 3·MAD) relative to the current rule.
Read-only. Outputs: REVISION_OUTPUTS_FOCUSED/source_data/LSFM_count_ranges_single_vs_consensus.csv (+ _summary.md)
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)
from pathlib import Path

import numpy as np
import pandas as pd

PRA = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis"))
SRC = PRA / "REVISION_OUTPUTS_FOCUSED" / "source_data"
UNTUNED = "Untuned start (median ± 3·MAD)"


def main():
    lab = pd.read_csv(SRC / "sample_number_mapping.csv")
    lab = dict(zip(lab.sample_id.astype(str), lab.label))
    t = pd.read_csv(PRA / "dataset_qc_comparison" / "original_lsfm_full_final_results_with_ssim.csv")
    t["Sample ID"] = t["Sample ID"].astype(str)
    cn = pd.read_csv(PRA / "QC_REGIME_FULLRES_AXIAL_MASKS" / "counts.csv", dtype={"sample": str})
    assert len(cn) == 80 and cn.rule.nunique() == 8, (len(cn), cn.rule.unique())
    rows = []
    for s in sorted(cn["sample"].unique(), key=lambda x: lab[x]):
        g = t[t["Sample ID"] == s]
        assert len(g) == 48, (s, len(g))
        for v in ("raw", "preprocessed"):
            single = g[f"{v}_snpc_CellCount"].astype(float)
            assert single.notna().all()
            c = cn[(cn["sample"] == s) & (cn.input == v)].set_index("rule")
            ok = c[(c.n_registrations.fillna(0) > 0) & (c.cell_count.fillna(0) > 0)]
            cur, unt = c.loc["Current", "cell_count"], c.loc[UNTUNED, "cell_count"]
            rows.append(dict(
                sample_id=s, sample_label=lab[s], input=v,
                single_n=len(single), single_min=single.min(), single_max=single.max(),
                single_range=single.max() - single.min(),
                single_range_pct_of_current=100 * (single.max() - single.min()) / cur,
                single_max_over_min=single.max() / single.min() if single.min() > 0 else np.nan,
                single_min_registration=g.loc[single.idxmin(), ["bending", "grid"]].tolist(),
                single_max_registration=g.loc[single.idxmax(), ["bending", "grid"]].tolist(),
                consensus_rules_used=len(ok), consensus_rules_excluded_empty=", ".join(c.index.difference(ok.index)),
                consensus_min=ok.cell_count.min(), consensus_max=ok.cell_count.max(),
                consensus_min_rule=ok.cell_count.idxmin(), consensus_max_rule=ok.cell_count.idxmax(),
                consensus_range=ok.cell_count.max() - ok.cell_count.min(),
                consensus_range_pct_of_current=100 * (ok.cell_count.max() - ok.cell_count.min()) / cur,
                current_count=cur, current_n_registrations=int(c.loc["Current", "n_registrations"]),
                untuned_count=unt, untuned_n_registrations=int(c.loc[UNTUNED, "n_registrations"]),
                untuned_change=unt - cur, untuned_change_pct=100 * (unt / cur - 1)))
    d = pd.DataFrame(rows)
    d["consensus_range_over_single_range"] = d.consensus_range / d.single_range
    d.to_csv(SRC / "LSFM_count_ranges_single_vs_consensus.csv", index=False)

    md = ["# Original LSFM SNpc counts: single registrations vs consensus across QC rules", "",
          "Script `validation_analysis/scripts/32_lsfm_count_ranges.py`; data in "
          "`LSFM_count_ranges_single_vs_consensus.csv`.", "",
          "| Sample | Input | Single registrations (n = 48): range | Spread (% of current) | "
          "Consensus over QC rules: range | Spread (% of current) | Rules used | Untuned start vs current |",
          "|---|---|---|---|---|---|---|---|"]
    for r in d.itertuples():
        md.append(f"| {r.sample_label} | {r.input} | {r.single_min:,.0f}–{r.single_max:,.0f} | "
                  f"{r.single_range_pct_of_current:.0f} % | {r.consensus_min:,.0f}–{r.consensus_max:,.0f} | "
                  f"{r.consensus_range_pct_of_current:.0f} % | {r.consensus_rules_used}/8 | "
                  f"{r.untuned_change:+,.0f} ({r.untuned_change_pct:+.1f} %) |")
    md += ["", f"Medians over the 10 combinations: single-registration spread "
               f"{d.single_range_pct_of_current.median():.0f} % of the current count "
               f"(range {d.single_range_pct_of_current.min():.0f}–{d.single_range_pct_of_current.max():.0f} %); "
               f"consensus spread across QC rules {d.consensus_range_pct_of_current.median():.0f} % "
               f"(range {d.consensus_range_pct_of_current.min():.0f}–{d.consensus_range_pct_of_current.max():.0f} %); "
               f"untuned start vs current {d.untuned_change_pct.median():+.1f} % "
               f"(range {d.untuned_change_pct.min():+.1f} to {d.untuned_change_pct.max():+.1f} %).", ""]
    (SRC / "LSFM_count_ranges_single_vs_consensus_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
