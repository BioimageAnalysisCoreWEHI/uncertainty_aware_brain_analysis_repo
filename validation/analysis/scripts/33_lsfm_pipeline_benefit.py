"""Original LSFM (SNpc TH+ counts): where, why and how much each pipeline step changes the count, and what the
consensus adds beyond the QC filters. No ground truth exists: everything here is spread / stability /
reproducibility, never accuracy.

Counting rule throughout: the manuscript's overlap rule (a cell counts if any voxel is inside the mask), exact from the
voxel-vote cache (07_lsfm_analysis.Counter). A single registration = the count of its own mask.
Pipeline (current rule): global QC IoU > 0.9 & grad_p95 < 0.045 -> SSIM top ceil(n/2) -> consensus >= 0.5.

Per sample x input:
  A  single-registration spread at each stage (all 48 / global QC / + SSIM top half);
  B  consensus of each pool (all 48, global QC, final) and leave-one-registration-out stability of each;
  C  sensitivity of the final consensus to its own choices (8 QC rules; consensus threshold 0.4-0.6, 0.25-0.75);
  D  brain ranking: one random registration per brain from each pool vs the consensus ranking;
  E  fixed parameter for all brains: deviation from the consensus and ranking;
  F  raw vs preprocessed reproducibility: consensus vs same-parameter single registrations.
Read-only. Outputs: REVISION_OUTPUTS_FOCUSED/source_data/LSFM_benefit_*.csv, LSFM_benefit_summary.md,
figures/LSFM_pipeline_benefit.{pdf,svg,png}
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import ast
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import CACHE, VA  # noqa: E402

spec = importlib.util.spec_from_file_location("l07", str(VA / "scripts" / "07_lsfm_analysis.py"))
l07 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l07)

PRA = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis"))
OUT = PRA / "REVISION_OUTPUTS_FOCUSED"
FIG, SRC = OUT / "figures", OUT / "source_data"
ST = ["raw", "preprocessed"]
RULES = ["No global QC", "Very lenient", "Lenient", "Untuned start (median ± 3·MAD)", "Current", "Strict",
         "Very strict", "Median IoU and grad_p95"]
RNG = np.random.default_rng(0)
NDRAW = 20000


def regimes():
    tree = ast.parse((VA / "scripts" / "22_focused_revision_figures.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "REGIMES" for t in node.targets):
            return ast.literal_eval(node.value)


def spread(x):
    x = np.asarray(x, float)
    return dict(n=len(x), mean=x.mean(), sd=x.std(ddof=1) if len(x) > 1 else np.nan,
                cv_pct=100 * x.std(ddof=1) / x.mean() if len(x) > 1 else np.nan, min=x.min(), max=x.max())


def main():
    runs = pd.read_csv(CACHE / "lsfm" / "runs.csv")
    runs["sample"] = runs["sample"].astype(str)
    lab = pd.read_csv(SRC / "sample_number_mapping.csv")
    lab = dict(zip(lab.sample_id.astype(str), lab.label))
    samples = sorted(runs["sample"].unique(), key=lambda s: lab[s])
    reg = regimes()
    st_tab = pd.read_csv(VA / "results" / "qc_sensitivity" / "paper" / "settings.csv").set_index("setting_id")
    rule_thr = {r: (st_tab.loc[reg[r]["paper"], "iou_thr"], st_tab.loc[reg[r]["paper"], "grad_thr"]) for r in RULES}
    C = {(s, v): l07.Counter(s, v) for s in samples for v in ST}

    stage_rows, cons_rows, knob_rows, single = [], [], [], {}
    final_sets, cons_final = {}, {}
    for s in samples:
        for v in ST:
            c = C[(s, v)]
            sub = runs[(runs["sample"] == s) & (runs.status == v)].copy()
            assert len(sub) == 48
            sub["count"] = [c.count([r], 0.5) for r in sub.run_key]
            single[(s, v)] = sub.set_index("run_key")
            pools = {"all 48": list(sub.run_key),
                     "global QC": l07.select(sub, "passed_qc", "all"),
                     "global QC + SSIM (final)": l07.select(sub, "passed_qc", "top_50pct")}
            final_sets[(s, v)] = pools["global QC + SSIM (final)"]
            P = c.count(pools["global QC + SSIM (final)"], 0.5)
            cons_final[(s, v)] = P
            # A + B
            for name, rs in pools.items():
                x = sub.set_index("run_key").loc[rs, "count"]
                cnt = c.count(rs, 0.5)
                loo = [c.count([r for r in rs if r != q], 0.5) for q in rs] if len(rs) > 1 else [np.nan]
                stage_rows.append(dict(sample=s, sample_label=lab[s], input=v, pool=name, **spread(x),
                                       single_range_pct_of_final=100 * (x.max() - x.min()) / P,
                                       single_iqr_pct_of_final=100 * (x.quantile(.75) - x.quantile(.25)) / P,
                                       consensus_count=cnt, consensus_vs_final_pct=100 * (cnt / P - 1),
                                       consensus_loo_maxdev_pct=100 * np.nanmax(np.abs(np.array(loo) / cnt - 1)),
                                       consensus_loo_range_pct=100 * (np.nanmax(loo) - np.nanmin(loo)) / cnt))
            # C: knobs of the final consensus
            rc = {}
            for r, (ti, tg) in rule_thr.items():
                if pd.isna(ti) and pd.isna(tg):
                    rs = l07.select(sub, "all_runs", "top_50pct")
                else:
                    rs = l07.select(sub, "passed_qc", "top_50pct", ti if pd.notna(ti) else -np.inf,
                                    tg if pd.notna(tg) else np.inf)
                rc[r] = c.count(rs, 0.5) if rs else np.nan
                cons_rows.append(dict(sample=s, sample_label=lab[s], input=v, rule=r, n_runs=len(rs), count=rc[r],
                                      change_vs_current_pct=100 * (rc[r] / P - 1) if rs else np.nan))
            ok = [x for x in rc.values() if pd.notna(x)]
            fin = pools["global QC + SSIM (final)"]
            th = {t: c.count(fin, t) for t in (0.25, 0.4, 0.5, 0.6, 0.75)}
            knob_rows.append(dict(sample=s, sample_label=lab[s], input=v, final_count=P, n_final=len(fin),
                                  qc_rule_range_pct=100 * (max(ok) - min(ok)) / P, n_rules_nonempty=len(ok),
                                  qc_rule_range_excl_extremes_pct=100 * (max(rc[r] for r in RULES[2:7] if pd.notna(rc[r]))
                                                                         - min(rc[r] for r in RULES[2:7] if pd.notna(rc[r]))) / P,
                                  thr_0p4_0p6_pct=100 * (th[0.4] - th[0.6]) / P,
                                  thr_0p25_0p75_pct=100 * (th[0.25] - th[0.75]) / P))
    stages = pd.DataFrame(stage_rows)
    rules = pd.DataFrame(cons_rows)
    knobs = pd.DataFrame(knob_rows)
    # check: Counter consensus at the current rule = full-resolution counts.csv (script 24)
    fr = pd.read_csv(PRA / "QC_REGIME_FULLRES_AXIAL_MASKS" / "counts.csv", dtype={"sample": str})
    frc = fr[fr.rule == "Current"].set_index(["sample", "input"]).cell_count
    chk = pd.DataFrame([dict(sample=s, input=v, counter=cons_final[(s, v)], fullres=frc[(s, v)]) for s, v in cons_final])
    chk["diff_pct"] = 100 * (chk.counter / chk.fullres - 1)

    # D: ranking with one random registration per brain, per pool
    rank_rows = []
    for v in ST:
        ref = np.array([cons_final[(s, v)] for s in samples])
        for pool in ("all 48", "global QC", "global QC + SSIM (final)"):
            cand = []
            for s in samples:
                sub = single[(s, v)]
                rs = {"all 48": list(sub.index), "global QC": l07.select(sub.reset_index(), "passed_qc", "all"),
                      "global QC + SSIM (final)": final_sets[(s, v)]}[pool]
                cand.append(sub.loc[rs, "count"].to_numpy(float))
            draws = np.stack([RNG.choice(x, NDRAW) for x in cand], 1)
            taus = np.array([stats.kendalltau(ref, d).statistic for d in draws[:2000]])
            same = (np.argsort(np.argsort(draws, 1), 1) == np.argsort(np.argsort(ref))).all(1)
            pairs = list(itertools.combinations(range(len(samples)), 2))
            flips = np.mean([np.mean(np.sign(draws[:, i] - draws[:, j]) != np.sign(ref[i] - ref[j])) for i, j in pairs])
            rank_rows.append(dict(input=v, pool=pool, p_identical_order=same.mean(), median_kendall_tau=np.median(taus),
                                  p_tau_ge_0p8=(taus >= 0.8).mean(), mean_frac_pairs_flipped=flips))
        # consensus at every rule, and the fixed parameters, as rankings too
        for r in RULES:
            x = rules[(rules.input == v) & (rules.rule == r)].set_index("sample").reindex(samples)["count"]
            if x.notna().all():
                rank_rows.append(dict(input=v, pool=f"consensus, rule: {r}",
                                      p_identical_order=float((x.rank().to_numpy() == stats.rankdata(ref)).all()),
                                      median_kendall_tau=stats.kendalltau(ref, x).statistic))
    ranks = pd.DataFrame(rank_rows)

    # E: one fixed parameter for every brain
    fx = []
    keys = list(single[(samples[0], "raw")].index)
    for k in keys:
        for v in ST:
            cnt = np.array([single[(s, v)].loc[k, "count"] for s in samples], float)
            ref = np.array([cons_final[(s, v)] for s in samples], float)
            pq = [bool(single[(s, v)].loc[k, "pass_final"]) for s in samples]
            inf = [k in final_sets[(s, v)] for s in samples]
            fx.append(dict(param=k, input=v, bending=single[(samples[0], v)].loc[k, "bending"],
                           grid=single[(samples[0], v)].loc[k, "grid"], n_brains_pass_global=sum(pq),
                           n_brains_in_final=sum(inf),
                           median_abs_dev_pct=100 * np.median(np.abs(cnt / ref - 1)),
                           max_abs_dev_pct=100 * np.max(np.abs(cnt / ref - 1)),
                           dev_spread_pct=100 * (np.max(cnt / ref) - np.min(cnt / ref)),
                           kendall_tau=stats.kendalltau(ref, cnt).statistic,
                           identical_order=bool((stats.rankdata(cnt) == stats.rankdata(ref)).all())))
    fixed = pd.DataFrame(fx)

    # F: raw vs preprocessed
    rp = []
    for s in samples:
        a, b = cons_final[(s, "raw")], cons_final[(s, "preprocessed")]
        r_, p_ = single[(s, "raw")], single[(s, "preprocessed")]
        d_all = (r_["count"] - p_["count"]).abs() / ((r_["count"] + p_["count"]) / 2)
        both = r_.pass_final.astype(bool) & p_.pass_final.astype(bool)
        rp.append(dict(sample=s, sample_label=lab[s], consensus_raw=a, consensus_pre=b,
                       consensus_abs_diff_pct=100 * abs(a - b) / ((a + b) / 2),
                       single_same_param_median_abs_diff_pct_all48=100 * d_all.median(),
                       single_same_param_median_abs_diff_pct_qc_both=100 * d_all[both].median(),
                       n_params_qc_both=int(both.sum())))
    rawpre = pd.DataFrame(rp)

    # Fig. 1B-C example of the manuscript: 0.95/-30 (2,628 cells) vs 0.5/-10 in one brain, 27.5 % apart
    ex = []
    for (s, v), sub in single.items():
        if "0.95_-30" in sub.index and int(sub.loc["0.95_-30", "snpc_CellCount"]) == 2628:
            for k in ("0.95_-30", "0.5_-10"):
                r = sub.loc[k]
                ex.append(dict(sample=s, sample_label=lab[s], input=v, param=k, count_table=r.snpc_CellCount,
                               count_overlap=r["count"], iou=r.iou, grad_p95=r.grad_p95,
                               passes_global_qc=bool(r.pass_final), in_final_set=k in final_sets[(s, v)],
                               final_consensus=cons_final[(s, v)],
                               dev_from_final_pct=100 * (r["count"] / cons_final[(s, v)] - 1)))
    example = pd.DataFrame(ex)
    if len(example) == 2:
        a, b = example.count_table.to_numpy(float)
        example["pair_diff_pct_of_larger"] = 100 * abs(a - b) / max(a, b)
        example["pair_diff_pct_of_smaller"] = 100 * abs(a - b) / min(a, b)
    example.to_csv(SRC / "LSFM_benefit_fig1_example.csv", index=False)
    print("FIG 1 EXAMPLE\n", example.round(3).to_string(index=False))

    # default parameter
    dflt = fixed[fixed.param == "0.95_-10"]

    for name, d in (("stages", stages), ("qc_rules", rules), ("knobs", knobs), ("ranking", ranks),
                    ("fixed_parameter", fixed), ("raw_vs_preprocessed", rawpre), ("check_vs_fullres", chk)):
        d.to_csv(SRC / f"LSFM_benefit_{name}.csv", index=False)
    figure(stages, knobs, lab, samples)
    summary(stages, knobs, ranks, fixed, rawpre, dflt, chk)


def figure(stages, knobs, lab, samples):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    cols = {"all 48": "#bdbdbd", "global QC": "#9e9ac8", "global QC + SSIM (final)": "#6a51a3"}
    x0 = 0
    ticks, labels = [], []
    for s in samples:
        for v in ST:
            g = stages[(stages["sample"] == s) & (stages.input == v)].set_index("pool")
            k = knobs[(knobs["sample"] == s) & (knobs.input == v)].iloc[0]
            P = k.final_count
            for j, pool in enumerate(cols):
                r = g.loc[pool]
                ax.plot([x0 + j * 0.16] * 2, [100 * (r["min"] / P - 1), 100 * (r["max"] / P - 1)], color=cols[pool],
                        lw=3.2, solid_capstyle="butt")
            rr = rules_range(s, v)
            ax.plot([x0 + 0.52] * 2, rr, color="#e6550d", lw=3.2, solid_capstyle="butt")
            ax.plot(x0 + 0.52, 0, marker="o", ms=3, color="k")
            ticks.append(x0 + 0.26)
            labels.append(f"{lab[s].replace('Sample ', 'S')}\n{v[:3]}")
            x0 += 1
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(ticks, labels, fontsize=5.5)
    ax.set_ylabel("Count relative to the final consensus (%)", fontsize=6.5)
    ax.tick_params(labelsize=5.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, lw=3, label=f"single registrations: {p}") for p, c in cols.items()]
    h += [Line2D([], [], color="#e6550d", lw=3, label="consensus across the 8 QC rules (non-empty)"),
          Line2D([], [], color="k", ls="", marker="o", ms=3, label="final consensus (current rule)")]
    ax.legend(handles=h, fontsize=5.3, frameon=False, loc="lower left", ncol=2)
    ax.set_title("Original LSFM SNpc: range of counts a user could report at each step", fontsize=7)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"LSFM_pipeline_benefit.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


_RULES = None


def rules_range(s, v):
    global _RULES
    if _RULES is None:
        _RULES = pd.read_csv(SRC / "LSFM_benefit_qc_rules.csv", dtype={"sample": str})
    g = _RULES[(_RULES["sample"] == s) & (_RULES.input == v)].change_vs_current_pct.dropna()
    return [g.min(), g.max()]


def summary(stages, knobs, ranks, fixed, rawpre, dflt, chk):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("CHECK counter vs fullres\n", chk.round(2).to_string(index=False))
    print("\nSTAGES (single-registration spread; consensus of each pool)")
    cols = ["sample_label", "input", "pool", "n", "cv_pct", "single_range_pct_of_final", "consensus_vs_final_pct",
            "consensus_loo_maxdev_pct"]
    print(stages[cols].round(1).to_string(index=False))
    print("\nmedian over the 10 combinations by pool\n",
          stages.groupby("pool")[["n", "cv_pct", "single_range_pct_of_final", "single_iqr_pct_of_final",
                                  "consensus_vs_final_pct", "consensus_loo_maxdev_pct"]].median().round(1).to_string())
    print("\nKNOBS of the final consensus\n", knobs.round(1).to_string(index=False))
    print(knobs[["qc_rule_range_pct", "qc_rule_range_excl_extremes_pct", "thr_0p4_0p6_pct", "thr_0p25_0p75_pct"]]
          .median().round(1).to_string())
    print("\nRANKING\n", ranks.round(3).to_string(index=False))
    f = fixed
    print("\nFIXED PARAMETER (one parameter for all brains)")
    for v in ST:
        g = f[f.input == v]
        gp = g[g.n_brains_pass_global == 5]
        print(v, f"params passing global QC in all 5 brains: {len(gp)}/48;",
              f"their median|dev| median {gp.median_abs_dev_pct.median():.1f}% (range {gp.median_abs_dev_pct.min():.1f}-"
              f"{gp.median_abs_dev_pct.max():.1f}); max|dev| median {gp.max_abs_dev_pct.median():.1f}%;",
              f"identical order {gp.identical_order.mean():.2f}; all 48: identical order {g.identical_order.mean():.2f}, "
              f"median|dev| median {g.median_abs_dev_pct.median():.1f}%")
    print("default\n", dflt.round(3).to_string(index=False))
    print("\nRAW vs PREPROCESSED\n", rawpre.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
