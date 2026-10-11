"""Shared paths, loaders, metrics and statistics for the validation analysis package.

Everything here is read-only with respect to the original data: inputs come from the aMAP reproduction outputs
(``aMAP_validation_python_outputs``), the uncertainty-aware pipeline's ``results`` folder and the LSFM study
(``<IMAGING>\\Ishrat\\JoM``). Outputs are written only under ``validation_analysis``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import ndimage as ndi
from scipy import stats

VA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(next(p for p in VA.parents if (p / "uaa_paths.py").exists())))
from uaa_paths import P  # noqa: E402  (resolves <IMAGING>/<REPO> placeholders in config.yaml)

CFG = yaml.safe_load((VA / "config.yaml").read_text())
CFG["paths"] = {k: P(v) for k, v in CFG["paths"].items()}
AMAP_PY = Path(CFG["paths"]["amap_python"])
if str(AMAP_PY) not in sys.path:
    sys.path.insert(0, str(AMAP_PY))

from amap_validation import compare as C  # noqa: E402
from amap_validation.qc import PAPER_NAME, REGION_ORDER  # noqa: E402
from amap_validation.scoring import bwperim8, dice  # noqa: E402

PIPE = Path(CFG["paths"]["pipeline_results"])
AMAP_OUT = Path(CFG["paths"]["amap_outputs"])
LSFM = Path(CFG["paths"]["lsfm_root"])
CACHE = VA / "cache"
TABLES = VA / "results" / "tables"
FIG_MAIN = VA / "figures" / "main"
FIG_SUPP = VA / "figures" / "supplementary"
SRC = VA / "source_data"
for _d in (CACHE, TABLES, FIG_MAIN, FIG_SUPP, SRC, VA / "text"):
    _d.mkdir(parents=True, exist_ok=True)

VOX = 25.0
BRAINS = ["MV_Ntsr1_165", "MV_Ntsr1_169", "CR_Syt6CreRfp_35", "ER_Glt25d2Cre_9", "AB_Gad67_223", "MV131017_7"]

# QC thresholds are taken from the pipeline's own config (single source of truth; the pipeline was recalibrated on
# 2026-09-29 from grad_p95 < 0.045 to < 0.058). config.yaml values are only a fallback.
_PC = yaml.safe_load(Path(CFG["paths"]["pipeline_config"]).read_text())["qc"]
CFG["amap"]["iou_threshold"] = float(_PC["iou_threshold"])
CFG["amap"]["grad_p95_threshold"] = float(_PC["grad_p95_threshold"])
CFG["amap"]["ssim_top_fraction"] = float(_PC["ssim_top_fraction"])


def _qc_brains():
    """Brains with at least one raw registration passing global QC (data-driven, not hard-coded)."""
    q = pd.read_csv(PIPE / "qc_tables" / "qc_results.csv", usecols=["sample_id", "iou", "grad_p95"])
    ok = q[(q.iou > CFG["amap"]["iou_threshold"]) & (q.grad_p95 < CFG["amap"]["grad_p95_threshold"])]
    return [b for b in BRAINS if ok.sample_id.str.startswith(C.SAMPLE_PREFIX[b]).any()]


QC_BRAINS = _qc_brains()
REJECTED_BRAINS = [b for b in BRAINS if b not in QC_BRAINS]
SHORT = {"MV_Ntsr1_165": "165", "MV_Ntsr1_169": "169", "CR_Syt6CreRfp_35": "CR", "ER_Glt25d2Cre_9": "ER",
         "AB_Gad67_223": "AB", "MV131017_7": "MV7"}
STRUCTS = [PAPER_NAME[r] for r in REGION_ORDER]            # ACA SSp RSP VISp VISam SUB AHN MV VPM DG-sg
STRUCTS_MAIN = [s for s in STRUCTS if s != "DG-sg"]
DEFAULT_RUN = CFG["amap"]["default_run"]
RNG_SEED = int(CFG["seed"])


def amap_brain(sample_id: str) -> str:
    """Pipeline sample_id (e.g. 'MV_Ntsr1_165_StitchedImagesPaths_...') -> brain name as in BRAINS."""
    hits = [b for b in BRAINS if str(sample_id).startswith(C.SAMPLE_PREFIX[b])]
    if len(hits) != 1:
        raise ValueError(f"cannot map aMAP sample_id {sample_id!r} to one brain: {hits}")
    return hits[0]


def log(*a):
    print(*a, flush=True)


def case_path(brain, structure, variant):
    return CACHE / "amap" / brain / f"{structure}_{variant}.npz"


def load_case(brain, structure, variant):
    """Unpack one cached case (01_cache_amap.py) -> dict; masks/h1/h2 are bool (n_runs, nA, H, W)."""
    z = np.load(case_path(brain, structure, variant), allow_pickle=True)  # own cache (rater names are objects)
    shp = tuple(int(s) for s in z["shape"])
    n = int(np.prod(shp))
    d = {k: z[k] for k in z.files if k not in ("masks", "h1", "h2")}
    for k in ("masks", "h1", "h2"):
        d[k] = np.unpackbits(z[k])[:n].reshape(shp).astype(bool)
    d["runs"] = [str(r) for r in d["runs"]]
    d["A_list"] = [int(a) for a in d["A_list"]]
    d["A0"], d["label"] = int(d["A0"]), int(d["label"])
    return d


def run_params(run: str):
    """'bend_0p95__grid_m10__smooth_m1p0' -> (0.95, -10)."""
    parts = run.split("__")
    be = float(parts[0].replace("bend_", "").replace("p", "."))
    g = -float(parts[1].replace("grid_m", "").replace("p", "."))
    return be, g


# -------------------------------------------------------------------------------------------------------------
# 2-D metrics (25 um pixels)
# -------------------------------------------------------------------------------------------------------------
def surface_distances(a: np.ndarray, b: np.ndarray):
    """Pooled symmetric boundary-to-boundary distances (um) between two 2-D masks (8-connected perimeters)."""
    a, b = a > 0, b > 0
    if not a.any() or not b.any():
        return None
    pa, pb = bwperim8(a), bwperim8(b)
    da = ndi.distance_transform_edt(~pb)[pa]
    db = ndi.distance_transform_edt(~pa)[pb]
    return np.concatenate([da, db]) * VOX


def seg_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    d = surface_distances(pred, truth)
    ap, at = float((pred > 0).sum()), float((truth > 0).sum())
    return dict(dice=dice(pred, truth) if ap + at else np.nan,
                hd_um=float(d.max()) if d is not None else np.nan,
                hd95_um=float(np.percentile(d, 95)) if d is not None else np.nan,
                msd_um=float(d.mean()) if d is not None else np.nan,
                area_ratio=ap / at if at else np.nan)


def dice_many(stack: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Dice of each 2-D mask in (n, H, W) against one truth mask."""
    t = truth.reshape(-1).astype(np.float32)
    s = stack.reshape(len(stack), -1).astype(np.float32)
    inter = s @ t
    den = s.sum(1) + t.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, 2 * inter / den, np.nan)


def binary_entropy(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


# -------------------------------------------------------------------------------------------------------------
# statistics (brains are the independent unit)
# -------------------------------------------------------------------------------------------------------------
def boot_brain(df: pd.DataFrame, col: str, stat=np.median, n=None, by="brain", seed=RNG_SEED):
    """Point estimate and 95 % percentile CI of stat(df[col]) with brains resampled with replacement."""
    n = n or int(CFG["bootstrap_n"])
    df = df.dropna(subset=[col])
    groups = {k: g[col].to_numpy() for k, g in df.groupby(by)}
    keys = list(groups)
    if not keys:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    est = float(stat(df[col].to_numpy()))
    vals = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys), replace=True)
        vals.append(stat(np.concatenate([groups[keys[i]] for i in pick])))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return est, float(lo), float(hi)


def boot_spearman(x, y, groups, n=None, seed=RNG_SEED):
    """Spearman rho with a brain-resampled 95 % CI."""
    n = n or int(CFG["bootstrap_n"])
    d = pd.DataFrame(dict(x=x, y=y, g=groups)).dropna()
    if len(d) < 4 or d.x.nunique() < 2 or d.y.nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan
    r = stats.spearmanr(d.x, d.y)
    keys = d.g.unique()
    idx = {k: np.flatnonzero(d.g.to_numpy() == k) for k in keys}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sel = np.concatenate([idx[k] for k in rng.choice(keys, len(keys), replace=True)])
        xs, ys = d.x.to_numpy()[sel], d.y.to_numpy()[sel]
        if np.unique(xs).size > 1 and np.unique(ys).size > 1:
            vals.append(stats.spearmanr(xs, ys).statistic)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return float(r.statistic), float(r.pvalue), float(lo), float(hi)


def centred_spearman(df, x, y, factors=("brain", "structure")):
    """Spearman after removing additive brain and structure effects from ranks (two-way fixed-effects check,
    used in place of a mixed model because statsmodels is not installed)."""
    d = df[[x, y, *factors]].dropna().copy()
    if len(d) < 6:
        return np.nan
    for v in (x, y):
        d[v] = d[v].rank()
        for _ in range(20):  # alternating projections = additive two-way demeaning
            for f in factors:
                d[v] = d[v] - d.groupby(f)[v].transform("mean")
    return float(stats.pearsonr(d[x], d[y]).statistic) if d[x].std() > 0 and d[y].std() > 0 else np.nan


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    order = np.argsort(pv)
    adj = np.empty_like(pv)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (len(pv) - k) * pv[i])
        adj[i] = min(1.0, running)
    out[ok] = adj
    return out


def wilcoxon_p(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x) & (x != 0)]
    return float(stats.wilcoxon(x).pvalue) if len(x) >= 5 else np.nan


def save_table(df: pd.DataFrame, name: str):
    p = TABLES / f"{name}.csv"
    df.to_csv(p, index=False)
    return p


def load_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv")


NUMBERS_FILE = TABLES / "numbers.json"


def record(key: str, value):
    """Key numbers quoted in the text drafts are recorded here, so every number is traceable to a script."""
    d = json.loads(NUMBERS_FILE.read_text()) if NUMBERS_FILE.exists() else {}
    d[key] = value
    NUMBERS_FILE.write_text(json.dumps(d, indent=1, default=float))
