"""Readable summary of the data-release provenance table, plus the private ID maps (written next to it, never into
the release folder).

    python tools/summarise_provenance.py --prov <IMAGING>/Ishrat/JoM/post_review_analysis/data_availability_PROVENANCE.csv
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uaa_paths import P  # noqa: E402

IDS = ["162", "527", "pp451", "xl353", "xl385"]


def group_key(rel: str) -> str:
    parts = rel.split("/")
    if parts[0].startswith("Sample"):
        return "SampleN/" + "/".join(parts[1:-1])
    return "/".join(parts[:-1]) or parts[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prov", required=True)
    a = ap.parse_args()
    prov = Path(P(a.prov))
    df = pd.read_csv(prov)
    out = prov.with_suffix(".md")
    lines = ["# Data release: where every file came from", "",
             f"Full table (one row per released file, with source paths and SHA-256): `{prov.name}`. "
             "This file and the table contain internal paths and IDs and are NOT part of the release.", "",
             "## Sample IDs", "", "| Release | Manuscript | Internal ID |", "|---|---|---|"]
    lines += [f"| Sample{i + 1} | Sample {i + 1} | {s} |" for i, s in enumerate(IDS)]
    ann = {}
    for r in df[df.release_path.str.startswith("expert_annotations/")].itertuples():
        m = re.search(r"annotator_\d+", Path(r.release_path).name)
        rel_name = m.group(0) if m else Path(r.release_path).name
        ann[rel_name] = Path(str(r.source_paths).split(" ; ")[0]).parent.name
    lines += ["", "## Expert annotators", "", "| Release | Source folder (annotator) |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(ann.items())]
    lines += ["", "## Files by folder", ""]
    df["group"] = df.release_path.map(group_key)
    for g, sub in df.groupby("group", sort=True):
        ex = sub.iloc[0]
        src = str(ex.source_paths).split(" ; ")[0]
        ops = ", ".join(sorted(set(sub.operation)))
        lines += [f"### `{g}/` ({len(sub)} files, {sub.release_bytes.fillna(0).astype(float).sum() / 1e9:.2f} GB)",
                  f"- operation: {ops}", f"- example source: `{src}`"]
        notes = sorted(set(str(n) for n in sub.notes if isinstance(n, str) and n))
        if notes:
            lines.append(f"- notes: {notes[0]}" + (f" (+{len(notes) - 1} variants)" if len(notes) > 1 else ""))
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame([dict(release_label=f"Sample{i + 1}", manuscript_label=f"Sample {i + 1}", internal_id=s)
                  for i, s in enumerate(IDS)] +
                 [dict(release_label=k, manuscript_label="", internal_id=v) for k, v in sorted(ann.items())]
                 ).to_csv(prov.with_name("data_availability_private_id_map.csv"), index=False)
    print("wrote", out, "and the private ID map")


if __name__ == "__main__":
    main()
