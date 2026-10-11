"""Storage roots for every script in this repository.

The analyses were run on an HPC cluster with two storage areas (project storage and per-user scratch), which are also
mounted as network drives on Windows workstations. Instead of hard-coded absolute paths, the scripts write paths with
placeholders that are resolved here at run time:

    <IMAGING>       project / imaging storage root      e.g. /data/project        or  V:\\
    <SCRATCH>       your scratch folder                 e.g. /scratch/users/me    or  U:\\users\\me
    <SCRATCH_ROOT>  scratch root                        e.g. /scratch             or  U:\\
    <HOME>          your home folder
    <DATA>          the downloaded data deposit (data_availability folder)
    <REPO>          this repository (resolved automatically)

Set them with environment variables (UAA_IMAGING_ROOT, UAA_SCRATCH, UAA_SCRATCH_ROOT, UAA_HOME) or in
config/local_paths.yaml (copy config/local_paths.example.yaml; that file is git-ignored).

Tables written on the cluster (job tables, QC tables) store cluster paths. List the cluster prefixes under
`cluster_prefixes` in local_paths.yaml and `to_local()` maps them to the local mounts, so the same tables can be read
on a workstation.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent
_ENV = {"<IMAGING>": "UAA_IMAGING_ROOT", "<SCRATCH>": "UAA_SCRATCH", "<SCRATCH_ROOT>": "UAA_SCRATCH_ROOT",
        "<HOME>": "UAA_HOME", "<DATA>": "UAA_DATA_ROOT", "<REPO>": "UAA_REPO"}
_KEYS = {"<IMAGING>": "imaging_root", "<SCRATCH>": "scratch", "<SCRATCH_ROOT>": "scratch_root", "<HOME>": "home",
         "<DATA>": "data_root", "<REPO>": "repo"}
_CFG = None


def _config() -> dict:
    global _CFG
    if _CFG is None:
        _CFG = {}
        f = Path(os.environ.get("UAA_LOCAL_PATHS", REPO / "config" / "local_paths.yaml"))
        if f.exists():
            import yaml
            _CFG = yaml.safe_load(f.read_text()) or {}
    return _CFG


def root(placeholder: str) -> str:
    """Configured value of one placeholder (environment variable first, then local_paths.yaml)."""
    v = os.environ.get(_ENV[placeholder]) or _config().get(_KEYS[placeholder])
    if v is None and placeholder == "<SCRATCH_ROOT>":
        v = root("<SCRATCH>")
    if v is None and placeholder == "<HOME>":
        v = str(Path.home())
    if v is None and placeholder == "<REPO>":
        v = str(REPO)
    if v is None:
        raise RuntimeError(f"{placeholder} is not configured: set {_ENV[placeholder]} or '{_KEYS[placeholder]}' in "
                           f"config/local_paths.yaml (see config/local_paths.example.yaml)")
    return str(v).rstrip("\\/")


def P(s: str) -> str:
    """Resolve placeholders in a path string and use the separators of the current operating system."""
    s = str(s)
    for ph in ("<SCRATCH_ROOT>", "<SCRATCH>", "<IMAGING>", "<HOME>", "<DATA>", "<REPO>"):
        if ph in s:
            s = s.replace(ph, root(ph))
    return s.replace("/", "\\") if os.name == "nt" else s.replace("\\", "/")


def to_cluster(s) -> str:
    """Inverse of to_local(): a local path written as the cluster sees it (for HPC job tables). Uses the first
    cluster prefix listed for each placeholder in cluster_prefixes; paths outside the configured roots are returned
    with forward slashes only."""
    s = str(s).replace("\\", "/")
    seen = set()
    for cluster, placeholder in (_config().get("cluster_prefixes") or {}).items():
        if placeholder in seen:
            continue
        seen.add(placeholder)
        try:
            local = root(placeholder).replace("\\", "/")
        except RuntimeError:
            continue
        if s.lower().startswith(local.lower() + "/"):
            return cluster.rstrip("/") + s[len(local):]
    return s


def to_local(s) -> str | None:
    """Map a stored path (placeholder form, or a cluster path listed in cluster_prefixes) to the local file system."""
    if s is None:
        return None
    s = str(s)
    for cluster, placeholder in (_config().get("cluster_prefixes") or {}).items():
        if s.startswith(cluster):
            s = placeholder + s[len(cluster):]
            break
    return P(s)
