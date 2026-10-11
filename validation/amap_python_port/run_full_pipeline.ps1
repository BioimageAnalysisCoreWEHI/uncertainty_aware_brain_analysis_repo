# Reproduce the whole aMAP annotation pipeline (see PIPELINE.md). Run from anywhere in PowerShell.
# Extra arguments are passed through, e.g.:  .\run_full_pipeline.ps1 --repeats first --out D:\other_outputs
$ErrorActionPreference = "Stop"
$py = "python"   # analysis_env has a broken BLAS; do not use it
Set-Location $PSScriptRoot
$env:PYTHONUNBUFFERED = "1"
& $py -m amap_validation full @args
if ($LASTEXITCODE -ne 0) { throw "pipeline failed with exit code $LASTEXITCODE" }
