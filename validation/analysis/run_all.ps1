# Runs the validation analysis end to end (spatialdata_env). Inputs are read-only; outputs stay in this folder.
# Caches (cache/) are reused if present; delete them to rebuild from the original data.
$py = "python"
Set-Location (Join-Path $PSScriptRoot "scripts")
$steps = "00_provenance.py", "01_cache_amap.py", "02_human_loo.py", "03_amap_metrics.py", "04_amap_analyses.py",
         "06_lsfm_cache.py", "07_lsfm_analysis.py", "09_scale_analysis.py", "10_residual_ambiguity.py", "08_figures.py"
foreach ($s in $steps) {
    Write-Output "===== $s ====="
    & $py -W ignore $s
    if ($LASTEXITCODE -ne 0) { throw "$s failed" }
}
