param(
    [string]$InDir = "C:\Users\awand\Desktop\reddit_pushshift_dump\subreddits24",
    [string]$OutParquetDir = "data/parquet",
    [string]$OutCsvDir = "",
    [string]$Start = "2005-01-01",
    [string]$End = "2025-06-30",
    [ValidateSet("comments", "submissions", "both")]
    [string]$Mode = "both",
    [string]$IdeologyMap = "config/subreddits.yaml",
    [string]$Keywords = "config/keywords.yaml",
    [string]$NegFilters = "config/neg_filters.yaml",
    [string]$LogLevel = "INFO"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$pythonScript = Join-Path $repoRoot "scripts/ingest_from_pushshiftdumps.py"

if (-not (Test-Path $pythonScript)) {
    throw "Unable to locate ingest_from_pushshiftdumps.py at $pythonScript"
}

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw "Python executable not found on PATH."
}

$arguments = @(
    $pythonScript,
    "--in_dir", $InDir,
    "--out_parquet_dir", $OutParquetDir,
    "--start", $Start,
    "--end", $End,
    "--mode", $Mode,
    "--ideology_map", $IdeologyMap,
    "--keywords", $Keywords,
    "--neg_filters", $NegFilters,
    "--log_level", $LogLevel
)

if ($OutCsvDir -and $OutCsvDir.Trim().Length -gt 0) {
    $arguments += @("--out_csv_dir", $OutCsvDir)
}

Write-Host "Launching ingest pipeline..."
Write-Host "python $($arguments -join ' ')"

$process = Start-Process -FilePath $python.Source -ArgumentList $arguments -Wait -NoNewWindow -PassThru

if ($process.ExitCode -ne 0) {
    throw "Ingest pipeline failed with exit code $($process.ExitCode)"
}
