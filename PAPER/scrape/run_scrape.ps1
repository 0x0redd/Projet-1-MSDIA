# Run paper scraper with the dl Python env (no conda activate needed).
param(
    [Parameter(Mandatory = $true)]
    [string]$Email,
    [string]$Refs = "",
    [switch]$DryRun,
    [switch]$CopyToPaper,
    [switch]$LookupMissingDoi
)

$Python = "C:\Users\FERRA\.conda\envs\dl\python.exe"
$Script = Join-Path $PSScriptRoot "scrape_papers.py"

if (-not (Test-Path $Python)) {
    Write-Error "Python not found: $Python`nInstall the dl env or edit `$Python in run_scrape.ps1"
    exit 1
}

$argsList = @($Script, "--email", $Email)
if ($Refs) { $argsList += @("--refs", $Refs) }
if ($DryRun) { $argsList += "--dry-run" }
if ($CopyToPaper) { $argsList += "--copy-to-paper" }
if ($LookupMissingDoi) { $argsList += "--lookup-missing-doi" }

& $Python @argsList
