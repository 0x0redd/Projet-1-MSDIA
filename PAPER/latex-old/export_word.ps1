# Export IEEE LaTeX paper to Word (.docx)
# Usage: .\export_word.ps1
#        .\export_word.ps1 -Output main_submission.docx
param(
    [string]$Output = "main.docx"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "[export] Installing Python deps (pypandoc-binary, pymupdf)..." -ForegroundColor Cyan
python -m pip install -q pypandoc-binary pymupdf

if (-not (Test-Path "main.tex")) {
    throw "main.tex not found. Run from PAPER\latex"
}

if (-not (Test-Path "main.pdf")) {
    Write-Host "[export] main.pdf missing - run .\build.ps1 first for up-to-date bibliography (main.bbl)." -ForegroundColor Yellow
}

python export_to_word.py -o $Output 2>&1 | Tee-Object -Variable exportLog | Write-Host
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$outPath = $null
$successLine = $exportLog | Where-Object { $_ -match '^\[export\] Success:' } | Select-Object -Last 1
if ($successLine -match 'Success:\s+(.+\.docx)\s*$') {
    $outPath = $Matches[1].Trim()
}
if (-not $outPath -or -not (Test-Path $outPath)) {
    $outPath = Join-Path $PSScriptRoot $Output
    if (-not (Test-Path $outPath)) {
        $fallback = Join-Path $PSScriptRoot "main_export.docx"
        if (Test-Path $fallback) { $outPath = $fallback }
    }
}

if (Test-Path $outPath) {
    $sizeMB = [math]::Round((Get-Item $outPath).Length / 1MB, 2)
    Write-Host ("[export] Word document: {0} ({1} MB)" -f $outPath, $sizeMB) -ForegroundColor Green
    if ($Output -eq "main.docx" -and $outPath -like "*main_export*") {
        Write-Host "[export] Close main.docx in Word if open, then re-run to overwrite main.docx." -ForegroundColor Yellow
    }
}
