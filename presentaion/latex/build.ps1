# Build Beamer presentation: pdflatex -> biber -> pdflatex x2
# Usage: .\build.ps1           (normal build)
#        .\build.ps1 -Setup    (install packages)
param([switch]$Setup)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Find-MiKTeXBin {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64",
        "$env:ProgramFiles\MiKTeX\miktex\bin\x64",
        "${env:ProgramFiles(x86)}\MiKTeX\miktex\bin\x64"
    )
    foreach ($dir in $candidates) {
        if (Test-Path (Join-Path $dir "pdflatex.exe")) { return $dir }
    }
    return $null
}

$miktexBin = Find-MiKTeXBin
if ($miktexBin -and -not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
    $env:PATH = "$miktexBin;$env:PATH"
    Write-Host "[build] Using MiKTeX from $miktexBin"
}

if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
    Write-Host "pdflatex not found. Install MiKTeX: winget install MiKTeX.MiKTeX" -ForegroundColor Yellow
    exit 1
}

$env:MIKTEX_ENABLE_INSTALLER = "t"

if ($Setup) {
    $packages = @(
        "beamer", "babel-english", "booktabs", "siunitx", "pgf",
        "biblatex", "biber", "metropolis", "appendixnumberbeamer",
        "csquotes", "microtype", "grffile"
    )
    foreach ($pkg in $packages) {
        $prev = $ErrorActionPreference
        $ErrorActionPreference = "SilentlyContinue"
        & miktex packages install $pkg 2>$null
        $ErrorActionPreference = $prev
    }
    Write-Host "[build] Setup complete."
    exit 0
}

function Invoke-LaTeX {
    param([string]$Label, [string[]]$Args = @("-interaction=nonstopmode", "main.tex"))
    Write-Host "[build] $Label..."
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & pdflatex @Args 2>&1 | Out-Null
    $ErrorActionPreference = $prev
    if (-not (Test-Path "main.log")) { throw "pdflatex failed: $Label" }
    $fatal = Select-String -Path "main.log" -Pattern "Fatal error occurred" -Quiet
    if ($fatal) {
        Get-Content "main.log" -Tail 35 | ForEach-Object { Write-Host $_ }
        throw "pdflatex failed: $Label"
    }
}

try {
    Invoke-LaTeX "pdflatex pass 1"
    if (Get-Command biber -ErrorAction SilentlyContinue) {
        Write-Host "[build] biber..."
        $prev = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & biber main 2>&1 | Out-Null
        $ErrorActionPreference = $prev
    } else {
        Write-Host "[build] biber not found - skipping bibliography pass" -ForegroundColor Yellow
    }
    Invoke-LaTeX "pdflatex pass 2"
    Invoke-LaTeX "pdflatex pass 3"
}
catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    if (Test-Path "main.log") { Get-Content "main.log" -Tail 35 }
    exit 1
}

if (Test-Path "main.pdf") {
    Write-Host "[build] Success: $((Resolve-Path main.pdf).Path)" -ForegroundColor Green
} else {
    Write-Host "[build] main.pdf not created." -ForegroundColor Red
    exit 1
}
