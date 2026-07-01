# Build IEEE paper: pdflatex -> bibtex -> pdflatex x2
# Usage: .\build.ps1           (normal build)
#        .\build.ps1 -Setup    (first-time MiKTeX package install)
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
        if (Test-Path (Join-Path $dir "pdflatex.exe")) {
            return $dir
        }
    }
    return $null
}

$miktexBin = Find-MiKTeXBin
if ($miktexBin -and -not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
    $env:PATH = "$miktexBin;$env:PATH"
    Write-Host "[build] Using MiKTeX from $miktexBin"
}

if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "pdflatex not found. Install MiKTeX with:" -ForegroundColor Yellow
    Write-Host '  winget install MiKTeX.MiKTeX --accept-package-agreements --accept-source-agreements'
    Write-Host "  Or download: https://miktex.org/download"
    Write-Host ""
    Write-Host "After install, close and reopen PowerShell, then run .\build.ps1 again."
    exit 1
}

$env:MIKTEX_ENABLE_INSTALLER = "t"

if ($Setup) {
    $initexmf = Get-Command initexmf -ErrorAction SilentlyContinue
    if ($initexmf) {
        & initexmf --set-config-value "[MPM] AutoInstall=1" 2>$null
        & initexmf --set-config-value "[MPM] AutoAdmin=0" 2>$null
    }
    $miktex = Get-Command miktex -ErrorAction SilentlyContinue
    if ($miktex) {
        $packages = @(
            "ieeetran", "booktabs", "multirow", "caption", "siunitx",
            "cite", "microtype", "translations"
        )
        Write-Host "[build] Installing LaTeX packages (first-time setup)..."
        foreach ($pkg in $packages) {
            $prev = $ErrorActionPreference
            $ErrorActionPreference = "SilentlyContinue"
            & miktex packages install $pkg
            $ErrorActionPreference = $prev
        }
        & initexmf --update-fndb 2>$null
        Write-Host "[build] Setup complete. Run .\build.ps1 to compile."
    }
    exit 0
}

function Invoke-LaTeX {
    param([string]$Label, [string[]]$Args)
    Write-Host "[build] $Label..."
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & pdflatex @Args 2>&1 | Out-Null
    $ErrorActionPreference = $prev
    if (-not (Test-Path "main.log")) {
        throw "pdflatex failed during: $Label (no main.log produced)"
    }
    $fatal = Select-String -Path "main.log" -Pattern "Fatal error occurred" -Quiet
    if ($fatal) {
        Get-Content "main.log" -Tail 25 | ForEach-Object { Write-Host $_ }
        throw "pdflatex failed during: $Label (see main.log)"
    }
}

try {
    Invoke-LaTeX "pdflatex pass 1" @("-interaction=nonstopmode", "main.tex")
    Write-Host "[build] bibtex..."
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & bibtex main 2>&1 | Out-Null
    $ErrorActionPreference = $prev
    Invoke-LaTeX "pdflatex pass 2" @("-interaction=nonstopmode", "main.tex")
    Invoke-LaTeX "pdflatex pass 3" @("-interaction=nonstopmode", "main.tex")
}
catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    if (Test-Path "main.log") {
        Write-Host "[build] Last lines of main.log:" -ForegroundColor Yellow
        Get-Content "main.log" -Tail 25
    }
    exit 1
}

if (Test-Path "main.pdf") {
    $pdfPath = (Resolve-Path "main.pdf").Path
    Write-Host "[build] Success: $pdfPath" -ForegroundColor Green
} else {
    Write-Host "[build] main.pdf was not created. Check main.log for errors." -ForegroundColor Red
    exit 1
}
