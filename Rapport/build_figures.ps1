# Regenerate all Rapport figures (matplotlib + TikZ).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$py = "C:\Users\FERRA\.conda\envs\dl\python.exe"
& $py figures/scripts/gen_dataset_figure.py

Push-Location figures/tikz
pdflatex -interaction=nonstopmode preprocessing_pipeline.tex | Out-Null
Copy-Item -Force preprocessing_pipeline.pdf ..\preprocessing_pipeline.pdf
Pop-Location

Write-Host "[build_figures] Done. Outputs in figures/"
