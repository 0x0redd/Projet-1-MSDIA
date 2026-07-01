# Convert a PDF to Markdown + references JSON
# Usage: .\run_pdf_to_md.ps1 "path\to\paper.pdf" [output_dir]

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$PdfPath,

    [Parameter(Mandatory = $false, Position = 1)]
    [string]$OutputDir = ""
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

python -m pip install -q pymupdf pymupdf4llm

$args = @("pdf_to_md_refs.py", (Resolve-Path $PdfPath).Path)
if ($OutputDir) {
    $args += @("-o", (Resolve-Path $OutputDir).Path)
}

python @args
