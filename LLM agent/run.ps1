# Run GraphRAG paper agent CLI with the dl conda environment.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
& "C:\Users\FERRA\.conda\envs\dl\python.exe" -m agents.cursor_agent @args
