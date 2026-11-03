$ErrorActionPreference = 'Stop'
& python (Join-Path $PSScriptRoot 'entrypoint.py') reproduce @args
if ($LASTEXITCODE -ne 0) { throw "Reproduction failed with exit code $LASTEXITCODE; inspect work/orchestration_logs" }
