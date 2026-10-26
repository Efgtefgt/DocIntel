<#
.SYNOPSIS
  Start the optional DocIntel REST API on http://127.0.0.1:8000 (docs at /docs).
  Requires requirements-optional.txt. The API has no authentication: keep it on localhost.
#>
[CmdletBinding()]
param(
    [string]$VenvPath = (Join-Path (Split-Path $PSScriptRoot -Parent) ".venv"),
    [int]$Port = 8000
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$venvPy = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $venvPy)) { throw "Virtual environment not found at $VenvPath. Run scripts\install_windows.ps1 first." }
& $venvPy -m uvicorn docintel.api.main:app --host 127.0.0.1 --port $Port --workers 1
