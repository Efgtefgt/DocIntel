<#
.SYNOPSIS
  Start the DocIntel Streamlit app on http://localhost:8501 (this computer only).
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1
  powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1 -VenvPath "$env:USERPROFILE\.venvs\docintel" -Port 8502
#>
[CmdletBinding()]
param(
    [string]$VenvPath = (Join-Path (Split-Path $PSScriptRoot -Parent) ".venv"),
    [int]$Port = 8501
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$venvPy = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $venvPy)) { throw "Virtual environment not found at $VenvPath. Run scripts\install_windows.ps1 first." }
& $venvPy -m streamlit run app.py --server.port $Port --server.address 127.0.0.1
