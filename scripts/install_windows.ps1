<#
.SYNOPSIS
  Create a virtual environment and install DocIntel's Python dependencies.

.DESCRIPTION
  Does NOT install Tesseract or Ghostscript (native programs); see README for those.
  Run from the project folder:
      powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1
      powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1 -WithOptional -WithDev
      powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1 -VenvPath "$env:USERPROFILE\.venvs\docintel"

.PARAMETER VenvPath
  Where to create the virtual environment (default: .venv in the project folder).
  If the project lives in OneDrive, prefer a path outside OneDrive to avoid syncing
  thousands of package files.
#>
[CmdletBinding()]
param(
    [string]$VenvPath = (Join-Path (Split-Path $PSScriptRoot -Parent) ".venv"),
    [string]$PythonVersion = "",
    [switch]$WithOptional,
    [switch]$WithPaddle,
    [switch]$WithDev
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

function Find-Python {
    $candidates = @()
    if ($PythonVersion) { $candidates += ,@("py", "-$PythonVersion") }
    $candidates += ,@("py", "-3.11")
    $candidates += ,@("py", "-3.12")
    $candidates += ,@("py", "-3.10")
    $candidates += ,@("python")
    foreach ($c in $candidates) {
        try {
            $exe = $c[0]; $rest = @($c | Select-Object -Skip 1)
            $ver = & $exe @rest -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $ver -match '^3\.(10|11|12)$') { return ,$c }
        } catch { }
    }
    throw "No compatible Python (3.10-3.12) found. Install Python 3.11 from https://www.python.org/downloads/windows/ (tick 'Add python.exe to PATH')."
}

$py = Find-Python
$pyExe = $py[0]; $pyArgs = @($py | Select-Object -Skip 1)
Write-Host "Using Python: $(& $pyExe @pyArgs --version)"

if (-not (Test-Path (Join-Path $VenvPath "Scripts\python.exe"))) {
    Write-Host "Creating virtual environment at $VenvPath"
    & $pyExe @pyArgs -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
}
$venvPy = Join-Path $VenvPath "Scripts\python.exe"

& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Installing requirements.txt failed." }
if ($WithOptional) { & $venvPy -m pip install -r requirements-optional.txt; if ($LASTEXITCODE -ne 0) { throw "Optional install failed." } }
if ($WithDev) { & $venvPy -m pip install -r requirements-dev.txt; if ($LASTEXITCODE -ne 0) { throw "Dev install failed." } }
if ($WithPaddle) { & $venvPy -m pip install -r requirements-paddle.txt; if ($LASTEXITCODE -ne 0) { throw "PaddleOCR install failed." } }

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example (edit it to set the Tesseract path or defaults)."
}

Write-Host ""
Write-Host "Checking environment..."
& $venvPy scripts\check_environment.py
Write-Host ""
Write-Host "Activate the environment with:  & '$VenvPath\Scripts\Activate.ps1'"
Write-Host "Start the app with:             powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1 -VenvPath '$VenvPath'"
